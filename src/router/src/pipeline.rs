//! Pipeline orchestrator — sequences stages as `WorkUnit` implementations.
//! Follows the `TierRegistry` pattern from `coral/src/tier_units.rs:528-570`.

use std::sync::Arc;
use std::time::Instant;

use fluent_wvr::prelude::*;
use serde::{Deserialize, Serialize};

use common_core::constants::default_true;

use crate::config::{strip_declaration_params, ModelEntry};

/// Router-local newtype for the `<base>` / `<base>:<qualifier>` model-id grammar
/// (`pipeline.rs:87`, `config.rs:47` `split_model_key`). The wire form is still a
/// single `model` string (OpenAI compat), but in-memory code uses the typed form
/// so bare vs qualified dispatch is checked at the call site rather than by
/// stringly-typed `format!("{base}:{q}")` / `split_once(':')` (Pattern 7,
/// `fluent-wvr/SKILL.md:1107`). Kept router-local per the import-boundary rule:
/// only the router speaks this grammar.
///
/// Ownership: `split_model_key` (`config/root.rs:33`) is the single `split_once(':')`
/// site; `QualifiedModelId::parse` delegates to it. Zero-alloc callers use
/// `split_model_key`; typed callers use `QualifiedModelId::parse`/`as_wire`.
/// Never duplicate the `split_once` literal elsewhere. Model-id parsing is
/// neither confidence nor task-value — no threshold.
///
/// `as_wire` is the single `format!("{base}:{q}")` site.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualifiedModelId {
    pub base: String,
    pub qualifier: Option<String>,
}

impl QualifiedModelId {
    pub fn new(base: impl Into<String>, qualifier: Option<impl Into<String>>) -> Self {
        Self {
            base: base.into(),
            qualifier: qualifier.map(Into::into),
        }
    }
    pub fn bare(base: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            qualifier: None,
        }
    }
    pub fn qualified(base: impl Into<String>, qualifier: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            qualifier: Some(qualifier.into()),
        }
    }
    /// Parse the wire form (`base` or `base:qualifier`) into the typed form.
    /// Mirrors `crate::config::split_model_key` but returns the owned newtype.
    pub fn parse(wire: &str) -> Self {
        match crate::config::split_model_key(wire) {
            (base, Some(q)) => Self {
                base: base.to_string(),
                qualifier: Some(q.to_string()),
            },
            (base, None) => Self {
                base: base.to_string(),
                qualifier: None,
            },
        }
    }
    /// Render the wire form (`base` or `base:qualifier`) — the single format site.
    pub fn as_wire(&self) -> String {
        match &self.qualifier {
            Some(q) => format!("{}:{}", self.base, q),
            None => self.base.clone(),
        }
    }
    pub fn is_qualified(&self) -> bool {
        self.qualifier.is_some()
    }
}

impl std::fmt::Display for QualifiedModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_wire())
    }
}
use crate::pipeline_types::{
    PipelineStage, StageDecision, StageDecisionProducer, StageMetadata, StageVerdict,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTarget {
    pub url: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_name: Option<String>,
    /// Model inference params to merge into the request body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
    /// Instance or group name to route to (explicit request field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// KV snapshot to switch into the target slot before serving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
    /// Slot to target for snapshot switching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_slot: Option<i32>,
    /// Whether to filter thinking blocks from idle timeout.
    #[serde(default)]
    pub filter_thinking: bool,
    /// Number of retry attempts.
    #[serde(default)]
    pub retry_count: u32,
    /// Base interval between retries in seconds.
    #[serde(default = "default_retry_interval")]
    pub retry_base_interval_s: u64,
    /// Whether the backend model supports streaming.
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Maximum idle time between stream chunks in milliseconds.
    #[serde(default = "default_idle_timeout_ms")]
    pub idle_timeout_ms: u64,
    /// Maximum total time for the entire request in milliseconds.
    #[serde(default = "default_total_timeout_ms")]
    pub total_timeout_ms: u64,
    /// Name of an environment variable holding the `Authorization: Bearer`
    /// token for an external OpenAI endpoint. Resolved to the actual token at
    /// dispatch time by the backend; `None` sends no auth header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Ordered fallback targets to try when the primary fails.
    /// Populated at route-resolution time from all available models,
    /// ordered by intelligence proximity to the request complexity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallbacks: Vec<RoutingTarget>,
    /// Whether this target is served by the in-process onnx backend (a role
    /// key like `onnx/llm`) rather than an HTTP model entry. The dispatch layer
    /// routes onnx targets to the onnx `ChatBackend`; `url` is empty for them.
    #[serde(default)]
    pub is_onnx: bool,
}

/// The model id grammar on the wire: `<base>` qualified with a group or exact
/// instance as `<base>:<qualifier>`. The fork resolves the default instance of
/// a `default: true` profile, else a group's first available member.
fn base_target(entry: &ModelEntry, model_key: &str) -> (String, Option<serde_json::Value>, String) {
    let base = entry.name.clone().unwrap_or_else(|| model_key.to_string());
    // The entry-default inference point — the shared step of the single
    // qualifier precedence, so dispatch wire ids agree with backend ids.
    // Routing entries arrive with fleet defaults materialized, hence `None`.
    let qualifier = crate::config::root::default_inference_point(entry, None);
    let model = QualifiedModelId::new(base.clone(), qualifier.clone()).as_wire();
    let params = qualifier
        .as_deref()
        .and_then(|q| entry.instance_params_for(q))
        .or_else(|| entry.params.clone().map(strip_declaration_params));
    (model, params, base)
}

impl RoutingTarget {
    /// Build a routing target from a configured model entry — the canonical
    /// mapping used by every dispatch path (direct-model requests and the
    /// classifier fallback). For a model with an instance pool the `model` id
    /// is qualified to the default dispatch point (`<base>:<qualifier>`), and
    /// declaration-only params (`num_ctx`/`parallel`/`sleep_idle_seconds`/
    /// `rope_freq_base`) are stripped from the body (the fork owns them via the
    /// instance grammar).
    pub fn from_model_entry(model_key: &str, entry: &ModelEntry) -> Self {
        let (model, params, _base) = base_target(entry, model_key);
        Self {
            url: entry.endpoint.clone(),
            model,
            group: None,
            target_name: Some(model_key.to_string()),
            params,
            instance: None,
            snapshot: None,
            id_slot: None,
            filter_thinking: entry.filter_thinking,
            retry_count: entry.retry_count,
            retry_base_interval_s: entry.retry_base_interval_s,
            stream: entry.stream,
            idle_timeout_ms: entry.idle_timeout_ms,
            total_timeout_ms: entry.total_timeout_ms,
            api_key: entry.api_key.clone(),
            fallbacks: vec![],
            is_onnx: false,
        }
    }

    /// Build a routing target for a specific named inference point
    /// (`<base>:<instance_or_group>`), used by callers that must target a
    /// particular instance (e.g. the ledger summarizer or on-demand scratch).
    /// The `instance` field is set so the request explicitly names the point.
    pub fn from_model_entry_instance(
        model_key: &str,
        entry: &ModelEntry,
        instance_or_group: &str,
    ) -> Self {
        let base = entry.name.clone().unwrap_or_else(|| model_key.to_string());
        let model = QualifiedModelId::qualified(base, instance_or_group).as_wire();
        Self {
            url: entry.endpoint.clone(),
            model,
            group: None,
            target_name: Some(model_key.to_string()),
            params: entry
                .instance_params_for(instance_or_group)
                .or_else(|| entry.params.clone().map(strip_declaration_params)),
            instance: Some(instance_or_group.to_string()),
            snapshot: None,
            id_slot: None,
            filter_thinking: entry.filter_thinking,
            retry_count: entry.retry_count,
            retry_base_interval_s: entry.retry_base_interval_s,
            stream: entry.stream,
            idle_timeout_ms: entry.idle_timeout_ms,
            total_timeout_ms: entry.total_timeout_ms,
            api_key: entry.api_key.clone(),
            fallbacks: vec![],
            is_onnx: false,
        }
    }

    /// Build a routing target for an in-process onnx role (e.g. the generative
    /// `onnx/llm` routing model). The target has no HTTP `url` — `is_onnx` is
    /// set so the dispatch layer serves it through the onnx `ChatBackend` — and
    /// uses the canonical default timeouts (the role's own knobs are applied at
    /// the onnx-backend call site).
    pub fn from_onnx_role(model_key: &str) -> Self {
        use fluent_llm::constants::{
            DEFAULT_IDLE_TIMEOUT_MS, DEFAULT_RETRY_INTERVAL_S, DEFAULT_TOTAL_TIMEOUT_MS,
        };
        Self {
            url: String::new(),
            model: model_key.to_string(),
            group: None,
            target_name: Some(model_key.to_string()),
            params: None,
            instance: None,
            snapshot: None,
            id_slot: None,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: DEFAULT_RETRY_INTERVAL_S,
            stream: false,
            idle_timeout_ms: DEFAULT_IDLE_TIMEOUT_MS,
            total_timeout_ms: DEFAULT_TOTAL_TIMEOUT_MS,
            api_key: None,
            fallbacks: vec![],
            is_onnx: true,
        }
    }
}

/// Canonical timeout/retry defaults, centralized in `fluent_llm::constants`.
fn default_retry_interval() -> u64 {
    fluent_llm::constants::DEFAULT_RETRY_INTERVAL_S
}

fn default_idle_timeout_ms() -> u64 {
    fluent_llm::constants::DEFAULT_IDLE_TIMEOUT_MS
}

fn default_total_timeout_ms() -> u64 {
    fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub decisions: Vec<StageDecision>,
    pub final_response: Option<String>,
    pub rejected: bool,
    pub reject_reason: Option<String>,
    /// Routing target from the classifier stage (URL + model to dispatch to).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_target: Option<RoutingTarget>,
    /// Direct response from the classifier stage (for trivial queries).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classifier_response: Option<String>,
}

pub type StageProducer = Arc<
    dyn Fn(&WorkContext, &[StageDecision]) -> Result<(StageDecision, Option<RoutingTarget>), WorkError>
        + Send
        + Sync,
>;

pub fn producer_for<T>(arc: Arc<T>) -> StageProducer
where
    T: StageDecisionProducer + 'static,
{
    Arc::new(move |ctx, prior| arc.evaluate(ctx, prior).map(|d| (d, None)))
}

/// Holds pipeline stages as `Arc<dyn Component>` and executes them sequentially.
pub struct PipelineOrchestrator {
    name: ArcIntern<str>,
    stages: Vec<Arc<dyn Component>>,
    producers: Vec<Option<StageProducer>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

/// Typed-store key under which `PipelineOrchestrator` publishes each stage's
/// `StageDecision` for in-process handoff. `handle_stage_verdict` reads the
/// decision back from the typed store (by reference — no
/// `WorkOutput.data`/`data_take` round-trip), and any downstream stage or
/// handler can read it the same way. See the decision-rule doc on
/// `WorkContext`: the typed store is the primary inter-unit channel; `data`
/// is reserved for serialization boundaries.
pub const STAGE_DECISION_KEY: &str = "stage.decision";

impl PipelineOrchestrator {
    pub fn new(stages: Vec<Arc<dyn Component>>) -> Self {
        Self {
            name: ArcIntern::from("pipeline.orchestrator"),
            stages,
            producers: vec![],
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.result")],
        }
    }

    pub fn with_producers(stages: Vec<Arc<dyn Component>>, producers: Vec<Option<StageProducer>>) -> Self {
        Self {
            name: ArcIntern::from("pipeline.orchestrator"),
            stages,
            producers,
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.result")],
        }
    }

    pub fn builder() -> PipelineOrchestratorBuilder {
        PipelineOrchestratorBuilder::default()
    }

    fn build_stage_context(base: &WorkContext, current_request: &serde_json::Value) -> WorkContext {
        let mut ctx = base.clone();
        ctx.structured
            .insert("request".into(), current_request.clone());
        ctx
    }

    /// Apply a stage verdict to the running pipeline state, reading the
    /// current `StageDecision` from the typed store (published by `execute`
    /// under `STAGE_DECISION_KEY`) rather than re-deserializing it from
    /// `WorkOutput.data`. Returns `Some(WorkOutput)` when the pipeline should
    /// short-circuit (rejected / error); `None` otherwise.
    fn handle_stage_verdict(
        ctx: &WorkContext,
        stage_name: PipelineStage,
        current_request: &mut serde_json::Value,
        routing_target: &mut Option<RoutingTarget>,
        classifier_response: &mut Option<String>,
    ) -> Option<Result<WorkOutput, WorkError>> {
        // Typed handoff: the orchestrator published the decision to the store,
        // so we read it by reference — no per-stage JSON round-trip.
        let decision = ctx.get::<StageDecision>(STAGE_DECISION_KEY)?;
        let metadata = StageMetadata::from(decision.metadata.clone());
        match decision.verdict {
            StageVerdict::Passed | StageVerdict::Skipped => {
                if stage_name == PipelineStage::Classifier {
                    if let Some(resp) = metadata.response() {
                        tracing::info!(target: "router.pipeline",
                            response_len = resp.len(),
                            "classifier direct response"
                        );
                        crate::audit::AuditRecord::route(
                            PipelineStage::Classifier,
                            StageVerdict::Passed,
                            None,
                            Some(resp.len()),
                            Some("direct_response"),
                        )
                        .emit();
                        *classifier_response = Some(resp.to_string());
                    }
                    // The typed channel is the only routing-target handoff.
                    if let Some(rt) = ctx
                        .get::<RoutingTarget>(crate::stages::common::ROUTING_TARGET_TYPED_KEY)
                        .cloned()
                    {
                        tracing::info!(target: "router.pipeline",
                            target_route = %rt.target_name.as_deref().unwrap_or("?"),
                            target_model = %rt.model,
                            target_url = %rt.url,
                            "classifier set routing target"
                        );
                        crate::audit::AuditRecord::route(
                            PipelineStage::Classifier,
                            StageVerdict::Passed,
                            Some(&rt),
                            None,
                            Some("passed"),
                        )
                        .emit();
                        *routing_target = Some(rt);
                    }
                }
                None
            }
            StageVerdict::Rerouted => {
                if let Some(rewritten) = metadata.rewritten_request() {
                    tracing::info!(target: "router.pipeline",
                        new_request_len = rewritten.len(),
                        "request rerouted"
                    );
                    crate::audit::emit(
                        "route",
                        serde_json::json!({
                            "stage": stage_name,
                            "verdict": "rerouted",
                            "new_request_len": rewritten.len(),
                        }),
                    );
                    // Boundary: the rewritten request arrives as a string
                    // (a re-serialized `RouterRequest`), so parse it back
                    // into the structured channel's Value form.
                    *current_request = serde_json::from_str(rewritten)
                        .unwrap_or_else(|_| serde_json::Value::String(rewritten.to_string()));
                }
                None
            }
            StageVerdict::Rejected => {
                tracing::info!(target: "router.pipeline",
                    stage = ?stage_name,
                    reason = %decision.reason,
                    "pipeline rejected request"
                );
                crate::audit::emit(
                    "route",
                    serde_json::json!({
                        "stage": stage_name,
                        "verdict": "rejected",
                        "reason": decision.reason,
                    }),
                );
                Some(WorkOutput::typed(
                    "rejected",
                    &PipelineResult {
                        decisions: vec![decision.clone()],
                        final_response: None,
                        rejected: true,
                        reject_reason: Some(decision.reason.clone()),
                        routing_target: None,
                        classifier_response: None,
                    },
                ))
            }
            StageVerdict::Error => {
                tracing::error!(target: "router.pipeline",
                    stage = ?stage_name,
                    reason = %decision.reason,
                    "stage error"
                );
                Some(WorkOutput::typed(
                    "pipeline_error",
                    &PipelineResult {
                        decisions: vec![decision.clone()],
                        final_response: None,
                        rejected: true,
                        reject_reason: Some(format!("stage error: {}", decision.reason)),
                        routing_target: None,
                        classifier_response: None,
                    },
                ))
            }
        }
    }
}

#[derive(Default)]
pub struct PipelineOrchestratorBuilder {
    stages: Vec<Arc<dyn Component>>,
    producers: Vec<Option<StageProducer>>,
}

impl PipelineOrchestratorBuilder {
    #[must_use]
    pub fn push(mut self, stage: Arc<dyn Component>) -> Self {
        self.stages.push(stage);
        self.producers.push(None);
        self
    }

    #[must_use]
    pub fn push_with_producer(
        mut self,
        stage: Arc<dyn Component>,
        producer: StageProducer,
    ) -> Self {
        self.stages.push(stage);
        self.producers.push(Some(producer));
        self
    }

    #[must_use]
    pub fn build(self) -> PipelineOrchestrator {
        PipelineOrchestrator::with_producers(self.stages, self.producers)
    }
}

impl WorkUnit for PipelineOrchestrator {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let mut decisions: Vec<StageDecision> = Vec::new();
        let mut current_request: serde_json::Value = ctx
            .structured
            .get("request")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let mut routing_target: Option<RoutingTarget> = None;
        let mut classifier_response: Option<String> = None;

        // The running accumulator clones the caller's context so each stage's
        // decision can be published to the typed store (`outputs`) and read by
        // reference — by `handle_stage_verdict` and by later stages. The typed
        // store, not `WorkOutput.data`, is the in-process handoff channel.
        let mut running = ctx.clone();

        for (idx, stage) in self.stages.iter().enumerate() {
            let stage_ctx = Self::build_stage_context(&running, &current_request);
            let start = Instant::now();

            let stage_name_human = stage.name().to_string();
            tracing::debug!(target: "router.pipeline", stage = %stage_name_human, "stage entering");

            // Typed handoff via builder registry. If a producer is registered
            // for this index, use it — it returns the target by value alongside
            // the decision; otherwise fall back to the `WorkOutput`
            // serialization boundary (test stubs, unknown components), which
            // carries no target.
            let outcome: Result<(StageDecision, Option<RoutingTarget>), WorkError> =
                if let Some(prod) = self.producers.get(idx).and_then(|p| p.as_ref()) {
                    prod(&stage_ctx, &decisions)
                } else {
                    stage
                        .execute(&stage_ctx)
                        .and_then(|output| {
                            output
                                .data_take()
                                .map_err(|e| WorkError::Execution(e.to_string()))
                        })
                        .map(|decision| (decision, None))
                };

            match outcome {
                Ok((mut decision, target)) => {
                    let latency_ms = start.elapsed().as_millis() as u64;
                    decision.latency_ms = latency_ms;
                    let verdict = decision.verdict.clone();
                    let stage_name = decision.stage;

                    let fallback = stage_name == PipelineStage::Classifier
                        && StageMetadata::from(decision.metadata.clone())
                            .fallback()
                            .unwrap_or(false);
                    tracing::info!(target: "router.pipeline",
                        stage = ?stage_name,
                        verdict = ?verdict,
                        latency_ms = latency_ms,
                        score = ?decision.score,
                        reason = %decision.reason,
                        fallback = fallback,
                        "stage complete"
                    );

                    decisions.push(decision.clone());

                    // Publish the typed decision to the store — the primary
                    // in-process handoff. `handle_stage_verdict` reads it back
                    // by reference instead of `data_take()`, and any downstream
                    // stage can do the same via `STAGE_DECISION_KEY`.
                    running.set(STAGE_DECISION_KEY, decision.clone());
                    // Publish a producer-returned target once through the
                    // single canonical typed write. No JSON bridge: the
                    // decision carries no routing-target shim.
                    if let Some(rt) = target {
                        crate::stages::common::publish_routing_target(
                            &mut running,
                            &mut decision,
                            rt,
                        );
                    }

                    if let Some(early_return) = Self::handle_stage_verdict(
                        &running,
                        stage_name,
                        &mut current_request,
                        &mut routing_target,
                        &mut classifier_response,
                    ) {
                        return early_return;
                    }
                }
                Err(e) => {
                    tracing::error!(target: "router.pipeline",
                        stage = %stage_name_human,
                        error = %e,
                        latency_ms = %start.elapsed().as_millis(),
                        "stage execution error"
                    );
                    // No synthetic decision is recorded: the `WorkUnit`
                    // contract propagates the error, and the caller surfaces
                    // the failing stage via `PipelineError::Stage`.
                    return Err(e);
                }
            }
        }

        let has_routing = routing_target.is_some();
        let has_classifier_resp = classifier_response.is_some();
        tracing::info!(target: "router.pipeline",
            stages = decisions.len(),
            has_routing_target = has_routing,
            has_classifier_response = has_classifier_resp,
            routing_model = ?routing_target.as_ref().map(|rt| &rt.model),
            routing_route = ?routing_target.as_ref().and_then(|rt| rt.target_name.as_ref()),
            "pipeline complete"
        );

        WorkOutput::typed(
            "pipeline_complete",
            &PipelineResult {
                decisions,
                final_response: None,
                rejected: false,
                reject_reason: None,
                routing_target,
                classifier_response,
            },
        )
    }
}

impl_fieldless!(PipelineOrchestrator);

impl Describable for PipelineOrchestrator {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(PipelineOrchestrator);

#[cfg(test)]
#[path = "../tests/pipeline.rs"]
mod tests;
#[cfg(test)]
#[path = "../tests/qualified_model_id_roundtrip.rs"]
mod qualified_model_id_roundtrip;
