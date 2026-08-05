//! Pipeline orchestrator — sequences stages as `WorkUnit` implementations.
//! Follows the `TierRegistry` pattern from `coral/src/tier_units.rs:528-570`.

use std::sync::Arc;
use std::time::Instant;

use fluent_wvr::component_downcast_ref;
use fluent_wvr::prelude::*;
use serde::{Deserialize, Serialize};

use common_core::constants::default_true;

use crate::config::ModelEntry;
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
    /// Ordered fallback targets to try when the primary fails.
    /// Populated at route-resolution time from all available models,
    /// ordered by intelligence proximity to the request complexity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallbacks: Vec<RoutingTarget>,
}

impl RoutingTarget {
    /// Build a routing target from a configured model entry — the canonical
    /// mapping used by every dispatch path (direct-model requests and the
    /// classifier fallback) so any call to a model carries its configured
    /// `params` (e.g. `num_ctx`/`parallel`/`sleep_idle_seconds` for llama.cpp
    /// slot sizing) plus its timeout/retry/streaming profile.
    pub fn from_model_entry(model_key: &str, entry: &ModelEntry) -> Self {
        Self {
            url: entry.endpoint.clone(),
            model: entry.name.clone().unwrap_or_else(|| model_key.to_string()),
            group: None,
            target_name: Some(model_key.to_string()),
            params: entry.params.clone(),
            filter_thinking: entry.filter_thinking,
            retry_count: entry.retry_count,
            retry_base_interval_s: entry.retry_base_interval_s,
            stream: entry.stream,
            idle_timeout_ms: entry.idle_timeout_ms,
            total_timeout_ms: entry.total_timeout_ms,
            fallbacks: vec![],
        }
    }
}

/// Canonical timeout/retry defaults, centralized in `common_core::constants`.
/// **D7 behavior change (ROADMAP_20260804_DRY M7.2):** the `RoutingTarget`
/// serde path moves from 120s/10s to the `config.rs` values 300s/30s — the
/// divergence was a latent bug. Both modules now read the same constant.
fn default_retry_interval() -> u64 {
    common_core::constants::DEFAULT_RETRY_INTERVAL_S
}

fn default_idle_timeout_ms() -> u64 {
    common_core::constants::DEFAULT_IDLE_TIMEOUT_MS
}

fn default_total_timeout_ms() -> u64 {
    common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS
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

/// Holds pipeline stages as `Arc<dyn Component>` and executes them sequentially.
pub struct PipelineOrchestrator {
    name: ArcIntern<str>,
    stages: Vec<Arc<dyn Component>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl PipelineOrchestrator {
    pub fn new(stages: Vec<Arc<dyn Component>>) -> Self {
        Self {
            name: ArcIntern::from("pipeline.orchestrator"),
            stages,
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

    fn handle_stage_verdict(
        verdict: &StageVerdict,
        stage_name: PipelineStage,
        decision: &StageDecision,
        current_request: &mut serde_json::Value,
        routing_target: &mut Option<RoutingTarget>,
        classifier_response: &mut Option<String>,
    ) -> Option<Result<WorkOutput, WorkError>> {
        let metadata = StageMetadata::from(decision.metadata.clone());
        match verdict {
            StageVerdict::Passed | StageVerdict::Skipped => {
                if stage_name == PipelineStage::Classifier {
                    if let Some(resp) = metadata.response() {
                        tracing::info!(target: "router.pipeline",
                            response_len = resp.len(),
                            "classifier direct response"
                        );
                        crate::audit::emit(
                            "route",
                            serde_json::json!({
                                "stage": "classifier",
                                "verdict": "direct_response",
                                "response_len": resp.len(),
                            }),
                        );
                        *classifier_response = Some(resp.to_string());
                    }
                    if let Some(rt) = metadata.routing_target() {
                        tracing::info!(target: "router.pipeline",
                            target_route = %rt.target_name.as_deref().unwrap_or("?"),
                            target_model = %rt.model,
                            target_url = %rt.url,
                            "classifier set routing target"
                        );
                        crate::audit::emit(
                            "route",
                            serde_json::json!({
                                "stage": "classifier",
                                "verdict": "passed",
                                "target_route": rt.target_name,
                                "target_model": rt.model,
                                "target_url": rt.url,
                            }),
                        );
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

/// Router-internal downcast to the typed decision producers (M5.4). The
/// pipelines built by `config::RouterConfigBuilder` contain exactly
/// `DeterministicPreFilter` and `ClassifierStage`; the `None` fallback keeps
/// the orchestrator usable with arbitrary components (test stubs, pipeline
/// refs), which then go through the `WorkOutput` channel unchanged.
fn as_producer(stage: &dyn Component) -> Option<&dyn StageDecisionProducer> {
    component_downcast_ref::<crate::stages::deterministic::DeterministicPreFilter>(stage)
        .map(|s| s as &dyn StageDecisionProducer)
        .or_else(|| {
            component_downcast_ref::<crate::stages::classifier::ClassifierStage>(stage)
                .map(|s| s as &dyn StageDecisionProducer)
        })
}

#[derive(Default)]
pub struct PipelineOrchestratorBuilder {
    stages: Vec<Arc<dyn Component>>,
}

impl PipelineOrchestratorBuilder {
    #[must_use]
    pub fn push(mut self, stage: Arc<dyn Component>) -> Self {
        self.stages.push(stage);
        self
    }

    #[must_use]
    pub fn build(self) -> PipelineOrchestrator {
        PipelineOrchestrator::new(self.stages)
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

        for stage in &self.stages {
            let stage_ctx = Self::build_stage_context(ctx, &current_request);
            let start = Instant::now();

            let stage_name_human = stage.name().to_string();
            tracing::debug!(target: "router.pipeline", stage = %stage_name_human, "stage entering");

            // M5.4: typed handoff. The known stages implement
            // `StageDecisionProducer`, so their `StageDecision` is produced by
            // a direct method call with the running decision accumulator
            // passed by reference — no per-stage serialize→deserialize through
            // `WorkOutput.data`. Arbitrary components (test stubs, pipeline
            // refs) fall back to the `WorkOutput` channel unchanged.
            let decision = if let Some(producer) = as_producer(stage.as_ref()) {
                producer.evaluate(&stage_ctx, &decisions)
            } else {
                stage.execute(&stage_ctx).and_then(|output| {
                    output
                        .data_take()
                        .map_err(|e| WorkError::Execution(e.to_string()))
                })
            };

            match decision {
                Ok(mut decision) => {
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

                    if let Some(early_return) = Self::handle_stage_verdict(
                        &verdict,
                        stage_name,
                        &decision,
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
                    decisions.push(StageDecision {
                        stage: PipelineStage::Router,
                        verdict: StageVerdict::Error,
                        score: None,
                        reason: e.to_string(),
                        latency_ms: start.elapsed().as_millis() as u64,
                        metadata: serde_json::json!({}),
                    });
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
mod tests {
    use super::*;

    fn test_entry() -> ModelEntry {
        serde_json::from_value(serde_json::json!({
            "endpoint": "http://localhost:8080/v1/chat/completions",
            "name": "unsloth/lfm2.5-1.2b-instruct",
            "intelligence": 2,
            "cost_input": 1e-6,
            "cost_output": 6e-6,
            "cost_cached_read": 4e-7,
            "speed": 8,
            "total_timeout_ms": 40000,
            "idle_timeout_ms": 8000,
            "stream": true,
            "filter_thinking": true,
            "retry_count": 2,
            "retry_base_interval_s": 1,
            "params": {
                "num_ctx": 98304,
                "parallel": 3,
                "sleep_idle_seconds": 7200
            }
        }))
        .expect("valid ModelEntry")
    }

    #[test]
    fn from_model_entry_forwards_configured_params_and_profile() {
        let rt = RoutingTarget::from_model_entry("lfm", &test_entry());

        assert_eq!(rt.url, "http://localhost:8080/v1/chat/completions");
        assert_eq!(rt.model, "unsloth/lfm2.5-1.2b-instruct");
        assert_eq!(rt.target_name.as_deref(), Some("lfm"));
        // The llama.cpp slot args must reach the request body unchanged —
        // dropping them yields a default-context slot and a second model line.
        assert_eq!(
            rt.params.as_ref().and_then(|p| p.get("num_ctx")),
            Some(&serde_json::json!(98304))
        );
        assert_eq!(
            rt.params.as_ref().and_then(|p| p.get("parallel")),
            Some(&serde_json::json!(3))
        );
        assert_eq!(
            rt.params.as_ref().and_then(|p| p.get("sleep_idle_seconds")),
            Some(&serde_json::json!(7200))
        );
        assert!(rt.filter_thinking);
        assert_eq!(rt.retry_count, 2);
        assert_eq!(rt.retry_base_interval_s, 1);
        assert!(rt.stream);
        assert_eq!(rt.idle_timeout_ms, 8000);
        assert_eq!(rt.total_timeout_ms, 40000);
    }

    #[test]
    fn from_model_entry_falls_back_to_key_when_name_missing() {
        let mut entry = test_entry();
        entry.name = None;
        let rt = RoutingTarget::from_model_entry("lfm", &entry);
        assert_eq!(rt.model, "lfm");
    }

    #[test]
    fn routing_target_serde_defaults_read_canonical_constants() {
        // Round-trips through the serde path (no explicit timeout/retry fields)
        // so the defaults actually exercised are the serde defaults — guards
        // against the 120s/10s-vs-300s/30s divergence recurring (D7).
        let rt: RoutingTarget = serde_json::from_str(r#"{"url":"u","model":"m"}"#).unwrap();
        assert_eq!(
            rt.total_timeout_ms,
            common_core::constants::DEFAULT_TOTAL_TIMEOUT_MS
        );
        assert_eq!(
            rt.idle_timeout_ms,
            common_core::constants::DEFAULT_IDLE_TIMEOUT_MS
        );
        assert_eq!(
            rt.retry_base_interval_s,
            common_core::constants::DEFAULT_RETRY_INTERVAL_S
        );
    }
}
