//! Pipeline orchestrator — sequences stages as `WorkUnit` implementations.
//! Follows the `TierRegistry` pattern from `coral/src/tier_units.rs:528-570`.

use std::sync::Arc;
use std::time::Instant;

use fluent_wvr::prelude::*;
use serde::{Deserialize, Serialize};

use common_core::constants::default_true;

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::get_metadata_string;

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

fn default_retry_interval() -> u64 {
    1
}

fn default_idle_timeout_ms() -> u64 {
    10_000
}

fn default_total_timeout_ms() -> u64 {
    120_000
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

    fn build_stage_context(
        base: &WorkContext,
        current_request: &str,
        _decisions: &[StageDecision],
    ) -> WorkContext {
        let mut ctx = base.clone();
        ctx.metadata
            .insert("request".into(), MetadataValue::String(current_request.to_string()));
        ctx
    }

    fn handle_stage_verdict(
        verdict: &StageVerdict,
        stage_name: PipelineStage,
        decision: &StageDecision,
        current_request: &mut String,
        routing_target: &mut Option<RoutingTarget>,
        classifier_response: &mut Option<String>,
    ) -> Option<Result<WorkOutput, WorkError>> {
        match verdict {
            StageVerdict::Passed | StageVerdict::Skipped => {
                if stage_name == PipelineStage::Classifier {
                    if let Some(resp) = classifier_response_from_decision(decision) {
                        tracing::info!(target: "router.pipeline",
                            response_len = resp.len(),
                            "classifier direct response"
                        );
                        *classifier_response = Some(resp);
                    }
                    if let Some(rt) = extract_routing_target(&decision.metadata) {
                        tracing::info!(target: "router.pipeline",
                            target_route = %rt.target_name.as_deref().unwrap_or("?"),
                            target_model = %rt.model,
                            target_url = %rt.url,
                            "classifier set routing target"
                        );
                        *routing_target = Some(rt);
                    }
                }
                None
            }
            StageVerdict::Rerouted => {
                if let Some(rewritten) = decision.metadata.get("rewritten_request") {
                    if let Some(s) = rewritten.as_str() {
                        tracing::info!(target: "router.pipeline",
                            new_request_len = s.len(),
                            "request rerouted"
                        );
                        *current_request = s.to_string();
                    }
                }
                None
            }
            StageVerdict::Rejected => {
                tracing::info!(target: "router.pipeline",
                    stage = ?stage_name,
                    reason = %decision.reason,
                    "pipeline rejected request"
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

fn extract_routing_target(metadata: &serde_json::Value) -> Option<RoutingTarget> {
    let rt = metadata.get("routing_target")?;
    serde_json::from_value(rt.clone()).ok()
}

fn classifier_response_from_decision(decision: &StageDecision) -> Option<String> {
    decision
        .metadata
        .get("response")
        .and_then(|v| v.as_str())
        .map(String::from)
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
        let mut current_request =
            get_metadata_string(ctx, "request").unwrap_or_default();
        let mut routing_target: Option<RoutingTarget> = None;
        let mut classifier_response: Option<String> = None;

        for stage in &self.stages {
            let stage_ctx = Self::build_stage_context(ctx, &current_request, &decisions);
            let start = Instant::now();

            let stage_name_human = stage.name().to_string();
            tracing::debug!(target: "router.pipeline", stage = %stage_name_human, "stage entering");

            match stage.execute(&stage_ctx) {
                Ok(output) => {
                    let mut decision: StageDecision = output
                        .data_take()
                        .map_err(|e| WorkError::Execution(e.to_string()))?;
                    let latency_ms = start.elapsed().as_millis() as u64;
                    decision.latency_ms = latency_ms;
                    let verdict = decision.verdict.clone();
                    let stage_name = decision.stage;

                    let fallback = stage_name == PipelineStage::Classifier
                        && decision.metadata.get("fallback").and_then(serde_json::Value::as_bool).unwrap_or(false);
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

impl FieldAccess for PipelineOrchestrator {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "PipelineOrchestrator has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "PipelineOrchestrator has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

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
