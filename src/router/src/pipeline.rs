//! Pipeline orchestrator — sequences stages as `WorkUnit` implementations.
//! Follows the `TierRegistry` pattern from `coral/src/tier_units.rs:528-570`.

use std::sync::Arc;
use std::time::Instant;

use fluent_wvr::prelude::*;
use serde::{Deserialize, Serialize};

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
}

fn default_retry_interval() -> u64 {
    1
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

                    match verdict {
                        StageVerdict::Passed | StageVerdict::Skipped => {
                            if stage_name == PipelineStage::Classifier {
                                if let Some(resp) = decision
                                    .metadata
                                    .get("response")
                                    .and_then(|v| v.as_str())
                                {
                                    tracing::info!(target: "router.pipeline",
                                        response_len = resp.len(),
                                        "classifier direct response"
                                    );
                                    classifier_response = Some(resp.to_string());
                                }
                                if let Some(rt) = decision.metadata.get("routing_target") {
                                    let model = rt.get("model").and_then(|v| v.as_str()).unwrap_or("?");
                                    let target_name = rt.get("target_name").and_then(|v| v.as_str()).unwrap_or("?");
                                    let url = rt.get("url").and_then(|v| v.as_str()).unwrap_or("?");
                                    tracing::info!(target: "router.pipeline",
                                        target_route = %target_name,
                                        target_model = %model,
                                        target_url = %url,
                                        "classifier set routing target"
                                    );
                                    routing_target = Some(RoutingTarget {
                                        url: url.into(),
                                        model: model.into(),
                                        group: rt
                                            .get("group")
                                            .and_then(|v| v.as_str())
                                            .map(String::from)
                                            .filter(|s| !s.is_empty()),
                                        target_name: rt
                                            .get("target_name")
                                            .and_then(|v| v.as_str())
                                            .map(String::from),
                                        params: rt
                                            .get("params")
                                            .cloned()
                                            .filter(|v| !v.is_null()),
                                        filter_thinking: rt
                                            .get("filter_thinking")
                                            .and_then(serde_json::Value::as_bool)
                                            .unwrap_or(false),
                                        retry_count: rt
                                            .get("retry_count")
                                            .and_then(serde_json::Value::as_u64)
                                            .unwrap_or(0) as u32,
                                        retry_base_interval_s: rt
                                            .get("retry_base_interval_s")
                                            .and_then(serde_json::Value::as_u64)
                                            .unwrap_or(1),
                                    });
                                }
                            }
                        }
                        StageVerdict::Rerouted => {
                            if let Some(rewritten) = decision.metadata.get("rewritten_request") {
                                if let Some(s) = rewritten.as_str() {
                                    tracing::info!(target: "router.pipeline", new_request_len = s.len(), "request rerouted");
                                    current_request = s.to_string();
                                }
                            }
                        }
                        StageVerdict::Rejected => {
                            tracing::info!(target: "router.pipeline",
                                stage = ?stage_name,
                                reason = %decision.reason,
                                "pipeline rejected request"
                            );
                            return WorkOutput::typed(
                                "rejected",
                                &PipelineResult {
                                    decisions,
                                    final_response: None,
                                    rejected: true,
                                    reject_reason: Some(decision.reason),
                                    routing_target: None,
                                    classifier_response: None,
                                },
                            );
                        }
                        StageVerdict::Error => {
                            tracing::error!(target: "router.pipeline",
                                stage = ?stage_name,
                                reason = %decision.reason,
                                "stage error"
                            );
                            return WorkOutput::typed(
                                "pipeline_error",
                                &PipelineResult {
                                    decisions,
                                    final_response: None,
                                    rejected: true,
                                    reject_reason: Some(format!(
                                        "stage error: {}",
                                        decision.reason
                                    )),
                                    routing_target: None,
                                    classifier_response: None,
                                },
                            );
                        }
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
