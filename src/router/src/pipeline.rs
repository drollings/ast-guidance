//! Pipeline orchestrator — sequences stages as `WorkUnit` implementations.
//! Follows the `TierRegistry` pattern from `coral/src/tier_units.rs:528-570`.

use std::sync::Arc;
use std::time::Instant;

use fluent_wvr::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pipeline_types::{StageDecision, StageVerdict};

/// Result returned by the `PipelineOrchestrator` after all stages complete.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub decisions: Vec<StageDecision>,
    pub final_response: Option<String>,
    pub rejected: bool,
    pub reject_reason: Option<String>,
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
        let mut current_request = get_metadata_string(ctx, "request")
            .unwrap_or_default();

        for stage in &self.stages {
            let stage_ctx = Self::build_stage_context(ctx, &current_request, &decisions);
            let start = Instant::now();

            match stage.execute(&stage_ctx) {
                Ok(output) => {
                    let decision: StageDecision = output
                        .data_take()
                        .map_err(|e| WorkError::Execution(e.to_string()))?;
                    let latency_ms = start.elapsed().as_millis() as u64;
                    let decision = StageDecision {
                        latency_ms,
                        ..decision
                    };
                    let verdict = decision.verdict.clone();
                    decisions.push(decision.clone());

                    match verdict {
                        StageVerdict::Passed | StageVerdict::Skipped => {}
                        StageVerdict::Rerouted => {
                            if let Some(rewritten) = decision.metadata.get("rewritten_request") {
                                if let Some(s) = rewritten.as_str() {
                                    current_request = s.to_string();
                                }
                            }
                        }
                        StageVerdict::Rejected => {
                            return WorkOutput::typed(
                                "rejected",
                                &PipelineResult {
                                    decisions,
                                    final_response: None,
                                    rejected: true,
                                    reject_reason: Some(decision.reason),
                                },
                            );
                        }
                        StageVerdict::Error => {
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
                                },
                            );
                        }
                    }
                }
                Err(e) => {
                    decisions.push(StageDecision {
                        stage: crate::pipeline_types::PipelineStage::Router,
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

        WorkOutput::typed(
            "pipeline_complete",
            &PipelineResult {
                decisions,
                final_response: None,
                rejected: false,
                reject_reason: None,
            },
        )
    }
}

fn get_metadata_string(ctx: &WorkContext, key: &str) -> Option<String> {
    ctx.metadata.get(key).and_then(|v| match v {
        MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    })
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