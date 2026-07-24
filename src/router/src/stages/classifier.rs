//! Stage 2: ClassifierStage — single LLM call that replaces QualityGate,
//! PlanningRefinement, and GuardrailCheck. Acts as an FSM switch: the LLM
//! returns either a direct response, a routing target, or a rejection.
//! Configurable via `RoutingFsm` from JSON.

use fluent_wvr::prelude::*;
use guidance_llm::{ChatMessage, LlmClient, LlmConfig};

use crate::config::{ClassifierOutput, RoutingFsm};
use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;

pub struct ClassifierStage {
    name: ArcIntern<str>,
    client: LlmClient,
    fsm: RoutingFsm,
    quality_threshold: f64,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl ClassifierStage {
    pub fn new(config: LlmConfig, fsm: RoutingFsm, quality_threshold: f64) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage2.classifier"),
            client: LlmClient::with_config(config),
            fsm,
            quality_threshold,
            depends: vec![ArcIntern::from("pipeline.stage1.output")],
            provides: vec![ArcIntern::from("pipeline.stage2.output")],
        }
    }
}

impl WorkUnit for ClassifierStage {
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
        let input = extract_user_message(ctx)?;

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: self.fsm.system_prompt.clone(),
            },
            ChatMessage {
                role: "user".into(),
                content: input,
            },
        ];

        let (output, ok) = match self.client.chat_complete(&messages) {
            Ok(response) => match serde_json::from_str::<ClassifierOutput>(&response) {
                Ok(o) => (o, true),
                Err(e) => {
                    tracing::warn!(target: "classifier", error = %e, "classifier parse error, falling back to default route");
                    (ClassifierOutput {
                        action: "route".into(),
                        response: None,
                        target: Some(self.fsm.default_route.clone()),
                        coherence_score: 1.0,
                        safety_score: 1.0,
                        intent: None,
                        reason: format!("parse error: {e}"),
                    }, false)
                }
            },
            Err(e) => {
                tracing::warn!(target: "classifier", error = %e, "classifier LLM error, falling back to default route");
                (ClassifierOutput {
                    action: "route".into(),
                    response: None,
                    target: Some(self.fsm.default_route.clone()),
                    coherence_score: 1.0,
                    safety_score: 1.0,
                    intent: None,
                    reason: format!("LLM error: {e}"),
                }, false)
            }
        };

        let coherence_ok = output.coherence_score >= self.quality_threshold;
        let safety_ok = output.safety_score >= self.fsm.safety_threshold;

        if !coherence_ok || !safety_ok {
            let reason = if coherence_ok {
                format!(
                    "rejected: safety {:.2} below threshold {:.2}",
                    output.safety_score, self.fsm.safety_threshold
                )
            } else {
                format!(
                    "rejected: coherence {:.2} below threshold {:.2}",
                    output.coherence_score, self.quality_threshold
                )
            };
            return WorkOutput::typed(
                "rejected",
                &StageDecision {
                    stage: PipelineStage::Classifier,
                    verdict: StageVerdict::Rejected,
                    score: Some(output.coherence_score),
                    reason,
                    latency_ms: 0,
                    metadata: serde_json::json!({
                        "coherence_score": output.coherence_score,
                        "safety_score": output.safety_score,
                        "intent": output.intent,
                        "action": output.action,
                    }),
                },
            );
        }

        let action = output.action.as_str();
        let routing_target = match action {
            "respond" => None,
            "route" => {
                let target_name = output.target.as_deref().unwrap_or(&self.fsm.default_route);
                let route = self.fsm.routes.get(target_name);
                route.map(|r| {
                    serde_json::json!({
                        "url": r.url,
                        "model": r.model,
                        "target_name": target_name,
                    })
                })
            }
            _ => {
                let route = self.fsm.routes.get(&self.fsm.default_route);
                route.map(|r| {
                    serde_json::json!({
                        "url": r.url,
                        "model": r.model,
                        "target_name": &self.fsm.default_route,
                    })
                })
            }
        };

        let mut metadata = serde_json::json!({
            "coherence_score": output.coherence_score,
            "safety_score": output.safety_score,
            "intent": output.intent,
            "action": output.action,
            "reason": output.reason,
            "fallback": !ok,
        });

        if let Some(ref resp) = output.response {
            metadata["response"] = serde_json::Value::String(resp.clone());
        }
        if let Some(ref rt) = routing_target {
            metadata["routing_target"] = rt.clone();
        }

        WorkOutput::typed(
            "classified",
            &StageDecision {
                stage: PipelineStage::Classifier,
                verdict: StageVerdict::Passed,
                score: Some(output.coherence_score),
                reason: format!(
                    "intent={}, action={}, coherence={:.2}",
                    output.intent.as_deref().unwrap_or("?"),
                    action,
                    output.coherence_score
                ),
                latency_ms: 0,
                metadata,
            },
        )
    }
}

impl FieldAccess for ClassifierStage {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "ClassifierStage has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "ClassifierStage has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for ClassifierStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(ClassifierStage);
