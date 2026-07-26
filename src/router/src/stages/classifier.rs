//! Stage 2: ClassifierStage — single LLM call that replaces QualityGate,
//! PlanningRefinement, and GuardrailCheck. Acts as an FSM switch: the LLM
//! returns either a direct response, a routing target, or a rejection.
//! Configurable via `RoutingConfig` from the top-level coral-router config.
//!
//! The LLM backend is injected as `Arc<dyn ChatBackend>` rather than a
//! concrete `LlmClient`, so mock/stub backends can be injected for testing
//! without duplicating the pipeline wiring.

use std::sync::Arc;

use fluent_wvr::prelude::*;
use guidance_llm::client::ChatBackend;
use guidance_llm::ChatMessage;

use crate::config::{ClassifierOutput, RoutingConfig};
use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::score_matrix::ScoreMatrix;
use crate::stages::common::extract_user_message;

pub struct ClassifierStage {
    name: ArcIntern<str>,
    client: Arc<dyn ChatBackend>,
    routing_config: RoutingConfig,
    coherence_threshold: f64,
    score_matrix: Option<ScoreMatrix>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl ClassifierStage {
    pub fn new(
        client: Arc<dyn ChatBackend>,
        routing_config: RoutingConfig,
        coherence_threshold: f64,
        score_matrix: Option<ScoreMatrix>,
    ) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage2.classifier"),
            client,
            routing_config,
            coherence_threshold,
            score_matrix,
            depends: vec![ArcIntern::from("pipeline.stage1.output")],
            provides: vec![ArcIntern::from("pipeline.stage2.output")],
        }
    }

    fn build_system_prompt(&self) -> String {
        self.routing_config.system_prompt
            .replace("{coherence_threshold}", &format!("{:.2}", self.coherence_threshold))
            .replace("{safety_threshold}", &format!("{:.2}", self.routing_config.safety_threshold))
    }

    fn build_routing_target_json(
        &self,
        route_name: &str,
        model: &crate::config::ModelEntry,
        model_name: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "url": model.endpoint,
            "model": model_name,
            "group": self.routing_config
                .routes
                .get(route_name)
                .or_else(|| self.routing_config.routes.get(&self.routing_config.default_route))
                .map_or("", |r| r.group.as_str()),
            "target_name": route_name,
            "params": model.params,
            "filter_thinking": model.filter_thinking,
            "retry_count": model.retry_count,
            "retry_base_interval_s": model.retry_base_interval_s,
            "stream": model.stream,
            "idle_timeout_ms": model.idle_timeout_ms,
            "total_timeout_ms": model.total_timeout_ms,
        })
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

        let system_prompt = ctx
            .metadata
            .get("classifier_system_prompt")
            .and_then(|v| match v {
                MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| self.build_system_prompt());

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: system_prompt.clone(),
            },
            ChatMessage {
                role: "user".into(),
                content: input.clone(),
            },
        ];

        tracing::debug!(target: "router.pipeline.stage2", input_len = input.len(), "classifier stage");

        let (output, ok) = match self.client.chat_complete(&messages) {
            Ok(response) => {
                tracing::debug!(target: "router.pipeline.stage2", raw_response_len = response.len(), raw_response = %response, "classifier LLM response received");
                match serde_json::from_str::<ClassifierOutput>(&response) {
                    Ok(o) => {
                        tracing::info!(target: "router.pipeline.stage2",
                            action = %o.action,
                            target = ?o.target,
                            response_direct = o.response.is_some(),
                            coherence = %o.coherence_score,
                            safety = %o.safety_score,
                            complexity = ?o.complexity,
                            intent = ?o.intent,
                            reason = %o.reason,
                            fallback = false,
                            "classifier verdict"
                        );
                        (o, true)
                    }
                    Err(e) => {
                        tracing::error!(target: "router.pipeline.stage2", error = %e, raw_response_len = response.len(), raw_response = %response, "classifier LLM response was not valid ClassifierOutput JSON — falling back to default route");
                        (ClassifierOutput {
                            action: "route".into(),
                            response: None,
                            target: Some(self.routing_config.default_route.clone()),
                            coherence_score: 1.0,
                            safety_score: 1.0,
                            complexity: None,
                            intent: None,
                            reason: format!("parse error: {e}"),
                            completeness: None,
                            risk: None,
                        }, false)
                    }
                }
            }
            Err(e) => {
                tracing::error!(target: "router.pipeline.stage2", error = %e, input_len = input.len(), "classifier LLM call failed — falling back to default route");
                (ClassifierOutput {
                    action: "route".into(),
                    response: None,
                    target: Some(self.routing_config.default_route.clone()),
                    coherence_score: 1.0,
                    safety_score: 1.0,
                    complexity: None,
                    intent: None,
                    reason: format!("LLM error: {e}"),
                    completeness: None,
                    risk: None,
                }, false)
            }
        };

        let coherence_ok = output.coherence_score >= self.coherence_threshold;
        let safety_ok = output.safety_score >= self.routing_config.safety_threshold;

        if !coherence_ok || !safety_ok {
            let reason = if coherence_ok {
                format!(
                    "rejected: safety {:.2} below threshold {:.2}",
                    output.safety_score, self.routing_config.safety_threshold
                )
            } else {
                format!(
                    "rejected: coherence {:.2} below threshold {:.2}",
                    output.coherence_score, self.coherence_threshold
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

        if action == "reject" {
            tracing::info!(target: "router.pipeline.stage2",
                reason = %output.reason,
                "classifier rejected request"
            );
            return WorkOutput::typed(
                "rejected",
                &StageDecision {
                    stage: PipelineStage::Classifier,
                    verdict: StageVerdict::Rejected,
                    score: Some(output.coherence_score),
                    reason: format!("blocked: {}", output.reason),
                    latency_ms: 0,
                    metadata: serde_json::json!({
                        "coherence_score": output.coherence_score,
                        "safety_score": output.safety_score,
                        "intent": output.intent,
                        "action": output.action,
                        "reason": output.reason,
                    }),
                },
            );
        }

        let min_complexity = output.complexity;
        let resolved_route = output.target.as_deref().unwrap_or(&self.routing_config.default_route);
        let routing_target = match action {
            "respond" => {
                tracing::info!(target: "router.pipeline.stage2", "direct response — no dispatch");
                None
            }
            "route" => {
                let resolved = self.routing_config.resolve_route(resolved_route, min_complexity);
                if let Some((model, model_name)) = &resolved {
                    tracing::info!(target: "router.pipeline.stage2",
                        route = %resolved_route,
                        model = %model_name,
                        endpoint = %model.endpoint,
                        group = ?self.routing_config.routes.get(resolved_route).map(|r| &r.group),
                        "routing target resolved"
                    );
                } else {
                    tracing::warn!(target: "router.pipeline.stage2", route = %resolved_route, "resolve_route returned None — no dispatch target");
                }
                resolved.map(|(model, model_name)| {
                    self.build_routing_target_json(
                        resolved_route,
                        model,
                        &model_name,
                    )
                })
            }
            _ => {
                let fallback_route = &self.routing_config.default_route;
                tracing::warn!(target: "router.pipeline.stage2", action = %action, fallback_route = %fallback_route, "unknown action, falling back to default route");
                let resolved = self.routing_config.resolve_route(fallback_route, min_complexity);
                if let Some((model, model_name)) = &resolved {
                    tracing::info!(target: "router.pipeline.stage2",
                        route = %fallback_route,
                        model = %model_name,
                        endpoint = %model.endpoint,
                        "fallback routing target"
                    );
                }
                resolved.map(|(model, model_name)| {
                    self.build_routing_target_json(
                        fallback_route,
                        model,
                        &model_name,
                    )
                })
            }
        };

        // ── Score-matrix resolution (MOA_ROUTER_SPEC §2.2) ──
        let scored_routes = self.score_matrix.as_ref().map(|sm| {
            let scores = std::collections::HashMap::from([
                ("coherence".into(), output.coherence_score),
                ("complexity".into(), f64::from(output.complexity.unwrap_or(5)) / 10.0),
                ("completeness".into(), output.completeness.unwrap_or(0.5)),
                ("risk".into(), output.risk.unwrap_or(0.0)),
            ]);
            sm.resolve(&scores)
        });

        let mut metadata = serde_json::json!({
            "coherence_score": output.coherence_score,
            "safety_score": output.safety_score,
            "complexity": output.complexity,
            "completeness": output.completeness,
            "risk": output.risk,
            "intent": output.intent,
            "action": output.action,
            "reason": output.reason,
            "fallback": !ok,
        });

        if let Some(ref routes) = scored_routes {
            if let Some(top) = routes.first() {
                metadata["scored_route"] = serde_json::json!({
                    "route": top.route_name,
                    "score": top.weighted_score,
                    "score_vector": top.score_vector.iter().map(|(d, s)| {
                        serde_json::json!({"dimension": d, "score": s})
                    }).collect::<Vec<_>>(),
                });
            }
            metadata["scored_routes"] = serde_json::Value::Array(
                routes.iter().map(|r| serde_json::json!({
                    "route": r.route_name,
                    "score": r.weighted_score,
                })).collect(),
            );
        }

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
