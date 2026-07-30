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

const DEFAULT_COMPLEXITY: u8 = 5;
const COMPLEXITY_SCALE: f64 = 10.0;
const DEFAULT_COMPLETENESS: f64 = 0.5;

fn parse_classifier_response(response: &str, default_route: &str) -> (ClassifierOutput, bool) {
    match serde_json::from_str::<ClassifierOutput>(response) {
        Ok(o) => (o, true),
        Err(e) => {
            tracing::error!(target: "router.pipeline.stage2", error = %e, raw_response_len = response.len(), raw_response = %response, "classifier LLM response was not valid ClassifierOutput JSON — falling back to default route");
            (
                ClassifierOutput {
                    action: "route".into(),
                    response: None,
                    target: Some(default_route.into()),
                    coherence_score: 1.0,
                    safety_score: 1.0,
                    complexity: None,
                    intent: None,
                    reason: format!("parse error: {e}"),
                    completeness: None,
                    risk: None,
                },
                false,
            )
        }
    }
}

fn check_thresholds(
    output: &ClassifierOutput,
    coherence_threshold: f64,
    safety_threshold: f64,
) -> Option<StageDecision> {
    let coherence_ok = output.coherence_score >= coherence_threshold;
    let safety_ok = output.safety_score >= safety_threshold;
    if coherence_ok && safety_ok {
        return None;
    }
    let reason = if coherence_ok {
        format!(
            "rejected: safety {:.2} below threshold {:.2}",
            output.safety_score, safety_threshold
        )
    } else {
        format!(
            "rejected: coherence {:.2} below threshold {:.2}",
            output.coherence_score, coherence_threshold
        )
    };
    Some(StageDecision {
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
    })
}

fn resolve_routing_target(
    action: &str,
    output: &ClassifierOutput,
    routing_config: &RoutingConfig,
) -> Option<serde_json::Value> {
    let min_complexity = output.complexity;
    let resolved_route = output.target.as_deref().unwrap_or(&routing_config.default_route);

    if action == "respond" {
        tracing::info!(target: "router.pipeline.stage2", "direct response — no dispatch");
        return None;
    }

    let route = if action == "route" {
        resolved_route
    } else {
        tracing::warn!(target: "router.pipeline.stage2", action = %action, fallback_route = %routing_config.default_route, "unknown action, falling back to default route");
        &routing_config.default_route
    };

    let resolved = routing_config.resolve_route(route, min_complexity);
    if let Some((model, model_name)) = &resolved {
        tracing::info!(target: "router.pipeline.stage2",
            route = %route,
            model = %model_name,
            endpoint = %model.endpoint,
            group = ?routing_config.routes.get(route).map(|r| &r.group),
            "routing target resolved"
        );
        Some(build_routing_target_value(route, model, model_name, routing_config))
    } else {
        tracing::warn!(target: "router.pipeline.stage2", route = %route, "resolve_route returned None — no dispatch target");
        None
    }
}

fn build_routing_target_value(
    route_name: &str,
    model: &crate::config::ModelEntry,
    model_name: &str,
    routing_config: &RoutingConfig,
) -> serde_json::Value {
    serde_json::json!({
        "url": model.endpoint,
        "model": model_name,
        "group": routing_config
            .routes
            .get(route_name)
            .or_else(|| routing_config.routes.get(&routing_config.default_route))
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
                content: system_prompt,
            },
            ChatMessage {
                role: "user".into(),
                content: input,
            },
        ];

        let response = match self.client.chat_complete(&messages) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(target: "router.pipeline.stage2", error = %e, "classifier LLM call failed");
                let output = ClassifierOutput {
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
                };
                let fallback_rt = resolve_routing_target(&output.action, &output, &self.routing_config);
                return Self::build_decision(&output, fallback_rt.as_ref(), false, self.score_matrix.as_ref());
            }
        };

    let (output, ok) = parse_classifier_response(&response, &self.routing_config.default_route);

        if let Some(decision) = check_thresholds(&output, self.coherence_threshold, self.routing_config.safety_threshold) {
            return WorkOutput::typed("rejected", &decision);
        }

        if output.action == "reject" {
            let decision = StageDecision {
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
            };
            return WorkOutput::typed("rejected", &decision);
        }

        let routing_target = resolve_routing_target(&output.action, &output, &self.routing_config);

        Self::build_decision(&output, routing_target.as_ref(), ok, self.score_matrix.as_ref())
    }
}

impl ClassifierStage {
    fn build_decision(
        output: &ClassifierOutput,
        routing_target: Option<&serde_json::Value>,
        ok: bool,
        score_matrix: Option<&ScoreMatrix>,
    ) -> Result<WorkOutput, WorkError> {
        let scored_routes = score_matrix.map(|sm| {
            let scores = std::collections::HashMap::from([
                ("coherence".into(), output.coherence_score),
                ("complexity".into(), f64::from(output.complexity.unwrap_or(DEFAULT_COMPLEXITY)) / COMPLEXITY_SCALE),
                ("completeness".into(), output.completeness.unwrap_or(DEFAULT_COMPLETENESS)),
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
        if let Some(rt) = routing_target {
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
                    output.action,
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
