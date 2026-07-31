//! Stage 2: ClassifierStage — single LLM call that replaces QualityGate,
//! PlanningRefinement, and GuardrailCheck. Acts as an FSM switch: the LLM
//! returns either a direct response, a routing target, or a rejection.
//! Configurable via `RoutingConfig` from the top-level coral-router config.
//!
//! The LLM backend is injected as `Arc<dyn ChatBackend>` rather than a
//! concrete `LlmClient`, so mock/stub backends can be injected for testing
//! without duplicating the pipeline wiring.

use std::fmt::Write;
use std::sync::Arc;

use fluent_concurrency::pool::Limiter;
use fluent_wvr::prelude::*;
use guidance_llm::client::ChatBackend;
use guidance_llm::ChatMessage;

use crate::config::{ClassifierOutput, RoutingConfig};
use crate::pipeline_types::{PipelineStage, StageDecision, StageMetadata, StageVerdict};
use crate::score_matrix::ScoreMatrix;
use crate::stages::common::extract_user_message;

const DEFAULT_COMPLEXITY: u8 = 5;
const COMPLEXITY_SCALE: f64 = 10.0;
const DEFAULT_COMPLETENESS: f64 = 0.5;

/// Sanitize a classifier JSON blob by filling in missing required fields
/// with defaults, and coercing string-valued numbers back to numeric.
/// This lets partial or slightly malformed responses (common from smaller LLMs)
/// survive parsing instead of falling back to the default route.
fn sanitize_classifier_json(mut v: serde_json::Value) -> serde_json::Value {
    let Some(obj) = v.as_object_mut() else {
        return v;
    };

    /// Ensure a floating-point field exists; coerce from string if needed.
    fn ensure_float(obj: &mut serde_json::Map<String, serde_json::Value>, key: &str, default: f64) {
        match obj.get(key) {
            None => {
                let n = serde_json::Number::from_f64(default)
                    .unwrap_or_else(|| serde_json::Number::from_f64(0.0).unwrap());
                obj.insert(key.into(), serde_json::Value::Number(n));
            }
            Some(serde_json::Value::String(s)) => {
                if let Ok(n) = s.parse::<f64>() {
                    if let Some(num) = serde_json::Number::from_f64(n) {
                        obj[key] = serde_json::Value::Number(num);
                    }
                }
            }
            _ => {}
        }
    }

    /// Ensure an unsigned-integer field exists; coerce from float or string.
    fn ensure_u8(obj: &mut serde_json::Map<String, serde_json::Value>, key: &str, default: u8) {
        match obj.get(key) {
            None => {
                obj.insert(
                    key.into(),
                    serde_json::Value::Number(serde_json::Number::from(default)),
                );
            }
            Some(serde_json::Value::Number(n)) => {
                // serde_json::Number can be float or integer; extract as u8
                if let Some(i) = n.as_u64() {
                    obj[key] = serde_json::Value::Number(serde_json::Number::from(
                        i.min(u64::from(u8::MAX)) as u8,
                    ));
                } else if let Some(f) = n.as_f64() {
                    let i = f.round() as u64;
                    obj[key] = serde_json::Value::Number(serde_json::Number::from(
                        i.min(u64::from(u8::MAX)) as u8,
                    ));
                }
            }
            Some(serde_json::Value::String(s)) => {
                if let Ok(i) = s.parse::<u8>() {
                    obj[key] = serde_json::Value::Number(serde_json::Number::from(i));
                } else if let Ok(f) = s.parse::<f64>() {
                    let i = f.round() as u8;
                    obj[key] = serde_json::Value::Number(serde_json::Number::from(i));
                }
            }
            _ => {}
        }
    }

    fn ensure_string(
        obj: &mut serde_json::Map<String, serde_json::Value>,
        key: &str,
        default: &str,
    ) {
        if !obj.contains_key(key) {
            obj.insert(key.into(), serde_json::Value::String(default.into()));
        }
    }

    ensure_float(obj, "coherence_score", 1.0);
    ensure_float(obj, "safety_score", 1.0);
    ensure_float(obj, "completeness", 0.5);
    ensure_float(obj, "risk", 0.0);
    ensure_u8(obj, "complexity", 5);
    ensure_string(obj, "action", "route");
    ensure_string(obj, "reason", "");

    // Normalize action: if it's not one of the three standard values but
    // looks like a plausible route name, treat it as a route action and
    // promote the value to target.
    if let Some(serde_json::Value::String(action)) = obj.get("action") {
        let action_lower = action.to_lowercase();
        let is_standard =
            action_lower == "route" || action_lower == "respond" || action_lower == "reject";
        let target_missing = obj
            .get("target")
            .and_then(|t| t.as_str())
            .is_none_or(str::is_empty);
        if !is_standard && target_missing {
            obj.insert(
                "target".into(),
                serde_json::Value::String(action_lower.clone()),
            );
            obj.insert("action".into(), serde_json::Value::String("route".into()));
        }
    }

    v
}

/// Log the raw classifier response with clear delimiters so multiline content
/// is visible in the log output instead of being hidden by structured key-value
/// formatting (which breaks on embedded newlines).
fn log_classifier_raw_response(response: &str) {
    tracing::debug!(
        target: "router.pipeline.stage2",
        "--- raw classifier response ({} bytes) ---\n{}--- end raw response ---",
        response.len(),
        response,
    );
}

fn parse_classifier_response(response: &str, default_route: &str) -> (ClassifierOutput, bool) {
    // Fast path: try direct parse first
    if let Ok(o) = serde_json::from_str::<ClassifierOutput>(response) {
        return (o, true);
    }

    // Slow path: sanitize partial JSON
    let raw: serde_json::Value = match serde_json::from_str(response) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                target: "router.pipeline.stage2",
                error = %e,
                raw_response_len = response.len(),
                error_line = e.line(),
                error_column = e.column(),
                "classifier LLM response was not valid JSON at all — falling back to default route",
            );
            log_classifier_raw_response(response);
            return fallback_parse(default_route, &format!("invalid JSON: {e}"));
        }
    };

    let sanitized = sanitize_classifier_json(raw);
    match serde_json::from_value::<ClassifierOutput>(sanitized) {
        Ok(o) => {
            // If the LLM set action=route but omitted target, use default
            let output = ClassifierOutput {
                target: o
                    .target
                    .clone()
                    .or_else(|| o.action.as_str().eq("route").then(|| default_route.into())),
                ..o
            };
            (output, false) // false = sanitized, not pristine
        }
        Err(e) => {
            tracing::error!(
                target: "router.pipeline.stage2",
                error = %e,
                raw_response_len = response.len(),
                "classifier response survived sanitization but still failed to parse — falling back to default route",
            );
            log_classifier_raw_response(response);
            fallback_parse(default_route, &format!("post-sanitize parse error: {e}"))
        }
    }
}

fn fallback_parse(default_route: &str, reason: &str) -> (ClassifierOutput, bool) {
    (
        ClassifierOutput {
            action: "route".into(),
            response: None,
            target: Some(default_route.into()),
            coherence_score: 1.0,
            safety_score: 1.0,
            complexity: None,
            intent: None,
            reason: reason.into(),
            completeness: None,
            risk: None,
        },
        false,
    )
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

/// Normalize non-standard `action` / `intent` values that look like route names.
///
/// The classifier LLM often identifies the correct intent (e.g. `"intent": "code"`)
/// but sets the wrong action (e.g. `"action": "respond"` or `"action": "code"`).
///
/// Override rules (applied in order):
///
/// 1. If `action` is `respond` AND `intent` is a known route AND the query
///    complexity exceeds the classifier's intelligence, promote to `route` with
///    `target=intent` — the classifier is not capable enough for this query.
/// 2. If `action` is a non-standard value that is a known route name, treat it
///    as `route` with that target.
fn normalize_classifier_action<'a>(
    action: &'a str,
    target: Option<&'a str>,
    intent: Option<&'a str>,
    complexity: Option<u8>,
    classifier_intelligence: u8,
    routes: &std::collections::HashMap<String, crate::config::RouteRef>,
) -> (&'a str, Option<&'a str>) {
    // Intent-driven override: promote respond → route when the query exceeds
    // the classifier's capability, even if the LLM thought it could handle it.
    if let Some(intent_route) = intent {
        let intent_is_route = routes.contains_key(intent_route) || intent_route == "local";
        let target_matches = target.is_none_or(|t| t == intent_route || t.is_empty());
        let exceeds_capability = complexity.is_none_or(|c| c > classifier_intelligence);
        if intent_is_route && target_matches && action != "reject" && exceeds_capability {
            return ("route", Some(intent_route));
        }
    }

    let is_standard = action == "route" || action == "respond" || action == "reject";
    if is_standard {
        return (action, target);
    }
    // Non-standard action: if it's a route name, treat it as route + target
    if routes.contains_key(action) || action == "local" {
        return ("route", Some(action));
    }
    // If target is missing and action isn't a route, set target to action anyway
    // so the downstream "unknown action" log is accurate
    if target.is_none() && routes.contains_key("local") {
        return (action, Some("local"));
    }
    (action, target)
}

fn resolve_routing_target(
    action: &str,
    output: &ClassifierOutput,
    routing_config: &RoutingConfig,
    classifier_intelligence: u8,
) -> Option<serde_json::Value> {
    let min_complexity = output.complexity;
    let (normalized_action, normalized_target) = normalize_classifier_action(
        action,
        output.target.as_deref(),
        output.intent.as_deref(),
        output.complexity,
        classifier_intelligence,
        &routing_config.routes,
    );
    let resolved_route = normalized_target.unwrap_or(&routing_config.default_route);

    if normalized_action == "respond" {
        tracing::info!(target: "router.pipeline.stage2", "direct response — no dispatch");
        return None;
    }

    let route = if normalized_action == "route" {
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
        Some(build_routing_target_value(
            route,
            model,
            model_name,
            routing_config,
            min_complexity,
        ))
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
    min_complexity: Option<u8>,
) -> serde_json::Value {
    /// Build a single routing target JSON value from a model entry + name pair.
    fn target_json(name: &str, entry: &crate::config::ModelEntry) -> serde_json::Value {
        serde_json::json!({
            "url": entry.endpoint,
            "model": name,
            "params": entry.params,
            "filter_thinking": entry.filter_thinking,
            "retry_count": entry.retry_count,
            "retry_base_interval_s": entry.retry_base_interval_s,
            "stream": entry.stream,
            "idle_timeout_ms": entry.idle_timeout_ms,
            "total_timeout_ms": entry.total_timeout_ms,
        })
    }

    let group = routing_config
        .routes
        .get(route_name)
        .or_else(|| routing_config.routes.get(&routing_config.default_route))
        .map_or(String::new(), |r| r.group.clone());

    let fallbacks: Vec<serde_json::Value> = routing_config
        .all_dispatch_targets(route_name, min_complexity)
        .into_iter()
        .skip(1) // skip the primary (already included)
        .map(|(name, entry)| target_json(&name, &entry))
        .collect();

    let mut rt = target_json(model_name, model);
    rt["group"] = serde_json::Value::String(group);
    rt["target_name"] = serde_json::Value::String(route_name.to_string());
    rt["fallbacks"] = serde_json::Value::Array(fallbacks);
    rt
}

pub struct ClassifierStage {
    name: ArcIntern<str>,
    client: Arc<dyn ChatBackend>,
    routing_config: RoutingConfig,
    coherence_threshold: f64,
    score_matrix: Option<ScoreMatrix>,
    classifier_intelligence: u8,
    /// Bounds concurrent classifier LLM calls. Each sync `chat_complete` runs
    /// through `run_sync`, which acquires a permit before invoking the backend.
    ///
    /// Long-term design (see `doc/router/VISION.md`) is a `ResultPool`-based
    /// parallel classifier fan-out; this limiter only bounds the current sync
    /// path so a burst cannot starve every tokio worker via `block_in_place`.
    limiter: Arc<Limiter>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl ClassifierStage {
    pub fn new(
        client: Arc<dyn ChatBackend>,
        routing_config: RoutingConfig,
        coherence_threshold: f64,
        score_matrix: Option<ScoreMatrix>,
        classifier_intelligence: u8,
        limiter: Arc<Limiter>,
    ) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage2.classifier"),
            client,
            routing_config,
            coherence_threshold,
            score_matrix,
            classifier_intelligence,
            limiter,
            depends: vec![ArcIntern::from("pipeline.stage1.output")],
            provides: vec![ArcIntern::from("pipeline.stage2.output")],
        }
    }

    fn build_system_prompt(&self) -> String {
        let mut prompt = String::new();

        // Preamble from config — variable substitution still applies
        let preamble = self
            .routing_config
            .system_prompt
            .replace(
                "{coherence_threshold}",
                &format!("{:.2}", self.coherence_threshold),
            )
            .replace(
                "{safety_threshold}",
                &format!("{:.2}", self.routing_config.safety_threshold),
            );
        if !preamble.is_empty() {
            let _ = writeln!(prompt, "{preamble}\n");
        }

        // Available routes — auto-generated from config to eliminate DRY violation
        prompt.push_str("Available routes:\n");
        let mut route_names: Vec<&String> = self.routing_config.routes.keys().collect();
        route_names.sort();
        for name in &route_names {
            let desc = &self.routing_config.routes[*name].description;
            if desc.is_empty() {
                let _ = writeln!(prompt, "  - {name}");
            } else {
                let _ = writeln!(prompt, "  - {name}: {desc}");
            }
        }
        prompt.push('\n');

        // Output schema — derived from ClassifierOutput
        let intent_values: Vec<String> = route_names.iter().map(|n| format!("\"{n}\"")).collect();
        let intent_enum = if intent_values.is_empty() {
            String::from("\"question\" | \"command\" | \"code\"")
        } else {
            intent_values.join(" | ")
        };

        let coherence = format!("{:.2}", self.coherence_threshold);
        let safety = format!("{:.2}", self.routing_config.safety_threshold);
        let intel = self.classifier_intelligence;
        let _ = write!(
            prompt,
            "Output schema:\n\
            {{\n\
            \x20 \"action\": \"respond\" | \"route\" | \"reject\",\n\
            \x20 \"response\": \"direct answer (only if action=respond)\",\n\
            \x20 \"target\": \"route name (only if action=route)\",\n\
            \x20 \"coherence_score\": 0.0-1.0,\n\
            \x20 \"safety_score\": 0.0-1.0,\n\
            \x20 \"complexity\": 0-10,\n\
            \x20 \"intent\": {intent_enum},\n\
            \x20 \"completeness\": 0.0-1.0,\n\
            \x20 \"risk\": 0.0-1.0,\n\
            \x20 \"reason\": \"brief explanation\"\n\
            }}\n\n\
            Response rules:\n\
            - If complexity <= {intel} (your intelligence level), set action=respond and answer directly.\n\
            - If complexity > {intel}, the query needs a more capable model: set action=route with target set to the matching route name. Code requests ALWAYS go to the \"code\" route. Translation requests ALWAYS go to the \"translation\" route.\n\
            - If content is incoherent (coherence_score < {coherence}), set action=reject.\n\
            - If content is unsafe (safety_score < {safety}), set action=reject.\n\
            - Safety score 1.0 = completely safe, 0.0 = dangerous.\n\
            - Complexity 0 = trivial, 10 = extremely complex.\n\
            - Only output JSON, no other text.\n"
        );

        prompt
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

        let response = match self
            .limiter
            .run_sync(|| async { self.client.chat_complete(&messages) })
        {
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
                let fallback_rt = resolve_routing_target(
                    &output.action,
                    &output,
                    &self.routing_config,
                    self.classifier_intelligence,
                );
                return Self::build_decision(
                    &output,
                    fallback_rt.as_ref(),
                    false,
                    self.score_matrix.as_ref(),
                );
            }
        };

        let (output, ok) = parse_classifier_response(&response, &self.routing_config.default_route);

        if let Some(decision) = check_thresholds(
            &output,
            self.coherence_threshold,
            self.routing_config.safety_threshold,
        ) {
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

        let routing_target = resolve_routing_target(
            &output.action,
            &output,
            &self.routing_config,
            self.classifier_intelligence,
        );

        Self::build_decision(
            &output,
            routing_target.as_ref(),
            ok,
            self.score_matrix.as_ref(),
        )
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
                (
                    "complexity".into(),
                    f64::from(output.complexity.unwrap_or(DEFAULT_COMPLEXITY)) / COMPLEXITY_SCALE,
                ),
                (
                    "completeness".into(),
                    output.completeness.unwrap_or(DEFAULT_COMPLETENESS),
                ),
                ("risk".into(), output.risk.unwrap_or(0.0)),
            ]);
            sm.resolve(&scores)
        });

        let mut metadata = StageMetadata::from(serde_json::json!({
            "coherence_score": output.coherence_score,
            "safety_score": output.safety_score,
            "complexity": output.complexity,
            "completeness": output.completeness,
            "risk": output.risk,
            "intent": output.intent,
            "action": output.action,
            "reason": output.reason,
            "fallback": !ok,
        }));

        if let Some(ref routes) = scored_routes {
            if let Some(top) = routes.first() {
                metadata.insert(
                    "scored_route",
                    serde_json::json!({
                        "route": top.route_name,
                        "score": top.weighted_score,
                        "score_vector": top.score_vector.iter().map(|(d, s)| {
                            serde_json::json!({"dimension": d, "score": s})
                        }).collect::<Vec<_>>(),
                    }),
                );
            }
            metadata.insert(
                "scored_routes",
                serde_json::Value::Array(
                    routes
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "route": r.route_name,
                                "score": r.weighted_score,
                            })
                        })
                        .collect(),
                ),
            );
        }

        if let Some(rt) = routing_target {
            // When we have a routing target, the response is from a misbehaving
            // LLM that output both action=respond + code.  Don't store it as a
            // classifier response — the handler will dispatch instead.
            if let Ok(typed) = serde_json::from_value::<crate::pipeline::RoutingTarget>(rt.clone())
            {
                metadata.set_routing_target(&typed);
            }
        } else if let Some(ref resp) = output.response {
            metadata.set_response(resp.clone());
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
                metadata: metadata.into_value(),
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
