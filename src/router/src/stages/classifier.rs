//! Stage 2: ClassifierStage — single LLM call that replaces QualityGate,
//! PlanningRefinement, and GuardrailCheck. Acts as an FSM switch: the LLM
//! returns either a direct response, a routing target, or a rejection.
//! Configurable via `RoutingConfig` from the top-level coral-router config.
//!
//! When `RouterConfig.classification` is `Some`, the stage becomes a thin
//! wrapper over the classification-tree engine (`stages::tree`): the engine
//! evaluates the nested tree recursively, auto-builds prompts from child keys
//! and descriptions, enforces per-node coherence/safety thresholds, and emits a
//! `StageDecision` per visited node plus a durable audit record. The flat path
//! below no longer guesses route names from classifier `action`/`intent`
//! strings — route selection is the tree's job.
//!
//! The LLM backend is injected as `Arc<dyn ChatBackend>` rather than a
//! concrete `LlmClient`, so mock/stub backends can be injected for testing
//! without duplicating the pipeline wiring.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;
use std::time::Instant;

use fluent_concurrency::pool::Limiter;
use fluent_llm::client::ChatBackend;
use fluent_llm::ChatMessage;
use fluent_wvr::prelude::*;

use crate::metrics::classify_error;

use crate::config::{ClassifierFailurePolicy, ClassifierOutput, RoutingConfig};
use crate::pipeline::RoutingTarget;
use crate::pipeline_types::{
    PipelineStage, StageDecision, StageDecisionProducer, StageMetadata, StageVerdict,
};
use crate::score_matrix::ScoreMatrix;
use crate::stages::common::{coerce_float, coerce_string, coerce_u8, extract_user_message};
use crate::stages::tree::ClassificationEngine;
use crate::target_match::TargetMatcher;

const DEFAULT_COMPLEXITY: u8 = 5;
const COMPLEXITY_SCALE: f64 = 10.0;
const DEFAULT_COMPLETENESS: f64 = 0.5;

/// Sanitize a classifier JSON blob by filling in missing required fields
/// with defaults, and coercing string-valued numbers back to numeric.
/// This lets partial or slightly malformed responses (common from smaller LLMs)
/// survive parsing instead of falling back to the default route.
///
/// The field-coercion lives in the shared `stages::common` helpers (the
/// "surviving normalization" both this stage and the tree engine use).
/// Route-name guessing is gone — the classification tree replaces it.
fn sanitize_classifier_json(v: &mut serde_json::Value) {
    if let Some(obj) = v.as_object_mut() {
        coerce_float(obj, "coherence_score", 1.0);
        coerce_float(obj, "safety_score", 1.0);
        coerce_float(obj, "completeness", 0.5);
        coerce_float(obj, "risk", 0.0);
        coerce_u8(obj, "complexity", 5);
        coerce_string(obj, "action", "route");
        coerce_string(obj, "reason", "");
    }
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

// LLM boundary: parse the classifier's raw LLM response text (JSON in a
// string from the model). Routes through the shared `fluent_llm::parse_typed`
// codec — direct-deserialize fast path, then fence-strip → parse →
// extract-first-value, then the shared `sanitize_classifier_json` coercion.
// `Ok((output, true))` = pristine parse; `Ok((output, false))` = recovered via
// sanitization; `Err(reason)` = total parse failure (the caller applies the
// `ClassifierFailurePolicy`).
fn parse_classifier_response(
    response: &str,
    default_route: &str,
) -> Result<(ClassifierOutput, bool), String> {
    match fluent_llm::parse_typed::<ClassifierOutput>(
        response,
        &serde_json::Value::Null,
        sanitize_classifier_json,
    ) {
        Ok(o) => {
            // `true` iff the raw text directly deserialized (the codec's fast
            // path); a small re-parse of the already-owned string, once per
            // classifier call — not on any hot loop.
            let ok = serde_json::from_str::<ClassifierOutput>(response).is_ok();
            let output = ClassifierOutput {
                target: o
                    .target
                    .clone()
                    .or_else(|| o.action.as_str().eq("route").then(|| default_route.into())),
                ..o
            };
            Ok((output, ok))
        }
        Err(fluent_llm::JsonParseError::NoJson) => {
            tracing::error!(
                target: "router.pipeline.stage2",
                raw_response_len = response.len(),
                "classifier LLM response was not valid JSON at all",
            );
            log_classifier_raw_response(response);
            Err("invalid JSON in LLM response".into())
        }
        Err(e) => {
            tracing::error!(
                target: "router.pipeline.stage2",
                error = %e,
                raw_response_len = response.len(),
                "classifier response survived sanitization but still failed to parse",
            );
            log_classifier_raw_response(response);
            Err(format!("post-sanitize parse error: {e}"))
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

/// Resolve a route to a typed `RoutingTarget` through the target-matching
/// ladder when one is available, falling back to the static
/// cheapest-qualifying pick.
///
/// The ladder only runs for 2+ member groups (a single-member group has
/// nothing to climb — it resolves statically, byte-identical to today, with no
/// extra LLM call). The matcher never fails hard: on absence, an unresolvable
/// group, an empty candidate list, or an internal `None`, the static
/// `routing_target` path runs (defense in depth).
///
/// This is the one selection algorithm shared by the LLM-driven flat path and
/// the matrix-authoritative branch (DRY — §4.3 of the roadmap).
fn resolve_via_matcher(
    routing_config: &RoutingConfig,
    matcher: Option<&TargetMatcher>,
    route: &str,
    complexity: Option<u8>,
    user_text: &str,
) -> Option<RoutingTarget> {
    if let Some(matcher) = matcher {
        if let Some(group) = routing_config.route_group(route) {
            let candidates = crate::target_match::candidates_for_group(routing_config, group);
            if candidates.len() >= 2 {
                if let Some(tm) = matcher.match_target(
                    route,
                    group,
                    routing_config,
                    &candidates,
                    complexity,
                    user_text,
                ) {
                    tracing::info!(
                        target: "router.pipeline.stage2",
                        route = %route,
                        group = %group,
                        model = %tm.primary.model,
                        assessments = tm.assessments.len(),
                        "routing target resolved via self-assessment ladder",
                    );
                    return Some(tm.primary);
                }
                tracing::warn!(
                    target: "router.pipeline.stage2",
                    route = %route,
                    "target-matching ladder produced no target — static fallback",
                );
            }
        }
    }
    routing_config.routing_target(route, complexity)
}

/// Resolve a classifier output to a typed `RoutingTarget`.
///
/// Only the three standard `action` values are honored:
/// - `respond` → `None` (direct response, handled by the caller),
/// - `route` → the explicit `target` or `default_route`,
/// - anything else → a warning and the `default_route` fallback.
///
/// The cleanup deleted `normalize_classifier_action`'s route-name
/// guessing from `action`/`intent` strings — route selection is the
/// classification tree's job now. Complexity-based model selection still flows
/// through the shared target resolver (`resolve_via_matcher`), which runs the
/// self-assessment ladder when the pipeline opts in.
fn resolve_routing_target(
    action: &str,
    output: &ClassifierOutput,
    routing_config: &RoutingConfig,
    matcher: Option<&TargetMatcher>,
    user_text: &str,
) -> Option<RoutingTarget> {
    if action == "respond" {
        tracing::info!(target: "router.pipeline.stage2", "direct response — no dispatch");
        return None;
    }

    let route = if action == "route" {
        output
            .target
            .as_deref()
            .unwrap_or(&routing_config.default_route)
    } else {
        tracing::warn!(target: "router.pipeline.stage2", action = %action, fallback_route = %routing_config.default_route, "unknown action, falling back to default route");
        &routing_config.default_route
    };

    if let Some(rt) = resolve_via_matcher(routing_config, matcher, route, output.complexity, user_text) {
        tracing::info!(target: "router.pipeline.stage2",
            route = %route,
            model = %rt.model,
            url = %rt.url,
            group = ?rt.group,
            idle_timeout_ms = rt.idle_timeout_ms,
            total_timeout_ms = rt.total_timeout_ms,
            retry_count = rt.retry_count,
            stream = rt.stream,
            filter_thinking = rt.filter_thinking,
            "routing target resolved"
        );
        return Some(rt);
    }
    tracing::warn!(target: "router.pipeline.stage2", route = %route, "resolve_route returned None — no dispatch target");
    None
}

pub struct ClassifierStage {
    name: ArcIntern<str>,
    client: Arc<dyn ChatBackend>,
    routing_config: RoutingConfig,
    coherence_threshold: f64,
    score_matrix: Option<ScoreMatrix>,
    /// When `true` and `score_matrix` is `Some`, the matrix's top route
    /// decides the dispatch target instead of the LLM's `action`/`target`
    /// being metadata-only. Thresholds/`reject` still gate first.
    score_matrix_authoritative: bool,
    classifier_intelligence: u8,
    /// Config key of the model the classifier dispatches to (e.g. "fast").
    /// Logged as structured context on every classifier LLM call so the
    /// inference sub-stage is attributable without re-deriving it from the
    /// client.
    classifier_model: String,
    /// Bounds concurrent classifier LLM calls. Each sync `chat_complete` runs
    /// through `run_sync`, which acquires a permit before invoking the backend.
    ///
    /// Long-term design (see `doc/router/VISION.md`) is a `ResultPool`-based
    /// parallel classifier fan-out; this limiter only bounds the current sync
    /// path so a burst cannot starve every tokio worker via `block_in_place`.
    limiter: Arc<Limiter>,
    /// The classification-tree engine. `Some` short-circuits `execute` into
    /// tree evaluation; the flat path below is then unused.
    tree_engine: Option<Arc<ClassificationEngine>>,
    /// The target-matching ladder. `Some` (pipeline `target_match:
    /// "self_assess"`) resolves routes through per-candidate self-assessment;
    /// `None` (`target_match: "static"`) keeps today's cheapest-qualifying
    /// pick. The tree engine shares the same matcher via this field.
    target_matcher: Option<TargetMatcher>,
    /// What to do when the classifier LLM call fails or its response cannot be
    /// parsed. Injected from `RouterConfig`; the stage never
    /// branches on config fields directly (DIP).
    failure_policy: ClassifierFailurePolicy,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl ClassifierStage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: Arc<dyn ChatBackend>,
        routing_config: RoutingConfig,
        coherence_threshold: f64,
        score_matrix: Option<ScoreMatrix>,
        score_matrix_authoritative: bool,
        classifier_intelligence: u8,
        classifier_model: impl Into<String>,
        limiter: Arc<Limiter>,
        target_matcher: Option<TargetMatcher>,
        failure_policy: ClassifierFailurePolicy,
    ) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage2.classifier"),
            client,
            routing_config,
            coherence_threshold,
            score_matrix,
            score_matrix_authoritative,
            classifier_intelligence,
            classifier_model: classifier_model.into(),
            limiter,
            tree_engine: None,
            target_matcher,
            failure_policy,
            depends: vec![ArcIntern::from("pipeline.stage1.output")],
            provides: vec![ArcIntern::from("pipeline.stage2.output")],
        }
    }

    /// Construct the stage in classification-tree mode.  The engine owns
    /// the tree, the routing table, the per-node backends, and the
    /// auto-constructed prompts; this stage only extracts the user message
    /// and converts the engine's final `StageDecision` into a `WorkOutput`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_tree(
        client: Arc<dyn ChatBackend>,
        routing_config: RoutingConfig,
        coherence_threshold: f64,
        score_matrix: Option<ScoreMatrix>,
        score_matrix_authoritative: bool,
        classifier_intelligence: u8,
        classifier_model: impl Into<String>,
        limiter: Arc<Limiter>,
        tree_engine: Arc<ClassificationEngine>,
        target_matcher: Option<TargetMatcher>,
        failure_policy: ClassifierFailurePolicy,
    ) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage2.classifier.tree"),
            client,
            routing_config,
            coherence_threshold,
            score_matrix,
            score_matrix_authoritative,
            classifier_intelligence,
            classifier_model: classifier_model.into(),
            limiter,
            tree_engine: Some(tree_engine),
            target_matcher,
            failure_policy,
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

        // Dispatch rules — auto-generated for routes configured `always_route`
        // (domains where the classifier model is overconfident and must never
        // answer directly, e.g. creative prose, code, translation, specialized
        // knowledge). Completely config-driven.
        let mut always_routes: Vec<&String> = route_names
            .iter()
            .copied()
            .filter(|n| {
                self.routing_config
                    .routes
                    .get(*n)
                    .is_some_and(|r| r.always_route)
            })
            .collect();
        if !always_routes.is_empty() {
            always_routes.sort();
            prompt.push_str("Dispatch rules (never answer these directly):\n");
            for name in &always_routes {
                let _ = writeln!(
                    prompt,
                    "  - Requests on the \"{name}\" route ALWAYS dispatch to \"{name}\". Set action=route and target=\"{name}\" even if the reasoning complexity seems low: these domains need the stronger model, not you."
                );
            }
            prompt.push('\n');
        }

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
            - If complexity <= {intel} (your intelligence level), set action=respond and answer directly — UNLESS the request matches a dispatch rule above.\n\
            - If complexity > {intel}, the query needs a more capable model: set action=route with target set to the matching route name. Code requests ALWAYS go to the \"code\" route. Translation requests ALWAYS go to the \"translation\" route.\n\
            - Dispatch rules above ALWAYS win: set action=route with the named target, never action=respond.\n\
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
        let (message, decision) = self.decide(ctx)?;
        WorkOutput::typed(message, &decision)
    }
}

impl ClassifierStage {
    fn decide(&self, ctx: &WorkContext) -> Result<(String, StageDecision), WorkError> {
        let input = extract_user_message(ctx)?;
        // The route the client requested (the request's `model`). Used to
        // enforce route-level `always_route`: a route configured to always
        // dispatch never lets the classifier answer directly.
        let requested_route: Option<String> = ctx
            .structured("request")
            .ok()
            .map(|r: crate::types::RouterRequest| r.model)
            .filter(|m| !m.is_empty());

        // Classification-tree mode — the engine produces the final
        // `StageDecision` directly (routing target, rejection, or the
        // `tree_path` of visited nodes in metadata).
        if let Some(engine) = &self.tree_engine {
            let evaluation = engine.evaluate(&input)?;
            return Ok(("classified".into(), evaluation.decision));
        }

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
                content: input.clone(),
            },
        ];

        tracing::info!(
            target: "router.pipeline.stage2.classifier",
            model = %self.classifier_model,
            input_len = messages[1].content.len(),
            system_prompt_len = messages[0].content.len(),
            retry_attempt = ctx
                .get::<i64>(crate::stages::retry_classifier::METADATA_RETRY_ATTEMPT)
                .copied(),
            "classifier LLM request"
        );

        let call_start = Instant::now();
        let mut llm_latency_ms = 0u64;
        let response = self.limiter.run_sync(|| async {
            let llm_start = Instant::now();
            let result = self.client.chat_complete(&messages);
            llm_latency_ms = llm_start.elapsed().as_millis() as u64;
            if let Ok(s) = &result {
                tracing::info!(
                    target: "router.pipeline.stage2.classifier",
                    model = %self.classifier_model,
                    llm_latency_ms = llm_latency_ms,
                    response_len = s.len(),
                    "classifier LLM call succeeded"
                );
            }
            result
        });
        let total_latency_ms = call_start.elapsed().as_millis() as u64;
        let limiter_wait_ms = total_latency_ms.saturating_sub(llm_latency_ms);

        let response = match response {
            Ok(r) => r,
            Err(e) => {
                let class = classify_error(&e.to_string());
                tracing::error!(target: "router.pipeline.stage2.classifier",
                    model = %self.classifier_model,
                    error = %e,
                    error_class = class.label(),
                    retryable = e.is_retryable(),
                    llm_latency_ms = llm_latency_ms,
                    total_latency_ms = total_latency_ms,
                    "classifier LLM call failed"
                );
                return Ok(self.failure_decision(&format!("LLM error: {e}")));
            }
        };

        tracing::info!(
            target: "router.pipeline.stage2.classifier",
            model = %self.classifier_model,
            total_latency_ms = total_latency_ms,
            llm_latency_ms = llm_latency_ms,
            limiter_wait_ms = limiter_wait_ms,
            "classifier call complete"
        );

        let (mut output, ok) = match parse_classifier_response(
            &response,
            &self.routing_config.default_route,
        ) {
            Ok(parsed) => parsed,
            Err(reason) => return Ok(self.failure_decision(&reason)),
        };

        if !ok {
            tracing::warn!(
                target: "router.pipeline.stage2.classifier",
                model = %self.classifier_model,
                raw_len = response.len(),
                recovered_reason = %output.reason,
                "classifier response recovered via sanitization/fallback"
            );
        }

        if let Some(decision) = check_thresholds(
            &output,
            self.coherence_threshold,
            self.routing_config.safety_threshold,
        ) {
            tracing::info!(
                target: "router.pipeline.stage2.classifier",
                model = %self.classifier_model,
                coherence_score = output.coherence_score,
                safety_score = output.safety_score,
                coherence_threshold = self.coherence_threshold,
                safety_threshold = self.routing_config.safety_threshold,
                reason = %decision.reason,
                "classifier threshold rejection"
            );
            return Ok(("rejected".into(), decision));
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
            return Ok(("rejected".into(), decision));
        }

        // Route-level `always_route` (config): a route configured to always
        // dispatch never lets the classifier answer directly. The classifier
        // model is often a small edge model overconfident in exactly these
        // domains (creative prose, code, translation, specialized knowledge),
        // so its `action=respond` is overridden to `action=route` toward the
        // requested route regardless of its complexity judgment. Simple prompts
        // on `local`-style routes keep the direct-answer path.
        if let Some(route) = requested_route.as_deref() {
            if self
                .routing_config
                .routes
                .get(route)
                .is_some_and(|r| r.always_route)
            {
                tracing::info!(
                    target: "router.pipeline.stage2",
                    model = %self.classifier_model,
                    route = %route,
                    classifier_action = %output.action,
                    "always_route override - dispatching instead of direct response",
                );
                output.action = "route".into();
                output.target = Some(route.to_string());
                output.response = None;
            }
        }

        // Matrix-authoritative routing. When opted in, the weighted score
        // matrix DECIDES the route — the top-scoring route's name is resolved
        // through the one shared dispatch path (`routing_config.routing_target`)
        // — instead of the LLM's `action`/`target` being metadata-only. The
        // coherence/safety thresholds and the `reject` action above run first:
        // gating checks still protect downstream models.
        if self.score_matrix_authoritative {
            if let Some(sm) = &self.score_matrix {
                let scores = Self::score_vector(&output);
                if let Some(top) = sm.resolve(&scores).first() {
                    tracing::info!(
                        target: "router.pipeline.stage2",
                        model = %self.classifier_model,
                        matrix_route = %top.route_name,
                        weighted_score = top.weighted_score,
                        "score matrix decided the route"
                    );
                    let routing_target = if top.route_name == "respond" {
                        // "respond" has no dispatch target — preserve a direct
                        // response (build_decision sets the response metadata).
                        None
                    } else {
                        resolve_via_matcher(
                            &self.routing_config,
                            self.target_matcher.as_ref(),
                            &top.route_name,
                            output.complexity,
                            &input,
                        )
                    };
                    return Ok(Self::build_decision(
                        &output,
                        routing_target.as_ref(),
                        ok,
                        Some(sm),
                    ));
                }
                // No route matched any band — fall through to the LLM path.
            }
        }

        let routing_target = resolve_routing_target(
            &output.action,
            &output,
            &self.routing_config,
            self.target_matcher.as_ref(),
            &input,
        );

        Ok(Self::build_decision(
            &output,
            routing_target.as_ref(),
            ok,
            self.score_matrix.as_ref(),
        ))
    }

    /// Apply the `ClassifierFailurePolicy` to a classifier outage (LLM error or
    /// total parse failure). The single DRY decision point — both fail-open
    /// paths funnel here.
    fn failure_decision(&self, reason: &str) -> (String, StageDecision) {
        match self.failure_policy {
            ClassifierFailurePolicy::Reject => {
                tracing::warn!(
                    target: "router.pipeline.stage2.classifier",
                    model = %self.classifier_model,
                    reason,
                    "classifier failure policy=reject — rejecting request"
                );
                let decision = StageDecision {
                    stage: PipelineStage::Classifier,
                    verdict: StageVerdict::Rejected,
                    score: None,
                    reason: format!("classifier failure: {reason}"),
                    latency_ms: 0,
                    metadata: serde_json::json!({
                        "failure": reason,
                        "failure_policy": "reject",
                        // `fallback: true` lets `RetryClassifier` detect this
                        // as a retryable outage and re-run with corrective
                        // prompts BEFORE the policy is applied. After retries
                        // exhaust, this Rejected verdict is returned as-is, so
                        // the safe default fails closed.
                        "fallback": true,
                    }),
                };
                ("rejected".into(), decision)
            }
            ClassifierFailurePolicy::RouteToDefaultTruthful => {
                let output = ClassifierOutput {
                    action: "route".into(),
                    response: None,
                    target: Some(self.routing_config.default_route.clone()),
                    coherence_score: 0.0,
                    safety_score: 0.0,
                    complexity: None,
                    intent: None,
                    reason: reason.into(),
                    completeness: None,
                    risk: None,
                };
                let fallback_rt = resolve_routing_target(
                    &output.action,
                    &output,
                    &self.routing_config,
                    self.target_matcher.as_ref(),
                    "",
                );
                Self::build_decision(
                    &output,
                    fallback_rt.as_ref(),
                    false,
                    self.score_matrix.as_ref(),
                )
            }
            ClassifierFailurePolicy::LegacyFailOpen => {
                let output = ClassifierOutput {
                    action: "route".into(),
                    response: None,
                    target: Some(self.routing_config.default_route.clone()),
                    coherence_score: 1.0,
                    safety_score: 1.0,
                    complexity: None,
                    intent: None,
                    reason: reason.into(),
                    completeness: None,
                    risk: None,
                };
                let fallback_rt = resolve_routing_target(
                    &output.action,
                    &output,
                    &self.routing_config,
                    self.target_matcher.as_ref(),
                    "",
                );
                Self::build_decision(
                    &output,
                    fallback_rt.as_ref(),
                    false,
                    self.score_matrix.as_ref(),
                )
            }
        }
    }
}

impl StageDecisionProducer for ClassifierStage {
    fn stage_kind(&self) -> PipelineStage {
        PipelineStage::Classifier
    }

    fn evaluate(
        &self,
        ctx: &WorkContext,
        _prior: &[StageDecision],
    ) -> Result<StageDecision, WorkError> {
        Ok(self.decide(ctx)?.1)
    }
}

impl ClassifierStage {
    /// The four-axis score vector the matrix ranks over: coherence, normalized
    /// complexity (0–1), completeness, and risk. The single source of the
    /// vector — shared by the matrix-authoritative `decide()` path and the
    /// `build_decision` audit metadata.
    fn score_vector(output: &ClassifierOutput) -> HashMap<String, f64> {
        std::collections::HashMap::from([
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
        ])
    }

    fn build_decision(
        output: &ClassifierOutput,
        routing_target: Option<&RoutingTarget>,
        ok: bool,
        score_matrix: Option<&ScoreMatrix>,
    ) -> (String, StageDecision) {
        let scored_routes = score_matrix.map(|sm| sm.resolve(&Self::score_vector(output)));

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
            metadata.set_routing_target(rt);
        } else if let Some(ref resp) = output.response {
            metadata.set_response(resp.clone());
        }

        (
            "classified".into(),
            StageDecision {
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

impl_fieldless!(ClassifierStage);

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
