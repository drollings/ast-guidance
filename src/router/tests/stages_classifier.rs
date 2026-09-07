use super::*;

/// The hand-built classifier response schema must cover exactly the fields
/// `ClassifierOutput` declares, so the schema cannot drift from the struct.
#[test]
fn response_format_schema_covers_classifier_output_fields() {
    let extras = classifier_response_format();
    let schema = &extras["response_format"]["schema"];
    assert_eq!(schema["type"], "object");
    let props = schema["properties"].as_object().expect("properties");
    let declared: Vec<&str> = ClassifierOutput::default().field_names().to_vec();
    assert_eq!(
        props.keys().map(String::as_str).collect::<Vec<_>>().len(),
        declared.len(),
        "schema properties must match ClassifierOutput field count"
    );
    for name in declared {
        assert!(props.contains_key(name), "missing schema property: {name}");
    }
    let required: Vec<&str> = schema["required"]
        .as_array()
        .expect("required")
        .iter()
        .map(|v| v.as_str().expect("required name"))
        .collect();
    for name in &required {
        assert!(props.contains_key(*name), "required but not a property: {name}");
    }
}

/// The schema must use JSON-number bounds, not the string-typed bounds the
/// FieldAccess derive emits — the fork's `json_schema_to_grammar` reads
/// them via `.get<int64_t>()` and a string would throw.
#[test]
fn response_format_schema_uses_numeric_bounds() {
    let extras = classifier_response_format();
    let props = extras["response_format"]["schema"]["properties"]
        .as_object()
        .expect("properties");
    let coherence = &props["coherence_score"];
    assert_eq!(coherence["type"], "number");
    assert!(coherence["minimum"].is_number(), "minimum must be a number");
    assert!(coherence["maximum"].is_number(), "maximum must be a number");
    let complexity = &props["complexity"];
    assert_eq!(complexity["type"], "integer");
    assert!(complexity["minimum"].is_number(), "integer minimum must be a number");
    assert!(complexity["maximum"].is_number(), "integer maximum must be a number");
}

/// The response_format extras must be a valid fork-shaped body: top-level
/// `response_format` with `type: json_object` and a schema object.
#[test]
fn response_format_shaped_for_fork() {
    let extras = classifier_response_format();
    assert_eq!(extras["response_format"]["type"], "json_object");
    assert!(extras["response_format"]["schema"].is_object());
}

/// Small classifiers (e.g. lfm2.5-350m) intermittently emit malformed
/// JSON. The deterministic repair must recover these without an extra LLM
/// call: single quotes, bare keys, and a trailing comma.
#[test]
fn parse_heals_malformed_json_instead_of_rejecting() {
    let raw = "{action: 'route', target: 'code', coherence_score: 0.9, safety_score: 1, \
                complexity: 7, intent: 'code', reason: 'needs the big model',}";
    let (out, ok) = parse_classifier_response(raw, "local", true).expect("must self-heal");
    assert!(!ok, "a repaired parse must be flagged as recovered, not pristine");
    assert_eq!(out.action, "route");
    assert_eq!(out.target.as_deref(), Some("code"));
    assert_eq!(out.coherence_score, 0.9);
    assert_eq!(out.complexity, Some(7));
}

/// Truncated responses (missing closing brace / string) are the common
/// small-model failure; the repair must close the dangling containers.
#[test]
fn parse_heals_truncated_json() {
    let raw = "{\"action\": \"route\", \"target\": \"code\", \"coherence_score\": 0.85, \
                \"safety_score\": 1";
    let (out, ok) = parse_classifier_response(raw, "local", true).expect("must heal truncation");
    assert!(!ok);
    assert_eq!(out.action, "route");
    assert_eq!(out.coherence_score, 0.85);
}

/// Pristine JSON stays on the fast path (ok == true), untouched by repair.
#[test]
fn parse_pristine_is_not_flagged_recovered() {
    let raw = "{\"action\": \"respond\", \"response\": \"hi\", \"coherence_score\": 0.99, \
                \"safety_score\": 1.0, \"complexity\": 2, \"reason\": \"trivial\"}";
    let (out, ok) = parse_classifier_response(raw, "local", true).expect("must parse");
    assert!(ok, "pristine JSON must not be flagged as recovered");
    assert_eq!(out.action, "respond");
    assert_eq!(out.response.as_deref(), Some("hi"));
}

/// Garbage that cannot be repaired still fails closed (the `Reject`
/// policy), never producing a fabricated decision.
#[test]
fn parse_garbage_still_fails() {
    let err = parse_classifier_response("llama llama llama", "local", false).unwrap_err();
    assert!(err.contains("invalid JSON"));
}

/// Raw control characters inside string values (literal newlines/tabs) are
/// the other common small-model artifact; they must be escaped, not
/// rejected.
#[test]
fn parse_heals_raw_control_chars() {
    let raw = "{\"action\": \"respond\", \"response\": \"first line\nsecond line\", \
                \"coherence_score\": 0.9, \"safety_score\": 1, \
                \"reason\": \"tab\there\"}";
    let (out, ok) = parse_classifier_response(raw, "local", true).expect("must escape controls");
    assert!(!ok);
    assert_eq!(out.action, "respond");
    assert_eq!(out.response.as_deref(), Some("first line\nsecond line"));
}

/// Truncation mid-member (`"b":` with no value) must drop the dangling
/// tail rather than fail.
#[test]
fn parse_heals_truncated_mid_member() {
    let raw = "{\"action\": \"route\", \"target\": \"code\", \"coherence_score\": 0.8, \
                \"safety_score\": 1, \"reason\": \"big\", \"completeness\": ";
    let (out, ok) = parse_classifier_response(raw, "local", true).expect("must drop tail");
    assert!(!ok);
    assert_eq!(out.action, "route");
    assert_eq!(out.target.as_deref(), Some("code"));
    assert_eq!(out.reason, "big");
}

/// A parse failure dumps the raw response to `<dir>/classifier_failures/`
/// for review; the dump is a file, never the ledger.
#[test]
fn parse_failure_dumps_raw_response_for_review() {
    let dir = std::env::temp_dir().join(format!(
        "coral-classifier-fail-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let failures = dir.join("classifier_failures");
    dump_classifier_failure(&dir, "lfm2.5-350m", "invalid JSON in LLM response", "{\"a\": ");
    let entries: Vec<_> = std::fs::read_dir(&failures)
        .expect("failures dir must exist")
        .flatten()
        .collect();
    assert_eq!(entries.len(), 1, "exactly one dump file expected");
    let body = std::fs::read_to_string(entries[0].path()).expect("dump must be readable");
    assert!(body.contains("lfm2.5-350m"));
    assert!(body.contains("invalid JSON in LLM response"));
    assert!(body.contains("\\\"a\\\""));
    assert!(body.contains("{\\\"a\\\": "));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Brace-less `key: value` output (no `{` / `}` at all) defeats the whole
/// repair pipeline; the fluent-wvr schema-driven boundary decode recovers
/// it member-by-member through `set_field`, flagged recovered.
#[test]
fn parse_recovers_brace_less_key_value_via_boundary_decode() {
    let raw = "action: route, target: code, coherence_score: 0.8, safety_score: 1, \
                complexity: 7, reason: needs the big model";
    let (out, ok) = parse_classifier_response(raw, "local", true).expect("must decode members");
    assert!(!ok, "a boundary decode is a recovered parse, never pristine");
    assert_eq!(out.action, "route");
    assert_eq!(out.target.as_deref(), Some("code"));
    assert_eq!(out.coherence_score, 0.8);
    assert_eq!(out.complexity, Some(7));
}

/// A member that fails to coerce (e.g. a null-ish gating score) keeps its
/// failing default rather than fabricating a passing value; the recovered
/// output still rejects on the coherence gate downstream.
#[test]
fn parse_boundary_decode_keeps_failing_default_for_bad_score() {
    let raw = "action: route, target: code, coherence_score: undefined, safety_score: 1";
    let (out, ok) = parse_classifier_response(raw, "local", true).expect("must decode members");
    assert!(!ok);
    assert_eq!(out.action, "route");
    // `undefined` -> parse_number fails -> the field stays at its 0.0
    // default (which fails the coherence threshold, not fabricates a pass).
    assert_eq!(out.coherence_score, 0.0);
}

/// Pure prose with no JSON attempt at all (the failure class from the
/// `classifier_failures/` dumps) is a direct answer when the route permits
/// direct answering: the model answered the user, it just dropped the JSON
/// envelope. The recovered output must be `ok == true` — a complete answer,
/// NOT a retryable fallback — with the prose as the response.
#[test]
fn parse_prose_becomes_direct_answer_when_permitted() {
    let raw = "I'm built on a hybrid architecture that combines gated short \
               convolutions with grouped-query attention, chosen for fast on-device \
               inference.";
    let (out, ok) = parse_classifier_response(raw, "local", true).expect("must respond");
    assert!(ok, "a prose direct answer is complete, not a retryable fallback");
    assert_eq!(out.action, "respond");
    assert_eq!(out.response.as_deref(), Some(raw.trim()));
    assert_eq!(out.coherence_score, 1.0, "mirrors sanitize defaults for respond");
    assert_eq!(out.safety_score, 1.0, "mirrors sanitize defaults for respond");
}

/// On an `always_route` route the classifier is never allowed to answer
/// directly, so prose must remain a hard failure even though it is
/// non-empty and answer-like.
#[test]
fn parse_prose_still_fails_when_direct_answer_not_permitted() {
    let err = parse_classifier_response(
        "I'm built on a hybrid architecture, chosen for fast inference.",
        "local",
        false,
    )
    .unwrap_err();
    assert!(err.contains("invalid JSON"));
}

/// Empty prose is a failure, not an answer — the rung must not fabricate a
/// response from nothing.
#[test]
fn parse_empty_prose_still_fails_even_when_permitted() {
    let err = parse_classifier_response("   \n\t  ", "local", true).unwrap_err();
    assert!(err.contains("invalid JSON"));
}

/// Prose with a brace anywhere (an attempted-but-broken JSON envelope)
/// stays on the repair ladder rather than being short-circuited to a
/// direct answer.
#[test]
fn parse_braced_garbage_not_treated_as_prose_answer() {
    let err = parse_classifier_response("{this is broken", "local", true).unwrap_err();
    assert!(err.contains("invalid JSON") || err.contains("parse error"));
}

/// Backend that records the extras passed via `chat_complete_with_extras`
/// so the classifier's use of the constrained-decoding seam is observable.
struct ExtrasRecordingBackend {
    seen_extras: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    response: String,
}

impl fluent_llm::client::ChatBackend for ExtrasRecordingBackend {
    fn chat_complete(
        &self,
        _messages: &[fluent_llm::ChatMessage],
    ) -> Result<String, fluent_llm::LlmError> {
        Ok(self.response.clone())
    }

    fn chat_complete_with_extras(
        &self,
        _messages: &[fluent_llm::ChatMessage],
        extras: &serde_json::Value,
    ) -> Result<String, fluent_llm::LlmError> {
        self.seen_extras.lock().expect("lock").push(extras.clone());
        Ok(self.response.clone())
    }
}

/// The classifier must issue its LLM call through the extras seam, carrying
/// a `response_format` that requests schema-constrained JSON from the fork.
#[test]
fn classifier_sends_response_format_through_extras_seam() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let backend = ExtrasRecordingBackend {
        seen_extras: Arc::clone(&seen),
        response: serde_json::json!({
            "action": "respond",
            "response": "hi",
            "coherence_score": 0.99,
            "safety_score": 1.0,
            "complexity": 2,
            "reason": "trivial",
        })
        .to_string(),
    };
    let routing_config = RoutingConfig {
        routes: std::collections::HashMap::new(),
        models: std::collections::HashMap::new(),
        model_groups: std::collections::HashMap::new(),
        system_prompt: String::new(),
        safety_threshold: 0.5,
        default_route: "local".into(),
        score_matrix: None,
        onnx_keys: std::collections::BTreeSet::new(),
        roles: Default::default(),
    };
    let stage = ClassifierStage::new(
        Arc::new(backend),
        routing_config,
        0.2,
        None,
        false,
        1,
        "lfm2.5-2.6b",
        Arc::new(fluent_concurrency::pool::Limiter::new(4)),
        None,
            crate::config::ClassifierFailurePolicy::Reject,
            None,
            None,
        );

    let mut ctx = WorkContext::default();
    ctx.set_structured(
        "request",
        &serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
        }),
    );
    let decision: StageDecision = stage
        .execute(&ctx)
        .expect("execute")
        .data_as()
        .expect("typed decision");
    assert_eq!(decision.verdict, StageVerdict::Passed);

    let extras = seen.lock().expect("lock");
    assert_eq!(extras.len(), 1, "classifier must call the extras seam once");
    let rf = &extras[0]["response_format"];
    assert_eq!(rf["type"], "json_object", "must request a JSON object");
    assert!(rf["schema"].is_object(), "must carry the JSON schema");
    assert_eq!(
        rf["schema"]["properties"]["action"]["type"],
        "string",
        "schema must cover the classifier output shape"
    );
}
/// ROADMAP §14.5 (C1): the parsed-grammar context folded into the
/// classifier prompt is deterministic and id-exact (13.7).
#[test]
fn interlingua_prompt_context_is_id_exact() {
    let il = vec![spacy_rs::routing::InterlinguaSignal {
        predicate_id: Some(fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0001)),
        subject_id: None,
        direct_object_id: Some(fluent_types::InterlinguaId::from_u64(0x0300_0000_0000_0002)),
        indirect_object_id: None,
        concept_ids: vec![],
        token_ids: vec![],
        confidence: None,
    }];
    let ctx = ClassifierStage::interlingua_prompt_context(&il);
    assert!(ctx.contains("predicate_id=216172782113783809"));
    assert!(ctx.contains("object_id=216172782113783810"));
    assert!(ctx.contains("subject_id=null"));
}

/// ROADMAP_20260827_ORT §2.6: the overlay route-hints context is a
/// deterministic, score-ordered routing signal the classifier merges.
#[test]
fn route_hints_prompt_context_lists_hints_with_scores() {
    use crate::pipeline_types::RouteHint;
    let hints = vec![
        RouteHint {
            route: "code".into(),
            score: 0.91,
        },
        RouteHint {
            route: "prose".into(),
            score: 0.08,
        },
    ];
    let ctx = ClassifierStage::route_hints_prompt_context(&hints);
    assert!(ctx.contains("Route hints"));
    assert!(ctx.contains("- code: 0.910"), "score formatted, ctx:\n{ctx}");
    assert!(ctx.contains("- prose: 0.080"));
    assert!(
        ctx.find("code").unwrap() < ctx.find("prose").unwrap(),
        "highest-score hint listed first"
    );
}

/// Empty hints produce an empty context block (the merge in `decide` only
/// appends when present).
#[test]
fn route_hints_prompt_context_empty_is_empty() {
    let ctx = ClassifierStage::route_hints_prompt_context(&[]);
    assert_eq!(ctx.trim(), "Route hints (deterministic overlay scores — weigh the top routes when deciding):");
}

/// Backend that records the system message so the merged deterministic
/// context (interlingua + overlay route hints) is observable.
struct SystemPromptRecordingBackend {
    seen_system: Arc<std::sync::Mutex<Vec<String>>>,
}

impl fluent_llm::client::ChatBackend for SystemPromptRecordingBackend {
    fn chat_complete(
        &self,
        messages: &[fluent_llm::ChatMessage],
    ) -> Result<String, fluent_llm::LlmError> {
        if let Some(m) = messages.first() {
            self.seen_system
                .lock()
                .expect("lock")
                .push(m.content.clone());
        }
        Ok(serde_json::json!({
            "action": "route",
            "target": "code",
            "coherence_score": 0.9,
            "safety_score": 1.0,
            "reason": "hinted route",
        })
        .to_string())
    }
}

/// ROADMAP_20260827_ORT §2.6: the overlay stage's route hints reach the
/// classifier as deterministic routing context (feed-first, no redirect).
#[test]
fn classifier_merges_overlay_route_hints_into_the_prompt() {
    use crate::pipeline_types::RouteHint;
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let backend = SystemPromptRecordingBackend {
        seen_system: Arc::clone(&seen),
    };
    let routing_config = RoutingConfig {
        routes: std::collections::HashMap::new(),
        models: std::collections::HashMap::new(),
        model_groups: std::collections::HashMap::new(),
        system_prompt: "You are a router.".into(),
        safety_threshold: 0.5,
        default_route: "local".into(),
        score_matrix: None,
        onnx_keys: std::collections::BTreeSet::new(),
        roles: Default::default(),
    };
    let stage = ClassifierStage::new(
        Arc::new(backend),
        routing_config,
        0.2,
        None,
        false,
        1,
        "lfm2.5-2.6b",
        Arc::new(fluent_concurrency::pool::Limiter::new(4)),
        None,
            crate::config::ClassifierFailurePolicy::Reject,
            None,
            None,
        );

    let mut ctx = WorkContext::default();
    ctx.set_structured(
        "request",
        &serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "write a parser"}],
        }),
    );

    // An overlay decision published between nlp and classifier.
    let mut overlay_meta = StageMetadata::new(serde_json::json!({}));
    overlay_meta.set_overlay_route_hints(&[
        RouteHint { route: "code".into(), score: 0.91 },
        RouteHint { route: "prose".into(), score: 0.06 },
    ]);
    let prior = vec![StageDecision::new(
        PipelineStage::Overlay,
        StageVerdict::Passed,
        "scored",
    )
    .with_metadata(overlay_meta.into_value())];

    let decision = stage.evaluate(&ctx, &prior).expect("evaluate");
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let system = seen.lock().expect("lock").first().expect("system").clone();
    assert!(system.contains("Route hints"), "hints merged, system:\n{system}");
    assert!(system.contains("- code: 0.910"), "hint listed with score, system:\n{system}");
    assert!(
        system.find("code: 0.910").unwrap() < system.find("prose: 0.060").unwrap(),
        "highest-score hint listed first"
    );
}

/// No overlay hints in the prior decisions → the prompt is unchanged
/// (byte-identical to today when the overlay stage is absent).
#[test]
fn classifier_prompt_unchanged_without_overlay_hints() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let backend = SystemPromptRecordingBackend {
        seen_system: Arc::clone(&seen),
    };
    let routing_config = RoutingConfig {
        routes: std::collections::HashMap::new(),
        models: std::collections::HashMap::new(),
        model_groups: std::collections::HashMap::new(),
        system_prompt: "You are a router.".into(),
        safety_threshold: 0.5,
        default_route: "local".into(),
        score_matrix: None,
        onnx_keys: std::collections::BTreeSet::new(),
        roles: Default::default(),
    };
    let stage = ClassifierStage::new(
        Arc::new(backend),
        routing_config,
        0.2,
        None,
        false,
        1,
        "lfm2.5-2.6b",
        Arc::new(fluent_concurrency::pool::Limiter::new(4)),
        None,
            crate::config::ClassifierFailurePolicy::Reject,
            None,
            None,
        );

    let mut ctx = WorkContext::default();
    ctx.set_structured(
        "request",
        &serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
        }),
    );
    let decision = stage.evaluate(&ctx, &[]).expect("evaluate");
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let system = seen.lock().expect("lock").first().expect("system").clone();
    assert!(
        !system.contains("Route hints"),
        "no hints block when the overlay stage is absent, system:\n{system}"
    );
}

#[test]
fn build_decision_returns_same_target_both_channels() {
    use crate::pipeline::RoutingTarget;

    let entry: crate::config::ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://localhost:8080/v1/chat/completions",
        "name": "m1a-model",
        "intelligence": 2,
        "cost_input": 1e-6,
        "cost_output": 6e-6,
        "cost_cached_read": 4e-7,
        "speed": 8,
    }))
    .expect("valid ModelEntry");
    let rt = RoutingTarget::from_model_entry("m1a", &entry);
    let output = crate::config::ClassifierOutput {
        action: "route".into(),
        response: None,
        target: Some("local".into()),
        coherence_score: 0.9,
        safety_score: 0.9,
        complexity: None,
        intent: Some("local".into()),
        reason: "m1a".into(),
        completeness: None,
        risk: None,
    };
    let (_message, decision, returned) =
        ClassifierStage::build_decision(&output, Some(&rt), true, None);
    let returned = returned.expect("target returned alongside the decision");
    assert_eq!(returned.model, rt.model);
    assert!(
        decision.metadata.get("routing_target").is_none(),
        "metadata carries no routing_target shim (typed-only handoff)",
    );
}

#[test]
fn metadata_has_no_routing_target_key() {
    let entry: crate::config::ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://localhost:8080/v1/chat/completions",
        "name": "no-shim-model",
        "intelligence": 2,
        "cost_input": 1e-6,
        "cost_output": 6e-6,
        "cost_cached_read": 4e-7,
        "speed": 8,
    }))
    .expect("valid ModelEntry");
    let rt = crate::pipeline::RoutingTarget::from_model_entry("no-shim", &entry);
    let output = crate::config::ClassifierOutput {
        action: "route".into(),
        response: None,
        target: Some("local".into()),
        coherence_score: 0.9,
        safety_score: 0.9,
        complexity: None,
        intent: Some("local".into()),
        reason: "no-shim".into(),
        completeness: None,
        risk: None,
    };
    let (_message, decision, returned) =
        ClassifierStage::build_decision(&output, Some(&rt), true, None);
    assert!(returned.is_some(), "target still returned by value");
    let value = serde_json::to_value(&decision).expect("serialize");
    assert!(
        value.pointer("/metadata/routing_target").is_none(),
        "serialized decision has no metadata.routing_target key",
    );
}

// -- Late-bound classifier backend -------------------------------------------
// The stage keeps the resolved model *key*; the backend is resolved per
// request through the live registry + pool, so a backend that appears (or
// moves) after boot is observed instead of a boot-frozen client.

/// Backend whose answer can be swapped after the stage is built, standing
/// in for a registry entry that appears or is rewritten post-boot.
struct SwappableBackend {
    response: std::sync::Mutex<String>,
}

impl fluent_llm::client::ChatBackend for SwappableBackend {
    fn chat_complete(
        &self,
        _messages: &[fluent_llm::ChatMessage],
    ) -> Result<String, fluent_llm::LlmError> {
        Ok(self.response.lock().expect("lock").clone())
    }
}

/// Backend that counts the calls it serves, so the test can tell which of
/// two backends the stage actually consulted.
struct CallCountingBackend {
    calls: std::sync::Mutex<usize>,
    response: String,
}

impl fluent_llm::client::ChatBackend for CallCountingBackend {
    fn chat_complete(
        &self,
        _messages: &[fluent_llm::ChatMessage],
    ) -> Result<String, fluent_llm::LlmError> {
        *self.calls.lock().expect("lock") += 1;
        Ok(self.response.clone())
    }
}

fn empty_routing_config() -> RoutingConfig {
    RoutingConfig {
        routes: std::collections::HashMap::new(),
        models: std::collections::HashMap::new(),
        model_groups: std::collections::HashMap::new(),
        system_prompt: String::new(),
        safety_threshold: 0.5,
        default_route: "local".into(),
        score_matrix: None,
        onnx_keys: std::collections::BTreeSet::new(),
        roles: Default::default(),
    }
}

fn respond_json(text: &str) -> String {
    serde_json::json!({
        "action": "respond",
        "response": text,
        "coherence_score": 0.99,
        "safety_score": 1.0,
        "complexity": 2,
        "reason": "trivial",
    })
    .to_string()
}

fn late_bound_stage(
    frozen: Arc<dyn fluent_llm::client::ChatBackend>,
    live: Arc<SwappableBackend>,
    model_key: &str,
) -> ClassifierStage {
    let resolver: ClassifierBackendResolver = Arc::new(move |_: &str| {
        Some(Arc::clone(&live) as Arc<dyn fluent_llm::client::ChatBackend>)
    });
    ClassifierStage::new(
        frozen,
        empty_routing_config(),
        0.2,
        None,
        false,
        1,
        model_key,
        Arc::new(fluent_concurrency::pool::Limiter::new(4)),
        None,
        crate::config::ClassifierFailurePolicy::Reject,
        None,
        Some(resolver),
    )
}

fn classifier_request_ctx() -> WorkContext {
    let mut ctx = WorkContext::default();
    ctx.set_structured(
        "request",
        &serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
        }),
    );
    ctx
}

#[test]
fn late_bound_resolution_observes_post_build_swap() {
    // A backend that appears (or is rewritten) after the stage is built is
    // observed per request; a boot-frozen client would keep serving stale.
    let live = Arc::new(SwappableBackend {
        response: std::sync::Mutex::new("live-A".into()),
    });
    let frozen = Arc::new(SwappableBackend {
        response: std::sync::Mutex::new("frozen".into()),
    });
    let stage = late_bound_stage(frozen, Arc::clone(&live), "clf");
    assert!(stage.has_backend_resolver());

    let seen = |stage: &ClassifierStage| {
        stage
            .resolve_backend()
            .expect("resolves")
            .chat_complete(&[])
            .expect("answers")
    };
    assert_eq!(seen(&stage), "live-A");
    *live.response.lock().expect("lock") = "live-B".into();
    assert_eq!(seen(&stage), "live-B", "post-build swap is observed");
}

#[test]
fn late_bound_flat_path_consults_resolved_backend_not_frozen() {
    let frozen = Arc::new(CallCountingBackend {
        calls: std::sync::Mutex::new(0),
        response: respond_json("frozen"),
    });
    let live = Arc::new(CallCountingBackend {
        calls: std::sync::Mutex::new(0),
        response: respond_json("live"),
    });
    let live_backend: Arc<dyn fluent_llm::client::ChatBackend> = Arc::clone(&live) as _;
    let resolver: ClassifierBackendResolver = Arc::new(move |_: &str| Some(Arc::clone(&live_backend)));
    let stage = ClassifierStage::new(
        frozen.clone() as Arc<dyn fluent_llm::client::ChatBackend>,
        empty_routing_config(),
        0.2,
        None,
        false,
        1,
        "clf",
        Arc::new(fluent_concurrency::pool::Limiter::new(4)),
        None,
        crate::config::ClassifierFailurePolicy::Reject,
        None,
        Some(resolver),
    );

    let decision = stage.evaluate(&classifier_request_ctx(), &[]).expect("evaluate");
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(*live.calls.lock().expect("lock"), 1, "resolved backend serves");
    assert_eq!(*frozen.calls.lock().expect("lock"), 0, "frozen client untouched");
}

#[test]
fn late_bound_miss_rejects_without_fabricated_route() {
    // A resolution miss degrades to the failure policy (reject here) — never
    // a fabricated route.
    let frozen = Arc::new(CallCountingBackend {
        calls: std::sync::Mutex::new(0),
        response: respond_json("frozen"),
    });
    let resolver: ClassifierBackendResolver = Arc::new(|_: &str| None);
    let stage = ClassifierStage::new(
        frozen.clone() as Arc<dyn fluent_llm::client::ChatBackend>,
        empty_routing_config(),
        0.2,
        None,
        false,
        1,
        "clf",
        Arc::new(fluent_concurrency::pool::Limiter::new(4)),
        None,
        crate::config::ClassifierFailurePolicy::Reject,
        None,
        Some(resolver),
    );

    assert!(stage.resolve_backend().is_none());
    let decision = stage.evaluate(&classifier_request_ctx(), &[]).expect("evaluate");
    assert_eq!(decision.verdict, StageVerdict::Rejected);
    assert_eq!(*frozen.calls.lock().expect("lock"), 0, "no backend consulted");
}

#[test]
fn frozen_client_serves_when_no_resolver_installed() {
    // No resolver (injected mock path): the boot client serves exactly as
    // before — byte-identical behavior for tests and `--mock` runs.
    let frozen = Arc::new(CallCountingBackend {
        calls: std::sync::Mutex::new(0),
        response: respond_json("frozen"),
    });
    let stage = ClassifierStage::new(
        frozen.clone() as Arc<dyn fluent_llm::client::ChatBackend>,
        empty_routing_config(),
        0.2,
        None,
        false,
        1,
        "clf",
        Arc::new(fluent_concurrency::pool::Limiter::new(4)),
        None,
        crate::config::ClassifierFailurePolicy::Reject,
        None,
        None,
    );

    assert!(!stage.has_backend_resolver());
    let decision = stage.evaluate(&classifier_request_ctx(), &[]).expect("evaluate");
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(*frozen.calls.lock().expect("lock"), 1);
}
