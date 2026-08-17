//! End-to-end tests for the router pipeline.
//!
//! All tests use the real `PipelineOrchestrator` with a `TranscriptProvider`
//! injected into the `ClassifierStage` — no LLM inference, no network, no GPU.
//! The full 3-stage pipeline (deterministic → classifier → router) is exercised.

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::RouterConfig;
use crate::pipeline::{PipelineOrchestrator, PipelineResult};
use crate::pipeline_types::{PipelineStage, StageVerdict};
use crate::testing::mock::TranscriptProvider;
use crate::testing::test_request;
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};
use fluent_llm::client::ChatBackend;
use fluent_wvr::prelude::*;

fn make_request(text: &str) -> RouterRequest {
    let mut req = test_request(text);
    req.model = "orchestrator:llama3.1".into();
    req.session_id = Some("e2e-test-session".into());
    req
}

fn classify_output(action: &str, coherence: f64, safety: f64, reason: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "action": action,
        "coherence_score": coherence,
        "safety_score": safety,
        "reason": reason,
        "intent": if action == "reject" { serde_json::Value::Null } else { serde_json::Value::String("question".into()) },
    }))
    .unwrap()
}

fn classify_with_target(target: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "action": "route",
        "coherence_score": 0.95,
        "safety_score": 0.9,
        "intent": "question",
        "reason": "well-formed factual query",
        "target": target,
    }))
    .unwrap()
}

fn default_provider() -> TranscriptProvider {
    TranscriptProvider::new(HashMap::new())
}

fn make_test_config() -> RouterConfig {
    match serde_json::from_str::<RouterConfig>(
        r#"{
        "pipelines": {"default": {"deterministic_prefilter": true, "classifier": true, "blacklist": "env/pii-patterns.json"}},
        "models": {"fast": {"endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.000001, "cost_output": 0.000006, "cost_cached_read": 0.0000004, "speed": 10, "total_timeout_ms": 5000, "idle_timeout_ms": 2000, "stream": false, "filter_thinking": false, "retry_count": 0, "retry_base_interval_s": 1}},
        "model_groups": {"fast": ["fast"]},
        "routes": {"fast": {"group": "fast", "pipelines": ["default"]}},
        "default_route": "fast"
    }"#,
    ) {
        Ok(c) => c,
        Err(e) => panic!("invalid test config: {e}"),
    }
}

fn make_pipeline(provider: TranscriptProvider) -> PipelineOrchestrator {
    let config = make_test_config();
    let backend = Arc::new(provider) as Arc<dyn ChatBackend>;
    config
        .build_named_pipeline_with_backend("default", Some(backend))
        .expect("default pipeline should build with transcript provider")
}

fn route(
    pipeline: &PipelineOrchestrator,
    request: &RouterRequest,
) -> Result<PipelineResult, WorkError> {
    let mut ctx = WorkContext::default();
    ctx.set_structured("request", request);
    let output = pipeline.execute(&ctx)?;
    output
        .data_take()
        .map_err(|e| WorkError::Execution(e.to_string()))
}

#[allow(dead_code)]
fn make_request_with_messages(messages: Vec<RouterMessage>) -> RouterRequest {
    RouterRequest {
        model: "orchestrator:llama3.1".into(),
        messages,
        temperature: None,
        max_tokens: None,
        stream: None,
        tools: None,
        tool_choice: None,
        session_id: Some("e2e-test-session".into()),
        agent_id: None,
        adapter: None,
        instance: None,
        snapshot: None,
        id_slot: None,
        metadata: Default::default(),
    }
}

// ── Normal Request ──────────────────────────────────────────────────────

#[test]
fn test_e2e_normal_request_passes_all_stages() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request("What is Rust?");
    let result = route(&pipeline, &request).expect("pipeline should complete");

    assert!(!result.rejected, "normal request should not be rejected");
    assert!(
        result.decisions.len() >= 2,
        "pipeline should run through all 2 stages, got {}",
        result.decisions.len()
    );

    let stage_order: Vec<PipelineStage> = result.decisions.iter().map(|d| d.stage).collect();
    assert_eq!(stage_order[0], PipelineStage::DeterministicPreFilter);
    assert_eq!(stage_order[1], PipelineStage::Classifier);
}

#[test]
fn test_e2e_all_stages_pass_verdict() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request("Explain monads in Haskell");
    let result = route(&pipeline, &request).expect("pipeline should complete");

    for decision in &result.decisions {
        assert_eq!(
            decision.verdict,
            StageVerdict::Passed,
            "stage {:?} should have Passed verdict",
            decision.stage
        );
    }
}

// ── Garbage Input Rejection ─────────────────────────────────────────────

#[test]
fn test_e2e_garbage_rejected_by_classifier() {
    let mut entries = HashMap::new();
    entries.insert(
        "asdfghjkl qwerty zxcvbnm".into(),
        classify_output("reject", 0.15, 0.9, "incoherent input"),
    );
    let pipeline = make_pipeline(TranscriptProvider::new(entries));
    let request = make_request("asdfghjkl qwerty zxcvbnm");
    let result = route(&pipeline, &request).expect("pipeline should handle rejection");

    assert!(result.rejected, "garbage input should be rejected");
    assert!(
        result
            .reject_reason
            .as_ref()
            .is_some_and(|r| r.contains("coherence")),
        "rejection reason should mention coherence, got: {:?}",
        result.reject_reason
    );
}

// ── Streaming Response Support ──────────────────────────────────────────

#[test]
fn test_e2e_streaming_flag_preserved() {
    let pipeline = make_pipeline(default_provider());
    let mut request = make_request("Tell me a story");
    request.stream = Some(true);
    let result = route(&pipeline, &request).expect("pipeline should complete");
    assert!(!result.rejected, "streaming request should not be rejected");
}

// ── Routing Decision ────────────────────────────────────────────────────

#[test]
fn test_e2e_routing_decision_included() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request("Help me debug Rust code");
    let result = route(&pipeline, &request).expect("pipeline should complete");

    let classifier_decision = result
        .decisions
        .last()
        .expect("should have classifier decision");
    assert_eq!(classifier_decision.stage, PipelineStage::Classifier);
    assert_eq!(classifier_decision.verdict, StageVerdict::Passed);

    let routing_target = classifier_decision
        .metadata
        .get("routing_target")
        .expect("classifier decision should have routing_target metadata");
    assert!(
        routing_target.get("url").is_some(),
        "routing target should have a url"
    );
}

// ── Classifier routing target ───────────────────────────────────────────

#[test]
fn test_e2e_classifier_provides_routing_target() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request("What is 2+2?");
    let result = route(&pipeline, &request).expect("pipeline should complete");
    assert!(!result.rejected, "normal request should not be rejected");
    assert!(
        result.routing_target.is_some(),
        "classifier should provide routing target"
    );
}

// ── Error Handling ──────────────────────────────────────────────────────

#[test]
fn test_e2e_empty_request_handled() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request_with_messages(vec![]);
    let result = route(&pipeline, &request);
    assert!(result.is_err(), "empty messages should produce an error");
}

#[test]
fn test_e2e_missing_user_message_handled() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request_with_messages(vec![RouterMessage {
        role: "system".into(),
        content: RouterMessageContent::Text("You are a helpful assistant.".into()),
        tool_calls: None,
        tool_call_id: None,
    }]);
    let result = route(&pipeline, &request);
    assert!(
        result.is_err(),
        "missing user message should produce an error"
    );
}

// ── Full Pipeline with Custom Fixtures ──────────────────────────────────

#[test]
fn test_e2e_custom_fixtures_produce_expected_results() {
    let mut entries = HashMap::new();
    entries.insert(
        "bad input that should be rejected".into(),
        classify_output("reject", 0.2, 0.9, "mock rejection: low quality"),
    );
    entries.insert("good quality input".into(), classify_with_target("fast"));

    let pipeline = make_pipeline(TranscriptProvider::new(entries));

    let bad_result = route(
        &pipeline,
        &make_request("bad input that should be rejected"),
    )
    .expect("pipeline should handle rejection");
    assert!(bad_result.rejected, "bad input should be rejected");

    let good_result =
        route(&pipeline, &make_request("good quality input")).expect("pipeline should complete");
    assert!(!good_result.rejected, "good input should not be rejected");
}

// ── Classification-tree config through mock mode ──────────────────────

/// A tree-shaped config: root classifier → terminal nodes, plus a fallback.
fn make_tree_config() -> RouterConfig {
    serde_json::from_str(
        r#"{
            "pipelines": {"default": {"deterministic_prefilter": true, "classifier": true}},
            "models": {
                "fast": {"endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 1e-6, "cost_output": 6e-6, "cost_cached_read": 4e-7, "speed": 10},
                "code-model": {"endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "code-model", "intelligence": 5, "cost_input": 5e-6, "cost_output": 3e-5, "cost_cached_read": 2e-6, "speed": 5}
            },
            "model_groups": {"fast": ["fast"], "code": ["code-model"]},
            "routes": {
                "code": {"group": "code", "pipelines": ["default"], "description": "code"},
                "local": {"group": "fast", "pipelines": ["default"], "description": "local"}
            },
            "default_route": "local",
            "classification": {
                "root": {
                    "type": "classifier",
                    "description": "request router",
                    "model": "fast",
                    "children": [
                        {
                            "key": "code",
                            "description": "programming and implementation",
                            "node": { "type": "terminal", "route": "code", "group": "code" }
                        },
                        {
                            "key": "general",
                            "description": "everything else",
                            "node": {
                                "type": "fallback",
                                "node": { "type": "terminal", "route": "local", "group": "fast" }
                            }
                        }
                    ]
                }
            }
        }"#,
    )
    .expect("valid tree config")
}

fn tree_verdict(route: &str, complexity: u8) -> String {
    serde_json::to_string(&serde_json::json!({
        "route": route,
        "coherence": 0.9,
        "safety": 0.9,
        "complexity": complexity,
        "reason": "tree verdict",
    }))
    .unwrap()
}

fn make_tree_pipeline(provider: TranscriptProvider) -> PipelineOrchestrator {
    let config = make_tree_config();
    let backend = Arc::new(provider) as Arc<dyn ChatBackend>;
    config
        .build_named_pipeline_with_backend("default", Some(backend))
        .expect("tree pipeline should build with transcript provider")
}

#[test]
fn test_e2e_tree_config_unknown_route_uses_fallback() {
    let mut entries = HashMap::new();
    entries.insert("hello there".into(), tree_verdict("does-not-exist", 2));
    let pipeline = make_tree_pipeline(TranscriptProvider::new(entries));
    let result = route(&pipeline, &make_request("hello there")).expect("pipeline should complete");

    assert!(!result.rejected);
    let rt = result
        .routing_target
        .expect("fallback terminal should produce a routing target");
    assert_eq!(rt.target_name.as_deref(), Some("local"));
    assert_eq!(rt.model, "fast");
}

// ── Checkpoint/Rewind Cycle (DAG session-level) ─────────────────────────
// Note: the checkpoint→rewind→status-reset behavior is owned by the inline
// `dag_session.rs` tests (`test_rewind_to_checkpoint`, `test_checkpoint_listing`),
// which subsume this formerly-duplicated e2e coverage. See ROADMAP M2.4.
