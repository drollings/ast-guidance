use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use common_core::sync::lock;
use fluent_concurrency::pool::Limiter;
use fluent_llm::{ChatMessage, LlmError};
use fluent_llm::client::ChatBackend;

use crate::config::{ClassificationTree, ModelEntry, ModelGroup, RouteRef, RoutingConfig};
use crate::pipeline::RoutingTarget;
use crate::pipeline_types::StageVerdict;
use crate::target_match::{TargetBackends, TargetMatcher};
use crate::test_stubs::{CountingBackend, StubChatBackend};

use super::*;

fn model_entry(key: &str, intelligence: u8, cost: f64) -> ModelEntry {
    ModelEntry {
        name: Some(key.into()),
        endpoint: "http://localhost:8080/v1/chat/completions".into(),
        intelligence,
        cost_input: cost,
        cost_output: cost * 6.0,
        cost_cached_read: cost * 0.4,
        speed: 8,
        total_timeout_ms: 40_000,
        idle_timeout_ms: 8_000,
        stream: true,
        filter_thinking: true,
        retry_count: 0,
        retry_base_interval_s: 1,
        params: None,
        instances: None,
        weights: None,
        hf_repo: None,
        hf_file: None,
            api_key: None,
    }
}

fn test_routing() -> RoutingConfig {
    RoutingConfig {
        routes: HashMap::from([
            (
                "code".into(),
                RouteRef {
                    group: "code".into(),
                    pipelines: vec!["default".into()],
                    description: "code".into(),
        always_route: false,
                },
            ),
            (
                "translation".into(),
                RouteRef {
                    group: "translation".into(),
                    pipelines: vec!["default".into()],
                    description: "translation".into(),
        always_route: false,
                },
            ),
            (
                "local".into(),
                RouteRef {
                    group: "question".into(),
                    pipelines: vec!["default".into()],
                    description: "local".into(),
        always_route: false,
                },
            ),
        ]),
        models: HashMap::from([
            ("fast".into(), model_entry("fast", 1, 1e-6)),
            ("small".into(), model_entry("small", 2, 2e-6)),
            ("code-model".into(), model_entry("code-model", 5, 5e-6)),
            (
                "translation-model".into(),
                model_entry("translation-model", 3, 3e-6),
            ),
            (
                "question-model".into(),
                model_entry("question-model", 2, 2e-6),
            ),
        ]),
        model_groups: HashMap::from([
            ("code".into(), ModelGroup::Array(vec!["code-model".into()])),
            (
                "translation".into(),
                ModelGroup::Array(vec!["translation-model".into()]),
            ),
            (
                "question".into(),
                ModelGroup::Array(vec!["question-model".into()]),
            ),
            (
                "fast".into(),
                ModelGroup::Array(vec!["fast".into(), "small".into()]),
            ),
        ]),
        system_prompt: String::new(),
        safety_threshold: 0.3,
        default_route: "local".into(),
        score_matrix: None,
        onnx_keys: std::collections::BTreeSet::new(),
    }
}

fn engine(tree: &ClassificationTree, backend: Arc<dyn ChatBackend>) -> ClassificationEngine {
    engine_with_matcher(tree, backend, None)
}

fn engine_with_matcher(
    tree: &ClassificationTree,
    backend: Arc<dyn ChatBackend>,
    matcher: Option<TargetMatcher>,
) -> ClassificationEngine {
    engine_with_routing(tree, backend, test_routing(), matcher)
}

fn engine_with_routing(
    tree: &ClassificationTree,
    backend: Arc<dyn ChatBackend>,
    routing: RoutingConfig,
    matcher: Option<TargetMatcher>,
) -> ClassificationEngine {
    ClassificationEngine::new(
        tree.clone(),
        routing,
        backend,
        HashMap::new(),
        Arc::new(Limiter::new(4)),
        0.5,
        matcher,
    )
}

fn verdict(route: &str, coherence: f64, safety: f64, complexity: u8) -> String {
    serde_json::to_string(&serde_json::json!({
        "route": route,
        "coherence": coherence,
        "safety": safety,
        "complexity": complexity,
        "reason": "test verdict",
    }))
    .unwrap()
}

/// A canned self-assessment response for the target-matching ladder.
fn self_assessment(complexity: u8, reason: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "complexity": complexity,
        "reason": reason,
    }))
    .unwrap()
}

/// A ladder matcher whose default backend serves the queued self-assessment
/// responses (empty per-key map → every candidate routes through the
/// default, mirroring mock/transcript injection).
fn ladder_matcher(responses: Vec<String>) -> TargetMatcher {
    TargetMatcher::new(
        TargetBackends::new(
            HashMap::new(),
            Arc::new(StubChatBackend::new(responses)),
        ),
        Arc::new(Limiter::new(4)),
        0,
    )
}

fn routed_target(evaluation: &TreeEvaluation) -> RoutingTarget {
    evaluation
        .target
        .clone()
        .expect("evaluation should carry a routing target")
}

fn simple_tree() -> ClassificationTree {
    serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "request router",
            "model": "fast",
            "children": [
                {
                    "key": "code",
                    "description": "programming",
                    "node": { "type": "terminal", "route": "code", "group": "code" }
                },
                {
                    "key": "translation",
                    "description": "translation",
                    "node": { "type": "terminal", "route": "translation", "group": "translation" }
                },
                {
                    "key": "general",
                    "description": "everything else",
                    "node": {
                        "type": "fallback",
                        "node": { "type": "terminal", "route": "local", "group": "question" }
                    }
                }
            ]
        }
    }))
    .unwrap()
}

// ── Terminal nodes ─────────────────────────────────────────────────

#[test]
fn terminal_node_resolves_route() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": { "type": "terminal", "route": "code" }
    }))
    .unwrap();
    let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
    let evaluation = engine.evaluate("write a rust function", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let rt = routed_target(&evaluation);
    assert_eq!(rt.target_name.as_deref(), Some("code"));
    assert_eq!(rt.model, "code-model");
    assert_eq!(rt.group.as_deref(), Some("code"));
}

#[test]
fn terminal_complexity_selects_model() {
    // complexity 8 > code-model intelligence 5, so the cheapest model in
    // the group whose intelligence meets it — none — falls back to the
    // cheapest in the group (code-model).
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": { "type": "terminal", "route": "code" }
    }))
    .unwrap();
    let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
    let evaluation = engine.evaluate("complex", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(routed_target(&evaluation).model, "code-model");
}

#[test]
fn terminal_unresolvable_route_rejects() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": { "type": "terminal", "route": "does-not-exist" }
    }))
    .unwrap();
    let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
    let evaluation = engine.evaluate("hi", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Rejected);
    assert!(decision.reason.contains("does-not-exist"));
}

#[test]
fn terminal_with_own_group_resolves_without_flat_route() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": { "type": "terminal", "route": "fresh", "group": "fast" }
    }))
    .unwrap();
    let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
    let evaluation = engine.evaluate("hi", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let rt = routed_target(&evaluation);
    assert_eq!(rt.target_name.as_deref(), Some("fresh"));
    // Cheapest in "fast" group meeting no-complexity: fast (cost 1e-6 vs small 2e-6).
    assert_eq!(rt.model, "fast");
}

#[test]
fn terminal_group_ladder_self_assesses_and_matches() {
    // The "fast" group has 2 members (fast intelligence 1, small
    // intelligence 2). A root terminal on that group climbs: fast
    // self-assesses above its intelligence, small matches.
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": { "type": "terminal", "route": "fresh", "group": "fast" }
    }))
    .unwrap();
    let matcher = ladder_matcher(vec![
        self_assessment(7, "too hard for fast"),
        self_assessment(1, "easy for small"),
    ]);
    let engine = engine_with_matcher(
        &tree,
        Arc::new(StubChatBackend::always("{}")),
        Some(matcher),
    );
    let evaluation = engine.evaluate("some task", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let rt = routed_target(&evaluation);
    assert_eq!(
        rt.model, "small",
        "ladder climbs past the too-weak cheap member",
    );
    assert_eq!(rt.target_name.as_deref(), Some("fresh"));
    assert_eq!(rt.group.as_deref(), Some("fast"));

    // The terminal's tree_path audit carries the ladder walk (additive over
    // the existing route/group/model/complexity fields).
    let path = decision.metadata["tree_path"].as_array().expect("tree_path");
    let terminal = path
        .iter()
        .find(|d| d["metadata"]["node_type"] == "terminal")
        .expect("terminal node decision");
    assert_eq!(terminal["metadata"]["matched_via"], "self_assess");
    let assessments = terminal["metadata"]["assessments"]
        .as_array()
        .expect("assessments");
    assert_eq!(assessments.len(), 2);
    assert_eq!(assessments[0]["model_name"], "fast");
    assert_eq!(assessments[0]["assessed"], serde_json::json!(7));
    assert_eq!(assessments[0]["matched"], serde_json::json!(false));
    assert_eq!(assessments[1]["model_name"], "small");
    assert_eq!(assessments[1]["assessed"], serde_json::json!(1));
    assert_eq!(assessments[1]["matched"], serde_json::json!(true));
}

#[test]
fn terminal_flat_route_ladder_matches_within_group() {
    // The route's own group ("code" is a single-member group — static).
    // Use a 2-member group via a flat route: "local" → group "question"
    // is single-member too. Build a flat route on the 2-member "fast"
    // group to exercise the resolve_route_with_matcher path.
    let mut routing = test_routing();
    routing.routes.insert(
        "fresh".into(),
        RouteRef {
            group: "fast".into(),
            pipelines: vec!["default".into()],
            description: "fresh".into(),
        always_route: false,
        },
    );
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": { "type": "terminal", "route": "fresh" }
    }))
    .unwrap();

    // fast self-assesses 2 > intelligence 1 → escalate to small, which
    // matches at 2 <= 2.
    let matcher = ladder_matcher(vec![
        self_assessment(2, "above fast"),
        self_assessment(2, "ok for small"),
    ]);
    let engine = engine_with_routing(
        &tree,
        Arc::new(StubChatBackend::always("{}")),
        routing,
        Some(matcher),
    );
    let evaluation = engine.evaluate("a task", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let rt = routed_target(&evaluation);
    assert_eq!(rt.model, "small");
    assert_eq!(rt.target_name.as_deref(), Some("fresh"));
    assert_eq!(rt.group.as_deref(), Some("fast"));

    let path = decision.metadata["tree_path"].as_array().expect("tree_path");
    let terminal = path
        .iter()
        .find(|d| d["metadata"]["node_type"] == "terminal")
        .expect("terminal node decision");
    assert_eq!(
        terminal["metadata"]["assessments"].as_array().map(Vec::len),
        Some(2),
    );
}

#[test]
fn terminal_single_member_group_never_self_assesses() {
    // A single-member group ("code") has nothing to climb — the ladder is
    // skipped entirely and no self-assessment call is made, even with a
    // matcher present (byte-identical to today's static pick).
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": { "type": "terminal", "route": "code", "group": "code" }
    }))
    .unwrap();
    let counting = Arc::new(CountingBackend::new("{}"));
    let matcher = TargetMatcher::new(
        TargetBackends::new(HashMap::new(), Arc::clone(&counting) as Arc<dyn ChatBackend>),
        Arc::new(Limiter::new(4)),
        0,
    );
    let engine = engine_with_matcher(
        &tree,
        Arc::new(StubChatBackend::always("{}")),
        Some(matcher),
    );
    let evaluation = engine.evaluate("hello", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(routed_target(&evaluation).model, "code-model");
    assert_eq!(
        counting.calls(),
        0,
        "single-member group must not run the ladder",
    );
}

#[test]
fn terminal_ladder_assessment_failure_escalates_to_last_member() {
    // The "fast" group: fast's self-assessment is unparseable (conservative
    // escalate), small matches as the last member regardless.
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": { "type": "terminal", "route": "fresh", "group": "fast" }
    }))
    .unwrap();
    let matcher = ladder_matcher(vec![
        "not json at all".into(),
        self_assessment(9, "hard even for small"),
    ]);
    let engine = engine_with_matcher(
        &tree,
        Arc::new(StubChatBackend::always("{}")),
        Some(matcher),
    );
    let evaluation = engine.evaluate("some task", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(routed_target(&evaluation).model, "small");

    let path = decision.metadata["tree_path"].as_array().expect("tree_path");
    let terminal = path
        .iter()
        .find(|d| d["metadata"]["node_type"] == "terminal")
        .expect("terminal node decision");
    let assessments = terminal["metadata"]["assessments"]
        .as_array()
        .expect("assessments");
    assert_eq!(assessments[0]["assessed"], serde_json::Value::Null);
    assert!(assessments[0]["error"].as_str().is_some());
    assert_eq!(assessments[0]["matched"], serde_json::json!(false));
    assert_eq!(assessments[1]["matched"], serde_json::json!(true));
}

// ── Filter nodes ───────────────────────────────────────────────────

#[test]
fn filter_hard_reject_short_circuits() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "router",
            "model": "fast",
            "children": [
                {
                    "key": "blocked",
                    "description": "blocks banned content",
                    "node": {
                        "type": "filter",
                        "patterns": ["\\bharmful pattern\\b"],
                        "outcome": "hard_reject"
                    }
                },
                {
                    "key": "code",
                    "description": "programming",
                    "node": { "type": "terminal", "route": "code", "group": "code" }
                }
            ]
        }
    }))
    .unwrap();
    let engine = engine(
        &tree,
        Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3))),
    );
    let decision = engine
        .evaluate("this is a harmful pattern test", None, None).unwrap()
        .decision;
    assert_eq!(decision.verdict, StageVerdict::Rejected);
    assert!(decision.reason.contains("blocked"));
}

#[test]
fn filter_non_match_falls_through_to_llm() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "router",
            "model": "fast",
            "children": [
                {
                    "key": "blocked",
                    "description": "blocks banned content",
                    "node": { "type": "filter", "patterns": ["\\bharmful\\b"], "outcome": "hard_reject" }
                },
                {
                    "key": "code",
                    "description": "programming",
                    "node": { "type": "terminal", "route": "code", "group": "code" }
                }
            ]
        }
    }))
    .unwrap();
    let engine = engine(
        &tree,
        Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3))),
    );
    let evaluation = engine.evaluate("write a function", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(
        routed_target(&evaluation).target_name.as_deref(),
        Some("code")
    );
}

#[test]
fn filter_soft_redirect_jumps_to_sibling() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "router",
            "model": "fast",
            "children": [
                {
                    "key": "redirect",
                    "description": "always code",
                    "node": {
                        "type": "filter",
                        "patterns": [".*"],
                        "outcome": "soft_redirect",
                        "redirect_to": "code"
                    }
                },
                {
                    "key": "code",
                    "description": "programming",
                    "node": { "type": "terminal", "route": "code", "group": "code" }
                }
            ]
        }
    }))
    .unwrap();
    let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
    let evaluation = engine.evaluate("anything at all", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(
        routed_target(&evaluation).target_name.as_deref(),
        Some("code")
    );
}

#[test]
fn filter_output_filter_continues() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "router",
            "model": "fast",
            "children": [
                {
                    "key": "redact",
                    "description": "flag pii",
                    "node": { "type": "filter", "patterns": ["\\d{3}-\\d{2}-\\d{4}"], "outcome": "output_filter" }
                },
                {
                    "key": "code",
                    "description": "programming",
                    "node": { "type": "terminal", "route": "code", "group": "code" }
                }
            ]
        }
    }))
    .unwrap();
    let engine = engine(
        &tree,
        Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3))),
    );
    let evaluation = engine
        .evaluate("my ssn is 123-45-6789 and I need code", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(
        routed_target(&evaluation).target_name.as_deref(),
        Some("code")
    );
}

// ── Classifier nodes ───────────────────────────────────────────────

#[test]
fn classifier_picks_child_and_routes() {
    let tree = simple_tree();
    let backend = Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3)));
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("help me debug rust", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(
        routed_target(&evaluation).target_name.as_deref(),
        Some("code")
    );
}

#[test]
fn classifier_threshold_rejects_incoherent_query() {
    let tree = simple_tree();
    let backend = Arc::new(StubChatBackend::always(verdict("code", 0.2, 0.9, 3)));
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("asdf qwerty", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Rejected);
    assert!(decision.reason.contains("coherence"));
}

#[test]
fn classifier_threshold_rejects_unsafe_query() {
    let tree = simple_tree();
    let backend = Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.05, 3)));
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("something unsafe", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Rejected);
    assert!(decision.reason.contains("safety"));
}

#[test]
fn classifier_unknown_route_falls_back() {
    let tree = simple_tree();
    let backend = Arc::new(StubChatBackend::always(verdict("nonexistent", 0.9, 0.9, 3)));
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("hello", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(
        routed_target(&evaluation).target_name.as_deref(),
        Some("local")
    );
}

#[test]
fn classifier_llm_failure_falls_back() {
    let tree = simple_tree();
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![]));
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("hello", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(
        routed_target(&evaluation).target_name.as_deref(),
        Some("local")
    );
}

#[test]
fn classifier_no_fallback_rejects_on_llm_failure() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "router",
            "model": "fast",
            "children": [
                {
                    "key": "code",
                    "description": "programming",
                    "node": { "type": "terminal", "route": "code", "group": "code" }
                }
            ]
        }
    }))
    .unwrap();
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![]));
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("hello", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Rejected);
    assert!(decision.reason.contains("LLM error"));
}

#[test]
fn classifier_empty_route_rejects_when_no_fallback() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "router",
            "model": "fast",
            "children": [
                {
                    "key": "code",
                    "description": "programming",
                    "node": { "type": "terminal", "route": "code", "group": "code" }
                }
            ]
        }
    }))
    .unwrap();
    let backend = Arc::new(StubChatBackend::always(verdict("", 0.9, 0.9, 3)));
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("hello", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Rejected);
}

// ── Multi-level trees ──────────────────────────────────────────────

#[test]
fn multi_level_domain_to_subdomain_to_terminal() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "domain router",
            "model": "fast",
            "children": [
                {
                    "key": "code",
                    "description": "programming domain",
                    "node": {
                        "type": "classifier",
                        "description": "code subdomain",
                        "model": "small",
                        "children": [
                            {
                                "key": "debug",
                                "description": "debugging help",
                                "node": { "type": "terminal", "route": "code", "group": "code" }
                            },
                            {
                                "key": "general",
                                "description": "general programming",
                                "node": { "type": "terminal", "route": "code", "group": "code" }
                            }
                        ]
                    }
                },
                {
                    "key": "prose",
                    "description": "general questions",
                    "node": { "type": "terminal", "route": "local", "group": "question" }
                }
            ]
        }
    }))
    .unwrap();
    // Call 1: root picks "code". Call 2: subdomain picks "debug".
    let backend = Arc::new(StubChatBackend::new(vec![
        verdict("code", 0.9, 0.9, 5),
        verdict("debug", 0.9, 0.9, 6),
    ]));
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("my program segfaults", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);
    assert_eq!(
        routed_target(&evaluation).target_name.as_deref(),
        Some("code")
    );

    // Both visited node types appear in the tree_path.
    let path = decision.metadata["tree_path"]
        .as_array()
        .expect("tree_path");
    let types: Vec<&str> = path
        .iter()
        .filter_map(|d| d["metadata"]["node_type"].as_str())
        .collect();
    assert!(types.contains(&"classifier"));
    assert!(types.contains(&"terminal"));
    assert!(
        path.len() >= 3,
        "root + sub + terminal decisions, got {path:?}"
    );
}

// ── Prompt auto-construction ───────────────────────────────────────

#[test]
fn auto_generated_prompt_lists_children() {
    let tree = simple_tree();
    let backend = Arc::new(StubChatBackend::always(verdict("code", 0.9, 0.9, 3)));
    let engine = engine(&tree, backend);
    let _ = engine.evaluate("hello", None, None).unwrap();
    // The prompt is only observable via the audit/log stream; assert the
    // pure `build_prompt` output is what the engine would send.
    let prompt = tree
        .root
        .build_prompt(0.5, 0.3)
        .expect("root classifier prompt");
    assert!(prompt.contains("You are a request router."));
    assert!(prompt.contains("- code: programming"));
    assert!(prompt.contains("- translation: translation"));
    assert!(prompt.contains("\"route\": \"<exactly one of: code, translation>\""));
    assert!(prompt.contains("\"coherence\": 0.0-1.0"));
    assert!(prompt.contains("\"complexity\": 0-10"));
}

// ── Prompt capture through the backend ─────────────────────────────

struct RecordingBackend {
    prompts: Arc<Mutex<Vec<String>>>,
    response: String,
}

impl ChatBackend for RecordingBackend {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        lock(&self.prompts).extend(
            messages
                .iter()
                .filter(|m| m.role == "system")
                .map(|m| m.content.clone()),
        );
        Ok(self.response.clone())
    }
}

#[test]
fn engine_sends_auto_generated_prompt_to_backend() {
    let tree = simple_tree();
    let prompts = Arc::new(Mutex::new(Vec::<String>::new()));
    let backend: Arc<dyn ChatBackend> = Arc::new(RecordingBackend {
        prompts: prompts.clone(),
        response: verdict("code", 0.9, 0.9, 3),
    });
    let engine = engine(&tree, backend);
    let evaluation = engine.evaluate("write code", None, None).unwrap();
    let decision = &evaluation.decision;
    assert_eq!(decision.verdict, StageVerdict::Passed);

    let captured = lock(&prompts).clone();
    assert_eq!(captured.len(), 1, "exactly one classifier call");
    assert!(captured[0].contains("You are a request router."));
    assert!(captured[0].contains("- code: programming"));
    assert!(captured[0].contains("- translation: translation"));
    assert!(
        captured[0].contains("\"route\": \"<exactly one of: code, translation>\""),
        "three-axis route enum, got: {}",
        captured[0]
    );
}
// ── match_interlingua Filter nodes (ROADMAP §14.6, C6) ──────────────

/// A tree whose root classifier carries a report-terminal and an
/// interlingua filter that soft-redirects "show/display/get the report"
/// deterministically (same ids, zero tokens).
fn report_tree() -> ClassificationTree {
    serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "request router",
            "model": "fast",
            "children": [
                {
                    "key": "report",
                    "description": "report requests (deterministic target)",
                    "node": { "type": "terminal", "route": "translation", "group": "translation" }
                },
                {
                    "key": "code",
                    "description": "programming",
                    "node": { "type": "terminal", "route": "code", "group": "code" }
                },
                {
                    "key": "route_to_report",
                    "description": "deterministic report dispatch",
                    "node": {
                        "type": "filter",
                        "description": "any report request",
                        "match_interlingua": {
                            "object_id": 2251799813685262_i64,
                            "confidence_min": 0.5
                        },
                        "outcome": "soft_redirect",
                        "redirect_to": "report"
                    }
                },
                {
                    "key": "general",
                    "description": "everything else",
                    "node": {
                        "type": "fallback",
                        "node": { "type": "terminal", "route": "local", "group": "question" }
                    }
                }
            ]
        }
    })).unwrap()
}

fn report_signal(object_id: u64) -> spacy_rs::routing::InterlinguaSignal {
    spacy_rs::routing::InterlinguaSignal {
        predicate_id: Some(fluent_types::InterlinguaId::from_u64(2251799813685260)),
        subject_id: None,
        direct_object_id: Some(fluent_types::InterlinguaId::from_u64(object_id)),
        indirect_object_id: None,
        concept_ids: vec![],
        token_ids: vec![],
        // The fixture's parse is high-confidence — above `report_tree`'s
        // `confidence_min: 0.5` floor, so the positive tests still match.
        confidence: Some(0.8),
    }
}

#[test]
fn interlingua_filter_short_circuits_on_matching_ids() {
    // The filter short-circuits the classifier's LLM call entirely.
    let engine = engine(&report_tree(), Arc::new(StubChatBackend::always("{}")));
    let signals = vec![report_signal(2251799813685262)];
    let evaluation = engine
        .evaluate("show me the sales report", Some(&signals), None)
        .unwrap();
    let decision = &evaluation.decision;
    // Soft-redirect to the report terminal → routed target.
    assert_eq!(decision.verdict, StageVerdict::Passed);
    let rt = routed_target(&evaluation);
    // The report terminal resolved through the (existing) translation
    // route — the point is the deterministic short-circuit, not the route.
    assert_eq!(rt.target_name.as_deref(), Some("translation"));
}

#[test]
fn interlingua_filter_passes_through_on_non_match() {
    // A different object id → no match → the filter passes and the
    // classifier LLM decides (fallback to local).
    let engine = engine(&report_tree(), Arc::new(StubChatBackend::always(verdict(
        "local", 0.9, 1.0, 2,
    ))));
    let signals = vec![report_signal(999)];
    let evaluation = engine
        .evaluate("some other request", Some(&signals), None)
        .unwrap();
    let rt = routed_target(&evaluation);
    assert_eq!(rt.target_name.as_deref(), Some("local"));
}

#[test]
fn interlingua_filter_passes_through_when_nlp_absent() {
    // `None` interlingua (no NlpStage) → graceful degradation: the filter
    // does not fire, the LLM path decides.
    let engine = engine(&report_tree(), Arc::new(StubChatBackend::always(verdict(
        "local", 0.9, 1.0, 2,
    ))));
    let evaluation = engine
        .evaluate("show me the sales report", None, None).unwrap();
    let decision = &evaluation.decision;
    let _ = decision;
    let rt = routed_target(&evaluation);
    assert_eq!(rt.target_name.as_deref(), Some("local"));
}

#[test]
fn interlingua_filter_requires_confidence_floor() {
    // The ids match `report_tree`'s `confidence_min: 0.5` filter, but the
    // parse confidence (0.2) is below the floor → no match → pass through
    // to the LLM path (fail-closed on the confidence gate).
    let engine = engine(&report_tree(), Arc::new(StubChatBackend::always(verdict(
        "local", 0.9, 1.0, 2,
    ))));
    let mut signal = report_signal(2251799813685262);
    signal.confidence = Some(0.2);
    let evaluation = engine
        .evaluate("show me the sales report", Some(&[signal]), None)
        .unwrap();
    let rt = routed_target(&evaluation);
    assert_eq!(rt.target_name.as_deref(), Some("local"));
}

#[test]
fn interlingua_filter_fails_closed_without_confidence() {
    // The ids match but the parse carried no confidence (`None`) → below
    // any positive floor → no match (low-confidence parses escalate).
    let engine = engine(&report_tree(), Arc::new(StubChatBackend::always(verdict(
        "local", 0.9, 1.0, 2,
    ))));
    let mut signal = report_signal(2251799813685262);
    signal.confidence = None;
    let evaluation = engine
        .evaluate("show me the sales report", Some(&[signal]), None)
        .unwrap();
    let rt = routed_target(&evaluation);
    assert_eq!(rt.target_name.as_deref(), Some("local"));
}

#[test]
fn interlingua_filter_can_hard_reject() {
    let tree: ClassificationTree = serde_json::from_value(serde_json::json!({
        "root": {
            "type": "classifier",
            "description": "router",
            "model": "fast",
            "children": [
                {
                    "key": "blocked",
                    "node": {
                        "type": "filter",
                        "description": "blocked predicate",
                        "match_interlingua": { "predicate_id": 2251799813685260_i64 },
                        "outcome": "hard_reject"
                    }
                },
                {
                    "key": "general",
                    "node": { "type": "terminal", "route": "local", "group": "question" }
                }
            ]
        }
    })).unwrap();
    let engine = engine(&tree, Arc::new(StubChatBackend::always("{}")));
    let signals = vec![report_signal(2251799813685262)];
    let decision = engine
        .evaluate("show me the report", Some(&signals), None)
        .unwrap()
        .decision;
    assert_eq!(decision.verdict, StageVerdict::Rejected);
}

#[test]
fn final_decision_returns_same_target_both_channels() {
    use crate::stages::tree::decisions::{final_decision, TreeOutcome};

    let entry: crate::config::ModelEntry = serde_json::from_value(serde_json::json!({
        "endpoint": "http://localhost:8080/v1/chat/completions",
        "name": "m1a-tree-model",
        "intelligence": 2,
        "cost_input": 1e-6,
        "cost_output": 6e-6,
        "cost_cached_read": 4e-7,
        "speed": 8,
    }))
    .expect("valid ModelEntry");
    let rt = RoutingTarget::from_model_entry("m1a-tree", &entry);
    let handoff = final_decision(TreeOutcome::Route(Box::new(rt.clone())), vec![]);
    assert_eq!(handoff.target.as_ref().expect("target").model, rt.model);
    assert!(
        handoff.decision.metadata.get("routing_target").is_none(),
        "metadata carries no routing_target shim (typed-only handoff)",
    );
}
