use std::collections::HashMap;

use crate::config::{ModelEntry, ModelGroup, RouteRef};
use crate::test_stubs::StubChatBackend;

use super::*;

fn candidate(key: &str, intelligence: u8, cost: f64) -> TargetCandidate {
    TargetCandidate {
        model_key: key.into(),
        model_name: key.into(),
        intelligence,
        cost,
    }
}

fn candidates(entries: &[(&str, u8, f64)]) -> Vec<TargetCandidate> {
    entries.iter().map(|(k, i, c)| candidate(k, *i, *c)).collect()
}

fn assessment(complexity: u8, reason: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "complexity": complexity,
        "reason": reason,
    }))
    .unwrap()
}

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

fn routing_with(group: &str, keys: &[&str]) -> RoutingConfig {
    RoutingConfig {
        routes: HashMap::from([(
            "local".into(),
            RouteRef {
                group: group.into(),
                pipelines: vec!["default".into()],
                description: "local".into(),
        always_route: false,
            },
        )]),
        models: keys
            .iter()
            .map(|k| (k.to_string(), model_entry(k, 0, 1.0)))
            .collect(),
        model_groups: HashMap::from([(
            group.into(),
            ModelGroup::Array(keys.iter().map(ToString::to_string).collect()),
        )]),
        system_prompt: String::new(),
        safety_threshold: 0.3,
        default_route: "local".into(),
        score_matrix: None,
        onnx_keys: std::collections::BTreeSet::new(),
    }
}

// ── Pure selection core ──────────────────────────────────────────────

#[test]
fn start_index_with_classifier_complexity() {
    let cands = candidates(&[("swarm", 2, 1.0), ("qwen", 6, 3.0)]);
    // complexity 5 → first member with intelligence >= 5 is qwen (index 1).
    assert_eq!(start_index(&cands, Some(5)), 1);
    // complexity 1 → swarm qualifies at index 0.
    assert_eq!(start_index(&cands, Some(1)), 0);
    // complexity exactly equal → that index.
    assert_eq!(start_index(&cands, Some(6)), 1);
}

#[test]
fn start_index_none_or_unqualified_returns_zero() {
    let cands = candidates(&[("swarm", 2, 1.0), ("qwen", 6, 3.0)]);
    assert_eq!(start_index(&cands, None), 0);
    // No member qualifies (complexity beyond the top intelligence) → 0:
    // the cheapest candidate self-assesses first; the climb proceeds.
    assert_eq!(start_index(&cands, Some(9)), 0);
}

#[test]
fn start_index_empty_group_is_zero() {
    let empty: Vec<TargetCandidate> = vec![];
    assert_eq!(start_index(&empty, Some(3)), 0);
    assert_eq!(start_index(&empty, None), 0);
}

#[test]
fn start_index_skips_weak_cheaper_candidates() {
    // Cost ordering governs within the group: index 0 is cheapest but too
    // weak; the classifier already ruled it out, so the climb starts at
    // the first qualifying member.
    let cands = candidates(&[("tiny", 1, 0.5), ("small", 3, 1.0), ("big", 6, 3.0)]);
    assert_eq!(start_index(&cands, Some(3)), 1);
    assert_eq!(start_index(&cands, Some(2)), 1);
    assert_eq!(start_index(&cands, Some(6)), 2);
}

#[test]
fn is_match_boundary_inclusive() {
    let cand = candidate("qwen", 6, 3.0);
    // assessed == intelligence matches (meets or exceeds).
    assert!(is_match(&cand, 6));
    assert!(is_match(&cand, 5));
    assert!(!is_match(&cand, 7));
}

// ── Prompt + parse ───────────────────────────────────────────────────

#[test]
fn self_assessment_prompt_embeds_user_text() {
    let prompt = build_self_assessment_prompt("what is 2+2?");
    assert!(prompt.contains("0 (trivial) to 10 (requires the most capable model available)"));
    assert!(prompt.contains("User request: what is 2+2?"));
    assert!(prompt.contains(r#"{"complexity": <integer 0-10>, "reason": "<brief justification>"}"#));
    assert!(prompt.ends_with("Only output JSON, no other text."));
}

#[test]
fn parse_pristine_json() {
    let s = parse_self_assessment(&assessment(4, "simple math")).unwrap();
    assert_eq!(s.complexity, 4);
    assert_eq!(s.reason, "simple math");
}

#[test]
fn parse_fenced_json() {
    let s = parse_self_assessment(&format!("```json\n{}\n```", assessment(7, "complex"))).unwrap();
    assert_eq!(s.complexity, 7);
}

#[test]
fn parse_json_inside_prose() {
    let s = parse_self_assessment(&format!(
        "Here you go: {} Hope that helps!",
        assessment(3, "moderate")
    ))
    .unwrap();
    assert_eq!(s.complexity, 3);
}

#[test]
fn parse_string_number_coerces() {
    let s = parse_self_assessment(r#"{"complexity": "6", "reason": "coerced"}"#).unwrap();
    assert_eq!(s.complexity, 6);
}

#[test]
fn parse_missing_fields_default() {
    let s = parse_self_assessment(r"{}").unwrap();
    assert_eq!(s.complexity, 5, "missing complexity defaults to 5");
    assert_eq!(s.reason, "");
}

#[test]
fn parse_unparseable_is_error() {
    assert!(parse_self_assessment("not json at all").is_err());
    assert!(parse_self_assessment("").is_err());
}

// ── Matcher climb ────────────────────────────────────────────────────

fn matcher_with(default_response: Vec<String>) -> TargetMatcher {
    TargetMatcher::new(
        TargetBackends::new(
            HashMap::new(),
            Arc::new(StubChatBackend::new(default_response)),
        ),
        Arc::new(Limiter::new(4)),
        0,
    )
}

#[test]
fn target_backends_get_prefers_dedicated_then_default() {
    let dedicated: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::always(assessment(2, "dedicated")));
    let default: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::always(assessment(8, "default")));
    let backends = TargetBackends::new(
        HashMap::from([("swarm".to_string(), Arc::clone(&dedicated))]),
        Arc::clone(&default),
    );

    assert!(
        Arc::ptr_eq(&backends.get("swarm"), &dedicated),
        "dedicated backend wins for a mapped key"
    );
    assert!(
        Arc::ptr_eq(&backends.get("qwen3.6-27b"), &default),
        "default backend (injected mock/transcript) serves keys absent from the map"
    );
}

#[test]
fn climb_matches_first_candidate_within_intelligence() {
    let cands = candidates(&[
        ("swarm", 2, 1.0),
        ("qwen3.5-9b", 4, 3.0),
        ("qwen3.6-27b", 6, 5.0),
    ]);
    let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
    let matcher = matcher_with(vec![assessment(4, "mid"), assessment(6, "hard")]);

    let tm = matcher
        .match_target("local", "default", &routing, &cands, Some(3), "hello")
        .expect("match");
    // start_index(Some(3)) = 1 (qwen3.5-9b, the first member with
    // intelligence >= 3). It self-assesses 4 <= 4 → match.
    assert_eq!(tm.primary.model, "qwen3.5-9b");
    assert_eq!(tm.primary.target_name.as_deref(), Some("local"));
    assert_eq!(tm.primary.group.as_deref(), Some("default"));
    // Exactly 1 self-assessment: swarm skipped by the start index, qwen3.5-9b
    // assessed and matched.
    assert_eq!(tm.assessments.len(), 1);
    let rec = &tm.assessments[0];
    assert_eq!(rec.assessed, Some(4));
    assert!(rec.matched);
}

#[test]
fn climb_escalates_to_more_intelligent_member() {
    let cands = candidates(&[
        ("swarm", 2, 1.0),
        ("qwen3.5-9b", 4, 3.0),
        ("qwen3.6-27b", 6, 5.0),
    ]);
    let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
    let matcher = matcher_with(vec![
        assessment(7, "hard for qwen3.5"),
        assessment(6, "ok for 27b"),
    ]);

    let tm = matcher
        .match_target("local", "default", &routing, &cands, Some(3), "hello")
        .expect("match");
    // start_index(Some(3)) = 1. qwen3.5-9b self-assesses 7 > 4 → escalate.
    // qwen3.6-27b self-assesses 6 <= 6 → match.
    assert_eq!(tm.primary.model, "qwen3.6-27b");
    assert_eq!(tm.assessments.len(), 2);
    assert_eq!(tm.assessments[0].assessed, Some(7));
    assert!(!tm.assessments[0].matched);
    assert_eq!(tm.assessments[1].assessed, Some(6));
    assert!(tm.assessments[1].matched);
}

#[test]
fn climb_starts_at_classifier_seed() {
    let cands = candidates(&[
        ("swarm", 2, 1.0),
        ("qwen3.5-9b", 4, 3.0),
        ("qwen3.6-27b", 6, 5.0),
    ]);
    let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
    let matcher = matcher_with(vec![assessment(1, "easy"), assessment(3, "ok")]);

    let tm = matcher
        .match_target("local", "default", &routing, &cands, None, "hello")
        .expect("match");
    // No classifier estimate → start at 0 (swarm). Swarm self-assesses 1 <= 2 → match.
    assert_eq!(tm.primary.model, "swarm");
    assert_eq!(tm.assessments.len(), 1);
}

#[test]
fn parse_failure_escalates_conservatively() {
    let cands = candidates(&[
        ("swarm", 2, 1.0),
        ("qwen3.5-9b", 4, 3.0),
        ("qwen3.6-27b", 6, 5.0),
    ]);
    let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
    // swarm's response is unparseable → treated as can't-confirm → escalate.
    let matcher = matcher_with(vec![
        "not json".into(),
        assessment(4, "ok for qwen3.5"),
        assessment(6, "ok for 27b"),
    ]);

    let tm = matcher
        .match_target("local", "default", &routing, &cands, None, "hello")
        .expect("match");
    assert_eq!(tm.primary.model, "qwen3.5-9b");
    assert_eq!(tm.assessments.len(), 2);
    assert_eq!(tm.assessments[0].assessed, None);
    assert!(tm.assessments[0].error.is_some());
    assert!(!tm.assessments[0].matched);
}

#[test]
fn llm_error_escalates_conservatively() {
    let cands = candidates(&[("swarm", 2, 1.0), ("qwen3.6-27b", 6, 3.0)]);
    let routing = routing_with("default", &["swarm", "qwen3.6-27b"]);
    // Empty queue → both self-assessment calls fail with NoResponse.
    // Conservative escalation: a candidate that cannot confirm (LLM error)
    // is skipped; the last member still matches (terminate-don't-loop).
    let matcher = matcher_with(vec![]);

    let tm = matcher
        .match_target("local", "default", &routing, &cands, None, "hello")
        .expect("match");
    assert_eq!(tm.primary.model, "qwen3.6-27b");
    assert_eq!(tm.assessments.len(), 2);
    assert_eq!(tm.assessments[0].assessed, None);
    assert!(tm.assessments[0].error.is_some());
    assert!(!tm.assessments[0].matched);
}

#[test]
fn last_member_always_matches() {
    let cands = candidates(&[("swarm", 2, 1.0), ("qwen3.6-27b", 6, 3.0)]);
    let routing = routing_with("default", &["swarm", "qwen3.6-27b"]);
    // The last member self-assesses 9 > 6 — still matches (terminate).
    let matcher = matcher_with(vec![assessment(3, "ok for swarm"), assessment(9, "hard")]);

    let tm = matcher
        .match_target("local", "default", &routing, &cands, None, "hello")
        .expect("match");
    assert_eq!(tm.primary.model, "qwen3.6-27b");
    assert_eq!(tm.assessments.len(), 2);
    assert!(tm.assessments[1].matched, "last member always matches");
    assert_eq!(tm.assessments[1].assessed, Some(9));
}

#[test]
fn exact_call_count_two_assessments() {
    let cands = candidates(&[
        ("swarm", 2, 1.0),
        ("qwen3.5-9b", 4, 3.0),
        ("qwen3.6-27b", 6, 5.0),
    ]);
    let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
    let matcher = matcher_with(vec![assessment(7, "too hard"), assessment(6, "match")]);

    let tm = matcher
        .match_target("local", "default", &routing, &cands, Some(3), "hello")
        .expect("match");
    // start index 1 → swarm never self-assesses; exactly 2 calls made.
    assert_eq!(tm.assessments.len(), 2);
    assert_eq!(tm.assessments[0].model_key, "qwen3.5-9b");
    assert_eq!(tm.primary.model, "qwen3.6-27b");
}

#[test]
fn fallbacks_are_group_tail_then_cross_group() {
    let cands = candidates(&[
        ("swarm", 2, 1.0),
        ("qwen3.5-9b", 4, 3.0),
        ("qwen3.6-27b", 6, 5.0),
    ]);
    let routing = routing_with("default", &["swarm", "qwen3.5-9b", "qwen3.6-27b"]);
    let matcher = matcher_with(vec![assessment(1, "easy")]);

    let tm = matcher
        .match_target("local", "default", &routing, &cands, None, "hello")
        .expect("match");
    assert_eq!(tm.primary.model, "swarm");
    // Fallback tail = the more-intelligent members of the group, in order.
    let fb_models: Vec<&str> = tm.primary.fallbacks.iter().map(|f| f.model.as_str()).collect();
    assert_eq!(fb_models, vec!["qwen3.5-9b", "qwen3.6-27b"]);
}

#[test]
fn empty_candidates_is_none() {
    let routing = routing_with("default", &[]);
    let matcher = matcher_with(vec![]);
    assert!(
        matcher
            .match_target("local", "default", &routing, &[], Some(3), "hello")
            .is_none()
    );
}
