use super::*;
use crate::test_stubs::StubChatBackend;

fn rubric(require: &[&str]) -> ChartRubric {
    ChartRubric {
        require_fields: require.iter().map(ToString::to_string).collect(),
        judge_model: None,
        min_score: 0.7,
    }
}

#[test]
fn empty_rubric_passes_any_output() {
    let out = serde_json::json!({"whatever": true});
    let v = check_rubric(&rubric(&[]), &out, None, None, "t").expect("no error");
    assert!(v.accepted);
    assert!(!v.judged);
}

#[test]
fn present_non_null_field_passes() {
    let out = serde_json::json!({"plan": "step 1", "cause": null});
    let v = check_rubric(&rubric(&["plan"]), &out, None, None, "t").expect("no error");
    assert!(v.accepted);
}

#[test]
fn missing_field_fails() {
    let out = serde_json::json!({"plan": "step 1"});
    let v = check_rubric(&rubric(&["cause"]), &out, None, None, "t").expect("no error");
    assert!(!v.accepted);
    assert!(v.reason.contains("cause"));
}

#[test]
fn null_field_fails() {
    let out = serde_json::json!({"plan": null});
    let v = check_rubric(&rubric(&["plan"]), &out, None, None, "t").expect("no error");
    assert!(!v.accepted);
    assert!(v.reason.contains("null"));
}

#[test]
fn nested_path_check() {
    let out = serde_json::json!({"answer": {"steps": ["a"], "verdict": "ok"}});
    let v =
        check_rubric(&rubric(&["answer.verdict"]), &out, None, None, "t").expect("no error");
    assert!(v.accepted);
}

#[test]
fn judge_consulted_only_when_configured() {
    let out = serde_json::json!({"x": 1});
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always(
        r#"{"score": 0.9, "accepted": true, "reason": "good"}"#,
    ));
    // No judge_model → backend ignored (judged = false, no LLM call).
    let v = check_rubric(&rubric(&["x"]), &out, Some(&backend), None, "t").expect("no error");
    assert!(v.accepted);
    assert!(!v.judged);

    // judge_model set → backend consulted.
    let mut r = rubric(&["x"]);
    r.judge_model = Some("judge".into());
    let v = check_rubric(&r, &out, Some(&backend), None, "t").expect("no error");
    assert!(v.accepted);
    assert!(v.judged);
    assert_eq!(v.score, Some(0.9));
}

#[test]
fn judge_below_min_score_rejects() {
    let out = serde_json::json!({"x": 1});
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always(
        r#"{"score": 0.4, "accepted": true, "reason": "weak"}"#,
    ));
    let mut r = rubric(&["x"]);
    r.judge_model = Some("judge".into());
    let v = check_rubric(&r, &out, Some(&backend), None, "t").expect("no error");
    assert!(!v.accepted, "judge accept below min_score must reject");
    assert!(v.reason.contains("below min"));
}

#[test]
fn judge_absent_backend_degrades_to_deterministic() {
    let out = serde_json::json!({"x": 1});
    let mut r = rubric(&["x"]);
    r.judge_model = Some("judge".into());
    // No backend provided → warn + accept on the deterministic gate.
    let v = check_rubric(&r, &out, None, None, "t").expect("no error");
    assert!(v.accepted);
    assert!(!v.judged);
}

#[test]
fn judge_parse_tolerates_fences() {
    let v = parse_judge_output("```json\n{\"score\": 0.95, \"accepted\": true}\n```", 0.7)
        .expect("parses");
    assert!(v.accepted);
    assert_eq!(v.score, Some(0.95));
    let v = parse_judge_output("noise {\"score\": 0.2, \"accepted\": false} trailing", 0.7)
        .expect("parses");
    assert!(!v.accepted);
}

#[test]
fn cache_short_circuits_and_records() {
    let cache = RubricCache::new();
    let out = serde_json::json!({"plan": "ok"});
    assert!(!cache.is_cached_accepted(&rubric(&["plan"]), &out));
    cache.record_accepted(&rubric(&["plan"]), &out);
    assert!(cache.is_cached_accepted(&rubric(&["plan"]), &out));
    assert_eq!(cache.len(), 1);

    // Same rubric, different output → not cached.
    let other = serde_json::json!({"plan": "different"});
    assert!(!cache.is_cached_accepted(&rubric(&["plan"]), &other));
}

#[test]
fn cached_pair_short_circuits_gate() {
    let cache = RubricCache::new();
    let out = serde_json::json!({"x": 1});
    let mut r = rubric(&["x"]);
    r.judge_model = Some("judge".into());
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::always(
        r#"{"score": 0.9, "accepted": true, "reason": "good"}"#,
    ));
    let v = check_rubric(&r, &out, Some(&backend), Some(&cache), "t").expect("no error");
    assert!(v.accepted);
    assert!(v.judged, "first run consults the judge");

    let v2 = check_rubric(&r, &out, Some(&backend), Some(&cache), "t").expect("no error");
    assert!(v2.accepted);
    assert!(!v2.judged, "cached pair skips the judge");
    assert!(v2.reason.contains("cached"));
}
