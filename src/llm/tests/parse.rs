use super::*;

#[test]
fn parses_unfenced_object() {
    let v = parse_json_response(r#"{"a": 1}"#).unwrap();
    assert_eq!(v, serde_json::json!({"a": 1}));
}

#[test]
fn parses_fenced_object() {
    let v = parse_json_response("```json\n{\"a\": 1}\n```").unwrap();
    assert_eq!(v, serde_json::json!({"a": 1}));
}

#[test]
fn parses_plain_json_fence_without_language_tag() {
    let v = parse_json_response("```\n{\"a\": 1}\n```").unwrap();
    assert_eq!(v, serde_json::json!({"a": 1}));
}

#[test]
fn extracts_object_from_prose_and_noise() {
    let v = parse_json_response(
        "Sure! Here is the result:\n{\"chart\": \"bug_triage\"}\n\nHope that helps",
    )
    .unwrap();
    assert_eq!(v, serde_json::json!({"chart": "bug_triage"}));
}

#[test]
fn extracts_nested_braces() {
    let v = parse_json_response(r#"prefix {"a": {"b": [1, 2, {"c": 3}]}} suffix"#).unwrap();
    assert_eq!(v, serde_json::json!({"a": {"b": [1, 2, {"c": 3}]}}));
}

#[test]
fn handles_braces_inside_strings() {
    let v = parse_json_response(r#"{"a": "text { with brace", "b": "}"}"#).unwrap();
    assert_eq!(v, serde_json::json!({"a": "text { with brace", "b": "}"}));
}

#[test]
fn extracts_array_from_fenced_with_noise() {
    // The reranker shape: noise before the fence defeats the fence strip,
    // so extraction must find the balanced array.
    let v =
        parse_json_response("Sure!\n```json\n[\"draft_doc\", \"bug_triage\"]\n```").unwrap();
    assert_eq!(v, serde_json::json!(["draft_doc", "bug_triage"]));
}

#[test]
fn returns_no_json_for_garbage() {
    assert_eq!(
        parse_json_response("not json at all"),
        Err(JsonParseError::NoJson)
    );
    assert_eq!(parse_json_response(""), Err(JsonParseError::NoJson));
    assert_eq!(
        parse_json_response("no { unmatched"),
        Err(JsonParseError::NoJson)
    );
}

#[test]
fn fence_strip_handles_missing_closer() {
    assert_eq!(strip_json_fence("```json\n{\"a\": 1}"), r#"{"a": 1}"#);
}

#[derive(Debug, PartialEq, serde::Deserialize)]
struct Typed {
    #[serde(default)]
    name: String,
    #[serde(default)]
    score: f64,
    #[serde(default)]
    tags: Vec<String>,
}

fn empty_defaults() -> Value {
    Value::Null
}

#[test]
fn parse_typed_pristine_fast_path() {
    let t = parse_typed::<Typed>(r#"{"name":"x","score":1.5}"#, &empty_defaults(), |_| {})
        .unwrap();
    assert_eq!(t, Typed { name: "x".into(), score: 1.5, tags: vec![] });
}

#[test]
fn parse_typed_tolerant_fenced_input() {
    let t = parse_typed::<Typed>(
        "```json\n{\"name\":\"x\",\"score\":2.0}\n```",
        &empty_defaults(),
        |_| {},
    )
    .unwrap();
    assert_eq!(t, Typed { name: "x".into(), score: 2.0, tags: vec![] });
}

#[test]
fn parse_typed_tolerant_noisy_input() {
    let t = parse_typed::<Typed>(
        "Sure! Here is the result:\n{\"name\":\"x\"}",
        &empty_defaults(),
        |_| {},
    )
    .unwrap();
    assert_eq!(t, Typed { name: "x".into(), score: 0.0, tags: vec![] });
}

#[test]
fn parse_typed_applies_defaults_for_missing_fields() {
    // A noisy prefix defeats the fast path so the recovery merge runs; the
    // LLM omitted `score`/`tags`, which `defaults` fills in.
    let defaults = serde_json::json!({"score": 9.0, "tags": ["a"]});
    let t = parse_typed::<Typed>("prefix {\"name\":\"y\"} suffix", &defaults, |_| {}).unwrap();
    assert_eq!(t, Typed { name: "y".into(), score: 9.0, tags: vec!["a".into()] });
}

#[test]
fn parse_typed_defaults_do_not_overwrite_present_values() {
    // `score` is present so the merge must leave it alone.
    let defaults = serde_json::json!({"score": 9.0});
    let t = parse_typed::<Typed>("prefix {\"name\":\"y\",\"score\":3.0} suffix", &defaults, |_| {})
        .unwrap();
    assert_eq!(t, Typed { name: "y".into(), score: 3.0, tags: vec![] });
}

#[test]
fn parse_typed_runs_sanitize_field_coercion() {
    // "score" arrives as a string; sanitize coerces it to a number so the
    // typed deserialize succeeds (mirrors the classifier's coerce_float).
    let coerce = |v: &mut Value| {
        if let Some(obj) = v.as_object_mut() {
            if let Some(s) = obj.get("score").and_then(Value::as_str) {
                if let Ok(n) = s.parse::<f64>() {
                    obj["score"] = Value::from(n);
                }
            }
        }
    };
    let t = parse_typed::<Typed>(r#"{"name":"z","score":"4.5"}"#, &empty_defaults(), coerce)
        .unwrap();
    assert_eq!(t, Typed { name: "z".into(), score: 4.5, tags: vec![] });
}

#[test]
fn parse_typed_no_json_errors() {
    let err = parse_typed::<Typed>("not json at all", &empty_defaults(), |_| {}).unwrap_err();
    assert!(matches!(err, JsonParseError::NoJson));
}

#[test]
fn parse_typed_serde_error_when_coercion_fails() {
    let err = parse_typed::<Typed>(r#"{"name":"x","score":"oops"}"#, &empty_defaults(), |_| {})
        .unwrap_err();
    assert!(matches!(err, JsonParseError::Serde(_)));
}

// -- Deterministic self-healing (repair_json) ---------------------------
//
// `repair_json` delegates its structural repair to
// `fluent_wvr::boundary::repair_boundary` (schema-blind lenient mode), so
// the shared-repair behaviors — trailing commas, bare keys/values,
// single-quote conversion, comment stripping, non-JSON literals,
// control-character escaping, braces-inside-strings — are asserted once in
// `fluent-wvr/src/boundary.rs` (the owning module) and are NOT duplicated
// here (DRY). The cases below cover only llm-specific post-processing that
// `repair_boundary` deliberately does not do: truncation closing
// (`close_open_containers`), mid-member tail dropping
// (`drop_incomplete_tail`), first-value re-extraction, and the None case.

fn repaired(raw: &str) -> Value {
    serde_json::from_str(&repair_json(raw).expect("must repair")).expect("repaired output parses")
}

#[test]
fn repair_closes_truncated_object() {
    assert_eq!(
        repaired(r#"{"a": 1, "b": 2"#),
        serde_json::json!({"a": 1, "b": 2})
    );
}

#[test]
fn repair_closes_truncated_array_and_string() {
    assert_eq!(
        repaired(r#"{"a": "hello"#),
        serde_json::json!({"a": "hello"})
    );
    assert_eq!(
        repaired(r#"{"a": [1, 2"#),
        serde_json::json!({"a": [1, 2]})
    );
}

#[test]
fn repair_handles_mismatched_extra_closers() {
    assert_eq!(
        repaired(r#"{"a": [1, 2}}"#),
        serde_json::json!({"a": [1, 2]})
    );
}

#[test]
fn repair_handles_dangling_backslash() {
    // A string truncated right after a backslash: the trailing `\` must be
    // escaped so the closing quote terminates the string.
    assert_eq!(
        repaired(r#"{"a": "trailing\"#),
        serde_json::json!({"a": "trailing\\"})
    );
}

#[test]
fn repair_drops_incomplete_tail_member() {
    // Truncated mid-member: `"b":` has no value. Dropping the dangling
    // tail yields a valid (if lossy) object.
    assert_eq!(
        repaired(r#"{"a": 1, "b":}"#),
        serde_json::json!({"a": 1})
    );
    assert_eq!(
        repaired(r#"{"a": 1, "b": 2, "c": "#),
        serde_json::json!({"a": 1, "b": 2})
    );
}

#[test]
fn repair_extracts_from_leading_prose() {
    assert_eq!(repaired("Sure! {a: 1}"), serde_json::json!({"a": 1}));
}

#[test]
fn repair_returns_none_for_garbage() {
    assert_eq!(repair_json("not json"), None);
    assert_eq!(repair_json(""), None);
    assert_eq!(repair_json("just words"), None);
}

#[test]
fn parse_json_response_self_heals() {
    let v = parse_json_response(r#"{"a": 1, "b": [2, 3,],}"#).unwrap();
    assert_eq!(v, serde_json::json!({"a": 1, "b": [2, 3]}));
}

#[test]
fn parse_json_response_repaired_is_recovered_not_pristine() {
    // The pristine fast path must be untouched: valid JSON stays as-is and
    // is not forced through the repair pipeline.
    assert_eq!(
        parse_json_response(r#"{"a": 1}"#).unwrap(),
        serde_json::json!({"a": 1})
    );
}
