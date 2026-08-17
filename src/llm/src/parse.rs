//! Tolerant JSON parsing for LLM outputs.
//!
//! LLM responses that are *supposed* to be JSON routinely come back
//! wrapped in markdown code fences or padded with prose.  This module owns
//! the single tolerant parse pipeline — strip fence → parse → extract first
//! balanced JSON value — so every caller (classifier, chart adjudicator,
//! reranker, rubric judge, chart-stage output) shares one implementation.
//! It is string-only: no LLM protocol types, so it stays a pure helper on
//! top of `serde_json`.
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Errors produced by [`parse_json_response`].
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum JsonParseError {
    /// The text contained no parseable JSON value at all.
    #[error("no JSON value found in LLM output")]
    NoJson,
    /// A candidate JSON value was found but failed to parse.
    #[error("JSON parse error: {0}")]
    Serde(String),
}

/// Strip a surrounding markdown code fence, if present.
///
/// Accepts ` ```json ` and ` ``` ` openers with a trailing ` ``` `. Leading
/// and trailing whitespace is trimmed; any prose outside the fence is left
/// in place (the extraction step in [`parse_json_response`] discards it).
pub fn strip_json_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let after_open = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .unwrap_or(trimmed);
    after_open.trim_end_matches("```").trim()
}

/// Extract the first balanced JSON value (`{...}` object or `[...]` array)
/// from otherwise-noisy text.
///
/// Scans left-to-right for the earliest `{` or `[`, then finds the matching
/// close with a depth counter that skips over string contents (so nested
/// braces inside values and `"{"` inside strings do not confuse the scan).
/// Returns the parsed value, or `None` if no balanced JSON was found.
pub fn extract_first_json_value(raw: &str) -> Option<Value> {
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if matches!(bytes[i], b'{' | b'[') {
            let start = i;
            let mut depth = 0usize;
            let mut in_string = false;
            let mut escaped = false;
            let mut end = None;
            let mut j = start;
            while j < bytes.len() {
                let b = bytes[j];
                if in_string {
                    if escaped {
                        escaped = false;
                    } else if b == b'\\' {
                        escaped = true;
                    } else if b == b'"' {
                        in_string = false;
                    }
                } else {
                    match b {
                        b'"' => in_string = true,
                        b'{' | b'[' => depth += 1,
                        b'}' | b']' => {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                end = Some(j);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                j += 1;
            }
            let end = end?;
            if let Ok(v) = serde_json::from_str(&raw[start..=end]) {
                return Some(v);
            }
            // Invalid JSON inside a balanced region: skip past it and keep
            // scanning for a later, valid value.
            i = end;
        }
        i += 1;
    }
    None
}

/// Close any open string literals and containers left dangling by a truncated
/// response, producing a well-formed document. Extra/mismatched closers are
/// dropped so a stray `}}` cannot poison an otherwise-truncated value.
fn close_open_containers(healed: &str) -> String {
    let mut out = String::with_capacity(healed.len() + 4);
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for c in healed.chars() {
        if in_string {
            if escaped {
                out.push(c);
                escaped = false;
            } else if c == '\\' {
                out.push('\\');
                escaped = true;
            } else if c == '"' {
                out.push('"');
                in_string = false;
            } else if let Some(e) = fluent_wvr::coerce::escaped_char(c) {
                out.push_str(&e);
            } else {
                out.push(c);
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push('"');
            }
            '{' | '[' => {
                stack.push(c);
                out.push(c);
            }
            '}' | ']' => {
                let matching = stack.last().is_some_and(|open| {
                    (*open == '{' && c == '}') || (*open == '[' && c == ']')
                });
                if matching {
                    stack.pop();
                    out.push(c);
                }
                // Mismatched or extra closer: dropped.
            }
            _ => {
                out.push(c);
            }
        }
    }

    if in_string {
        // A dangling `\` would swallow the closing quote; escape it first.
        if escaped {
            out.push('\\');
        }
        out.push('"');
    }
    while let Some(open) = stack.pop() {
        out.push(if open == '[' { ']' } else { '}' });
    }

    out
}

/// Truncation repair for a response cut off mid-member. Collects the
/// top-level comma positions, then (rightmost first) drops the dangling tail
/// after each comma, closes the remaining containers, and returns the first
/// candidate that parses. Handles `{"a": 1, "b":}`-style truncation that
/// closing containers alone cannot fix. Returns `None` when no truncation
/// point heals.
fn drop_incomplete_tail(healed: &str) -> Option<String> {
    let chars: Vec<char> = healed.chars().collect();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    let mut commas: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => depth = (depth - 1).max(0),
            ',' if depth == 1 => commas.push(i),
            _ => {}
        }
        i += 1;
    }
    for pos in commas.into_iter().rev() {
        let candidate: String = chars[..pos].iter().collect();
        let closed = close_open_containers(&candidate);
        if serde_json::from_str::<Value>(&closed).is_ok() {
            return Some(closed);
        }
    }
    None
}

/// Deterministically heal common malformed JSON that small LLMs emit — the
/// self-healing step. Fixes trailing commas, unquoted keys, single-quoted
/// strings, comments, non-JSON literals (`undefined`, `None`, `NaN`,
/// `Infinity`) and truncation (a dangling string/brace/bracket), all with
/// string-awareness so legitimate contents are never altered.
///
/// The structural repair itself lives in `fluent_wvr::boundary::repair_boundary`
/// (schema-blind mode) — the shared, fluent-wvr-native replacement for the
/// historical `heal_lexically` pass, with value normalization factored through
/// `fluent_wvr::coerce`.
///
/// Conservative by construction: it only repairs structure outside string
/// values, verifies the result parses, and returns `None` when nothing heals
/// to valid JSON. Purely deterministic string manipulation — no LLM
/// round-trip — so it never spends context-window budget the way a
/// corrective-prompt retry would. Callers should try a direct parse first and
/// treat a successful repair as "recovered" (not pristine) output.
pub fn repair_json(raw: &str) -> Option<String> {
    let start = raw.find(['{', '['])?;
    let healed =
        fluent_wvr::boundary::repair_boundary(&raw[start..], &fluent_wvr::BoundaryOptions::lenient())
            .text;
    if let Some(v) = extract_first_json_value(&healed) {
        return Some(v.to_string());
    }
    let closed = close_open_containers(&healed);
    if serde_json::from_str::<Value>(&closed).is_ok() {
        return Some(closed);
    }
    // Truncated mid-member: drop the dangling tail at a top-level comma.
    drop_incomplete_tail(&healed)
}

/// Tolerant parse of an LLM response into JSON.
///
/// Pipeline: strip a surrounding ` ```json ` fence, try a direct parse of the
/// cleaned text, then fall back to extracting the first balanced JSON value,
/// then — last resort — deterministically repair common malformations
/// ([`repair_json`]) and re-parse. Returns [`JsonParseError::NoJson`] when
/// nothing parses.
pub fn parse_json_response(text: &str) -> Result<Value, JsonParseError> {
    let cleaned = strip_json_fence(text);
    if let Ok(v) = serde_json::from_str::<Value>(cleaned) {
        return Ok(v);
    }
    if let Some(v) = extract_first_json_value(cleaned) {
        return Ok(v);
    }
    if let Some(repaired) = repair_json(cleaned) {
        if let Ok(v) = serde_json::from_str::<Value>(&repaired) {
            return Ok(v);
        }
    }
    Err(JsonParseError::NoJson)
}

/// Tolerant parse + coerce into a typed value.
///
/// Pipeline: (1) try a direct deserialize of `raw` — the fast path, so
/// pristine LLM JSON skips every recovery step; (2) fall back to the shared
/// tolerant parse ([`parse_json_response`]: fence-strip → extract-first-value);
/// (3) merge `defaults` into missing object fields; (4) run `sanitize` field
/// coercion; (5) deserialize to `T`. Never re-implements fence-stripping.
///
/// `defaults` is a JSON object whose fields are inserted into the parsed
/// object only where the LLM omitted them (existing values are never
/// overwritten). `sanitize` receives the parsed `Value` — after `defaults` —
/// and mutates it in place; the canonical `coerce_float`/`coerce_u8`/
/// `coerce_string` helpers operate on an object map, so a typical closure is
/// `|v| if let Some(o) = v.as_object_mut() { coerce_float(o, "score", 1.0) }`.
/// Use a no-op (`|_| {}`) when the target type's `serde` defaults already
/// cover missing fields.
///
/// This is the single codec entry for the "build prompt → call → tolerant
/// parse → coerce → defaults → deserialize" round-trip shared by every router
/// LLM feature.
pub fn parse_typed<T>(
    raw: &str,
    defaults: &Value,
    sanitize: impl FnOnce(&mut Value),
) -> Result<T, JsonParseError>
where
    T: DeserializeOwned,
{
    if let Ok(v) = serde_json::from_str::<T>(raw) {
        return Ok(v);
    }
    let mut value = parse_json_response(raw)?;
    if let (Value::Object(map), Value::Object(defaults)) = (&mut value, defaults) {
        for (k, dv) in defaults {
            map.entry(k.clone()).or_insert_with(|| dv.clone());
        }
    }
    sanitize(&mut value);
    serde_json::from_value(value).map_err(|e| JsonParseError::Serde(e.to_string()))
}

#[cfg(test)]
mod tests {
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
}
