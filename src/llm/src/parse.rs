//! Tolerant JSON parsing for LLM outputs.
//!
//! LLM responses that are *supposed* to be JSON routinely come back wrapped
//! in markdown code fences or padded with prose. This module owns the single
//! tolerant parse pipeline — strip fence → parse → extract first balanced
//! JSON value — so every caller (classifier, chart adjudicator, reranker,
//! rubric judge, chart-stage output) shares one implementation
//! (ROADMAP_20260804_DRY M7.4). It is string-only: no LLM protocol types,
//! so it stays a pure helper on top of `serde_json`.

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

/// Tolerant parse of an LLM response into JSON.
///
/// Pipeline: strip a surrounding ` ```json ` fence, try a direct parse of the
/// cleaned text, then fall back to extracting the first balanced JSON value.
/// Returns [`JsonParseError::NoJson`] when nothing parses.
pub fn parse_json_response(text: &str) -> Result<Value, JsonParseError> {
    let cleaned = strip_json_fence(text);
    if let Ok(v) = serde_json::from_str::<Value>(cleaned) {
        return Ok(v);
    }
    extract_first_json_value(cleaned).ok_or(JsonParseError::NoJson)
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
}
