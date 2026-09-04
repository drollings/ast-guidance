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
#[path = "../tests/parse.rs"]
mod tests;
