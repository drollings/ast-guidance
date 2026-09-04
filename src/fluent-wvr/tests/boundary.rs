#![allow(unused_imports)]
#[allow(unused_imports)]
use fluent_wvr::boundary::*;
use fluent_wvr::prelude::*;
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};


use fluent_wvr::coerce::{coerce, Coercion};

#[derive(Debug, Clone, Default, PartialEq)]
struct Verdict {
    action: String,
    target: Option<String>,
    coherence: f64,
    complexity: u8,
    reason: String,
    tags: Option<String>,
}

impl FieldAccess for Verdict {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        let value = coerce(
            value,
            &[Coercion::StripQuotes, Coercion::Trim, Coercion::JsonEscape],
        );
        match name {
            "action" => {
                self.action = value;
                Ok(())
            }
            "target" => {
                self.target = (!value.is_empty()).then_some(value);
                Ok(())
            }
            "coherence" => {
                self.coherence = fluent_wvr::coerce::parse_number(&value)
                    .ok_or_else(|| FieldError::Parse(format!("invalid f64: {value}")))?;
                Ok(())
            }
            "complexity" => {
                self.complexity = fluent_wvr::coerce::parse_number(&value)
                    .ok_or_else(|| FieldError::Parse(format!("invalid u8: {value}")))?
                    as u8;
                Ok(())
            }
            "reason" => {
                self.reason = value;
                Ok(())
            }
            "tags" => {
                self.tags = (!value.is_empty()).then_some(value);
                Ok(())
            }
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "action" => Ok(self.action.clone()),
            "target" => Ok(self.target.clone().unwrap_or_default()),
            "coherence" => Ok(self.coherence.to_string()),
            "complexity" => Ok(self.complexity.to_string()),
            "reason" => Ok(self.reason.clone()),
            "tags" => Ok(self.tags.clone().unwrap_or_default()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["action", "target", "coherence", "complexity", "reason", "tags"]
    }
}

impl Describable for Verdict {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": "string",
                "target": "string",
                "coherence": "number",
                "complexity": "integer",
                "reason": "string",
                "tags": "string",
            },
        })
    }
}

// -- repair_boundary: schema-blind mode is the historical heal pass ------

fn repaired(raw: &str) -> String {
    repair_boundary(raw, &BoundaryOptions::lenient()).text
}

#[test]
fn repair_trailing_commas() {
    assert_eq!(
        repaired(r#"{"a": 1, "b": [1, 2,],}"#),
        r#"{"a": 1, "b": [1, 2]}"#
    );
}

#[test]
fn repair_quotes_bare_keys_and_values() {
    assert_eq!(
        repaired(r#"{action: route, target: code}"#),
        r#"{"action": "route", "target": "code"}"#
    );
}

#[test]
fn repair_converts_single_quotes() {
    assert_eq!(
        repaired("{'a': 'it\\'s', 'b': \"ok\"}"),
        r#"{"a": "it's", "b": "ok"}"#
    );
}

#[test]
fn repair_strips_comments() {
    // The `//` comment is stripped up to (not including) the newline, which
    // is JSON whitespace and survives; the block comment is fully removed.
    assert_eq!(
        repaired("{\"a\": 1, // note\n \"b\": 2 /* inline */}"),
        "{\"a\": 1, \n \"b\": 2 }"
    );
}

#[test]
fn repair_normalizes_non_json_literals() {
    assert_eq!(
        repaired(r#"{"a": undefined, "b": NaN, "c": Infinity, "d": -Infinity, "e": None, "f": true}"#),
        r#"{"a": null, "b": null, "c": null, "d": null, "e": null, "f": true}"#
    );
}

#[test]
fn repair_escapes_raw_control_chars_in_strings() {
    assert_eq!(
        repaired("{\"response\": \"line one\nline two\"}"),
        "{\"response\": \"line one\\nline two\"}"
    );
    assert_eq!(
        repaired("{'response': 'a\nb'}"),
        "{\"response\": \"a\\nb\"}"
    );
}

#[test]
fn repair_handles_dangling_backslash() {
    // The string stays unterminated in the repair text; the container
    // closer (llm's `close_open_containers`) fixes it afterwards.
    assert_eq!(repaired(r#"{"a": "trailing\"#), r#"{"a": "trailing\"#);
}

#[test]
fn repair_drops_lone_slash_and_semicolon() {
    assert_eq!(repaired(r#"{"a": 1; / "b": 2}"#), r#"{"a": 1  "b": 2}"#);
}

#[test]
fn repair_preserves_braces_inside_strings() {
    assert_eq!(
        repaired(r#"{"a": "text { with brace", "b": "}"}"#),
        r#"{"a": "text { with brace", "b": "}"}"#
    );
}

#[test]
fn repair_drops_leading_prose() {
    assert_eq!(repaired("Sure! {a: 1}"), r#"{"a": 1}"#);
}

// -- schema-aware member extraction + typed decode ------------------------

const VERDICT_SCHEMA: &[&str] = &[
    "action",
    "target",
    "coherence",
    "complexity",
    "reason",
    "tags",
];

#[test]
fn extract_members_from_bare_and_quoted() {
    let opts = BoundaryOptions::for_schema(VERDICT_SCHEMA);
    let members = extract_members("{action: route, 'target': code, reason: 'why'}", &opts)
        .expect("members");
    assert_eq!(
        members,
        vec![
            ("action".to_string(), "route".to_string()),
            ("target".to_string(), "code".to_string()),
            ("reason".to_string(), "'why'".to_string()),
        ]
    );
}

#[test]
fn extract_members_brace_less_key_value() {
    let opts = BoundaryOptions::for_schema(VERDICT_SCHEMA);
    let members = extract_members("action: route, target: code", &opts).expect("members");
    assert_eq!(
        members,
        vec![
            ("action".to_string(), "route".to_string()),
            ("target".to_string(), "code".to_string()),
        ]
    );
}

#[test]
fn extract_members_filters_to_schema() {
    let opts = BoundaryOptions::for_schema(VERDICT_SCHEMA);
    let members = extract_members("{action: route, unknown: 1, target: code}", &opts)
        .expect("members");
    // `unknown` is not a schema field and is not emitted.
    assert_eq!(
        members,
        vec![
            ("action".to_string(), "route".to_string()),
            ("target".to_string(), "code".to_string()),
        ]
    );
}

#[test]
fn extract_members_no_members_errors() {
    let opts = BoundaryOptions::for_schema(VERDICT_SCHEMA);
    assert_eq!(extract_members("llama llama llama", &opts), Err(BoundaryError::NoMembers));
    assert_eq!(extract_members("", &opts), Err(BoundaryError::NoMembers));
}

#[test]
fn decode_boundary_coerces_through_set_field() {
    let opts = BoundaryOptions::for_schema(VERDICT_SCHEMA);
    let (v, decoded) = decode_boundary::<Verdict>(
        "{action: 'route', target: code, coherence: '0.9', complexity: 7, reason: 'big model',}",
        &opts,
    )
    .expect("decode");
    assert_eq!(v.action, "route");
    assert_eq!(v.target.as_deref(), Some("code"));
    assert_eq!(v.coherence, 0.9);
    assert_eq!(v.complexity, 7);
    assert_eq!(v.reason, "big model");
    assert!(v.tags.is_none());
    assert_eq!(decoded.len(), 5);
}

#[test]
fn decode_boundary_garbage_errors() {
    let opts = BoundaryOptions::for_schema(VERDICT_SCHEMA);
    assert_eq!(
        decode_boundary::<Verdict>("not key value at all", &opts),
        Err(BoundaryError::NoMembers)
    );
}

#[test]
fn decode_boundary_skips_bad_member_keeps_default() {
    let opts = BoundaryOptions::for_schema(VERDICT_SCHEMA);
    // `coherence: undefined` fails to coerce (parse_number -> None), so it
    // is skipped and stays 0.0 (the failing default) rather than fabricating
    // a passing score.
    let (v, decoded) = decode_boundary::<Verdict>(
        "{action: route, coherence: undefined, complexity: 7}",
        &opts,
    )
    .expect("decode");
    assert_eq!(v.action, "route");
    assert_eq!(v.coherence, 0.0);
    assert_eq!(v.complexity, 7);
    assert!(!decoded.contains(&"coherence".to_string()));
}

#[test]
fn decode_boundary_typed_uses_field_names_as_schema() {
    // `decode_boundary_typed` derives the schema from the type itself.
    let (v, _) = decode_boundary_typed::<Verdict>("action: respond").expect("decode");
    assert_eq!(v.action, "respond");
    assert!(v.target.is_none());
}
