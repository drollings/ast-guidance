use super::*;
use std::collections::HashMap;

fn mapped(original: &str, anonymized: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    build_anonymize_map(original, anonymized, &mut map);
    map
}

#[test]
fn identical_texts_yield_empty_map() {
    assert!(mapped("hello world", "hello world").is_empty());
    assert!(mapped("", "").is_empty());
}

#[test]
fn single_placeholder_keys_by_placeholder_name() {
    // M6.2 verbatim semantics: the walk records one entry per placeholder
    // occurrence, keyed `[NAME]` first. (The matched original slice is the
    // verbatim `find_matching_len` result — a single leading char — preserved
    // exactly as the router helper produced it; improving the slice is a
    // behavior change, filed separately.)
    let map = mapped("hello alice world", "hello [NAME] world");
    assert_eq!(map.len(), 1);
    assert_eq!(map.get("[NAME]").map(String::as_str), Some("a"));
}

#[test]
fn repeated_placeholder_gets_suffixed_keys() {
    let map = mapped("a alice b alice c", "a [NAME] b [NAME] c");
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("[NAME]"));
    assert!(map.contains_key("[NAME]_1"));
}

#[test]
fn distinct_placeholders_map_independently() {
    let map = mapped(
        "mail a@b.com call 555-123-4567",
        "mail [EMAIL] call [PHONE]",
    );
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("[EMAIL]"));
    assert!(map.contains_key("[PHONE]"));
}

#[test]
fn second_call_overwrites_same_key() {
    // Per-call counters restart, so the second call overwrites `[NAME]`;
    // accumulation across calls is the caller's shape (verbatim).
    let mut map = HashMap::new();
    build_anonymize_map("hi alice", "hi [NAME]", &mut map);
    build_anonymize_map("hi bob", "hi [NAME]", &mut map);
    assert_eq!(map.len(), 1);
    assert_eq!(map.get("[NAME]").map(String::as_str), Some("b"));
}

#[test]
fn overlong_bracket_span_is_not_a_placeholder() {
    // `[...]` longer than 64 chars is walked byte-wise, never a map key.
    let long = format!("[{}]", "x".repeat(64));
    assert!(long.len() > 64);
    let map = mapped(&format!("v {long} w"), &format!("v {long} w"));
    assert!(map.is_empty());
}

#[test]
fn round_trips_through_anonymize() {
    // The production pairing: `anonymize` output diffed against its input
    // yields one entry per redaction, keyed by placeholder.
    let original = "Contact user@example.com or call 555-123-4567";
    let anonymized = anonymize(original);
    assert_ne!(anonymized, original);
    let map = mapped(original, &anonymized);
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("[EMAIL]"));
    assert!(map.contains_key("[PHONE]"));
}
