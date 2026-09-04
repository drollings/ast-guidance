use super::*;

#[test]
fn uuid_is_formatted_correctly() {
    let id = common_core::hash::uuid_v4();
    assert_eq!(id.len(), 36);
    assert_eq!(id.chars().filter(|&c| c == '-').count(), 4);
}

#[test]
fn fallback_has_stop_reason() {
    let r = fallback_completion("test");
    assert_eq!(r.choices.len(), 1);
    assert_eq!(r.choices[0].finish_reason, "stop");
}

#[test]
fn answer_text_extracts_first_choice() {
    let c = make_text_completion("fast", "the answer");
    assert_eq!(answer_text(&c).as_deref(), Some("the answer"));
}

#[test]
fn answer_text_is_none_without_choices() {
    let c = RouterResponse {
        id: String::new(),
        object: "chat.completion".into(),
        created: 0,
        model: "fast".into(),
        choices: vec![],
        usage: Usage::default(),
    };
    assert_eq!(answer_text(&c), None);
}

#[test]
fn answer_text_concatenates_text_parts() {
    let mut c = make_text_completion("fast", "ignored");
    c.choices[0].message.content = RouterMessageContent::Parts(vec![
        crate::types::ContentPart::Text { text: "hello".into() },
        crate::types::ContentPart::Text { text: "world".into() },
    ]);
    assert_eq!(answer_text(&c).as_deref(), Some("hello world"));
}

// ── M7: CORS headers unchanged through the move ──────────────────────────

#[test]
fn cors_headers_unchanged() {
    // Canonical owner is `crate::server::cors`; values are locked here
    // (M11 deleted the `common_core::http` shims these were pinned against).
    use crate::server::cors::{
        add_cors_headers, CORS_ALLOW_HEADERS, CORS_ALLOW_METHODS, CORS_ALLOW_ORIGIN,
    };
    assert_eq!(CORS_ALLOW_ORIGIN, "*");
    assert_eq!(CORS_ALLOW_METHODS, "POST, GET, OPTIONS");
    assert_eq!(CORS_ALLOW_HEADERS, "Content-Type, Authorization");
    let mut map = http::HeaderMap::new();
    add_cors_headers(&mut map);
    assert_eq!(map.get("access-control-allow-origin").unwrap(), "*");
    assert_eq!(
        map.get("access-control-allow-methods").unwrap(),
        "POST, GET, OPTIONS"
    );
    assert_eq!(
        map.get("access-control-allow-headers").unwrap(),
        "Content-Type, Authorization"
    );
}
