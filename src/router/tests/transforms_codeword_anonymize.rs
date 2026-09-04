use super::*;
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};

fn make_request_with_matches(text: &str, matches: &[MatchEntry]) -> RouterRequest {
    let mut req = RouterRequest {
        model: "test-model".into(),
        messages: vec![RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text(text.into()),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        stream: None,
        tools: None,
        tool_choice: None,
        session_id: None,
        agent_id: None,
        adapter: None,
        instance: None,
        snapshot: None,
        id_slot: None,
        metadata: Default::default(),
    };
    req.metadata.insert(
        "pii_filter".into(),
        serde_json::json!({
            "matches": matches,
        }),
    );
    req
}

fn make_match(pattern: &str, text: &str, start: usize, end: usize) -> MatchEntry {
    MatchEntry {
        pattern_name: pattern.to_string(),
        matched_text: text.to_string(),
        start,
        end,
        action: "Anonymize".to_string(),
    }
}

#[test]
fn same_email_gets_same_codeword() {
    let text = "Contact user@example.com or admin@example.com";
    let matches = vec![
        make_match("email", "user@example.com", 8, 23),
        make_match("email", "admin@example.com", 27, 44),
    ];

    let anon = CodewordAnonymizer::new();
    let request = make_request_with_matches(text, &matches);
    let result = anon.transform(&request).unwrap();
    let output = result.messages[0].content.to_string_lossy();

    // The first email gets CODEWORD_EMAIL_1, second gets CODEWORD_EMAIL_2
    assert!(
        output.contains("CODEWORD_EMAIL_1"),
        "first email should become CODEWORD_EMAIL_1, got: {output}"
    );
    assert!(
        output.contains("CODEWORD_EMAIL_2"),
        "second email should become CODEWORD_EMAIL_2, got: {output}"
    );
    assert!(
        !output.contains("user@example.com"),
        "original email should be replaced"
    );
}

#[test]
fn same_text_gets_same_codeword() {
    let text = "email1@test.com and email1@test.com again";
    let matches = vec![
        make_match("email", "email1@test.com", 0, 15),
        make_match("email", "email1@test.com", 20, 35),
    ];

    let anon = CodewordAnonymizer::new();
    let request = make_request_with_matches(text, &matches);
    let result = anon.transform(&request).unwrap();
    let output = result.messages[0].content.to_string_lossy();

    // Both occurrences of "email1@test.com" should become CODEWORD_EMAIL_1
    let count = output.matches("CODEWORD_EMAIL_1").count();
    assert_eq!(
        count, 2,
        "same email should map to same codeword (CODEWORD_EMAIL_1) appearing twice, got: {output}"
    );
}

#[test]
fn reverse_substitution_restores_original() {
    let text = "My email is user@example.com";
    let matches = vec![make_match("email", "user@example.com", 12, 28)];

    let anon = CodewordAnonymizer::new();
    let request = make_request_with_matches(text, &matches);
    let result = anon.transform(&request).unwrap();
    let output = result.messages[0].content.to_string_lossy();

    assert!(
        output.contains("CODEWORD_EMAIL_1"),
        "should contain codeword"
    );

    let reversed = anon.reverse(&output);
    assert!(
        reversed.contains("user@example.com"),
        "reverse should restore original, got: {reversed}"
    );
}

#[test]
fn no_matches_passes_unchanged() {
    let text = "What is the capital of France?";
    let request = make_request_with_matches(text, &[]);
    let anon = CodewordAnonymizer::new();
    let result = anon.transform(&request).unwrap();
    let output = result.messages[0].content.to_string_lossy();
    assert_eq!(output, "What is the capital of France?");
}

#[test]
fn no_matches_yields_no_metadata_key() {
    // C2.M0: empty codeword map ⇒ no spurious empty object under the key.
    let text = "What is the capital of France?";
    let request = make_request_with_matches(text, &[]);
    let anon = CodewordAnonymizer::new();
    let result = anon.transform(&request).unwrap();
    assert!(
        result.metadata.get("codeword_map").is_none(),
        "no matches ⇒ no codeword_map key"
    );
}

#[test]
fn stores_codeword_map_in_metadata() {
    let text = "email is user@example.com";
    let matches = vec![make_match("email", "user@example.com", 9, 25)];

    let anon = CodewordAnonymizer::new();
    let request = make_request_with_matches(text, &matches);
    let result = anon.transform(&request).unwrap();

    let map = result.metadata.get("codeword_map");
    assert!(map.is_some(), "codeword_map should be present in metadata");
}

#[test]
fn name_returns_correct_value() {
    let anon = CodewordAnonymizer::new();
    assert_eq!(anon.name(), "codeword_anonymize");
}
