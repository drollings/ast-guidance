use super::*;
use crate::testing::{test_request, text_of};
use crate::types::RouterRequest;

fn make_request(text: &str) -> RouterRequest {
    let mut req = test_request(text);
    req.model = "test".into();
    req
}

#[test]
fn bearer_token_masked_prefix_preserved() {
    let m = SecretMask;
    let req = make_request("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9");
    let result = m.transform(&req).unwrap();
    let output = text_of(&result);
    assert!(output.contains("Bearer ****"), "got: {output}");
    assert!(!output.contains("eyJhbGci"), "got: {output}");
}

#[test]
fn basic_auth_masked() {
    let m = SecretMask;
    let req = make_request("Basic dXNlcjpwYXNzd29yZA==");
    let result = m.transform(&req).unwrap();
    let output = text_of(&result);
    assert!(output.contains("Basic ****"), "got: {output}");
    assert!(!output.contains("dXNlcjpwYXNzd29yZA=="), "got: {output}");
}

#[test]
fn password_keyvalue_masked() {
    let m = SecretMask;
    let req = make_request("password=superSecret123!");
    let result = m.transform(&req).unwrap();
    let output = text_of(&result);
    assert!(output.contains("password=****"), "got: {output}");
    assert!(!output.contains("superSecret123"), "got: {output}");
}

#[test]
fn sk_prefix_key_masked() {
    let m = SecretMask;
    let req = make_request("Use key sk-abc123def456ghijklmnopqrstuvwxyz");
    let result = m.transform(&req).unwrap();
    let output = text_of(&result);
    assert!(
        !output.contains("sk-abc123def456ghijklmnopqrstuvwxyz"),
        "got: {output}"
    );
}

#[test]
fn akia_key_masked() {
    let m = SecretMask;
    let req = make_request("AWS key: AKIA1234567890ABCDEF");
    let result = m.transform(&req).unwrap();
    let output = text_of(&result);
    assert!(!output.contains("AKIA1234567890ABCDEF"), "got: {output}");
}

#[test]
fn github_token_masked() {
    let m = SecretMask;
    let req = make_request("Token: ghp_abc123def456ghijklmnopqrstuvwxyz123456");
    let result = m.transform(&req).unwrap();
    let output = text_of(&result);
    assert!(!output.contains("ghp_abc"), "got: {output}");
}

#[test]
fn multiple_secrets_all_masked() {
    let m = SecretMask;
    let req = make_request("Bearer tok123 password=abc api_key=xyz");
    let result = m.transform(&req).unwrap();
    let output = text_of(&result);
    assert!(output.contains("Bearer ****"), "got: {output}");
    assert!(output.contains("password=****"), "got: {output}");
    assert!(output.contains("api_key=****"), "got: {output}");
    assert!(!output.contains("tok123"), "got: {output}");
    assert!(!output.contains("=abc"), "got: {output}");
    assert!(!output.contains("=xyz"), "got: {output}");
}

#[test]
fn clean_text_passes_unchanged() {
    let m = SecretMask;
    let req = make_request("Hello, how are you?");
    let result = m.transform(&req).unwrap();
    assert_eq!(text_of(&result), "Hello, how are you?");
}

#[test]
fn matching_case_insensitive_bearer() {
    let m = SecretMask;
    let req = make_request("BEARER token123");
    let result = m.transform(&req).unwrap();
    let output = text_of(&result);
    assert!(
        output.to_lowercase().contains("bearer ****"),
        "got: {output}"
    );
}

#[test]
fn all_key_names_handled() {
    let m = SecretMask;
    for name in &[
        "apikey",
        "access_token",
        "private_key",
        "token",
        "key",
        "secret",
    ] {
        let req = make_request(&format!("{name}=value123"));
        let result = m.transform(&req).unwrap();
        let output = text_of(&result);
        assert!(
            !output.contains("value123"),
            "{name} value not masked: {output}"
        );
    }
}
