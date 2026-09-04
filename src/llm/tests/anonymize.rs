use super::*;

#[test]
fn test_anonymize_email() {
    let result = anonymize("Contact me at user@example.com");
    assert_eq!(result, "Contact me at [EMAIL]");
}

#[test]
fn test_anonymize_credit_card() {
    let result = anonymize("Card: 1234-5678-9012-3456");
    assert!(result.contains("[CREDIT_CARD]"));
}

#[test]
fn test_anonymize_ssn_us() {
    let result = anonymize("SSN: 123-45-6789");
    assert!(result.contains("[SSN]"));
    assert!(!result.contains("123-45-6789"));
}

#[test]
fn test_anonymize_nino_uk() {
    let result = anonymize("NINO: AB123456C");
    assert!(result.contains("[NINO]"));
}

#[test]
fn test_anonymize_sin_ca() {
    let result = anonymize("SIN: 046-454-286");
    assert!(result.contains("[SIN]"));
}

#[test]
fn test_anonymize_bearer_token() {
    let result = anonymize("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9");
    assert!(result.contains("[BEARER_TOKEN]"));
}

#[test]
fn test_anonymize_aws_key() {
    let result = anonymize("Key: AKIAIOSFODNN7EXAMPLE");
    assert!(result.contains("[AWS_KEY]"));
    assert!(!result.contains("AKIAIOSFODNN7EXAMPLE"));
}

#[test]
fn test_anonymize_generic_api_key() {
    let result = anonymize("api_key=abcdef1234567890abcdef1234567890");
    assert!(result.contains("[API_KEY]"));
}

#[test]
fn test_anonymize_ipv6() {
    let result = anonymize("IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334");
    assert!(result.contains("[IPv6]"));
}

#[test]
fn test_anonymize_ipv4() {
    let result = anonymize("Server at 192.168.1.1");
    assert_eq!(result, "Server at [IP_ADDRESS]");
}

#[test]
fn test_anonymize_phone() {
    let result = anonymize("Call 555-123-4567 for info");
    assert!(result.contains("[PHONE]"));
}

#[test]
fn test_anonymize_multiple() {
    let result = anonymize("user@test.com api_key=abcdefghijklmnop12345 from 10.0.0.1");
    assert!(result.contains("[EMAIL]"));
    assert!(result.contains("[REDACTED]"));
    assert!(result.contains("[IP_ADDRESS]"));
}

#[test]
fn test_no_pii_unchanged() {
    let text = "This is a normal sentence with no PII.";
    let result = anonymize(text);
    assert_eq!(result, text);
}

#[test]
fn test_double_apply_is_idempotent() {
    // M6.1 leak taxonomy: placeholders must never match a pattern on a
    // second pass (otherwise scrubbed text would morph downstream).
    let cases = [
        "Contact me at user@example.com",
        "Card: 1234-5678-9012-3456",
        "SSN: 123-45-6789",
        "NINO: AB123456C",
        "SIN: 046-454-286",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9",
        "Key: AKIAIOSFODNN7EXAMPLE",
        "api_key=abcdef1234567890abcdef1234567890",
        "IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        "Server at 192.168.1.1",
        "Call 555-123-4567 for info",
    ];
    for input in cases {
        let once = anonymize(input);
        assert_ne!(once, input, "case must actually redact: {input}");
        assert_eq!(anonymize(&once), once, "double-apply must be fixed-point: {input}");
    }
}

#[test]
fn test_codeword_and_placeholder_as_input_pass_through() {
    // M6.1: already-scrubbed surfaces (codewords, placeholders) must not be
    // rewritten — the pipeline never double-scrubs its own vocabulary.
    for input in [
        "CODEWORD_SSN_1",
        "replace CODEWORD_EMAIL_2 soon",
        "[EMAIL]",
        "[REDACTED:ssn]",
        "the [PHONE] number",
    ] {
        assert_eq!(anonymize(input), input, "must pass through: {input}");
    }
}

#[test]
fn test_mixed_adjacent_matches_all_redacted() {
    // M6.1: adjacent matches of different kinds in one span.
    let result = anonymize("user@test.com 555-123-4567 123-45-6789");
    assert!(result.contains("[EMAIL]"), "{result}");
    assert!(result.contains("[PHONE]"), "{result}");
    assert!(result.contains("[SSN]"), "{result}");
    assert!(!result.contains("user@test.com"));
    assert!(!result.contains("555-123-4567"));
    assert!(!result.contains("123-45-6789"));
}

#[test]
fn test_whitespace_obfuscated_pii_is_a_known_limit() {
    // M6.1: the regex baseline does not catch whitespace-split PII
    // (documents the limit — the ORT classifier rung covers these).
    assert_eq!(anonymize("user @ example.com"), "user @ example.com");
    assert_eq!(anonymize("Call me at 555 123 4567"), "Call me at 555 123 4567");
}

#[test]
fn test_placeholder_golden_table() {
    // M6.2 golden: the exact pattern → placeholder vocabulary. Any new
    // pattern MUST extend this table (never a second replace path).
    let table = [
        ("a@b.com", "[EMAIL]"),
        ("4111 1111 1111 1111", "[CREDIT_CARD]"),
        ("123-45-6789", "[SSN]"),
        ("AB123456C", "[NINO]"),
        ("046-454-286", "[SIN]"),
        ("Bearer abcdefgh", "[BEARER_TOKEN]"),
        ("AKIAIOSFODNN7EXAMPLE", "[AWS_KEY]"),
        ("abcdef1234567890abcdef1234567890", "[API_KEY]"),
        ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "[IPv6]"),
        ("192.168.1.1", "[IP_ADDRESS]"),
        ("555-123-4567", "[PHONE]"),
        ("api_key=abcdefghijklmnop12345", "[REDACTED]"),
    ];
    for (input, placeholder) in table {
        let out = anonymize(input);
        assert!(
            out.contains(placeholder),
            "{input:?} must yield {placeholder}, got {out:?}"
        );
        assert!(!out.contains(input), "{input:?} must not survive, got {out:?}");
    }
}
