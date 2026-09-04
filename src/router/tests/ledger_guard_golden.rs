use super::*;
use crate::test_support::capture_logs;

#[test]
fn pii_corpus_scrubbed_and_flagged() {
    let cases = [
        ("Contact user@example.com for help.", "email", "Contact [REDACTED:email] for help."),
        ("Call me at 555-123-4567 soon.", "phone", "Call me at [REDACTED:phone] soon."),
        ("My ssn is 123-45-6789.", "ssn", "My ssn is [REDACTED:ssn]."),
    ];
    for (input, expected_pattern, expected_text) in cases {
        let s = scrub_for_ledger(input);
        assert!(s.flagged, "flagged {input}");
        assert_eq!(s.pattern.as_deref(), Some(expected_pattern), "pattern for {input}");
        assert_eq!(s.text, expected_text);
        assert!(!s.text.contains(expected_pattern) || s.text.contains("[REDACTED:"), "marker present");
    }
}

#[test]
fn scrub_golden_exact_outputs() {
    let s = scrub_for_ledger("Contact user@example.com for help.");
    assert_eq!(s.text, "Contact [REDACTED:email] for help.");
    assert!(s.flagged);
    let s = scrub_for_ledger("My ssn is 123-45-6789.");
    assert_eq!(s.text, "My ssn is [REDACTED:ssn].");
}

#[test]
fn emit_write_audit_on_flagged() {
    let (_, logs) = capture_logs(|| {
        let ledger = crate::ledger::ContentNodeLedger::open(&std::env::temp_dir().join(format!("golden-{}", common_core::hash::uuid_v4()))).unwrap();
        ledger.record_request("sess", "req", "Contact user@example.com now").unwrap();
    });
    let joined = logs.join("\n");
    assert!(joined.contains("write_path") || joined.contains("router.audit"), "audit emitted {joined}");
}

#[test]
fn clean_text_not_flagged() {
    let s = scrub_for_ledger("What is the capital of France?");
    assert!(!s.flagged);
    assert_eq!(s.text, "What is the capital of France?");
}
