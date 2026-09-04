use super::*;
use crate::filters::RegexMatch;

fn output_decision(
    action: FilterAction,
    pattern: &str,
    matches: Vec<(usize, usize)>,
) -> FilterDecision {
    FilterDecision::OutputFilter {
        action: action.clone(),
        matched_pattern: pattern.to_string(),
        codewords: Default::default(),
        matches: matches
            .into_iter()
            .map(|(start, end)| RegexMatch {
                pattern_name: pattern.to_string(),
                matched_text: "x".to_string(),
                start,
                end,
                action: action.clone(),
            })
            .collect(),
    }
}

#[test]
fn clean_text_unchanged_and_unflagged() {
    let s = scrub_for_ledger("What is the capital of France?");
    assert!(!s.flagged);
    assert_eq!(s.text, "What is the capital of France?");
    assert_eq!(s.pattern, None);
}

#[test]
fn email_is_redacted() {
    let s = scrub_for_ledger("Contact user@example.com for help.");
    assert!(s.flagged);
    assert_eq!(s.pattern.as_deref(), Some("email"));
    assert_eq!(s.text, "Contact [REDACTED:email] for help.");
    assert!(!s.text.contains("user@example.com"), "email must be gone");
}

#[test]
fn phone_is_redacted() {
    let s = scrub_for_ledger("Call me at 555-123-4567 soon.");
    assert!(s.flagged);
    assert_eq!(s.pattern.as_deref(), Some("phone"));
    assert_eq!(s.text, "Call me at [REDACTED:phone] soon.");
}

#[test]
fn ssn_is_redacted() {
    let s = scrub_for_ledger("My ssn is 123-45-6789.");
    assert!(s.flagged);
    assert_eq!(s.pattern.as_deref(), Some("ssn"));
    assert_eq!(s.text, "My ssn is [REDACTED:ssn].");
}

#[test]
fn api_key_is_rejected() {
    let s = scrub_for_ledger("here is the key: api_key = abcdefghijklmnop");
    assert!(s.flagged);
    assert_eq!(s.pattern.as_deref(), Some("api_key"));
    assert_eq!(s.text, "[rejected: api_key]");
}

#[test]
fn adjacent_matches_replaced_rightmost_first_without_index_corruption() {
    let decision = output_decision(FilterAction::Redact, "p1", vec![(0, 3), (3, 6), (6, 9)]);
    let s = apply_filter_decision("aaaBBBccc", &decision);
    assert_eq!(s.text, "[REDACTED:p1][REDACTED:p1][REDACTED:p1]");
    assert!(s.flagged);
}

#[test]
fn omit_action_deletes_the_span() {
    let decision = output_decision(FilterAction::Omit, "secret", vec![(7, 18)]);
    let s = apply_filter_decision("prefix SECRETVALUE suffix", &decision);
    assert_eq!(s.text, "prefix  suffix");
    assert!(s.flagged);
}

#[test]
fn anonymize_collapses_to_redact_marker() {
    // Anonymize is irreversible on the write path: same marker as Redact.
    let decision = output_decision(FilterAction::Anonymize, "email", vec![(8, 24)]);
    let s = apply_filter_decision("Contact user@example.com for help.", &decision);
    assert_eq!(s.text, "Contact [REDACTED:email] for help.");
}
