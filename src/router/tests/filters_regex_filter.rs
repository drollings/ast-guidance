use super::*;
use crate::config::{
    ConfidenceGate, FilterAction, FilterOutcome, FilterScope, PatternEntry,
};

fn entry(name: &str, outcome: FilterOutcome, regexes: &[&str]) -> PatternEntry {
    PatternEntry {
        name: name.into(),
        outcome,
        filter_action: None,
        confidence_gate: ConfidenceGate::None,
        scope: vec![FilterScope::Any],
        http_code: 403,
        error_message: None,
        regexes: regexes.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
fn from_entry_compiles_valid_regexes() {
    let f = RegexFilter::from_entry(&entry("b", FilterOutcome::HardReject, &["se\\d+"])).unwrap();
    assert_eq!(f.name, "b");
    assert_eq!(f.kind(), FilterKind::Regex);
    assert_eq!(f.regexes.len(), 1);
}

#[test]
fn from_entry_none_when_no_valid_regex() {
    // An entry whose regexes all fail to compile produces no filter.
    assert!(RegexFilter::from_entry(&entry("bad", FilterOutcome::HardReject, &["[unclosed"])).is_none());
    assert!(RegexFilter::from_entry(&entry("empty", FilterOutcome::HardReject, &[])).is_none());
}

#[test]
fn hard_reject_matches_first_pattern() {
    let f = RegexFilter::from_entry(&entry("blocked", FilterOutcome::HardReject, &["secret", "other"])).unwrap();
    let decision = f.evaluate(&FilterContext::pipeline("this contains secret here".into())).expect("decision");
    match decision {
        FilterDecision::HardReject { pattern, message } => {
            assert_eq!(pattern, "blocked");
            assert!(message.contains("blocked"));
        }
        other => panic!("expected hard reject, got {other:?}"),
    }
}

#[test]
fn hard_reject_none_when_no_match() {
    let f = RegexFilter::from_entry(&entry("b", FilterOutcome::HardReject, &["secret"])).unwrap();
    assert!(f.evaluate(&FilterContext::pipeline("nothing here".into())).is_none());
}

#[test]
fn hard_reject_respects_scope() {
    // Filter scoped to FrontierBound must not fire in the default pipeline
    // scope.
    let mut e = entry("b", FilterOutcome::HardReject, &["secret"]);
    e.scope = vec![FilterScope::FrontierBound];
    let f = RegexFilter::from_entry(&e).unwrap();
    assert!(f.evaluate(&FilterContext::pipeline("secret".into())).is_none());
    // It fires when the active scopes include FrontierBound.
    assert!(f.evaluate(&FilterContext::frontier("secret".into())).is_some());
}

#[test]
fn hard_reject_with_luhn_gate() {
    let mut e = entry("card", FilterOutcome::HardReject, &["\\d{4}-\\d{4}-\\d{4}-\\d{4}"]);
    e.confidence_gate = ConfidenceGate::LuhnValid;
    let f = RegexFilter::from_entry(&e).unwrap();
    // "1234-5678-9012-3456" fails Luhn -> not rejected.
    assert!(f.evaluate(&FilterContext::pipeline("card 1234-5678-9012-3456".into())).is_none());
    // "4111-1111-1111-1111" is Luhn-valid -> rejected.
    assert!(f.evaluate(&FilterContext::pipeline("card 4111-1111-1111-1111".into())).is_some());
}

#[test]
fn output_filter_collects_all_matches() {
    let mut e = entry("code", FilterOutcome::OutputFilter, &["\\d{4}"]);
    e.filter_action = Some(FilterAction::Redact);
    let f = RegexFilter::from_entry(&e).unwrap();
    let decision = f.evaluate(&FilterContext::pipeline("1234 and 5678".into())).expect("decision");
    match decision {
        FilterDecision::OutputFilter { action, matches, matched_pattern, .. } => {
            assert_eq!(action, FilterAction::Redact);
            assert_eq!(matched_pattern, "code");
            assert_eq!(matches.len(), 2);
            assert_eq!(matches[0].start, 0);
            assert_eq!(matches[0].end, 4);
        }
        other => panic!("expected output filter, got {other:?}"),
    }
}

#[test]
fn output_filter_none_when_no_matches() {
    let f = RegexFilter::from_entry(&entry("code", FilterOutcome::OutputFilter, &["\\d{4}"])).unwrap();
    assert!(f.evaluate(&FilterContext::pipeline("no digits".into())).is_none());
}

#[test]
fn output_filter_default_action_redacts() {
    let f = RegexFilter::from_entry(&entry("code", FilterOutcome::OutputFilter, &["x+"])).unwrap();
    let decision = f.evaluate(&FilterContext::pipeline("xxx".into())).expect("decision");
    assert_eq!(decision, FilterDecision::OutputFilter {
        action: FilterAction::Redact,
        matched_pattern: "code".into(),
        codewords: HashMap::new(),
        matches: vec![RegexMatch {
            pattern_name: "code".into(),
            matched_text: "xxx".into(),
            start: 0,
            end: 3,
            action: FilterAction::Redact,
        }],
    });
}

#[test]
fn soft_redirect_not_wired_for_regex() {
    let f = RegexFilter::from_entry(&entry("r", FilterOutcome::SoftRedirect, &["x"])).unwrap();
    assert!(f.evaluate(&FilterContext::pipeline("x".into())).is_none());
}
