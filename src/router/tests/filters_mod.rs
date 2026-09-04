use super::*;
use crate::config::{ConfidenceGate, FilterOutcome, PatternEntry};

fn hard_reject_filter(name: &str, pattern: &str) -> Box<dyn Filter> {
    let e = PatternEntry {
        name: name.into(),
        outcome: FilterOutcome::HardReject,
        filter_action: None,
        confidence_gate: ConfidenceGate::None,
        scope: vec![FilterScope::Any],
        http_code: 403,
        error_message: None,
        regexes: vec![pattern.into()],
    };
    Box::new(regex_filter::RegexFilter::from_entry(&e).unwrap())
}

#[test]
fn empty_engine_returns_none() {
    let engine = DeterministicFilterEngine::new();
    assert!(engine.is_empty());
    assert!(engine.evaluate(&FilterContext::pipeline("anything".into())).is_none());
}

#[test]
fn engine_runs_filters_in_insertion_order_and_short_circuits() {
    let mut engine = DeterministicFilterEngine::new();
    engine.add_filter(hard_reject_filter("first", "first-secret"));
    engine.add_filter(hard_reject_filter("second", "second-secret"));
    // First filter matches -> short-circuits, first decision wins.
    let d = engine
        .evaluate(&FilterContext::pipeline("first-secret second-secret".into()))
        .expect("decision");
    match d {
        FilterDecision::HardReject { pattern, .. } => assert_eq!(pattern, "first"),
        other => panic!("unexpected {other:?}"),
    }
    // Only the second matches -> its decision surfaces.
    let d = engine
        .evaluate(&FilterContext::pipeline("second-secret".into()))
        .expect("decision");
    match d {
        FilterDecision::HardReject { pattern, .. } => assert_eq!(pattern, "second"),
        other => panic!("unexpected {other:?}"),
    }
    // Neither matches -> None.
    assert!(engine.evaluate(&FilterContext::pipeline("clean".into())).is_none());
}

#[test]
fn engine_scope_applies_to_each_filter() {
    let mut engine = DeterministicFilterEngine::new();
    let mut e = PatternEntry {
        name: "frontier-only".into(),
        outcome: FilterOutcome::HardReject,
        filter_action: None,
        confidence_gate: ConfidenceGate::None,
        scope: vec![FilterScope::FrontierBound],
        http_code: 403,
        error_message: None,
        regexes: vec!["secret".into()],
    };
    engine.add_filter(Box::new(regex_filter::RegexFilter::from_entry(&e).unwrap()));
    assert!(engine.evaluate(&FilterContext::pipeline("secret".into())).is_none());
    assert!(engine.evaluate(&FilterContext::frontier("secret".into())).is_some());
    // Ledger-write scope adds ContentNodeWrite but not FrontierBound.
    e.scope = vec![FilterScope::ContentNodeWrite];
    let mut engine2 = DeterministicFilterEngine::new();
    engine2.add_filter(Box::new(regex_filter::RegexFilter::from_entry(&e).unwrap()));
    assert!(engine2.evaluate(&FilterContext::ledger_write("secret".into())).is_some());
    assert!(engine2.evaluate(&FilterContext::frontier("secret".into())).is_none());
}

#[test]
fn filter_context_scope_helpers() {
    assert_eq!(FilterContext::pipeline("m".into()).active_scopes, &[FilterScope::Any]);
    assert_eq!(
        FilterContext::frontier("m".into()).active_scopes,
        &[FilterScope::Any, FilterScope::FrontierBound]
    );
    assert_eq!(
        FilterContext::ledger_write("m".into()).active_scopes,
        &[FilterScope::Any, FilterScope::ContentNodeWrite]
    );
}

#[test]
fn regex_match_serde_round_trip() {
    let m = RegexMatch {
        pattern_name: "p".into(),
        matched_text: "x".into(),
        start: 1,
        end: 2,
        action: FilterAction::Redact,
    };
    let back: RegexMatch =
        serde_json::from_str(&serde_json::to_string(&m).expect("serialize")).expect("round trip");
    assert_eq!(back, m);
}
