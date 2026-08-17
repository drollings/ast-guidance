pub mod injection_detect;
pub mod luhn;
pub mod regex_filter;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::{FilterAction, FilterScope};

/// A single regex match with positional info for codeword substitution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegexMatch {
    pub pattern_name: String,
    pub matched_text: String,
    pub start: usize,
    pub end: usize,
    pub action: FilterAction,
}

/// Filter kinds per MOA_ROUTER_SPEC §2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    Regex,
    Whitelist,
    HnswSimilarity,
    ModelClassification,
}

/// Filter outcomes per MOA_ROUTER_SPEC §2 table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterDecision {
    HardReject {
        pattern: String,
        message: String,
    },
    SoftRedirect {
        route: String,
        reason: String,
    },
    OutputFilter {
        action: FilterAction,
        matched_pattern: String,
        codewords: HashMap<String, String>,
        matches: Vec<RegexMatch>,
    },
}

/// Context passed to every filter at evaluation time.
///
/// `active_scopes` carries the set of scopes active for the current
/// evaluation; a filter applies when its declared `scopes` intersect the
/// active set. The constructor helpers keep the call sites readable.
pub struct FilterContext {
    pub user_message: String,
    pub active_scopes: &'static [FilterScope],
}

impl FilterContext {
    /// The default pipeline scope: every `[Any]` filter applies.
    pub fn pipeline(user_message: String) -> Self {
        Self {
            user_message,
            active_scopes: &[FilterScope::Any],
        }
    }

    /// The escalation-ladder output re-scan: adds the `FrontierBound` scope.
    pub fn frontier(user_message: String) -> Self {
        Self {
            user_message,
            active_scopes: &[FilterScope::Any, FilterScope::FrontierBound],
        }
    }

    /// The ledger write path: adds the `ContentNodeWrite` scope so the builtin
    /// PII engine always scrubs durable content.
    pub fn ledger_write(user_message: String) -> Self {
        Self {
            user_message,
            active_scopes: &[FilterScope::Any, FilterScope::ContentNodeWrite],
        }
    }
}

pub trait Filter: Send + Sync {
    fn kind(&self) -> FilterKind;
    fn evaluate(&self, ctx: &FilterContext) -> Option<FilterDecision>;
}

/// Owns all filter instances and runs them in insertion order.
/// Returns the first non-None decision (filters are short-circuit).
pub struct DeterministicFilterEngine {
    filters: Vec<Box<dyn Filter>>,
}

impl Default for DeterministicFilterEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DeterministicFilterEngine {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    pub fn add_filter(&mut self, filter: Box<dyn Filter>) {
        self.filters.push(filter);
    }

    pub fn evaluate(&self, ctx: &FilterContext) -> Option<FilterDecision> {
        for filter in &self.filters {
            if let Some(decision) = filter.evaluate(ctx) {
                return Some(decision);
            }
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

#[cfg(test)]
mod tests {
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
}
