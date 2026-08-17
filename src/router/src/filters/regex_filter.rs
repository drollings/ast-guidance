use std::collections::HashMap;

use regex::Regex;

use crate::config::{ConfidenceGate, FilterAction, FilterOutcome, FilterScope, PatternEntry};

use super::{Filter, FilterContext, FilterDecision, FilterKind, RegexMatch};

pub struct RegexFilter {
    pub name: String,
    pub regexes: Vec<Regex>,
    pub outcome: FilterOutcome,
    pub action: Option<FilterAction>,
    pub confidence_gate: ConfidenceGate,
    pub scopes: Vec<FilterScope>,
    pub error_message: String,
}

impl RegexFilter {
    pub fn from_entry(entry: &PatternEntry) -> Option<Self> {
        let regexes: Vec<Regex> = entry
            .regexes
            .iter()
            .filter_map(|r| Regex::new(r).ok())
            .collect();
        if regexes.is_empty() {
            return None;
        }
        Some(Self {
            name: entry.name.clone(),
            regexes,
            outcome: entry.outcome.clone(),
            action: entry.filter_action.clone(),
            confidence_gate: entry.confidence_gate.clone(),
            scopes: entry.scope.clone(),
            error_message: entry
                .error_message
                .clone()
                .unwrap_or_else(|| format!("blocked by '{}'", entry.name)),
        })
    }
}

impl Default for RegexFilter {
    fn default() -> Self {
        Self {
            name: String::new(),
            regexes: Vec::new(),
            outcome: FilterOutcome::HardReject,
            action: None,
            confidence_gate: ConfidenceGate::None,
            scopes: Vec::new(),
            error_message: String::new(),
        }
    }
}

impl Filter for RegexFilter {
    fn kind(&self) -> FilterKind {
        FilterKind::Regex
    }

    fn evaluate(&self, ctx: &FilterContext) -> Option<FilterDecision> {
        // Scope check: a filter applies only when its declared scopes
        // intersect the context's active scope set.
        let scope_ok = self.scopes.iter().any(|s| ctx.active_scopes.contains(s));
        if !scope_ok {
            return None;
        }

        match self.outcome {
            FilterOutcome::HardReject => {
                // First matching regex → hard reject
                let matched = self
                    .regexes
                    .iter()
                    .find(|re| re.is_match(&ctx.user_message))?;
                let m = matched.find(&ctx.user_message)?;

                if let ConfidenceGate::LuhnValid = self.confidence_gate {
                    if !super::luhn::luhn_valid(m.as_str()) {
                        return None;
                    }
                }

                Some(FilterDecision::HardReject {
                    pattern: self.name.clone(),
                    message: self.error_message.clone(),
                })
            }
            FilterOutcome::OutputFilter => {
                // Collect ALL regex matches for codeword substitution
                let mut matches = Vec::new();
                let action = self.action.clone().unwrap_or(FilterAction::Redact);

                for re in &self.regexes {
                    for m in re.find_iter(&ctx.user_message) {
                        if let ConfidenceGate::LuhnValid = self.confidence_gate {
                            if !super::luhn::luhn_valid(m.as_str()) {
                                continue;
                            }
                        }
                        matches.push(RegexMatch {
                            pattern_name: self.name.clone(),
                            matched_text: m.as_str().to_string(),
                            start: m.start(),
                            end: m.end(),
                            action: action.clone(),
                        });
                    }
                }

                if matches.is_empty() {
                    return None;
                }

                Some(FilterDecision::OutputFilter {
                    action,
                    matched_pattern: self.name.clone(),
                    codewords: HashMap::new(),
                    matches,
                })
            }
            FilterOutcome::SoftRedirect => {
                // SoftRedirect needs a target route — for now, only
                // hard_reject and output_filter are wired through
                // regex patterns. SoftRedirect is model-classification only.
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
}
