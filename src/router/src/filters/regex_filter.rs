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
        // Scope check: if filter is frontier_bound only and this isn't
        // a frontier-bound request, skip.
        let scope_ok = self.scopes.iter().any(|s| match s {
            FilterScope::Any => true,
            FilterScope::FrontierBound => ctx.is_frontier_bound,
        });
        if !scope_ok {
            return None;
        }

        match self.outcome {
            FilterOutcome::HardReject => {
                // First matching regex → hard reject
                let matched = self.regexes.iter().find(|re| re.is_match(&ctx.user_message))?;
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
