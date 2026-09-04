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
#[path = "../../tests/filters_mod.rs"]
mod tests;
