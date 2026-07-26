pub mod injection_detect;
pub mod regex_filter;
pub mod luhn;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::FilterAction;

/// A single regex match with positional info for codeword substitution (M4.2).
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
    HardReject { pattern: String, message: String },
    SoftRedirect { route: String, reason: String },
    OutputFilter {
        action: FilterAction,
        matched_pattern: String,
        codewords: HashMap<String, String>,
        matches: Vec<RegexMatch>,
    },
}

/// Context passed to every filter at evaluation time.
pub struct FilterContext {
    pub user_message: String,
    pub is_frontier_bound: bool,
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
