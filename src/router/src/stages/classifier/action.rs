//! ClassifierAction — typed dispatch for the classifier's LLM output.
//!
//! Confidence vs. task-value split: an LLM saying `{"action":"route","target":"local"}`
//! with low coherence is confident but wrong routing. Silently coercing unknown
//! action to Route makes confused classifier look decisive — unknown must be Error.

use std::str::FromStr;

use crate::config::ClassifierOutput;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifierAction {
    Respond(String),
    Route { target: Option<String> },
    Reject { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownAction(pub String);

impl std::fmt::Display for UnknownAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown classifier action: {}", self.0)
    }
}
impl std::error::Error for UnknownAction {}

impl FromStr for ClassifierAction {
    type Err = UnknownAction;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "respond" => Ok(Self::Respond(String::new())),
            "route" => Ok(Self::Route { target: None }),
            "reject" => Ok(Self::Reject { reason: String::new() }),
            other => Err(UnknownAction(other.to_string())),
        }
    }
}

impl ClassifierAction {
    /// Strict parse of ClassifierOutput.action into typed action.
    /// Unknown strings become Err, not silent Route fallback.
    pub fn from_output(output: &ClassifierOutput) -> Result<Self, UnknownAction> {
        match output.action.as_str() {
            "respond" => Ok(Self::Respond(output.response.clone().unwrap_or_default())),
            "route" => Ok(Self::Route {
                target: output.target.clone(),
            }),
            "reject" => Ok(Self::Reject {
                reason: output.reason.clone(),
            }),
            other => Err(UnknownAction(other.to_string())),
        }
    }

    pub fn is_respond(&self) -> bool {
        matches!(self, Self::Respond(_))
    }
    pub fn is_route(&self) -> bool {
        matches!(self, Self::Route { .. })
    }
    pub fn is_reject(&self) -> bool {
        matches!(self, Self::Reject { .. })
    }
}
#[cfg(test)]
#[path = "../../../tests/stages_classifier_action.rs"]
mod tests;
