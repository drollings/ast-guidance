//! Escalation-ladder configuration
//!
//! `model_groups[group]` accepts either the shipped array form
//! (`["fast", "small"]`) or a new object form
//! (`{"models": [...], "escalation": {...}}`) that attaches a
//! [`EscalationLadderConfig`] to the group. The ladder types are plain data;
//! the runtime that consumes them lives in `crate::dispatch::escalation`.

use serde::{Deserialize, Serialize};

use crate::frontier::modes::EscalationMode;

/// A `model_groups[group]` value. The array form (a list of model keys) keeps
/// existing configs deserializing byte-identically; the object form attaches
/// an optional escalation ladder to the group.
///
/// The size difference between variants is accepted: both are constructed
/// once at config load, never in a hot loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum ModelGroup {
    /// The shipped shape: a plain list of model keys.
    Array(Vec<String>),
    /// The escalated shape: an explicit `models` list plus an optional
    /// `escalation` ladder.
    Object {
        #[serde(default)]
        models: Vec<String>,
        #[serde(default)]
        escalation: Option<EscalationLadderConfig>,
    },
}

impl ModelGroup {
    /// The member model keys, regardless of which form was configured.
    pub fn models(&self) -> &[String] {
        match self {
            ModelGroup::Array(models) | ModelGroup::Object { models, .. } => models,
        }
    }

    /// The escalation ladder configured for this group, if any. Array-form
    /// groups never have one.
    pub fn escalation(&self) -> Option<&EscalationLadderConfig> {
        match self {
            ModelGroup::Array(_) => None,
            ModelGroup::Object { escalation, .. } => escalation.as_ref(),
        }
    }
}

/// The frontier endpoint a ladder dispatches to, plus the model name it asks
/// that endpoint to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierConfig {
    /// OpenAI-compatible chat-completions base URL. Reuses the canonical
    /// `dispatch/backend.rs` transport — no third HTTP path.
    pub endpoint: String,
    /// Name of the environment variable holding the API key. Optional for
    /// local OpenAI-compatible endpoints; when present the key is injected as
    /// a `Bearer` token on the frontier request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// The model name sent to the frontier endpoint.
    pub model: String,
}

/// The full escalation ladder for one model group
/// (`model_groups[group].escalation`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLadderConfig {
    /// Ordered escalation modes; tried filter → question → team → turnover.
    /// Empty/absent → no frontier escalation for this group.
    #[serde(default)]
    pub modes: Vec<EscalationMode>,
    #[serde(default)]
    pub frontier: Option<FrontierConfig>,
    /// Local model keys (referencing `models`) for the per-mode roles.
    /// Missing roles disable the modes that require them.
    #[serde(default)]
    pub decomposer_model: Option<String>,
    #[serde(default)]
    pub assembler_model: Option<String>,
    #[serde(default)]
    pub classifier_model: Option<String>,
    /// Number of parallel classifier slots in team mode.
    #[serde(default = "default_classifier_parallel")]
    pub classifier_parallel: usize,
    #[serde(default)]
    pub draft_model: Option<String>,
    #[serde(default)]
    pub judge_model: Option<String>,
}

impl Default for EscalationLadderConfig {
    fn default() -> Self {
        Self {
            modes: Vec::new(),
            frontier: None,
            decomposer_model: None,
            assembler_model: None,
            classifier_model: None,
            classifier_parallel: default_classifier_parallel(),
            draft_model: None,
            judge_model: None,
        }
    }
}

const fn default_classifier_parallel() -> usize {
    3
}
#[cfg(test)]
#[path = "../../tests/config_escalation.rs"]
mod tests;
