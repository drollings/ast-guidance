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

/// One member of a `model_groups` list: either a literal model key or an
/// availability-ordering sentinel. `Last` expands to the group's
/// most-recently-successful key (when still a member); `Any` orders the
/// currently-loaded members first, in config order. Both only *order*
/// candidates — the intelligence climb underneath is unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupMember {
    /// A literal model key (possibly qualified, e.g. `code:default`).
    Key(String),
    /// The group's most-recently-successful member.
    Last,
    /// Any loaded member first, then the rest in config order.
    Any,
}

impl GroupMember {
    /// Parse one raw group member. Only the bare, lowercase `last`/`any`
    /// spellings are sentinels (case-sensitive); every other spelling —
    /// including qualified forms like `base:last` — stays a literal key and
    /// keeps today's fail-closed lookup when unknown.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "last" => GroupMember::Last,
            "any" => GroupMember::Any,
            _ => GroupMember::Key(raw.to_string()),
        }
    }

    /// The raw member spelling this parsed from (identity round trip).
    pub fn raw(&self) -> &str {
        match self {
            GroupMember::Key(key) => key,
            GroupMember::Last => "last",
            GroupMember::Any => "any",
        }
    }
}

impl ModelGroup {
    /// The member model keys, regardless of which form was configured.
    pub fn models(&self) -> &[String] {
        match self {
            ModelGroup::Array(models) | ModelGroup::Object { models, .. } => models,
        }
    }

    /// The members parsed into literal keys vs. availability sentinels, in
    /// config order. Pure parsing — no registry or residency reads — so the
    /// raw `models()` shape stays the dispatch-neutral source of truth.
    pub fn members(&self) -> Vec<GroupMember> {
        self.models().iter().map(|m| GroupMember::parse(m)).collect()
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
