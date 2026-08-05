//! Escalation-ladder configuration (ROADMAP_20260805_REVIEW M3.1/M3.2).
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
mod tests {
    use super::*;

    #[test]
    fn array_form_deserializes_and_lists_models() {
        let group: ModelGroup = serde_json::from_str(r#"["fast", "small"]"#).unwrap();
        assert_eq!(group.models(), &["fast".to_string(), "small".to_string()]);
        assert!(group.escalation().is_none());
    }

    #[test]
    fn object_form_models_only() {
        let group: ModelGroup = serde_json::from_str(r#"{"models": ["code-model"]}"#).unwrap();
        assert_eq!(group.models(), &["code-model".to_string()]);
        assert!(group.escalation().is_none());
    }

    #[test]
    fn object_form_with_escalation() {
        let group: ModelGroup = serde_json::from_str(
            r#"{
                "models": ["code-model"],
                "escalation": {
                    "modes": ["filter", "question", "team", "turnover"],
                    "frontier": {"endpoint": "https://frontier.example/v1/chat/completions", "model": "claude-sonnet", "api_key_env": "ANTHROPIC_KEY"},
                    "decomposer_model": "fast",
                    "assembler_model": "fast",
                    "classifier_model": "small",
                    "classifier_parallel": 5,
                    "draft_model": "small",
                    "judge_model": "fast"
                }
            }"#,
        )
        .unwrap();
        assert_eq!(group.models(), &["code-model".to_string()]);
        let ladder = group.escalation().expect("escalation present");
        assert_eq!(
            ladder.modes,
            vec![
                EscalationMode::Filter,
                EscalationMode::Question,
                EscalationMode::Team,
                EscalationMode::Turnover
            ]
        );
        let front = ladder.frontier.as_ref().unwrap();
        assert_eq!(front.endpoint, "https://frontier.example/v1/chat/completions");
        assert_eq!(front.model, "claude-sonnet");
        assert_eq!(front.api_key_env.as_deref(), Some("ANTHROPIC_KEY"));
        assert_eq!(ladder.decomposer_model.as_deref(), Some("fast"));
        assert_eq!(ladder.classifier_parallel, 5);
        assert_eq!(ladder.draft_model.as_deref(), Some("small"));
        assert_eq!(ladder.judge_model.as_deref(), Some("fast"));
    }

    #[test]
    fn array_form_round_trips() {
        let group: ModelGroup = serde_json::from_str(r#"["fast", "small"]"#).unwrap();
        let serialized = serde_json::to_string(&group).unwrap();
        let back: ModelGroup = serde_json::from_str(&serialized).unwrap();
        assert_eq!(back.models(), group.models());
    }

    #[test]
    fn ladder_defaults() {
        let ladder = EscalationLadderConfig::default();
        assert!(ladder.modes.is_empty());
        assert!(ladder.frontier.is_none());
        assert_eq!(ladder.classifier_parallel, 3);
    }

    #[test]
    fn ladder_missing_fields_default() {
        let ladder: EscalationLadderConfig =
            serde_json::from_str(r#"{"frontier": {"endpoint": "u", "model": "m"}}"#).unwrap();
        assert!(ladder.modes.is_empty());
        assert_eq!(ladder.classifier_parallel, 3, "unset parallel defaults");
        assert!(ladder.decomposer_model.is_none());
    }

    #[test]
    fn escalation_mode_list_deserializes() {
        let modes: Vec<EscalationMode> = serde_json::from_str(r#"["filter","team"]"#).unwrap();
        assert_eq!(modes, vec![EscalationMode::Filter, EscalationMode::Team]);
    }

    #[test]
    fn empty_models_object_deserializes() {
        let group: ModelGroup = serde_json::from_str("{}").unwrap();
        assert!(group.models().is_empty());
        assert!(group.escalation().is_none());
    }

    #[test]
    fn router_config_shipped_array_shape_still_parses() {
        // The shipped `env/coral-router.json` shape: array-form model_groups.
        let cfg: crate::config::RouterConfig = serde_json::from_str(
            r#"{
                "model_groups": {"fast": ["fast"], "code": ["code-model"]},
                "models": {}
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.model_groups.len(), 2);
        assert_eq!(cfg.model_groups["fast"].models(), &["fast".to_string()]);
        assert!(cfg.model_groups["fast"].escalation().is_none());
    }
}
