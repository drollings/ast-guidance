//! Filter types and reject patterns for the deterministic pre-filter stage.
//! Defines the filter outcome taxonomy (hard_reject / soft_redirect / output_filter),
//! actions (redact / anonymize / omit), confidence gates, and configurable pattern entries.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use common_core::constants::default_true;

// ── Filter outcome taxonomy ─────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterOutcome {
    #[default]
    HardReject,
    SoftRedirect,
    OutputFilter,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterAction {
    Redact,
    Anonymize,
    Omit,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceGate {
    #[serde(rename = "luhn_valid")]
    LuhnValid,
    #[default]
    #[serde(rename = "none")]
    None,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FilterScope {
    #[default]
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "frontier_bound")]
    FrontierBound,
    #[serde(rename = "content_node_write")]
    ContentNodeWrite,
}

// ── Pattern entries ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEntry {
    pub name: String,
    #[serde(default)]
    pub outcome: FilterOutcome,
    #[serde(default)]
    pub filter_action: Option<FilterAction>,
    #[serde(default)]
    pub confidence_gate: ConfidenceGate,
    #[serde(default)]
    pub scope: Vec<FilterScope>,
    pub http_code: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default)]
    pub regexes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RejectPatterns {
    #[serde(default)]
    pub patterns: Vec<PatternEntry>,
    #[serde(default)]
    pub commands: Option<CommandConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandConfig {
    pub pattern: String,
    #[serde(default)]
    pub handlers: HashMap<String, String>,
}

// ── Mock configuration ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockConfig {
    pub transcript_path: String,
    #[serde(default = "default_true")]
    pub fail_on_unexpected: bool,
    #[serde(default = "default_mock_base_url")]
    pub base_url: String,
}

fn default_mock_base_url() -> String {
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_outcome_serde_round_trip() {
        for (variant, name) in [
            (FilterOutcome::HardReject, "hard_reject"),
            (FilterOutcome::SoftRedirect, "soft_redirect"),
            (FilterOutcome::OutputFilter, "output_filter"),
        ] {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(json, format!("\"{name}\""));
            let back: FilterOutcome = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn filter_action_serde_round_trip() {
        for variant in [FilterAction::Redact, FilterAction::Anonymize, FilterAction::Omit] {
            let json = serde_json::to_string(&variant).expect("serialize");
            let back: FilterAction = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn confidence_gate_serde_round_trip() {
        assert_eq!(
            serde_json::from_str::<ConfidenceGate>("\"luhn_valid\"").unwrap(),
            ConfidenceGate::LuhnValid
        );
        assert_eq!(
            serde_json::from_str::<ConfidenceGate>("\"none\"").unwrap(),
            ConfidenceGate::None
        );
        assert_eq!(ConfidenceGate::default(), ConfidenceGate::None);
    }

    #[test]
    fn filter_scope_serde_round_trip() {
        for (variant, name) in [
            (FilterScope::Any, "any"),
            (FilterScope::FrontierBound, "frontier_bound"),
            (FilterScope::ContentNodeWrite, "content_node_write"),
        ] {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(json, format!("\"{name}\""));
            let back: FilterScope = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, variant);
        }
        assert_eq!(FilterScope::default(), FilterScope::Any);
    }

    #[test]
    fn pattern_entry_serde_defaults() {
        let e: PatternEntry = serde_json::from_value(serde_json::json!({
            "name": "block",
            "http_code": 403,
            "regexes": ["secret"],
        }))
        .expect("deserialize");
        assert_eq!(e.outcome, FilterOutcome::HardReject);
        assert_eq!(e.filter_action, None);
        assert_eq!(e.confidence_gate, ConfidenceGate::None);
        assert!(e.scope.is_empty());
        assert_eq!(e.error_message, None);
    }

    #[test]
    fn pattern_entry_serde_round_trip() {
        let e: PatternEntry = serde_json::from_value(serde_json::json!({
            "name": "pii",
            "outcome": "output_filter",
            "filter_action": "anonymize",
            "confidence_gate": "luhn_valid",
            "scope": ["any", "frontier_bound"],
            "http_code": 200,
            "error_message": "scrubbed",
            "regexes": ["\\d{4}"],
        }))
        .expect("deserialize");
        assert_eq!(e.outcome, FilterOutcome::OutputFilter);
        assert_eq!(e.filter_action, Some(FilterAction::Anonymize));
        assert_eq!(e.confidence_gate, ConfidenceGate::LuhnValid);
        assert_eq!(e.scope, vec![FilterScope::Any, FilterScope::FrontierBound]);
        let back: PatternEntry =
            serde_json::from_str(&serde_json::to_string(&e).expect("serialize")).expect("round trip");
        assert_eq!(back.name, "pii");
        assert_eq!(back.regexes, e.regexes);
    }

    #[test]
    fn reject_patterns_serde_round_trip() {
        let r: RejectPatterns = serde_json::from_value(serde_json::json!({
            "patterns": [{"name": "block", "http_code": 403, "regexes": ["x"]}],
            "commands": {"pattern": "\\w+", "handlers": {"help": "h"}},
        }))
        .expect("deserialize");
        assert_eq!(r.patterns.len(), 1);
        let commands = r.commands.as_ref().expect("commands");
        assert_eq!(commands.handlers["help"], "h");
        // commands/patterns optional -> defaults.
        let empty: RejectPatterns = serde_json::from_str("{}").expect("empty");
        assert!(empty.patterns.is_empty());
        assert!(empty.commands.is_none());
    }

    #[test]
    fn mock_config_serde_defaults() {
        let m: MockConfig = serde_json::from_value(serde_json::json!({
            "transcript_path": "/tmp/t.json"
        }))
        .expect("deserialize");
        assert!(m.fail_on_unexpected);
        assert_eq!(m.base_url, "");
    }
}
