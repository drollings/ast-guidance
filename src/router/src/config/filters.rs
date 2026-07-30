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
