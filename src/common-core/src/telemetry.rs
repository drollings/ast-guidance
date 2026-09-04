//! Failure taxonomy — the generic observability contract.
//!
//! NOTE (ROADMAP_20260903_LLM M11): the LLM/router vocabulary
//! (`ToolName`, `ProviderCategory`, `FeatureName`) and the structured event
//! contract (`TelemetryEvent`, `TelemetrySink`, `TracingSink`, `NoopSink`)
//! lived here through M10 — the vocabulary as deprecated shims of
//! `fluent_llm::telemetry`, the event contract as the staying generic
//! type. M11 completed the move: the event contract is typed over the
//! vocabulary, so it moved with it to `fluent_llm::telemetry` (the single
//! owner), and M11 deleted the common-core copies. What stays here is the
//! domain-free failure taxonomy below, which `fluent_llm::http_class`
//! re-exports as the single `FailureClass` definition (do not reintroduce
//! a duplicate).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    Network,
    Authentication,
    RateLimit,
    InputValidation,
    Storage,
    Timeout,
    Internal,
    Unknown,
}

impl FailureClass {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Network => "network",
            Self::Authentication => "authentication",
            Self::RateLimit => "rate_limit",
            Self::InputValidation => "input_validation",
            Self::Storage => "storage",
            Self::Timeout => "timeout",
            Self::Internal => "internal",
            Self::Unknown => "unknown",
        }
    }
}
