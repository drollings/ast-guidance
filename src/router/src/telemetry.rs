//! Structured telemetry events — thin adaptor over `fluent_llm::telemetry`.
//!
//! Re-exports the LLM telemetry contract from `fluent_llm` (M11 completed
//! the M8 move out of `common_core`) so existing `crate::telemetry::*`
//! imports stay byte-identical. `FailureClass` (via
//! `fluent_llm::http_class`, itself the single `common_core` definition)
//! is the same type on both the metrics and telemetry paths — one enum
//! definition, zero converters.

pub use fluent_llm::http_class::FailureClass as TelemetryFailureClass;
pub use fluent_llm::telemetry::{
    FeatureName, NoopSink, ProviderCategory, TelemetryEvent, TelemetrySink, ToolName, TracingSink,
};

// For backward compat, keep `FailureClass` alias pointing to the shared type.
// Callers using `crate::telemetry::FailureClass` now get the shared type.
pub use fluent_llm::http_class::FailureClass;

#[cfg(test)]
#[path = "../tests/telemetry.rs"]
mod tests;
