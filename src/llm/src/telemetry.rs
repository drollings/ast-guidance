//! LLM telemetry vocabulary + structured events — the single owner
//! (ROADMAP_20260903_LLM M8, completed by M11).
//!
//! M8 moved `ToolName` (the router-relevant tool names, including the LLM
//! spells `Embed`, `LlmComplete`, `LlmStream`, `Classify`, `Summarize`),
//! `ProviderCategory` (routing-decision categories — never a provider name),
//! and `FeatureName` (usage-tracking features) verbatim out of
//! `common_core::telemetry`. M11 moved the structured event contract with
//! them — `TelemetryEvent`, `TelemetrySink`, `TracingSink`, `NoopSink` —
//! because the event type is typed over the vocabulary and could not stay
//! behind without either a second vocabulary definition or a
//! `common-core → fluent-llm` dependency cycle (invariant 8: `fluent-llm →
//! `common-core` already exists). M11 then deleted the `common-core`
//! copies. Dependencies are `serde`/`serde_json`/`tracing` only.
//!
//! What stays in `common_core::telemetry`: the generic failure taxonomy —
//! `FailureClass` + `label()` — which `llm::http_class` composes via its
//! single-definition re-export (do not reintroduce a duplicate). This
//! module owns the LLM/router vocabulary: the one `snake_case` label set
//! in the workspace, plus the PII-free event envelope over it.
//!
//! Calibration (roadmap §1, M10): these are task-value routing/usage labels,
//! not producer confidence — a `llm_complete` event label never endorses the
//! output's correctness. The labels moved unchanged; trusting them is M10.

use serde::Serialize;

/// Router-relevant tool names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolName {
    Search,
    Embed,
    LlmComplete,
    LlmStream,
    Classify,
    Summarize,
    Score,
    Filter,
    Transform,
}

/// Provider categories for routing decisions — never the provider name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderCategory {
    FirstPartyCloud,
    ThirdPartyCloud,
    Local,
    Proxy,
    Unknown,
}

/// Feature names for usage tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureName {
    OutputSanitization,
    InjectionDetection,
    SearchFusion,
    SecretMasking,
    PiiAnonymize,
    PlanRoute,
    RigorRoute,
    KvCacheRestore,
    FrontierFallback,
    DiscoveryPoll,
}

// ─── Structured events (moved verbatim from `common_core::telemetry`) ───────
// M11: the event contract follows the vocabulary — it is typed over the
// three enums above, so deleting the common-core vocabulary copies would
// have stranded it. `FailureClass` (the one generic piece) stays in
// `common-core`; the variants below reference it through the single
// `http_class` re-export, never a duplicate definition.

/// A structured telemetry event. Every variant carries only controlled-vocabulary or numeric fields. No free strings, no PII.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum TelemetryEvent {
    ToolInvoked { tool: ToolName },
    ToolCompleted { tool: ToolName, duration_ms: u64, success: bool },
    Routing { category: ProviderCategory },
    Error {
        class: crate::http_class::FailureClass,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    FeatureUsed { feature: FeatureName },
}

/// A sink for structured telemetry events.
pub trait TelemetrySink: Send + Sync {
    fn emit(&self, event: &TelemetryEvent);
}

/// Sink that writes events to the tracing system as structured info logs.
pub struct TracingSink;

impl TelemetrySink for TracingSink {
    fn emit(&self, event: &TelemetryEvent) {
        tracing::info!(
            target: "router.telemetry",
            event = %serde_json::to_string(event).unwrap_or_else(|_| "serialization_error".into()),
            "telemetry event"
        );
    }
}

/// Sink that discards all events (for tests and when telemetry is disabled).
pub struct NoopSink;

impl TelemetrySink for NoopSink {
    fn emit(&self, _event: &TelemetryEvent) {}
}
