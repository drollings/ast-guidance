//! Structured telemetry events with controlled-vocabulary fields.
//!
//! No free strings, no PII. Every field is either a controlled enum
//! variant or a fixed-schema numeric value.

use serde::Serialize;

use crate::metrics::FailureClass;

// ── Controlled-vocabulary enums ────────────────────────────────────────

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

// ── TelemetryEvent ─────────────────────────────────────────────────────

/// A structured telemetry event.
///
/// Every variant carries only controlled-vocabulary or numeric fields.
/// No free strings, no PII.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum TelemetryEvent {
    /// A tool was invoked.
    ToolInvoked {
        tool: ToolName,
    },
    /// A tool completed, with duration and outcome.
    ToolCompleted {
        tool: ToolName,
        duration_ms: u64,
        success: bool,
    },
    /// A routing decision was made, with provider category only.
    Routing {
        category: ProviderCategory,
    },
    /// An error occurred, with failure class.
    Error {
        class: FailureClass,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    /// A feature was used.
    FeatureUsed {
        feature: FeatureName,
    },
}

// ── TelemetrySink ──────────────────────────────────────────────────────

/// A sink for structured telemetry events.
pub trait TelemetrySink: Send + Sync {
    /// Emit a telemetry event.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn event_to_json(event: &TelemetryEvent) -> serde_json::Value {
        serde_json::to_value(event).unwrap()
    }

    #[test]
    fn tool_invoked_is_pii_free() {
        let json = event_to_json(&TelemetryEvent::ToolInvoked {
            tool: ToolName::Search,
        });
        assert_eq!(json["type"].as_str(), Some("ToolInvoked"));
        assert_eq!(json["tool"].as_str(), Some("search"));
        assert!(json.get("user_input").is_none());
        assert!(json.get("message").is_none());
    }

    #[test]
    fn tool_completed_has_duration_and_outcome() {
        let json = event_to_json(&TelemetryEvent::ToolCompleted {
            tool: ToolName::LlmComplete,
            duration_ms: 150,
            success: true,
        });
        assert_eq!(json["tool"].as_str(), Some("llm_complete"));
        assert_eq!(json["duration_ms"].as_u64(), Some(150));
        assert_eq!(json["success"].as_bool(), Some(true));
    }

    #[test]
    fn routing_has_category_not_name() {
        let json = event_to_json(&TelemetryEvent::Routing {
            category: ProviderCategory::Local,
        });
        assert_eq!(json["category"].as_str(), Some("local"));
        assert!(json.get("provider_name").is_none());
    }

    #[test]
    fn error_event_has_class() {
        let json = event_to_json(&TelemetryEvent::Error {
            class: FailureClass::Timeout,
            message: None,
        });
        assert_eq!(json["class"].as_str(), Some("timeout"));
        assert!(json.get("user_data").is_none());
    }

    #[test]
    fn feature_used_is_controlled() {
        let json = event_to_json(&TelemetryEvent::FeatureUsed {
            feature: FeatureName::InjectionDetection,
        });
        assert_eq!(json["feature"].as_str(), Some("injection_detection"));
    }

    #[test]
    fn failure_class_round_trips_through_json() {
        for (class, label) in &[
            (FailureClass::Network, "network"),
            (FailureClass::Authentication, "authentication"),
            (FailureClass::RateLimit, "rate_limit"),
            (FailureClass::InputValidation, "input_validation"),
            (FailureClass::Storage, "storage"),
            (FailureClass::Timeout, "timeout"),
            (FailureClass::Internal, "internal"),
            (FailureClass::Unknown, "unknown"),
        ] {
            let json = event_to_json(&TelemetryEvent::Error {
                class: *class,
                message: None,
            });
            assert_eq!(json["class"].as_str(), Some(*label));
        }
    }

    #[test]
    fn tracing_sink_emits_without_panic() {
        let sink = TracingSink;
        let event = TelemetryEvent::ToolCompleted {
            tool: ToolName::Classify,
            duration_ms: 42,
            success: true,
        };
        sink.emit(&event); // should not panic
    }

    #[test]
    fn noop_sink_emits_without_panic() {
        let sink = NoopSink;
        let event = TelemetryEvent::FeatureUsed {
            feature: FeatureName::OutputSanitization,
        };
        sink.emit(&event); // should not panic
    }
}
