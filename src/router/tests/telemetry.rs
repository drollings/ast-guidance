use super::*;

fn event_to_json(event: &TelemetryEvent) -> serde_json::Value {
    serde_json::to_value(event).unwrap()
}

#[test]
fn tool_invoked_is_pii_free() {
    let json = event_to_json(&TelemetryEvent::ToolInvoked { tool: ToolName::Search });
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
    let json = event_to_json(&TelemetryEvent::Routing { category: ProviderCategory::Local });
    assert_eq!(json["category"].as_str(), Some("local"));
    assert!(json.get("provider_name").is_none());
}

#[test]
fn error_event_has_class() {
    let json = event_to_json(&TelemetryEvent::Error { class: FailureClass::Timeout, message: None });
    assert_eq!(json["class"].as_str(), Some("timeout"));
    assert!(json.get("user_data").is_none());
}

#[test]
fn feature_used_is_controlled() {
    let json = event_to_json(&TelemetryEvent::FeatureUsed { feature: FeatureName::InjectionDetection });
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
        let json = event_to_json(&TelemetryEvent::Error { class: *class, message: None });
        assert_eq!(json["class"].as_str(), Some(*label));
    }
}

#[test]
fn tracing_sink_emits_without_panic() {
    let sink = TracingSink;
    let event = TelemetryEvent::ToolCompleted { tool: ToolName::Classify, duration_ms: 42, success: true };
    sink.emit(&event);
}

#[test]
fn noop_sink_emits_without_panic() {
    let sink = NoopSink;
    let event = TelemetryEvent::FeatureUsed { feature: FeatureName::OutputSanitization };
    sink.emit(&event);
}

#[test]
fn metrics_and_telemetry_failure_class_are_one_type() {
    // M1: single enum definition — `crate::metrics::FailureClass` and
    // `crate::telemetry::FailureClass` are the same type, so no converter
    // exists or is needed. This locks the unification: assigning one to the
    // other must compile, and every variant must be identical on both paths.
    fn assert_same_type(c: crate::metrics::FailureClass) -> FailureClass {
        c
    }
    for c in &[
        crate::metrics::FailureClass::Network,
        crate::metrics::FailureClass::Authentication,
        crate::metrics::FailureClass::RateLimit,
        crate::metrics::FailureClass::InputValidation,
        crate::metrics::FailureClass::Storage,
        crate::metrics::FailureClass::Timeout,
        crate::metrics::FailureClass::Internal,
        crate::metrics::FailureClass::Unknown,
    ] {
        assert_eq!(*c, assert_same_type(*c));
    }
}
