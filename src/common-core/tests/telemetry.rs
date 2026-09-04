// NOTE (ROADMAP_20260903_LLM M11): the LLM/router vocabulary
// (`ToolName`/`ProviderCategory`/`FeatureName`) and the structured event
// contract (`TelemetryEvent`/sinks) moved to `fluent_llm::telemetry`
// (canonical owner; goldens in `fluent-llm --test telemetry`), and M11
// deleted the `common_core::telemetry` shims with the envelope tests that
// exercised them. This file keeps the staying `FailureClass` taxonomy
// suites — the single generic piece owned by `common-core`.

use common_core::telemetry::*;

#[test]
fn failure_class_labels_match() {
        assert_eq!(FailureClass::Network.label(), "network");
        assert_eq!(FailureClass::Authentication.label(), "authentication");
        assert_eq!(FailureClass::RateLimit.label(), "rate_limit");
        assert_eq!(FailureClass::InputValidation.label(), "input_validation");
        assert_eq!(FailureClass::Storage.label(), "storage");
        assert_eq!(FailureClass::Timeout.label(), "timeout");
        assert_eq!(FailureClass::Internal.label(), "internal");
        assert_eq!(FailureClass::Unknown.label(), "unknown");
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
            let json = serde_json::to_value(class).unwrap();
            assert_eq!(json, serde_json::json!(label));
            let parsed: FailureClass = serde_json::from_value(json).unwrap();
            assert_eq!(*class, parsed);
        }
}
