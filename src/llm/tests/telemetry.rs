//! ROADMAP_20260903_LLM M8.2 — LLM telemetry-vocabulary goldens (moved, not copied).
//!
//! Canonical home for the `ToolName` / `ProviderCategory` / `FeatureName`
//! label snapshots (moved from `src/common-core/tests/telemetry.rs`); M11
//! also moved the structured event contract (`TelemetryEvent` / sinks) here
//! and deleted the `common_core` copies with the dual-path parity test.
//!
//! What stays behind in `common_core::telemetry`: `FailureClass` + `label()`
//! (the generic taxonomy); the `classify_http_status` mapping tests stay in
//! `tests/http_class.rs` and there is exactly one `FailureClass` definition
//! (locked below).
//!
//! Calibration (roadmap §1, M10): these are task-value routing/usage labels,
//! not producer confidence — a `llm_complete` label on an event never endorses
//! the output's correctness, and the labels move unchanged here.

use fluent_llm::telemetry::{FeatureName, ProviderCategory, ToolName};

// ── Label snapshots (moved goldens; snake_case must stay stable) ────────────

#[test]
fn tool_name_labels_match_spec() {
    let cases = [
        (ToolName::Search, "search"),
        (ToolName::Embed, "embed"),
        (ToolName::LlmComplete, "llm_complete"),
        (ToolName::LlmStream, "llm_stream"),
        (ToolName::Classify, "classify"),
        (ToolName::Summarize, "summarize"),
        (ToolName::Score, "score"),
        (ToolName::Filter, "filter"),
        (ToolName::Transform, "transform"),
    ];
    for (tool, label) in cases {
        let json = serde_json::to_value(tool).unwrap();
        assert_eq!(json, serde_json::json!(label), "{tool:?}");
    }
}

#[test]
fn provider_category_labels_match_spec() {
    let cases = [
        (ProviderCategory::FirstPartyCloud, "first_party_cloud"),
        (ProviderCategory::ThirdPartyCloud, "third_party_cloud"),
        (ProviderCategory::Local, "local"),
        (ProviderCategory::Proxy, "proxy"),
        (ProviderCategory::Unknown, "unknown"),
    ];
    for (category, label) in cases {
        let json = serde_json::to_value(category).unwrap();
        assert_eq!(json, serde_json::json!(label), "{category:?}");
    }
}

#[test]
fn feature_name_labels_match_spec() {
    let cases = [
        (FeatureName::OutputSanitization, "output_sanitization"),
        (FeatureName::InjectionDetection, "injection_detection"),
        (FeatureName::SearchFusion, "search_fusion"),
        (FeatureName::SecretMasking, "secret_masking"),
        (FeatureName::PiiAnonymize, "pii_anonymize"),
        (FeatureName::PlanRoute, "plan_route"),
        (FeatureName::RigorRoute, "rigor_route"),
        (FeatureName::KvCacheRestore, "kv_cache_restore"),
        (FeatureName::FrontierFallback, "frontier_fallback"),
        (FeatureName::DiscoveryPoll, "discovery_poll"),
    ];
    for (feature, label) in cases {
        let json = serde_json::to_value(feature).unwrap();
        assert_eq!(json, serde_json::json!(label), "{feature:?}");
    }
}

#[test]
fn vocab_labels_are_distinct_per_variant() {
    // The owner enums are `Serialize`-only (verbatim from `common-core` —
    // no `Deserialize` impl exists on either copy), so injectivity is locked
    // via serialization: every variant must produce a distinct label, or two
    // tools/features would collapse in telemetry.
    let tools = [
        ToolName::Search,
        ToolName::Embed,
        ToolName::LlmComplete,
        ToolName::LlmStream,
        ToolName::Classify,
        ToolName::Summarize,
        ToolName::Score,
        ToolName::Filter,
        ToolName::Transform,
    ];
    let tool_labels: Vec<String> = tools
        .iter()
        .map(|t| serde_json::to_string(t).unwrap())
        .collect();
    let mut sorted = tool_labels.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), tools.len());
    let categories = [
        ProviderCategory::FirstPartyCloud,
        ProviderCategory::ThirdPartyCloud,
        ProviderCategory::Local,
        ProviderCategory::Proxy,
        ProviderCategory::Unknown,
    ];
    let cat_labels: Vec<String> = categories
        .iter()
        .map(|c| serde_json::to_string(c).unwrap())
        .collect();
    let mut sorted = cat_labels.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), categories.len());
    let features = [
        FeatureName::OutputSanitization,
        FeatureName::InjectionDetection,
        FeatureName::SearchFusion,
        FeatureName::SecretMasking,
        FeatureName::PiiAnonymize,
        FeatureName::PlanRoute,
        FeatureName::RigorRoute,
        FeatureName::KvCacheRestore,
        FeatureName::FrontierFallback,
        FeatureName::DiscoveryPoll,
    ];
    let feat_labels: Vec<String> = features
        .iter()
        .map(|f| serde_json::to_string(f).unwrap())
        .collect();
    let mut sorted = feat_labels.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), features.len());
}

// ── Single FailureClass definition (stays in common-core) ───────────────────

#[test]
fn failure_class_has_single_definition() {
    // `fluent_llm::http_class::FailureClass` must BE the
    // `common_core::telemetry::FailureClass` type (one enum definition, zero
    // converters). This locks the unification: assigning one to the other
    // must compile.
    fn assert_same_type(
        c: common_core::telemetry::FailureClass,
    ) -> fluent_llm::http_class::FailureClass {
        c
    }
    for c in [
        fluent_llm::http_class::FailureClass::Network,
        fluent_llm::http_class::FailureClass::Authentication,
        fluent_llm::http_class::FailureClass::RateLimit,
        fluent_llm::http_class::FailureClass::InputValidation,
        fluent_llm::http_class::FailureClass::Storage,
        fluent_llm::http_class::FailureClass::Timeout,
        fluent_llm::http_class::FailureClass::Internal,
        fluent_llm::http_class::FailureClass::Unknown,
    ] {
        assert_eq!(c, assert_same_type(c));
    }
}

// NOTE (ROADMAP_20260903_LLM M11): the `parity_new_eq_old` dual-path test
// died with the `common_core::telemetry` vocabulary shims it pinned. The
// label snapshots + single-`FailureClass` lock above are the lasting
// contract.
