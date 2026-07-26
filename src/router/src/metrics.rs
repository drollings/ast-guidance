//! Metrics and monitoring for the router pipeline.
//!
//! Labels are tagged by model, agent, role, and adapter — never by
//! session ID or request ID (cardinality). Uses
//! `common_core::metrics::LatencyHistogram` for latency tracking.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use common_core::metrics::LatencyHistogram;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use common_core::watchdog::{WatchdogEvent, WatchdogEventType};

use crate::telemetry::TelemetrySink;

// ── FailureClass ───────────────────────────────────────────────────────

/// High-level failure classes for error classification.
///
/// Every variant serializes to a stable string label for backward
/// compatibility with existing metrics consumers.
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
    /// Stable string label for the failure class.
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

/// Classify an error message string into a `FailureClass`.
///
/// Uses regex patterns ordered by specificity, with a fast prefix
/// check for compiler diagnostic codes as an optimization.
pub fn classify_error(message: &str) -> FailureClass {
    if message.is_empty() {
        return FailureClass::Unknown;
    }

    // Fast prefix check for compiler diagnostic codes (e.g. [E0425])
    if message.starts_with('[') {
        if let Some(end) = message.find(']') {
            let code = &message[1..end];
            if code.len() >= 3 && code.chars().all(char::is_alphanumeric) {
                return FailureClass::Internal;
            }
        }
    }

    static PATTERNS: LazyLock<Vec<(Regex, FailureClass)>> = LazyLock::new(|| {
        vec![
            (Regex::new(r"(?i)\b(?:timeout|timed?\s*out|deadline exceeded|deadline_exceeded)\b").unwrap(), FailureClass::Timeout),
            (Regex::new(r"(?i)\b(?:rate.?limit|too many requests|429|503|throttle)\b").unwrap(), FailureClass::RateLimit),
            (Regex::new(r"(?i)\b(?:auth|unauthorized|forbidden|401|403|invalid.*(?:key|token|credential)|access denied)\b").unwrap(), FailureClass::Authentication),
            (Regex::new(r"(?i)\b(?:network|connection.*(?:reset|refused|refuse|timeout|closed)|dns.*(?:fail|error)|tcp|socket)\b").unwrap(), FailureClass::Network),
            (Regex::new(r"(?i)\b(?:io error|file.*not.*found|no such file|disk.*full|read.?only|permission.*denied)\b").unwrap(), FailureClass::Storage),
            (Regex::new(r"(?i)\b(?:parse.*error|invalid.*(?:input|format|argument|parameter|request|json)|malformed|bad request|422|400)\b").unwrap(), FailureClass::InputValidation),
            (Regex::new(r"(?i)\b(?:syntax error|type error|compiler? error|compilation failed|build.*(?:fail|error)|panic|internal error)\b").unwrap(), FailureClass::Internal),
            (Regex::new(r"(?i)\b(?:test.*(?:fail|error)|assert.*failed|assertion.*failed)\b").unwrap(), FailureClass::Internal),
            (Regex::new(r"(?i)\b(?:not found|404)\b").unwrap(), FailureClass::InputValidation),
            (Regex::new(r"(?i)\b(?:runtime error|segfault|segmentation fault|abort|fatal)\b").unwrap(), FailureClass::Internal),
        ]
    });

    for (re, class) in PATTERNS.iter() {
        if re.is_match(message) {
            return *class;
        }
    }

    FailureClass::Unknown
}

// ── RouterMetrics ──────────────────────────────────────────────────────

/// Aggregated router metrics.
///
/// All counters and histograms are thread-safe. Labels use bounded
/// cardinality dimensions (model, agent, role, adapter) — session
/// and request IDs are never used as metric labels.
pub struct RouterMetrics {
    /// Per-model latency histograms. Keyed by model name.
    pub model_latency: RwLock<HashMap<String, Arc<LatencyHistogram>>>,

    /// Per-agent latency histograms. Keyed by agent identity.
    pub agent_latency: RwLock<HashMap<String, Arc<LatencyHistogram>>>,

    /// Per-stage verdict counts. Keyed by `(PipelineStage, StageVerdict)`.
    pub stage_verdicts: RwLock<HashMap<(PipelineStage, StageVerdict), AtomicU64>>,

    /// Error rate counters. Keyed by `FailureClass` label.
    pub error_counts: RwLock<HashMap<FailureClass, AtomicU64>>,

    /// Watchdog fire counts. Keyed by watchdog event type.
    pub watchdog_events: RwLock<HashMap<WatchdogEventType, AtomicU64>>,

    /// Optional telemetry sink for structured event emission.
    pub telemetry_sink: RwLock<Option<Arc<dyn TelemetrySink>>>,
}

impl RouterMetrics {
    pub fn new() -> Self {
        Self {
            model_latency: RwLock::new(HashMap::new()),
            agent_latency: RwLock::new(HashMap::new()),
            stage_verdicts: RwLock::new(HashMap::new()),
            error_counts: RwLock::new(HashMap::new()),
            watchdog_events: RwLock::new(HashMap::new()),
            telemetry_sink: RwLock::new(None),
        }
    }

    /// Set the telemetry sink. Events will be emitted to this sink.
    pub fn set_telemetry_sink(&self, sink: Option<Arc<dyn TelemetrySink>>) {
        *self
            .telemetry_sink
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = sink;
    }

    /// Record a pipeline stage decision.
    ///
    /// Increments the verdict counter for `(decision.stage, decision.verdict)`.
    pub fn record_stage_decision(&self, decision: &StageDecision) {
        let key = (decision.stage, decision.verdict.clone());
        self.stage_verdicts
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a watchdog fire event.
    pub fn record_watchdog_fire(&self, event: &WatchdogEvent) {
        let event_type = WatchdogEventType::from(event);
        self.watchdog_events
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(event_type)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record an error by `FailureClass`.
    pub fn record_error(&self, class: FailureClass) {
        self.error_counts
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(class)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        // Emit telemetry if sink is configured
        if let Ok(sink_guard) = self.telemetry_sink.read() {
            if let Some(ref sink) = *sink_guard {
                sink.emit(&crate::telemetry::TelemetryEvent::Error {
                    class,
                    message: None,
                });
            }
        }
    }

    /// Record an error from a free-string message, auto-classifying it.
    ///
    /// This is the fallback for call sites that only have an error string.
    pub fn record_error_str(&self, message: &str) {
        let class = classify_error(message);
        self.record_error(class);
    }

    /// Record model latency.
    ///
    /// Gets or creates a `LatencyHistogram` for the given model name
    /// and records the duration.
    pub fn record_model_latency(&self, model: &str, latency_ms: u64) {
        self.model_latency
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(model.to_string())
            .or_insert_with(|| Arc::new(LatencyHistogram::new()))
            .observe(latency_ms);
    }

    /// Record agent latency.
    ///
    /// Gets or creates a `LatencyHistogram` for the given agent identity
    /// and records the duration.
    pub fn record_agent_latency(&self, agent: &str, latency_ms: u64) {
        self.agent_latency
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(agent.to_string())
            .or_insert_with(|| Arc::new(LatencyHistogram::new()))
            .observe(latency_ms);
    }

    /// Snapshot stage verdict counts (non-atomic read of all counters).
    pub fn snapshot_stage_verdicts(&self) -> HashMap<(PipelineStage, StageVerdict), u64> {
        let map = self.stage_verdicts.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        map.iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect()
    }

    /// Snapshot watchdog event counts.
    pub fn snapshot_watchdog_events(&self) -> HashMap<WatchdogEventType, u64> {
        let map = self.watchdog_events.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        map.iter()
            .map(|(k, v)| (*k, v.load(Ordering::Relaxed)))
            .collect()
    }

    /// Snapshot error counts. Keys are the stable string labels of `FailureClass`.
    pub fn snapshot_errors(&self) -> HashMap<String, u64> {
        let map = self.error_counts.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        map.iter()
            .map(|(k, v)| (k.label().to_string(), v.load(Ordering::Relaxed)))
            .collect()
    }
}

impl Default for RouterMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 4.1 FailureClass label round-trip ───────────────────────────

    #[test]
    fn failure_class_labels() {
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
        for class in &[
            FailureClass::Network,
            FailureClass::Authentication,
            FailureClass::RateLimit,
            FailureClass::InputValidation,
            FailureClass::Storage,
            FailureClass::Timeout,
            FailureClass::Internal,
            FailureClass::Unknown,
        ] {
            let json = serde_json::to_string(class).unwrap();
            let parsed: FailureClass = serde_json::from_str(&json).unwrap();
            assert_eq!(*class, parsed);
        }
    }

    // ── 4.2 Error classification ────────────────────────────────────

    #[test]
    fn classify_type_error() {
        assert_eq!(classify_error("type error: expected i32, found String"), FailureClass::Internal);
    }

    #[test]
    fn classify_syntax_error() {
        assert_eq!(classify_error("syntax error near unexpected token"), FailureClass::Internal);
    }

    #[test]
    fn classify_test_failure() {
        assert_eq!(classify_error("test failed: assertion failed at line 42"), FailureClass::Internal);
    }

    #[test]
    fn classify_build_failure() {
        assert_eq!(classify_error("build failed: compilation error"), FailureClass::Internal);
    }

    #[test]
    fn classify_permission_denied() {
        assert_eq!(classify_error("permission denied: cannot open file"), FailureClass::Storage);
    }

    #[test]
    fn classify_timeout() {
        assert_eq!(classify_error("timeout: request took too long"), FailureClass::Timeout);
    }

    #[test]
    fn classify_not_found() {
        assert_eq!(classify_error("not found: resource does not exist"), FailureClass::InputValidation);
    }

    #[test]
    fn classify_runtime_error() {
        assert_eq!(classify_error("runtime error: segmentation fault"), FailureClass::Internal);
    }

    #[test]
    fn classify_unknown_gibberish() {
        assert_eq!(classify_error("xyzzy flobnob glup"), FailureClass::Unknown);
    }

    #[test]
    fn classify_compiler_diagnostic_code() {
        assert_eq!(classify_error("[E0425] cannot find value `x` in this scope"), FailureClass::Internal);
    }

    #[test]
    fn classify_rate_limit_429() {
        assert_eq!(classify_error("429 Too Many Requests"), FailureClass::RateLimit);
    }

    #[test]
    fn classify_auth_401() {
        assert_eq!(classify_error("401 Unauthorized"), FailureClass::Authentication);
    }

    #[test]
    fn classify_network_refused() {
        assert_eq!(classify_error("connection refused (os error 111)"), FailureClass::Network);
    }

    #[test]
    fn classify_empty_returns_unknown() {
        assert_eq!(classify_error(""), FailureClass::Unknown);
    }

    // ── 4.1 Typed error recording ───────────────────────────────────

    #[test]
    fn record_error_increments_typed_category() {
        let m = RouterMetrics::new();
        m.record_error(FailureClass::Timeout);
        m.record_error(FailureClass::Timeout);
        m.record_error(FailureClass::Network);

        let snap = m.snapshot_errors();
        assert_eq!(snap.get("timeout"), Some(&2));
        assert_eq!(snap.get("network"), Some(&1));
    }

    #[test]
    fn record_error_str_classifies_and_records() {
        let m = RouterMetrics::new();
        m.record_error_str("connection reset by peer");
        m.record_error_str("connection reset by peer");
        m.record_error_str("429 rate limited");

        let snap = m.snapshot_errors();
        assert_eq!(snap.get("network"), Some(&2));
        assert_eq!(snap.get("rate_limit"), Some(&1));
    }

    #[test]
    fn record_stage_decision_increments_counter() {
        let m = RouterMetrics::new();
        let d = crate::pipeline_types::StageDecision::new(
            PipelineStage::Classifier,
            StageVerdict::Passed,
            "test",
        );
        m.record_stage_decision(&d);
        m.record_stage_decision(&d);

        let snap = m.snapshot_stage_verdicts();
        assert_eq!(
            snap.get(&(PipelineStage::Classifier, StageVerdict::Passed)),
            Some(&2)
        );
    }

    #[test]
    fn record_watchdog_fire_increments_counter() {
        let m = RouterMetrics::new();
        let event = WatchdogEvent::BudgetExceeded {
            limit: 4096,
            actual: 4097,
        };
        m.record_watchdog_fire(&event);
        m.record_watchdog_fire(&event);

        let snap = m.snapshot_watchdog_events();
        assert_eq!(snap.get(&WatchdogEventType::BudgetExceeded), Some(&2));
    }

    #[test]
    fn record_model_latency_tracks_histogram() {
        let m = RouterMetrics::new();
        m.record_model_latency("llama3.1:8b", 100);
        m.record_model_latency("llama3.1:8b", 200);

        let map = m.model_latency.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        let hist = map.get("llama3.1:8b").unwrap();
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.sum_ms(), 300);
    }

    #[test]
    fn record_agent_latency_tracks_histogram() {
        let m = RouterMetrics::new();
        m.record_agent_latency("agent-code-review", 50);
        m.record_agent_latency("agent-code-review", 150);

        let map = m.agent_latency.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        let hist = map.get("agent-code-review").unwrap();
        assert_eq!(hist.count(), 2);
        assert_eq!(hist.sum_ms(), 200);
    }

    #[test]
    fn default_creates_empty_metrics() {
        let m = RouterMetrics::default();
        assert!(m.snapshot_stage_verdicts().is_empty());
        assert!(m.snapshot_watchdog_events().is_empty());
        assert!(m.snapshot_errors().is_empty());
    }
}
