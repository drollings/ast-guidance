//! Metrics and monitoring for the router pipeline.
//!
//! Labels are tagged by model, agent, role, and adapter — never by
//! session ID or request ID (cardinality). Uses
//! `common_core::metrics::LatencyHistogram` for latency tracking.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use common_core::metrics::LatencyHistogram;

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use common_core::watchdog::{WatchdogEvent, WatchdogEventType};

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

    /// Error rate counters. Keyed by error category name.
    pub error_counts: RwLock<HashMap<String, AtomicU64>>,

    /// Watchdog fire counts. Keyed by watchdog event type.
    pub watchdog_events: RwLock<HashMap<WatchdogEventType, AtomicU64>>,
}

impl RouterMetrics {
    pub fn new() -> Self {
        Self {
            model_latency: RwLock::new(HashMap::new()),
            agent_latency: RwLock::new(HashMap::new()),
            stage_verdicts: RwLock::new(HashMap::new()),
            error_counts: RwLock::new(HashMap::new()),
            watchdog_events: RwLock::new(HashMap::new()),
        }
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

    /// Record an error by category name.
    pub fn record_error(&self, category: &str) {
        self.error_counts
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(category.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
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

    /// Snapshot error counts.
    pub fn snapshot_errors(&self) -> HashMap<String, u64> {
        let map = self.error_counts.read().unwrap_or_else(std::sync::PoisonError::into_inner);
        map.iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
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
    use crate::pipeline_types::StageDecision;

    #[test]
    fn record_stage_decision_increments_counter() {
        let m = RouterMetrics::new();
        let d = StageDecision::new(
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
    fn record_error_increments_category() {
        let m = RouterMetrics::new();
        m.record_error("timeout");
        m.record_error("timeout");
        m.record_error("parse");

        let snap = m.snapshot_errors();
        assert_eq!(snap.get("timeout"), Some(&2));
        assert_eq!(snap.get("parse"), Some(&1));
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