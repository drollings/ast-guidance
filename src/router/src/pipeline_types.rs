//! Pipeline decision types — structured decision records emitted by each
//! stage during request processing.

use serde::{Deserialize, Serialize};

/// Emitted by every pipeline stage. Flows through tracing spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDecision {
    pub stage: PipelineStage,
    pub verdict: StageVerdict,
    pub score: Option<f64>,
    pub reason: String,
    pub latency_ms: u64,
    pub metadata: serde_json::Value,
}

impl StageDecision {
    pub fn new(stage: PipelineStage, verdict: StageVerdict, reason: impl Into<String>) -> Self {
        Self {
            stage,
            verdict,
            score: None,
            reason: reason.into(),
            latency_ms: 0,
            metadata: serde_json::Value::Object(Default::default()),
        }
    }

    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }

    #[must_use]
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStage {
    DeterministicPreFilter,
    QualityGate,
    PlanningRefinement,
    GuardrailCheck,
    Router,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageVerdict {
    Passed,
    Rejected,
    Rerouted,
    Skipped,
    Error,
}