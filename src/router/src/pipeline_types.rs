//! Pipeline decision types — structured decision records emitted by each
//! stage during request processing.

use fluent_wvr::{WorkContext, WorkError};
use serde::{Deserialize, Serialize};

use crate::config::FilterAction;
use crate::filters::RegexMatch;
use crate::pipeline::RoutingTarget;

/// A pipeline stage that emits a typed `StageDecision`.
///
/// The `PipelineOrchestrator` calls `evaluate` directly, passing the running
/// decision accumulator (`prior`) by reference — a typed handoff that removes
/// the per-stage `StageDecision` serialize→deserialize through
/// `WorkOutput.data`. The `WorkUnit` path (`execute`) remains for composition
/// (wrappers, dependency graph, tests) and serializes the decision into
/// `WorkOutput.data` exactly as before.
pub trait StageDecisionProducer: Send + Sync + 'static {
    /// The pipeline stage this producer emits decisions for.
    fn stage_kind(&self) -> PipelineStage;

    /// Produce the typed decision for `ctx`, given the decisions already
    /// accumulated by earlier stages (`prior`).
    fn evaluate(
        &self,
        ctx: &WorkContext,
        prior: &[StageDecision],
    ) -> Result<StageDecision, WorkError>;
}

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
    Classifier,
    /// Synthetic error marker only — NOT a real pipeline stage. Retained so
    /// telemetry/rejection paths have a stable value to report when the
    /// pipeline itself fails to produce a verdict (F9); the pipeline never
    /// enters a `Router` stage, and no `PipelineStage` of this name is ever
    /// executed.
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

/// Structured PII verdict recorded by the deterministic pre-filter for
/// output-filter decisions (the `"pii_filter"` handoff key).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PiiVerdict {
    pub pattern: String,
    pub action: FilterAction,
    #[serde(default)]
    pub codewords: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub matches: Vec<RegexMatch>,
}

/// Typed access to the documented `StageDecision.metadata` handoff keys.
///
/// Producers build metadata through the typed setters and consumers read it
/// through the typed getters — no raw string-key traversal of the handoff
/// vocabulary outside this type. The underlying `serde_json::Value` storage
/// is unchanged, so the wire/serde shape of `StageDecision` is identical.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StageMetadata(serde_json::Value);

impl StageMetadata {
    pub fn new(inner: serde_json::Value) -> Self {
        Self(inner)
    }

    /// Borrow the underlying diagnostic fields (for logging/latency/verdict
    /// metadata that genuinely varies and has no typed accessor).
    pub fn as_value(&self) -> &serde_json::Value {
        &self.0
    }

    /// Consume the wrapper, returning the underlying `Value` for storage in
    /// `StageDecision.metadata`.
    pub fn into_value(self) -> serde_json::Value {
        self.0
    }

    // ── Typed getters ────────────────────────────────────────────────────

    pub fn routing_target(&self) -> Option<RoutingTarget> {
        self.0
            .get("routing_target")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    pub fn response(&self) -> Option<&str> {
        self.0.get("response").and_then(serde_json::Value::as_str)
    }

    pub fn rewritten_request(&self) -> Option<&str> {
        self.0
            .get("rewritten_request")
            .and_then(serde_json::Value::as_str)
    }

    pub fn command_result(&self) -> Option<&str> {
        self.0
            .get("command_result")
            .and_then(serde_json::Value::as_str)
    }

    pub fn pii_filter(&self) -> Option<PiiVerdict> {
        self.0
            .get("pii_filter")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    pub fn fallback(&self) -> Option<bool> {
        self.0.get("fallback").and_then(serde_json::Value::as_bool)
    }

    // ── Typed setters (producers) ────────────────────────────────────────

    pub fn set_routing_target(&mut self, rt: &RoutingTarget) {
        if let Ok(v) = serde_json::to_value(rt) {
            self.0["routing_target"] = v;
        }
    }

    pub fn set_response(&mut self, response: impl Into<String>) {
        self.0["response"] = serde_json::Value::String(response.into());
    }

    pub fn set_rewritten_request(&mut self, s: impl Into<String>) {
        self.0["rewritten_request"] = serde_json::Value::String(s.into());
    }

    pub fn set_command_result(&mut self, s: impl Into<String>) {
        self.0["command_result"] = serde_json::Value::String(s.into());
    }

    pub fn set_pii_filter(&mut self, verdict: &PiiVerdict) {
        if let Ok(v) = serde_json::to_value(verdict) {
            self.0["pii_filter"] = v;
        }
    }

    pub fn set_fallback(&mut self, fallback: bool) {
        self.0["fallback"] = serde_json::Value::Bool(fallback);
    }

    /// Write an arbitrary diagnostic field (not a documented handoff key).
    pub fn insert(&mut self, key: impl AsRef<str>, value: serde_json::Value) {
        self.0[key.as_ref()] = value;
    }
}

impl From<serde_json::Value> for StageMetadata {
    fn from(v: serde_json::Value) -> Self {
        Self(v)
    }
}
