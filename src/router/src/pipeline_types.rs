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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_stage_serde_round_trip() {
        for stage in [
            PipelineStage::DeterministicPreFilter,
            PipelineStage::Classifier,
            PipelineStage::Router,
        ] {
            let json = serde_json::to_string(&stage).expect("serialize");
            let back: PipelineStage = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, stage, "round trip for {json}");
        }
        // The `Router` marker exists only for telemetry/rejection paths.
        assert_eq!(
            serde_json::from_str::<PipelineStage>("\"Router\"").unwrap(),
            PipelineStage::Router
        );
    }

    #[test]
    fn stage_verdict_serde_round_trip() {
        for verdict in [
            StageVerdict::Passed,
            StageVerdict::Rejected,
            StageVerdict::Rerouted,
            StageVerdict::Skipped,
            StageVerdict::Error,
        ] {
            let json = serde_json::to_string(&verdict).expect("serialize");
            let back: StageVerdict = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, verdict, "round trip for {json}");
        }
    }

    #[test]
    fn stage_decision_builders_and_serde() {
        let d = StageDecision::new(PipelineStage::DeterministicPreFilter, StageVerdict::Rejected, "blocked")
            .with_score(0.9)
            .with_latency(12)
            .with_metadata(serde_json::json!({"x": 1}));
        assert_eq!(d.score, Some(0.9));
        assert_eq!(d.latency_ms, 12);
        assert_eq!(d.metadata["x"], 1);
        let back: StageDecision =
            serde_json::from_str(&serde_json::to_string(&d).expect("serialize")).expect("deserialize");
        assert_eq!(back.stage, d.stage);
        assert_eq!(back.verdict, d.verdict);
        assert_eq!(back.score, d.score);
    }

    #[test]
    fn pii_verdict_serde_round_trip() {
        let v = PiiVerdict {
            pattern: "email".into(),
            action: FilterAction::Anonymize,
            codewords: [("a".to_string(), "b".to_string())].into_iter().collect(),
            matches: vec![RegexMatch {
                pattern_name: "email".into(),
                matched_text: "x@y.z".into(),
                start: 0,
                end: 6,
                action: FilterAction::Redact,
            }],
        };
        let back: PiiVerdict =
            serde_json::from_str(&serde_json::to_string(&v).expect("serialize")).expect("deserialize");
        assert_eq!(back, v);
    }

    #[test]
    fn stage_metadata_typed_accessors() {
        let mut m = StageMetadata::new(serde_json::json!({}));
        m.set_response("hello");
        m.set_rewritten_request("rewritten");
        m.set_command_result("result");
        m.set_fallback(true);
        assert_eq!(m.response(), Some("hello"));
        assert_eq!(m.rewritten_request(), Some("rewritten"));
        assert_eq!(m.command_result(), Some("result"));
        assert_eq!(m.fallback(), Some(true));

        let pii = PiiVerdict {
            pattern: "ssn".into(),
            action: FilterAction::Redact,
            codewords: Default::default(),
            matches: vec![],
        };
        m.set_pii_filter(&pii);
        assert_eq!(m.pii_filter().expect("pii"), pii);

        // `RoutingTarget`'s non-defaulted fields are `url`/`model`; serde
        // fills the rest with the struct's `#[serde(default)]`s.
        let rt: RoutingTarget = serde_json::from_value(serde_json::json!({
            "url": "http://upstream",
            "model": "fast",
        }))
        .expect("routing target from json");
        assert_eq!(rt.model, "fast");
        m.set_routing_target(&rt);
        assert_eq!(m.routing_target().expect("routing target").model, "fast");

        m.insert("custom", serde_json::json!(true));
        assert_eq!(m.as_value()["custom"], true);
        // `into_value` unwraps to the underlying map.
        assert_eq!(m.into_value()["response"], "hello");
    }

    #[test]
    fn stage_metadata_missing_accessors_return_none() {
        let m = StageMetadata::new(serde_json::json!({}));
        assert!(m.routing_target().is_none());
        assert_eq!(m.response(), None);
        assert!(m.pii_filter().is_none());
        assert_eq!(m.fallback(), None);
    }

    #[test]
    fn stage_metadata_from_value_and_transparent_serde() {
        // `#[serde(transparent)]` means the wrapper (de)serializes as the
        // underlying JSON value alone.
        let json = serde_json::json!({"routing_target": {"model": "fast", "group": "fast"}});
        let m: StageMetadata = serde_json::from_value(json.clone()).expect("from value");
        let back = serde_json::to_value(m).expect("to value");
        assert_eq!(back, json);
    }
}
