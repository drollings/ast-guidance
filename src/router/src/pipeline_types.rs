//! Pipeline decision types — structured decision records emitted by each
//! stage during request processing.

use fluent_wvr::{WorkContext, WorkError};
use serde::{Deserialize, Serialize};

use crate::config::FilterAction;
use crate::filters::RegexMatch;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum PipelineStage {
    DeterministicPreFilter,
    /// Milestone 6: the deterministic NLP parse stage (enrichment — it never
    /// gates the request). Parses the user text with `spacy-rs` and publishes
    /// the `RoutingSignal`s under the `"nlp_parse"` metadata handoff key.
    Nlp,
    /// The overlay stage (ROADMAP_20260827_ORT §2.4): consumes the parse
    /// residuals a deterministic-first ordering leaves and scores them with a
    /// configured model overlay. Enrichment — it publishes route hints and
    /// never gates the request.
    Overlay,
    Classifier,
}

impl<'de> serde::Deserialize<'de> for PipelineStage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let label = String::deserialize(deserializer)?;
        match label.as_str() {
            "DeterministicPreFilter" => Ok(PipelineStage::DeterministicPreFilter),
            "Nlp" => Ok(PipelineStage::Nlp),
            "Overlay" => Ok(PipelineStage::Overlay),
            "Classifier" => Ok(PipelineStage::Classifier),
            // Historical payloads predate the variant's removal: the synthetic
            // `Router` error/audit marker reads back as `Classifier` (with a
            // warn) so stored decisions keep deserializing. Fresh code never
            // emits this label.
            "Router" => {
                tracing::warn!(
                    target: "router.pipeline",
                    "historical PipelineStage::Router payload mapped to Classifier",
                );
                Ok(PipelineStage::Classifier)
            }
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["DeterministicPreFilter", "Nlp", "Overlay", "Classifier"],
            )),
        }
    }
}

/// Pipeline-level error (M9): replaces the synthetic `PipelineStage::Router`
/// `StageDecision` push. `Stage` carries the failing stage + source `WorkError`;
/// `Internal` covers non-stage pipeline failures.
///
/// Compatibility: historical `"Router"` stage labels deserialize to
/// `PipelineStage::Classifier` with a `warn!` (see the `Deserialize` impl
/// above); fresh telemetry never emits `"Router"`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PipelineError {
    #[error("stage {stage:?} failed: {source}")]
    Stage { stage: PipelineStage, source: WorkError },
    #[error("pipeline internal error: {0}")]
    Internal(String),
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

/// Flat-string serde for `Option<RefineReason>` (Plan B F2): wire format stays
/// the flat `as_str()` string (`"confidence_overall"` etc.) for backward
/// compat, but the in-memory type is the exhaustive enum.
mod optional_flat_refine_reason {
    use serde::{Deserialize, Deserializer, Serializer};
    use spacy_rs::RefineReason;
    #[allow(clippy::ref_option, clippy::trivially_copy_pass_by_ref)]
    pub fn serialize<S>(opt: &Option<RefineReason>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match opt {
            Some(r) => s.serialize_str(r.as_str()),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D>(d: D) -> Result<Option<RefineReason>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<String>::deserialize(d)?;
        match opt {
            None => Ok(None),
            Some(s) => RefineReason::from_flat_str(&s)
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom(format!("unknown refine_reason {s:?}"))),
        }
    }
}

/// The NLP stage's confidence handoff (ROADMAP §14.5, C1): the producing rung
/// and the margin-aware parse confidence, plus how many interlingua collisions
/// the resolve step surfaced. Consumed by the escalation ladder — a low
/// confidence / ArcEager (or encoder) source marks the request "needs
/// disambiguation".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NlpConfidenceSummary {
    /// The rung that produced the parse (`Llm | ArcEager | RuleRung |
    /// HumanReview | Encoder`).
    pub source: spacy_rs::AnnotationSource,
    /// The parse-level confidence (ArcEager/encoder fill it; LLM/rule = 1.0).
    pub overall: f64,
    /// Fraction of `{nsubj, dobj}` roles filled.
    pub role_coverage: f64,
    /// Oracle tie count (ArcEager margin-awareness, §9.3).
    pub oracle_tie_count: usize,
    /// Interlingua collisions surfaced by the resolve step.
    pub collision_count: usize,
    /// YaGO type-plausibility from `YagoResolveStage` (Alt C, separate field, never blended into `overall`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_plausibility: Option<f64>,
    /// Why the refine decision fired (M5.1) — the `RefineReason` the ladder
    /// evaluated on the deterministic base, if any.  `None` on older payloads.
    /// Typed as the exhaustive enum (F2) but serialized as the flat string for
    /// wire compat.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_flat_refine_reason"
    )]
    pub refine_reason: Option<spacy_rs::RefineReason>,
}

impl NlpConfidenceSummary {
    /// Whether the parse is uncertain enough to route around (ROADMAP §14.5,
    /// C1): a heuristic ArcEager (or trained-encoder) parse below `threshold`,
    /// or any surfaced interlingua collision. The ladder treats this as "needs
    /// disambiguation" and prefers a more capable model — never a rejection
    /// and never a vocabulary mutation (§9.4).
    #[must_use]
    pub fn needs_disambiguation(&self, threshold: f64) -> bool {
        (self.source.is_confidence_bearing() && self.overall < threshold)
            || self.collision_count > 0
    }
}

/// A route hint the overlay stage publishes for the classifier: a route name
/// and the two-tower score that recommends it (ROADMAP_20260827_ORT §2.6).
/// The classifier merges these as deterministic routing context; a redirect
/// from a top hint is a separate opt-in behind `overlay_redirect_threshold`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteHint {
    pub route: String,
    pub score: f64,
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

    /// The per-sentence routing signals parsed by the `Nlp` stage (§10.4).
    pub fn nlp_parse(&self) -> Option<Vec<spacy_rs::routing::RoutingSignal>> {
        self.0
            .get("nlp_parse")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// The per-sentence interlingua frames (ROADMAP §14.5, C1).
    pub fn nlp_interlingua(&self) -> Option<Vec<spacy_rs::routing::InterlinguaSignal>> {
        self.0
            .get("nlp_interlingua")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// The NLP confidence summary (the escalation ladder's "needs
    /// disambiguation" signal, §14.5).
    pub fn nlp_confidence(&self) -> Option<NlpConfidenceSummary> {
        self.0
            .get("nlp_confidence")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// The NLP stage's per-token confidence vector (L3 — the review endpoint
    /// reads it back so a rebuild keeps per-token fidelity). `None` when the
    /// producing rung filled none (LLM/rule rungs).
    pub fn nlp_token_confidence(&self) -> Option<Vec<f64>> {
        self.0
            .get("nlp_token_confidence")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
    }

    /// The overlay stage's route hints (ROADMAP_20260827_ORT §2.6) — the
    /// classifier merges them into its routing context when present.
    pub fn overlay_route_hints(&self) -> Option<Vec<RouteHint>> {
        self.0
            .get("overlay_route_hints")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// The overlay stage's raw contributions (kind/score/payload) for audit.
    pub fn overlay_contributions(&self) -> Option<Vec<fluent_llm::backend::OverlayContribution>> {
        self.0
            .get("overlay_contributions")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    // ── Typed setters (producers) ────────────────────────────────────────

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

    /// Publish the `Nlp` stage's per-sentence routing signals.
    pub fn set_nlp_parse(&mut self, signals: &[spacy_rs::routing::RoutingSignal]) {
        if let Ok(v) = serde_json::to_value(signals) {
            self.0["nlp_parse"] = v;
        }
    }

    /// Publish the `Nlp` stage's per-sentence interlingua frames (C1).
    pub fn set_nlp_interlingua(&mut self, signals: &[spacy_rs::routing::InterlinguaSignal]) {
        if let Ok(v) = serde_json::to_value(signals) {
            self.0["nlp_interlingua"] = v;
        }
    }

    /// Publish the `Nlp` stage's confidence summary (C1).
    pub fn set_nlp_confidence(&mut self, summary: &NlpConfidenceSummary) {
        if let Ok(v) = serde_json::to_value(summary) {
            self.0["nlp_confidence"] = v;
        }
    }

    /// Publish the `Nlp` stage's per-token confidence vector (L3).
    pub fn set_nlp_token_confidence(&mut self, token_confidence: &[f64]) {
        self.0["nlp_token_confidence"] =
            serde_json::Value::Array(token_confidence.iter().map(|c| serde_json::json!(c)).collect());
    }

    /// Publish the `Overlay` stage's route hints for the classifier (§2.6).
    pub fn set_overlay_route_hints(&mut self, hints: &[RouteHint]) {
        if let Ok(v) = serde_json::to_value(hints) {
            self.0["overlay_route_hints"] = v;
        }
    }

    /// Publish the `Overlay` stage's raw contributions for audit (§2.4).
    pub fn set_overlay_contributions(&mut self, contributions: &[fluent_llm::backend::OverlayContribution]) {
        if let Ok(v) = serde_json::to_value(contributions) {
            self.0["overlay_contributions"] = v;
        }
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
#[path = "../tests/pipeline_types.rs"]
mod tests;
