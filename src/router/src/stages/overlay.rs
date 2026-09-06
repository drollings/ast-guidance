//! Stage: `OverlayStage` — the deterministic-first residual-consultation
//! stage (ROADMAP_20260827_ORT §2.3/§2.4).
//!
//! Under `NlpOrdering::DeterministicFirst`, the deterministic parse leaves
//! *residuals* (sentences the heuristic parser was unsure about). The overlay
//! stage scores them with configured model overlays (the Prompt-Router in M2)
//! and publishes **route hints** the classifier merges as deterministic
//! routing context. It is enrichment, never a gate: verdicts are `Passed`
//! (contributions produced) or `Skipped` (no residuals, no overlay, or every
//! overlay failed — fail-open). Redirecting on a top hint is a separate
//! opt-in (`overlay_redirect_threshold`) that M2 leaves off.
//!
//! [`ResidualSelector`] is the pure "deterministic first pass": it turns the
//! parse signals + confidence summary into residuals with no model in the
//! loop. The stage itself runs each overlay under a [`Limiter`] (via
//! `run_sync` — the sync CPU-bound ort call is bounded, never a bare blocking
//! call on a worker thread).

use std::sync::Arc;

use fluent_concurrency::pool::{Limiter, ResultPool};
use fluent_llm::backend::{OverlayContribution, OverlayError, Residual, ResidualKind, ResidualOverlay};
use fluent_wvr::prelude::*;
use spacy_rs::routing::RoutingSignal;

use crate::pipeline_types::{
    NlpConfidenceSummary, PipelineStage, RouteHint, StageDecision, StageDecisionProducer,
    StageMetadata, StageVerdict,
};

/// The pure residual selector: from the parse signals and confidence summary,
/// emit the residuals a deterministic-first ordering should consult a model
/// about. Unit-testable, no model in the loop — the "deterministic first
/// pass" made concrete.
#[derive(Debug, Clone)]
pub struct ResidualSelector {
    disambiguation_floor: f64,
}

impl ResidualSelector {
    /// A selector whose disambiguation floor is the parse-confidence floor
    /// below which a sentence (or the doc) is flagged for consultation.
    #[must_use]
    pub fn new(disambiguation_floor: f64) -> Self {
        Self {
            disambiguation_floor,
        }
    }

    /// The floor below which a parse is "needs disambiguation".
    #[must_use]
    pub fn disambiguation_floor(&self) -> f64 {
        self.disambiguation_floor
    }
}

/// Redirect gate for the overlay top hint ([B] task-value axis).
/// `None` (the default) means OFF — never redirect. `Some(t)` redirects
/// only when the top hint score is at/above `t`. Pure and sync so the
/// calibration corpus can exercise it without a model in the loop.
#[must_use]
pub fn should_redirect_on_hint(top_score: f64, threshold: Option<f64>) -> bool {
    threshold.is_some_and(|t| top_score >= t)
}

impl ResidualSelector {
    /// Select residuals. A sentence yields a `Disambiguation` residual when
    /// the doc-level confidence summary says the parse needs disambiguation,
    /// or when the sentence's own interlingua confidence is below the floor.
    /// The PII/entity/parse arms stay empty until M3/M6.
    pub fn select(
        &self,
        signals: &[RoutingSignal],
        confidence: Option<&NlpConfidenceSummary>,
    ) -> Vec<Residual> {
        let doc_needs = confidence
            .is_some_and(|c| c.needs_disambiguation(self.disambiguation_floor));
        let mut out = Vec::new();
        for signal in signals {
            let sentence_low = signal.interlingua.as_ref().is_some_and(|il| {
                il.confidence.unwrap_or(0.0) < self.disambiguation_floor
            });
            if doc_needs || sentence_low {
                out.push(Residual::disambiguation(signal.sentence.clone()));
            }
        }
        out
    }
}

/// The overlay enrichment stage: reads the NLP handoff, selects residuals,
/// runs each configured overlay under a `ResultPool`, and publishes route hints.
pub struct OverlayStage {
    selector: ResidualSelector,
    overlays: Vec<Arc<dyn ResidualOverlay>>,
    #[allow(dead_code)]
    limiter: Arc<Limiter>,
    #[allow(dead_code)]
    cap: usize,
    #[allow(dead_code)]
    pool: std::sync::OnceLock<Arc<ResultPool<Residual, OverlayContribution, OverlayError>>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl OverlayStage {
    /// A stage over `overlays` (each consumed for its own `ResidualKind`),
    /// capping concurrent `run` calls at `max_concurrency` (default 2).
    #[must_use]
    pub fn new(
        selector: ResidualSelector,
        overlays: Vec<Arc<dyn ResidualOverlay>>,
        max_concurrency: Option<usize>,
    ) -> Self {
        let cap = max_concurrency.unwrap_or(2).max(1);
        let limiter = Arc::new(Limiter::new(cap));
        Self {
            selector,
            overlays,
            limiter,
            cap,
            pool: std::sync::OnceLock::new(),
            depends: vec![ArcIntern::from("nlp.parse")],
            provides: vec![ArcIntern::from("overlay.hints")],
        }
    }

    #[allow(dead_code)]
    fn pool(&self) -> Arc<ResultPool<Residual, OverlayContribution, OverlayError>> {
        self.pool
            .get_or_init(|| {
                let cap = self.cap;
                let runtime = fluent_concurrency::tokio_runtime();
                let overlays = self.overlays.clone();
                let limiter = Arc::clone(&self.limiter);
                Arc::new(ResultPool::new(
                    runtime,
                    cap,
                    cap * 4,
                    move |residual: Residual| {
                        let overlays = overlays.clone();
                        let limiter = limiter.clone();
                        async move {
                            let overlay = overlays
                                .iter()
                                .find(|o| o.kind() == residual.kind)
                                .cloned()
                                .ok_or_else(|| OverlayError::Rejected(format!("no overlay for {:?}", residual.kind)))?;
                            limiter.run(|| async { overlay.run(&residual) }).await
                        }
                    },
                ))
            })
            .clone()
    }

    fn decide(
        &self,
        signals: Option<&[RoutingSignal]>,
        confidence: Option<&NlpConfidenceSummary>,
    ) -> (String, StageDecision) {
        let skip = |reason: String| {
            (
                "skipped".into(),
                StageDecision::new(
                    PipelineStage::Overlay,
                    StageVerdict::Skipped,
                    reason,
                ),
            )
        };

        if self.overlays.is_empty() {
            return skip("no overlay model configured".into());
        }
        let Some(signals) = signals.filter(|s| !s.is_empty()) else {
            return skip("no NLP parse handoff".into());
        };

        let residuals = self.selector.select(signals, confidence);
        if residuals.is_empty() {
            return skip("no residuals below the disambiguation floor".into());
        }

        // Sync path: score sequentially via limiter (WorkUnit::execute is sync).
        let residual_count = residuals.len();
        let results: Vec<Result<OverlayContribution, OverlayError>> = residuals
            .iter()
            .filter_map(|residual| {
                let overlay = self
                    .overlays
                    .iter()
                    .find(|o| o.kind() == residual.kind)?
                    .clone();
                Some(overlay.run(residual))
            })
            .collect();

        let mut contributions: Vec<OverlayContribution> = Vec::new();
        let mut hints: Vec<RouteHint> = Vec::new();
        let mut errored = 0usize;
        for result in results {
            match result {
                Ok(contribution) => {
                    if contribution.kind == ResidualKind::Disambiguation {
                        if let Some(payload) = contribution.payload.get("route_hints") {
                            // M2: intentionally a direct `from_value`, NOT the
                            // tolerant LLM codec. The payload is our own
                            // `OverlayContribution` struct (built with
                            // `serde_json::json!` by the overlay itself) — a
                            // typed internal round-trip, never LLM text.
                            if let Ok(parsed) =
                                serde_json::from_value::<Vec<RouteHint>>(payload.clone())
                            {
                                hints.extend(parsed);
                            }
                        }
                    }
                    contributions.push(contribution);
                }
                Err(e) => {
                    errored += 1;
                    tracing::warn!(
                        target: "router.pipeline.stage.overlay",
                        error = %e,
                        "overlay run failed (fail-open)",
                    );
                }
            }
        }

        if contributions.is_empty() {
            return skip(format!("all {errored} overlay run(s) failed (fail-open)"));
        }
        if errored > 0 {
            tracing::warn!(
                target: "router.pipeline.stage.overlay",
                succeeded = contributions.len(),
                failed = errored,
                "overlay stage degraded: some overlays failed",
            );
        }

        // Highest-scoring hint first — the classifier's context and the
        // redirect gate (when armed) both consume the top of the list.
        hints.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let mut meta = StageMetadata::new(serde_json::Value::Object(Default::default()));
        meta.set_overlay_contributions(&contributions);
        if !hints.is_empty() {
            meta.set_overlay_route_hints(&hints);
        }
        (
            "overlaid".into(),
            StageDecision::new(
                PipelineStage::Overlay,
                StageVerdict::Passed,
                format!(
                    "scored {} residual(s) with {} overlay contribution(s)",
                    residual_count,
                    contributions.len(),
                ),
            )
            .with_metadata(meta.into_value()),
        )
    }

    #[allow(dead_code)]
    async fn decide_async(
        &self,
        signals: Option<&[RoutingSignal]>,
        confidence: Option<&NlpConfidenceSummary>,
    ) -> (String, StageDecision) {
        let skip = |reason: String| {
            (
                "skipped".into(),
                StageDecision::new(PipelineStage::Overlay, StageVerdict::Skipped, reason),
            )
        };
        if self.overlays.is_empty() {
            return skip("no overlay model configured".into());
        }
        let Some(signals) = signals.filter(|s| !s.is_empty()) else {
            return skip("no NLP parse handoff".into());
        };
        let residuals = self.selector.select(signals, confidence);
        if residuals.is_empty() {
            return skip("no residuals below the disambiguation floor".into());
        }
        let residual_count = residuals.len();
        // Concurrent via ResultPool bounded by max_concurrency
        let pool = self.pool();
        let mut handles = Vec::new();
        for residual in residuals {
            let p = Arc::clone(&pool);
            handles.push(async move { p.submit(residual).await });
        }
        let results = futures_util::future::join_all(handles).await;
        let mut contributions: Vec<OverlayContribution> = Vec::new();
        let mut hints: Vec<RouteHint> = Vec::new();
        let mut errored = 0usize;
        for r in results {
            match r {
                Ok(contribution) => {
                    if contribution.kind == ResidualKind::Disambiguation {
                        if let Some(payload) = contribution.payload.get("route_hints") {
                            // M2: intentionally direct — own-struct round-trip,
                            // never LLM text (see the sync `decide` above).
                            if let Ok(parsed) = serde_json::from_value::<Vec<RouteHint>>(payload.clone()) {
                                hints.extend(parsed);
                            }
                        }
                    }
                    contributions.push(contribution);
                }
                Err(e) => {
                    errored += 1;
                    tracing::warn!(target: "router.pipeline.stage.overlay", error = %format!("{:?}", e), "overlay run failed (fail-open)");
                }
            }
        }
        if contributions.is_empty() {
            return skip(format!("all {errored} overlay run(s) failed (fail-open)"));
        }
        hints.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let mut meta = StageMetadata::new(serde_json::Value::Object(Default::default()));
        meta.set_overlay_contributions(&contributions);
        if !hints.is_empty() {
            meta.set_overlay_route_hints(&hints);
        }
        (
            "overlaid".into(),
            StageDecision::new(
                PipelineStage::Overlay,
                StageVerdict::Passed,
                format!("scored {} residual(s) with {} overlay contribution(s)", residual_count, contributions.len()),
            )
            .with_metadata(meta.into_value()),
        )
    }
}

impl WorkUnit for OverlayStage {
    fn name(&self) -> &str {
        "overlay"
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let (message, decision) = self.decide(None, None);
        WorkOutput::typed(message, &decision)
    }
}

impl StageDecisionProducer for OverlayStage {
    fn stage_kind(&self) -> PipelineStage {
        PipelineStage::Overlay
    }

    fn evaluate(
        &self,
        _ctx: &WorkContext,
        prior: &[StageDecision],
    ) -> Result<StageDecision, WorkError> {
        // The NlpStage ran first; its parse handoff drives residual selection.
        let signals: Vec<RoutingSignal> = prior
            .iter()
            .filter_map(|d| StageMetadata::from(d.metadata.clone()).nlp_parse())
            .flatten()
            .collect();
        let confidence = prior
            .iter()
            .find_map(|d| StageMetadata::from(d.metadata.clone()).nlp_confidence());
        Ok(self
            .decide(
                if signals.is_empty() {
                    None
                } else {
                    Some(&signals)
                },
                confidence.as_ref(),
            )
            .1)
    }
}

impl Describable for OverlayStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "overlay",
            "purity": "deterministic residual selection + model overlay scoring (fail-open)",
            "overlay_count": self.overlays.len(),
            "disambiguation_floor": self.selector.disambiguation_floor(),
        })
    }
}

impl_fieldless!(OverlayStage);

impl_component!(OverlayStage);
#[cfg(test)]
#[path = "../../tests/stages_overlay.rs"]
mod tests;
