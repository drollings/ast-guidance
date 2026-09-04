//! Stage: `NlpStage` — the deterministic NLP parse (roadmap §6).
//!
//! Runs `spacy-rs`'s pipeline on the inbound user text and publishes the
//! per-sentence [`spacy_rs::routing::RoutingSignal`]s under the `"nlp_parse"`
//! handoff key. This is an **enrichment** stage, never a gate: verdicts are
//! `Passed` (parse published) or `Skipped` (no text / parse failure) — a
//! parse problem must not reject the request (VISION: deterministic layers are
//! a cost floor, not a failure point).
//!
//! The annotation ladder is **deterministic-first** (ROADMAP_20260831_ARCEAGER
//! M4): a base phase over `[ArcEager, Rule]` always runs first and produces a
//! validated parse; a refine phase over `[Encoder, Llm]` is **gated** by a
//! [`spacy_rs::RefinePolicy`] (confidence OR task-value) and its output is
//! adopted only when it passes the 7-check gate and does not regress
//! `frame_coverage` — otherwise the base is kept. The LLM/encoder calls are
//! bounded by a [`Limiter`] via `run_sync` (the classifier-stage pattern),
//! never a bare blocking call on a worker thread.

use std::path::PathBuf;
use std::sync::Arc;

use fluent_concurrency::pool::Limiter;
use fluent_wvr::prelude::*;

use crate::pipeline_types::{
    PipelineStage, StageDecision, StageDecisionProducer, StageMetadata, StageVerdict,
};
use crate::stages::common::extract_user_message;

/// Build the spacy-rs synchronous annotation fetch from a fluent-llm backend:
/// prompts with the canonical [`spacy_rs::AnnotationRecord::prompt`] contract
/// and returns the raw §10.1 JSON array (or an error the sync ladder swallows
/// in favor of the rule rung).
///
/// The call goes through `chat_complete_with_extras` carrying the
/// `AnnotationSet` JSON schema as `response_format.schema` (ROADMAP M2.4), so a
/// grammar-constrained backend (the onnx LLM) cannot emit a structurally
/// invalid annotation array — `parse_json` cannot structurally fail. A backend
/// without the extras seam (HTTP, stubs) ignores the extras unchanged, leaving
/// the post-hoc validator as the backstop.
pub(crate) fn annotation_fetch(
    backend: Arc<dyn fluent_llm::client::ChatBackend>,
) -> spacy_rs::pipeline::LlmFetchSync {
    Arc::new(move |tokens: Vec<String>| {
        use fluent_llm::ChatMessage;
        let system = spacy_rs::AnnotationRecord::prompt(&tokens);
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: system,
            },
            ChatMessage {
                role: "user".into(),
                content: "Annotate the given tokens.".into(),
            },
        ];
        let extras = serde_json::json!({
            "response_format": {
                "type": "json_object",
                "schema": spacy_rs::AnnotationRecord::contract(),
            }
        });
        backend
            .chat_complete_with_extras(&messages, &extras)
            .map_err(|e| spacy_rs::pipeline::AnnotateError::Fetch(e.to_string()))
    })
}

/// The NLP enrichment stage: parses the request text with `spacy-rs` and
/// publishes the routing signals. Additive — only present when a pipeline opts
/// in via `PipelineParams.nlp`.
pub struct NlpStage {
    pipeline: Arc<spacy_rs::NlpPipeline>,
    fetch: Option<spacy_rs::pipeline::LlmFetchSync>,
    encoder: Option<spacy_rs::pipeline::EncoderFetchSync>,
    strings_path: Option<PathBuf>,
    refine_policy: spacy_rs::RefinePolicy,
    limiter: Limiter,
    metrics: std::sync::Arc<spacy_rs::RefineMetrics>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl NlpStage {
    /// A stage over `pipeline`, attempting the LLM annotation rung only when
    /// `fetch` is `Some`, and the trained-encoder rung only when `encoder` is
    /// `Some` (ROADMAP_20260827_ORT §4.4). The refine policy defaults to
    /// `Off` unconditionally; the builder selects `Always`/`OnUncertain` from
    /// `NlpOrdering`/`refine_policy`.
    #[must_use]
    pub fn new(
        pipeline: Arc<spacy_rs::NlpPipeline>,
        fetch: Option<spacy_rs::pipeline::LlmFetchSync>,
        encoder: Option<spacy_rs::pipeline::EncoderFetchSync>,
    ) -> Self {
        Self::with_strings(pipeline, fetch, encoder, None)
    }

    /// [`Self::new`] plus a durable StringStore path (G9): the pipeline's
    /// vocab is persisted to `strings_path` after each successful parse so
    /// newly interned lemmas survive restarts. `None` disables persistence.
    #[must_use]
    pub fn with_strings(
        pipeline: Arc<spacy_rs::NlpPipeline>,
        fetch: Option<spacy_rs::pipeline::LlmFetchSync>,
        encoder: Option<spacy_rs::pipeline::EncoderFetchSync>,
        strings_path: Option<PathBuf>,
    ) -> Self {
        let refine_policy = spacy_rs::RefinePolicy::default();
        Self {
            pipeline,
            fetch,
            encoder,
            strings_path,
            refine_policy,
            limiter: Limiter::new(
                std::thread::available_parallelism().map_or(1, |n| n.get().max(1)),
            ),
            metrics: std::sync::Arc::clone(&GLOBAL_REFINE_METRICS),
            depends: vec![],
            provides: vec![ArcIntern::from("nlp.parse")],
        }
    }

    /// Override the refine policy (ROADMAP_20260831_ARCEAGER M4.2). The builder
    /// uses this to thread the effective policy derived from
    /// `PipelineParams.nlp_ordering` / `refine_policy` into the stage.
    #[must_use]
    pub fn with_refine_policy(mut self, policy: spacy_rs::RefinePolicy) -> Self {
        self.refine_policy = policy;
        self
    }

    /// The effective refine policy for this stage.
    #[must_use]
    pub fn refine_policy(&self) -> spacy_rs::RefinePolicy {
        self.refine_policy
    }

    /// Parse the user text and return `(message, decision)`.
    fn decide(&self, ctx: &WorkContext) -> (String, StageDecision) {
        let text = match extract_user_message(ctx) {
            Ok(t) => t,
            Err(e) => {
                return (
                    "skipped".into(),
                    StageDecision::new(
                        PipelineStage::Nlp,
                        StageVerdict::Skipped,
                        format!("no user message: {e}"),
                    ),
                )
            }
        };
        if text.trim().is_empty() {
            return (
                "skipped".into(),
                StageDecision::new(
                    PipelineStage::Nlp,
                    StageVerdict::Skipped,
                    "empty user message",
                ),
            );
        }

        // Bounded sync-LLM bridge (the classifier-stage pattern): the refine
        // phase (encoder/LLM) runs only when the stored `RefinePolicy` says so
        // — `Always` for `LlmFirst` (today's behavior), `OnUncertain` for
        // `DeterministicFirst` (confidence-or-task-value gated), `Off` for no
        // refinement. The deterministic base always runs.  M5.1: the pipeline
        // also returns the `RefineReason` so the confidence summary can carry it
        // for observability; no behavior change.
        let result = if self.fetch.is_some() || self.encoder.is_some() {
            let pipeline = Arc::clone(&self.pipeline);
            let policy = self.refine_policy;
            let fetch = self.fetch.clone();
            let encoder = self.encoder.clone();
            let text_owned = text.clone();
            // `LlmFetchSync` / `EncoderFetchSync` are `Arc<dyn Fn>` — clone the
            // `Arc`s so the `run_sync` closure can own them across the await
            // point (`block_in_place` requires `'static`).
            self.limiter.run_sync(move || {
                let pipeline = Arc::clone(&pipeline);
                let text = text_owned.clone();
                async move {
                    pipeline.process_sync_with_refine_and_reason(
                        &text,
                        fetch.as_ref(),
                        encoder.as_ref(),
                        &spacy_rs::RefineSeams::default(),
                        None,
                        policy,
                    )
                }
            })
        } else {
            self.pipeline.process_sync_with_refine_and_reason(
                &text,
                None,
                None,
                &spacy_rs::RefineSeams::default(),
                None,
                self.refine_policy,
            )
        };

        match result {
            Ok((doc, annotation, refine_reason)) => {
                // Durable StringStore (G9): when a strings path is configured,
                // persist the vocab after each parse so newly interned lemmas
                // (from annotation) survive restarts. Best-effort — a failed
                // write only logs.
                if let Some(path) = &self.strings_path {
                    if let Err(e) = self.pipeline.persist_strings(path) {
                        tracing::warn!(
                            target: "router.nlp",
                            error = %e,
                            "string store persist failed",
                        );
                    }
                }
                let signals = spacy_rs::routing::extract_routing_signals(&doc);
                if signals.is_empty() {
                    return (
                        "skipped".into(),
                        StageDecision::new(
                            PipelineStage::Nlp,
                            StageVerdict::Skipped,
                            "empty parse",
                        ),
                    );
                }
                // The C1 handoff: per-sentence interlingua frames + the
                // confidence summary (source/overall/roles/collisions) so the
                // classifier and the escalation ladder can route on the parse.
                let interlingua: Vec<_> = signals
                    .iter()
                    .filter_map(|s| s.interlingua.clone())
                    .collect();
                let summary = confidence_summary_with_reason(&annotation, Some(refine_reason));
                // A3a: router owns the per-reason counters; pipeline is pure
                // and only returns the reason — record here.
                self.metrics.record(refine_reason);
                tracing::debug!(
                    target: "router.nlp",
                    reason = %refine_reason.as_str(),
                    source = %format!("{:?}", summary.source),
                    "nlp refine decision"
                );
                let mut meta = StageMetadata::new(serde_json::Value::Object(Default::default()));
                meta.set_nlp_parse(&signals);
                meta.set_nlp_interlingua(&interlingua);
                meta.set_nlp_confidence(&summary);
                // L3: the per-token confidence rides along so the review
                // endpoint can rebuild the parse with token fidelity.
                if let Some(tc) = annotation.token_confidence() {
                    meta.set_nlp_token_confidence(tc);
                }
                (
                    "parsed".into(),
                    StageDecision::new(
                        PipelineStage::Nlp,
                        StageVerdict::Passed,
                        format!("parsed {} sentence(s)", signals.len()),
                    )
                    .with_metadata(meta.into_value()),
                )
            }
            Err(e) => (
                "skipped".into(),
                StageDecision::new(
                    PipelineStage::Nlp,
                    StageVerdict::Skipped,
                    format!("nlp parse failed: {e}"),
                ),
            ),
        }
    }
}

/// The C1 confidence summary from a ladder handoff: the producing rung (the
/// serde-serialized [`spacy_rs::AnnotationSource`]), the margin-aware parse
/// confidence (ArcEager/encoder), and the role coverage / tie count. LLM and
/// rule rungs report 1.0 (no oracle to doubt). When `refine_reason` is
/// supplied (M5.1), it is carried verbatim for observability.
#[allow(dead_code)]
pub(crate) fn confidence_summary(
    annotation: &spacy_rs::AnnotationResult,
) -> crate::pipeline_types::NlpConfidenceSummary {
    confidence_summary_with_reason(annotation, None)
}

/// Like [`confidence_summary`] but carrying the refine decision reason (M5.1)
/// that the ladder evaluated on the deterministic base.
pub(crate) fn confidence_summary_with_reason(
    annotation: &spacy_rs::AnnotationResult,
    refine_reason: Option<spacy_rs::RefineReason>,
) -> crate::pipeline_types::NlpConfidenceSummary {
    let source = annotation.source();
    // Confidence-bearing rungs (ArcEager/encoder) report their margin-aware
    // parse confidence; LLM/rule/human-review report 1.0 by convention (no
    // oracle to doubt). Branching on `is_confidence_bearing` — the closed-set
    // classifier in `spacy-rs` — keeps a future confidence-bearing rung a
    // compile error rather than a silent fail-open.
    let confidence = if source.is_confidence_bearing() {
        annotation
            .parse_confidence
            .as_ref()
            .map_or((1.0, 1.0, 0), |pc| {
                (pc.overall, pc.role_coverage, pc.oracle_tie_count)
            })
    } else {
        (1.0, 1.0, 0)
    };
    crate::pipeline_types::NlpConfidenceSummary {
        source,
        overall: confidence.0,
        role_coverage: confidence.1,
        oracle_tie_count: confidence.2,
        collision_count: annotation.collision_count,
        semantic_plausibility: annotation
            .parse_confidence
            .as_ref()
            .and_then(|pc| pc.semantic_plausibility),
        refine_reason,
    }
}

impl WorkUnit for NlpStage {
    fn name(&self) -> &str {
        "nlp"
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let (message, decision) = self.decide(ctx);
        WorkOutput::typed(message, &decision)
    }
}

impl StageDecisionProducer for NlpStage {
    fn stage_kind(&self) -> PipelineStage {
        PipelineStage::Nlp
    }

    fn evaluate(
        &self,
        ctx: &WorkContext,
        _prior: &[StageDecision],
    ) -> Result<StageDecision, WorkError> {
        Ok(self.decide(ctx).1)
    }
}

impl_fieldless!(NlpStage);

impl Describable for NlpStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "nlp",
            "purity": "deterministic parse; publishes per-sentence routing signals (deterministic base + gated refine)",
            "llm_rung": self.fetch.is_some(),
            "encoder_rung": self.encoder.is_some(),
            "refine_policy": format!("{:?}", self.refine_policy.mode),
        })
    }
}

/// Instance-owned per-reason counters (A3a). The router owns the
/// metrics; `spacy-rs` keeps only the types and pure `refine_reason`
/// function — no global atomics.
static GLOBAL_REFINE_METRICS: std::sync::LazyLock<std::sync::Arc<spacy_rs::RefineMetrics>> =
    std::sync::LazyLock::new(|| std::sync::Arc::new(spacy_rs::RefineMetrics::new()));

/// Per-reason trigger-rate histogram for the refine decision (M5.4): the
/// current snapshot of the router-owned `RefineMetrics`, exposed so
/// `refine_on_*` thresholds can be tuned from observed production rates
/// without a code change.  Call from an admin/metrics handler.
#[must_use]
pub fn refine_metrics_snapshot() -> spacy_rs::RefineMetricsSnapshot {
    GLOBAL_REFINE_METRICS.snapshot()
}

/// Reset the global refine metrics (test-only).
#[cfg(test)]
pub fn reset_refine_metrics() {
    GLOBAL_REFINE_METRICS.reset();
}

impl_component!(NlpStage);
#[cfg(test)]
#[path = "../../tests/stages_nlp.rs"]
mod tests;
