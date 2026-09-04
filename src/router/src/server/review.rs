//! The async review worker (ROADMAP §12.6 — C4).
//!
//! `ReviewWorker` owns a [`fluent_concurrency::credit_pool::CreditGatedPool`]
//! of [`ReviewJob`]s (the shared credit-gated `WorkerPool` primitive) — the
//! ledger-tier-worker pattern. The hot path never blocks: `enqueue` returns
//! immediately (202 + job id) and the worker processes jobs on its own
//! workers.
//!
//! Per job:
//! 1. [`CorrectionIndex::query_previous_corrections`] is consulted first —
//!    a reviewed `(lemma, entity)` pattern is **zero LLM cost** on reuse.
//! 2. On a miss, the taxonomy-grounded [`review_prompt`] is sent to the
//!    review model (an independent, more-capable tier), the corrections are
//!    applied, recorded in the index, and a `parse_review` ledger node is
//!    written (C7 — the workflow-extraction handoff).
//! 3. The parse node's `review_status` is updated.
//!
//! **Lifecycle (C4):** constructed at boot, spawned into the server's tracked
//! task set, and **drained on graceful shutdown** — in-flight jobs flush
//! corrections + status before the router exits. Never `tokio::spawn` it
//! detached.

use std::sync::Arc;

use fluent_concurrency::credit_pool::CreditGatedPool;
use fluent_concurrency::pool::PoolError;
use fluent_onnx::pii::{PiiSpan, PiiSpanDetector};
use fluent_wvr::Runtime;
use fluent_types::{AnnotationClaim, InterlinguaId, NodeId, ProvenanceTier};
use spacy_rs::review::{apply_corrections, review_prompt, ParseReview};
use spacy_rs::{AnnotationResult, ConceptStore, CorrectionIndex};
use thiserror::Error;

use crate::ledger::correction_index::CorrectionRow;
use crate::ledger::nlp::review_node;
use crate::ledger::ContentNodeLedger;

/// One review job: a parse node plus the interlingua pattern keys to
/// (re)check against the correction index.
#[derive(Debug, Clone)]
pub struct ReviewJob {
    pub node_id: NodeId,
    pub session_id: String,
    pub request_id: String,
    pub text: String,
    /// The original parse (provenance + records) to correct.
    pub parse: AnnotationResult,
    /// Aligned per-token `(lemma_id, entity_id)` correction-cache patterns.
    /// `entity_id` is `None` until entity linking lands (today always `None`).
    pub patterns: Vec<(InterlinguaId, Option<InterlinguaId>)>,
    pub review_model: String,
    /// PII spans detected on `text` by the review pre-filter
    /// (ROADMAP_20260827_ORT §3.3). Additive candidates — the pre-filter only
    /// *adds* these, never drops a manually enqueued job. Empty when no
    /// pre-filter is configured or nothing matched.
    pub pii_spans: Vec<PiiSpan>,
}

impl ReviewJob {
    /// Whether the pre-filter flagged PII-shaped content in this job's text.
    #[must_use]
    pub fn has_pii_spans(&self) -> bool {
        !self.pii_spans.is_empty()
    }
}

/// The review-model seam: given the taxonomy-grounded prompt, return the
/// corrections JSON (the model's reply). Injected so the worker is hermetic
/// and unit-testable without a real endpoint.
pub type ReviewFetch = Arc<dyn Fn(String) -> Result<String, String> + Send + Sync>;

/// Errors surfaced by enqueue (the worker's own processing is best-effort and
/// logged — a failed review must never take the hot path down).
#[derive(Debug, Error)]
pub enum ReviewError {
    #[error("review queue closed: {0}")]
    Closed(String),
    #[error("review queue full: {0}")]
    Full(String),
}

/// The review worker: a credit-gated bounded worker pool (the shared
/// [`CreditGatedPool`] primitive) over the shared ledger connection, the
/// correction index, and the concept store.
pub struct ReviewWorker {
    pool: CreditGatedPool<ReviewJob>,
    review_model: String,
    /// The PII pre-filter seam (ROADMAP_20260827_ORT §3.3): additive only —
    /// it annotates job texts with `PiiSpan`s, never drops a job, never
    /// rejects, never blocks the hot path (an error surfaces as an empty set).
    ///
    /// M6: this is the `fluent_onnx::pii::PiiSpanDetector` trait — genuinely
    /// model-capable (the `OrtPiiClassifier` token-classification rung when
    /// an onnx PII model is registered, else the `RegexPiiDetector` baseline
    /// over the same `llm::pii_patterns` table). It is NOT unified with the
    /// request-path `anonymize` scrub by design: detection spans (byte
    /// offsets + labels for the ledger `pii_spans` handoff) and replacement
    /// scrubbing (`[TYPE]` placeholders) are different shapes for different
    /// consumers. Documented seam — do not force-merge.
    prefilter: Option<Arc<dyn PiiSpanDetector>>,
    /// Opt-in PII auto-enqueue (ROADMAP_20260827_ORT §3.4): when `true`, the
    /// request path enqueues a review candidate after a parse whose text the
    /// pre-filter flags with PII spans. Credit-gated like every other job.
    auto_enqueue: bool,
}

impl ReviewWorker {
    /// Construct the worker. `credit_limit` bounds in-flight reviews (chain
    /// backpressure, §12.6); `queue_capacity` bounds the pool queue.
    /// `prefilter` is the PII pre-filter seam (fail-open — it only adds
    /// `PiiSpan` candidates to a job, never drops one); `auto_enqueue` opts
    /// the request path into enqueuing a candidate when the pre-filter flags
    /// the parse text. The credit receiver releases a token after each
    /// processed job (handled internally by the primitive; kept out of the
    /// public seam).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ledger: &Arc<ContentNodeLedger>,
        index: &Arc<dyn CorrectionIndex>,
        concepts: &Arc<dyn ConceptStore>,
        fetch: &ReviewFetch,
        prefilter: Option<Arc<dyn PiiSpanDetector>>,
        auto_enqueue: bool,
        review_model: String,
        queue_capacity: usize,
        credit_limit: usize,
        runtime: Arc<dyn Runtime>,
    ) -> Self {
        let index_h = Arc::clone(index);
        let concepts_h = Arc::clone(concepts);
        let fetch_h = Arc::clone(fetch);
        let ledger_h = Arc::clone(ledger);
        let model_h = review_model.clone();
        let prefilter_h = prefilter.clone();
        let pool = CreditGatedPool::new(
            runtime,
            credit_limit,
            queue_capacity,
            move |job: ReviewJob| {
                let index = Arc::clone(&index_h);
                let concepts = Arc::clone(&concepts_h);
                let fetch = Arc::clone(&fetch_h);
                let ledger = Arc::clone(&ledger_h);
                let model = model_h.clone();
                let prefilter = prefilter_h.clone();
                async move {
                    process_job(&job, &index, &concepts, &fetch, &ledger, &model, prefilter.as_deref());
                }
            },
        );

        Self {
            pool,
            review_model,
            prefilter,
            auto_enqueue,
        }
    }

    /// The review-model key this worker uses.
    #[must_use]
    pub fn review_model(&self) -> &str {
        &self.review_model
    }

    /// The configured PII pre-filter, when one is attached.
    #[must_use]
    pub fn prefilter(&self) -> Option<&Arc<dyn PiiSpanDetector>> {
        self.prefilter.as_ref()
    }

    /// Whether the PII auto-enqueue path is enabled.
    #[must_use]
    pub fn auto_enqueue_enabled(&self) -> bool {
        self.auto_enqueue
    }

    /// Detect PII spans on `text` via the pre-filter. Fail-open: no pre-filter
    /// or a detector error yields an empty set (never a job drop, never a
    /// rejection, never a block).
    #[must_use]
    pub fn detect_spans(&self, text: &str) -> Vec<PiiSpan> {
        let Some(prefilter) = &self.prefilter else {
            return Vec::new();
        };
        match prefilter.detect(text) {
            Ok(spans) => spans,
            Err(e) => {
                tracing::warn!(
                    target: "router.review",
                    error = %e,
                    "PII pre-filter failed (fail-open): no spans flagged",
                );
                Vec::new()
            }
        }
    }

    /// The opt-in PII auto-enqueue path (ROADMAP_20260827_ORT §3.4): when
    /// enabled AND the pre-filter flags PII spans on `job.text`, the job is
    /// enqueued (bounded by the existing credit gate). `false` when disabled,
    /// nothing was flagged, or the enqueue failed (logged — the hot path is
    /// never taken down). `POST /v1/sessions/{id}/review-parse` is unchanged —
    /// manual enqueues always proceed regardless of this gate.
    pub async fn maybe_auto_enqueue(&self, mut job: ReviewJob) -> bool {
        if !self.auto_enqueue || self.prefilter.is_none() {
            return false;
        }
        job.pii_spans = self.detect_spans(&job.text);
        if job.pii_spans.is_empty() {
            return false;
        }
        let node_id = job.node_id;
        let span_count = job.pii_spans.len();
        match self.enqueue(job).await {
            Ok(()) => {
                tracing::info!(
                    target: "router.review",
                    node_id = node_id.as_int(),
                    spans = span_count,
                    "PII pre-filter auto-enqueued a review candidate",
                );
                true
            }
            Err(e) => {
                tracing::warn!(
                    target: "router.review",
                    error = %e,
                    "PII auto-enqueue failed (fail-open)",
                );
                false
            }
        }
    }

    /// Enqueue a review job, bounded by the credit gate (blocks only when
    /// credit is exhausted — the hot path returns immediately otherwise). The
    /// PII pre-filter annotates the job's text (additive only — a manually
    /// enqueued job is never dropped).
    pub async fn enqueue(&self, mut job: ReviewJob) -> Result<(), ReviewError> {
        if job.pii_spans.is_empty() && self.prefilter.is_some() {
            job.pii_spans = self.detect_spans(&job.text);
        }
        self.pool
            .submit(job)
            .await
            .map_err(|e| match e {
                PoolError::Closed => ReviewError::Closed("worker drained".into()),
                PoolError::Full => ReviewError::Full(e.to_string()),
            })
    }

    /// Whether the credit gate is currently blocking producers.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.pool.is_blocked()
    }

    /// Drain in-flight jobs and shut the pool down (graceful shutdown, C4).
    /// The worker is unusable afterward.
    pub async fn drain(self: Arc<Self>) {
        self.pool.drain().await;
    }
}

/// Process one job: reuse-check → (miss) LLM review → apply → persist.
/// Best-effort by design — a review failure logs and never blocks the pool.
#[allow(clippy::too_many_arguments)]
/// A HumanReview-tier claim recording an applied review (ROADMAP M4): the
/// authoritative human-verified outcome for the parse node's current content
/// hash. Best-effort side effect, keyed to the node's hash — a later content
/// mutation invalidates it automatically.
fn human_review_claim(status: &str, model: &str) -> AnnotationClaim {
    AnnotationClaim::confirmed(
        "review",
        ProvenanceTier::HumanReview,
        serde_json::json!({ "status": status, "model": model }),
        "human-review",
        common_core::now_secs(),
    )
}

/// Best-effort tiered annotation write after a review lands. Never fails the
/// review itself; a failure is surfaced (not silently swallowed) and leaves
/// the review node committed.
fn record_review_annotation(
    ledger: &ContentNodeLedger,
    node_id: NodeId,
    status: &str,
    model: &str,
) {
    if let Err(e) = ledger.write_annotation(node_id, &human_review_claim(status, model)) {
        tracing::warn!(
            target: "router.review",
            node_id = node_id.as_int(),
            error = %e,
            "review annotation write failed",
        );
    }
}

fn process_job(
    job: &ReviewJob,
    index: &Arc<dyn CorrectionIndex>,
    concepts: &Arc<dyn ConceptStore>,
    fetch: &ReviewFetch,
    ledger: &Arc<ContentNodeLedger>,
    review_model: &str,
    prefilter: Option<&dyn PiiSpanDetector>,
) {
    // 0. The PII pre-filter's candidate spans (fail-open: a detector error was
    //    already reduced to an empty set at enqueue; this is the fallback for
    //    jobs that carried none). Additive only — never a job drop.
    let pii_spans: Vec<PiiSpan> = if job.pii_spans.is_empty() {
        prefilter
            .map(|p| p.detect(&job.text).unwrap_or_default())
            .unwrap_or_default()
    } else {
        job.pii_spans.clone()
    };

    // 1. Correction reuse: every (lemma, entity) pattern already reviewed.
    let mut reuse: Vec<spacy_rs::Correction> = Vec::new();
    let mut misses: Vec<(InterlinguaId, Option<InterlinguaId>)> = Vec::new();
    for (lemma, entity) in &job.patterns {
        match index.query_previous_corrections(*lemma, *entity) {
            Some(cs) => reuse.extend(cs),
            None => misses.push((*lemma, *entity)),
        }
    }

    // The reuse branch is load-bearing (red-team H1): it must require an
    // actual reuse set. With an empty one the review model is always called —
    // a review with nothing cached is never skipped.
    if !job.patterns.is_empty() && misses.is_empty() && !reuse.is_empty() {
        // Zero LLM cost: reuse the recorded corrections.
        let corrected = apply_corrections(&job.parse, &reuse);
        let meta = review_metadata(ledger, job.node_id, "reused", &corrected, review_model, &pii_spans);
        if let Err(e) = ledger.apply_review(job.node_id, meta, None, &[]) {
            // Best-effort, never blocks the pool — but the failure is visible.
            tracing::warn!(
                target: "router.review",
                node_id = job.node_id.as_int(),
                error = %e,
                "review reuse apply failed",
            );
            return;
        }
        record_review_annotation(ledger, job.node_id, "reused", review_model);
        tracing::debug!(
            target: "router.review",
            node_id = job.node_id.as_int(),
            "review reuse — zero LLM cost",
        );
        return;
    }

    // 2. Miss → taxonomy-grounded LLM review.
    let candidates = candidate_concepts(concepts, &job.parse);
    let prompt = review_prompt(&job.text, &job.parse, &candidates);
    let reply = match fetch(prompt.clone()) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                target: "router.review",
                node_id = job.node_id.as_int(),
                error = %e,
                "review model call failed — leaving parse unreviewed",
            );
            return;
        }
    };
    // M2: LLM-produced text goes through the shared tolerant codec
    // (`parse_typed`: pristine fast path → fence-strip → extract → repair).
    // Fail-open is preserved: `None` still means "no usable review".
    let parsed: Option<ParseReview> =
        fluent_llm::parse_typed(&reply, &serde_json::Value::Null, |_| {}).ok();
    let review = parsed.unwrap_or_else(|| ParseReview {
        corrections: Vec::new(),
        linked_entities: Vec::new(),
        note: None,
    });

    // 3. Apply + persist atomically (C4/§12.6): the correction-cache rows,
    //    the `parse_review` node (C7), and the parse node's `review_status`
    //    update commit in ONE SQLite transaction — a crash mid-review never
    //    half-applies.
    let corrected = apply_corrections(&job.parse, &review.corrections);
    let mut correction_rows: Vec<CorrectionRow> = Vec::new();
    if !review.corrections.is_empty() {
        let json = serde_json::to_string(&review.corrections).unwrap_or_default();
        for (lemma, entity) in &misses {
            correction_rows.push(CorrectionRow {
                lemma_id: lemma.as_i64(),
                entity_id: entity.map_or(0, InterlinguaId::as_i64),
                corrections_json: json.clone(),
            });
        }
    }
    let review_node = if review.corrections.is_empty() {
        None
    } else {
        Some(review_node(
            job.node_id,
            &job.session_id,
            &job.request_id,
            &job.text,
            &prompt,
            &reply,
            job.patterns
                .first()
                .map_or(InterlinguaId::from_u64(0), |(lemma, _)| *lemma),
            job.patterns.first().and_then(|(_, entity)| *entity),
            review_model,
        ))
    };
    let meta = review_metadata(ledger, job.node_id, "reviewed", &corrected, review_model, &pii_spans);
    match ledger.apply_review(
        job.node_id,
        meta,
        review_node.as_ref(),
        &correction_rows,
    ) {
        Ok(_) => {}
        // Best-effort, never blocks the pool — but the failure is visible
        // (red-team M1: no silent `let _ =` swallowing).
        Err(e) => {
            tracing::warn!(
                target: "router.review",
                node_id = job.node_id.as_int(),
                error = %e,
                "review apply failed — parse left unreviewed",
            );
            return;
        }
    }
    record_review_annotation(ledger, job.node_id, "reviewed", review_model);
    if !review.linked_entities.is_empty() {
        // M6.5: promote matching overlay candidates for this node. The review
        // model's linked-entities handoff (the same `apply_corrections`-adjacent
        // path) promotes the corresponding `EntityLink` candidates in the
        // candidate plane — never a doc-id write.
        let linked: Vec<InterlinguaId> = review
            .linked_entities
            .iter()
            .map(|le| le.interlingua_id)
            .collect();
        if let Some(candidates) = crate::ledger::overlay::candidate_store(ledger) {
            if let Err(e) = candidates.promote_linked_for_node(job.node_id, &linked) {
                tracing::warn!(
                    target: "router.review",
                    node_id = job.node_id.as_int(),
                    error = %e,
                    "candidate promotion failed (best-effort)",
                );
            }
        }
    }
}

/// Best-effort candidate concepts for the review prompt: the NOUN/PROPN lemma
/// names from the parse resolved through the store (bounded), falling back to
/// a bounded `iter_ids()` scan when nothing resolved. Grounded in the parse —
/// never an arbitrary slice of the store (red-team M4).
fn candidate_concepts(
    concepts: &Arc<dyn ConceptStore>,
    parse: &AnnotationResult,
) -> Vec<fluent_types::ConceptMetadata> {
    let mut resolved: Vec<fluent_types::ConceptMetadata> = Vec::new();
    for rec in parse.records().records() {
        if !matches!(rec.pos.to_ascii_lowercase().as_str(), "noun" | "propn") {
            continue;
        }
        if resolved.len() >= 8 {
            break;
        }
        if let Ok(id) = concepts.resolve_name(&rec.lemma) {
            if let Ok(meta) = concepts.get(id) {
                if !resolved.iter().any(|m| m.id == meta.id) {
                    resolved.push(meta);
                }
            }
        }
    }
    if !resolved.is_empty() {
        return resolved;
    }
    // Nothing in the parse resolved against the store — surface a bounded
    // sample of registered concepts so the prompt is still taxonomy-grounded.
    concepts
        .iter_ids()
        .take(8)
        .filter_map(|id| concepts.get(id).ok())
        .collect()
}

/// Merge the `review_status` overlay onto the parse node's existing metadata
/// (kind/signals/ids are preserved — the node is read first so the parse data
/// is not clobbered). The value passed to [`ContentNodeLedger::apply_review`].
/// The pre-filter's PII spans ride along as additive candidates (never a
/// write to any interlingua doc field).
fn review_metadata(
    ledger: &Arc<ContentNodeLedger>,
    node_id: NodeId,
    status: &str,
    corrected: &AnnotationResult,
    review_model: &str,
    pii_spans: &[PiiSpan],
) -> serde_json::Value {
    let mut merged = ledger
        .get_node(node_id)
        .and_then(|n| n.metadata)
        .unwrap_or_else(|| serde_json::json!({}));
    if let Some(obj) = merged.as_object_mut() {
        obj.insert(
            "review_status".into(),
            serde_json::json!({ "Reviewed": {
                "source": "human_review",
                "review_model": review_model,
                "corrected": serde_json::to_value(corrected.records()).unwrap_or_default(),
                "pii_spans": serde_json::to_value(pii_spans).unwrap_or_default(),
            } }),
        );
        obj.insert("status_note".into(), serde_json::json!(status));
    }
    merged
}

/// Extract lemma interlingua IDs from signals. The `InterlinguaId(0)` none-
/// sentinel (RESERVED namespace — a token whose lemma was never resolved) is
/// filtered out so the correction cache never keys on a fake id.
pub(crate) fn extract_lemma_ids(
    signals: &[spacy_rs::routing::RoutingSignal],
) -> Vec<fluent_types::InterlinguaId> {
    signals
        .iter()
        .filter_map(|s| s.interlingua.as_ref())
        .flat_map(|il| il.token_ids.iter().copied())
        .filter(|id| id.as_u64() != 0)
        .collect()
}

/// Build an `AnnotationResult` from the parse signals, the confidence summary,
/// and the per-token confidence vector (the auto-enqueue and manual-review
/// paths both feed the same builder, so the reconstructed parse is identical).
pub(crate) fn build_annotation_result_from_signals(
    signals: &[spacy_rs::routing::RoutingSignal],
    confidence: Option<&crate::pipeline_types::NlpConfidenceSummary>,
    token_confidence: Option<&[f64]>,
) -> spacy_rs::AnnotationResult {
    let mut records = Vec::new();
    for s in signals {
        for (i, token) in s.tokens.iter().enumerate() {
            let lemma = s.lemmas.get(i).cloned().unwrap_or_default();
            let pos = s.pos.get(i).cloned().unwrap_or_default();
            let dep = s.deps.get(i).cloned().unwrap_or_default();
            let head = s.heads.get(i).copied().unwrap_or(0);
            records.push(spacy_rs::llm::AnnotationRecord {
                text: token.clone(),
                pos,
                tag: String::new(),
                dep,
                head,
                lemma,
                morph: String::new(),
                ent_iob: String::new(),
                ent_type: String::new(),
            });
        }
    }
    let source = confidence
        .as_ref()
        .map_or(spacy_rs::AnnotationSource::RuleRung, |c| c.source);
    let record_count = records.len();
    let mut result = spacy_rs::AnnotationResult::new(
        spacy_rs::llm::AnnotationSet(records),
        source,
    );
    if let Some(c) = confidence {
        let token_conf = token_confidence
            .filter(|v| !v.is_empty())
            .map_or_else(|| vec![c.overall; record_count], <[f64]>::to_vec);
        result = result.with_confidence(
            Some(token_conf.clone()),
            Some(spacy_rs::ParseConfidence {
                overall: c.overall,
                token_scores: token_conf,
                role_coverage: c.role_coverage,
                oracle_tie_count: c.oracle_tie_count,
                oracle_margins: Vec::new(),
                semantic_plausibility: None,
            }),
        );
    }
    result
}

/// Build a review job from a parse node's data: the correction-cache patterns
/// are the aligned `(lemma_id, entity_id)` pairs from the sentinel-filtered
/// token ids; `entity_id` is `None` until entity linking lands. Shared by the
/// manual `POST /v1/sessions/{id}/review-parse` path and the PII auto-enqueue
/// path (ROADMAP_20260827_ORT §3.4).
pub(crate) fn build_review_job(
    review_model: &str,
    node_id: NodeId,
    session_id: &str,
    request_id: &str,
    text: &str,
    signals: &[spacy_rs::routing::RoutingSignal],
    confidence: Option<&crate::pipeline_types::NlpConfidenceSummary>,
    token_confidence: Option<&[f64]>,
) -> ReviewJob {
    ReviewJob {
        node_id,
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        text: text.to_string(),
        parse: build_annotation_result_from_signals(signals, confidence, token_confidence),
        patterns: extract_lemma_ids(signals).into_iter().map(|l| (l, None)).collect(),
        review_model: review_model.to_string(),
        pii_spans: Vec::new(),
    }
}
#[cfg(test)]
#[path = "../../tests/server_review.rs"]
mod tests;
