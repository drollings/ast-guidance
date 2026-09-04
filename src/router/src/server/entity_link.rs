//! The async entity-link overlay worker (ROADMAP_20260827_ORT §6.2).
//!
//! A credit-gated bounded worker pool (the shared
//! [`fluent_concurrency::credit_pool::CreditGatedPool`] primitive, mirroring
//! `ReviewWorker`'s construction) consuming `EntityLink` residuals: a PROPN
//! span with no resolved entity. The worker scores the span text against
//! boot-cached concept-label embeddings (via an injected [`EntityLinkScorer`]
//! seam — the ColBERT `EntitySimilarityIndex` at boot, blocked until the M5
//! ONNX export), applies a config threshold and an `is_subclass_of(YagoEntity)`
//! gate, and writes **candidates only** to the `overlay_candidates` table — it
//! never writes `interlingua_entity_id` / `concept_ids` on the doc (boot-only
//! registration is an invariant).
//!
//! **Fail-open.** A scorer error surfaces as "no candidates" (never a block,
//! never a drop, never a rejection). The hot path never waits: `submit` returns
//! immediately (bounded by the credit gate), and the worker processes jobs on
//! its own workers. Drained on graceful shutdown like the review worker.

use std::sync::Arc;

use fluent_concurrency::credit_pool::CreditGatedPool;
use fluent_concurrency::pool::PoolError;
use fluent_types::{InterlinguaId, NodeId};
use fluent_wvr::Runtime;
use thiserror::Error;

use crate::ledger::overlay::OverlayCandidateStore;
use fluent_concept::ConceptStore;

/// One entity-link job: a PROPN span (with no resolved entity) to link.
#[derive(Debug, Clone)]
pub struct EntityLinkJob {
    pub node_id: NodeId,
    /// Byte span into the source text.
    pub span_start: usize,
    pub span_end: usize,
    /// The span text (surface form / lemma) to score against concept labels.
    pub text: String,
}

/// Build the entity-link jobs for a parse: one per PROPN token that has no
/// resolved entity (at runtime, entity resolution never writes a doc id, so
/// every PROPN token is unresolved by construction). The byte span is located
/// by searching `request_text` for the token's surface form (deterministic —
/// the routing transcript indexes the exact request bytes). Fail-open: an
/// unlocatable token contributes no job.
#[must_use]
pub fn entity_link_jobs_from_signals(
    request_text: &str,
    node_id: NodeId,
    signals: &[spacy_rs::routing::RoutingSignal],
) -> Vec<EntityLinkJob> {
    let mut jobs = Vec::new();
    for signal in signals {
        for (i, token) in signal.tokens.iter().enumerate() {
            let is_propn = signal
                .pos
                .get(i)
                .is_some_and(|p| p.eq_ignore_ascii_case("propn"));
            if !is_propn {
                continue;
            }
            // Runtime never writes a doc-level entity id, so every PROPN is an
            // unresolved span by construction (the residual is always emitted).
            let Some((start, _)) = request_text.match_indices(token).next() else {
                continue;
            };
            let end = start + token.len();
            jobs.push(EntityLinkJob {
                node_id,
                span_start: start,
                span_end: end,
                text: token.clone(),
            });
        }
    }
    jobs
}

/// The entity-link scoring seam: given a span text, return ranked `(entity_id,
/// score)` candidates. Injected so the worker is hermetic and unit-testable
/// without a model — the boot wires it from the ColBERT `EntitySimilarityIndex`
/// over boot-cached concept-label encodings (M5.4). A scorer that yields no
/// candidates is fail-open.
pub type EntityLinkScorer = Arc<dyn Fn(&str) -> Vec<(InterlinguaId, f64)> + Send + Sync>;

/// Errors surfaced by `submit` (the worker's own processing is best-effort and
/// logged — an entity-link failure must never take the hot path down).
#[derive(Debug, Error)]
pub enum EntityLinkError {
    #[error("entity-link queue closed: {0}")]
    Closed(String),
    #[error("entity-link queue full: {0}")]
    Full(String),
}

/// The async entity-link overlay worker: a credit-gated bounded worker pool
/// (the shared [`CreditGatedPool`] primitive) over the candidate plane, the
/// concept store, and an injected scorer.
pub struct EntityLinkWorker {
    pool: CreditGatedPool<EntityLinkJob>,
}

impl EntityLinkWorker {
    /// Construct the worker. `credit_limit` bounds in-flight links (chain
    /// backpressure); `queue_capacity` bounds the pool queue. Every candidate
    /// whose score clears `threshold` AND whose `entity_id` is a subclass of
    /// `entity_root` (the YaGO `Entity` reference class) is written to the
    /// candidate plane. A `Pending` candidate is never overwritten (first-wins
    /// in the store).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        candidates: &OverlayCandidateStore,
        concepts: &Arc<dyn ConceptStore>,
        scorer: &EntityLinkScorer,
        threshold: f64,
        entity_root: InterlinguaId,
        queue_capacity: usize,
        credit_limit: usize,
        runtime: Arc<dyn Runtime>,
    ) -> Self {
        let candidates_h = candidates.clone();
        let concepts_h = Arc::clone(concepts);
        let scorer_h = Arc::clone(scorer);
        let pool = CreditGatedPool::new(
            runtime,
            credit_limit,
            queue_capacity,
            move |job: EntityLinkJob| {
                let candidates = candidates_h.clone();
                let concepts = Arc::clone(&concepts_h);
                let scorer = Arc::clone(&scorer_h);
                async move {
                    process_job(&job, &candidates, &concepts, &scorer, threshold, entity_root);
                }
            },
        );

        Self { pool }
    }

    /// Enqueue an entity-link job, bounded by the credit gate (blocks only when
    /// credit is exhausted — the hot path returns immediately otherwise).
    pub async fn submit(&self, job: EntityLinkJob) -> Result<(), EntityLinkError> {
        self.pool
            .submit(job)
            .await
            .map_err(|e| match e {
                PoolError::Closed => EntityLinkError::Closed("worker drained".into()),
                PoolError::Full => EntityLinkError::Full(e.to_string()),
            })
    }

    /// Whether the credit gate is currently blocking producers.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.pool.is_blocked()
    }

    /// Drain in-flight jobs and shut the pool down (graceful shutdown). The
    /// worker is unusable afterward.
    pub async fn drain(self: Arc<Self>) {
        self.pool.drain().await;
    }
}

/// Process one entity-link job: score → threshold → `is_subclass_of(YagoEntity)`
/// → write candidates only. Best-effort by design — a failure logs and never
/// blocks the pool.
#[allow(clippy::too_many_arguments)]
fn process_job(
    job: &EntityLinkJob,
    candidates: &OverlayCandidateStore,
    concepts: &Arc<dyn ConceptStore>,
    scorer: &EntityLinkScorer,
    threshold: f64,
    entity_root: InterlinguaId,
) {
    // Fail-open: a scorer error degrades to "no candidates" — never a block,
    // never a drop.
    let scored = scorer(&job.text);
    let mut written = 0usize;
    for (entity_id, score) in scored {
        if score < threshold {
            continue;
        }
        // Gate: the candidate must genuinely be an entity (a subclass of the
        // YaGO Entity reference class). Identity and transitive subclasses pass.
        if !concepts.is_subclass_of(entity_id, entity_root) {
            continue;
        }
        let candidate = crate::ledger::overlay::OverlayCandidate::entity_link(
            job.node_id,
            job.span_start,
            job.span_end,
            entity_id,
            score,
            "entity_link",
        );
        match candidates.write_candidate(&candidate) {
            Ok(true) => written += 1,
            Ok(false) => {
                // First-wins: a candidate for this span/entity already exists.
            }
            Err(e) => {
                tracing::warn!(
                    target: "router.overlay",
                    node_id = job.node_id.as_int(),
                    error = %e,
                    "entity-link candidate write failed (fail-open)",
                );
            }
        }
    }
    tracing::debug!(
        target: "router.overlay",
        node_id = job.node_id.as_int(),
        span_start = job.span_start,
        span_end = job.span_end,
        written = written,
        "entity-link overlay wrote candidates",
    );
}
#[cfg(test)]
#[path = "../../tests/server_entity_link.rs"]
mod tests;
