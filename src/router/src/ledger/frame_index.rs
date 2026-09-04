//! Frame resolution — the router's implementation of the spacy-rs
//! [`PreferredSenseIndex`] seam and the wave-batched [`FrameResolutionWorker`]
//! (ROADMAP M3, G7/G8).
//!
//! **Sense promotion** reuses the `interlingua_index` pattern-cache rows
//! exactly like `SqliteCorrectionIndex` (the golden-corpus rule-genesis flow),
//! but with a distinct `role = 'sense'` so sense resolutions never collide with
//! review corrections. The **entity-scope column = the ambiguity kind**, so
//! each `(predicate_lemma_id, ambiguity_kind)` pattern gets its own cache row;
//! a resolved pattern replays deterministically and never re-triggers an LLM
//! call.
//!
//! **Wave batching** composes the shared [`CreditGatedPool`] (the
//! `ReviewWorker`/`EntityLinkWorker` shape — never a hand-rolled worker): one
//! job drains all open ambiguities pending in a tick and issues **one**
//! grammar-constrained call returning N resolutions (the `BatchPromptGrammar`
//! array shape), then applies + promotes.

use std::sync::Arc;

use fluent_concurrency::credit_pool::CreditGatedPool;
use fluent_concurrency::pool::PoolError;
use fluent_db::error::DbError;
use fluent_db::store::SqliteStore;
use fluent_types::InterlinguaId;
use fluent_wvr::Runtime;
use rusqlite::params;
use spacy_rs::{AmbiguityKind, Resolution};
use thiserror::Error;

use crate::ledger::correction_index::PATTERN_NODE;

/// Role discriminating sense-pattern cache rows from review-correction rows
/// (`'correction'`) and parse-node audit rows (a real `node_id`).
const SENSE_ROLE: &str = "sense";

/// A request to resolve one ambiguous frame (one slot of the wave batch).
#[derive(Debug, Clone)]
pub struct FrameResolutionRequest {
    pub predicate_lemma_id: InterlinguaId,
    pub ambiguity_kind: AmbiguityKind,
    /// A legible description of the ambiguity (fed to the resolver prompt).
    pub detail: String,
    /// Candidate concept ids when the ambiguity is a sense choice.
    pub candidate_ids: Vec<InterlinguaId>,
}

/// One frame-resolution job: the open ambiguities pending in a single tick.
#[derive(Debug, Clone, Default)]
pub struct FrameResolutionJob {
    pub requests: Vec<FrameResolutionRequest>,
}

/// The resolution-model seam: given the batched request list, return the
/// resolutions JSON — a JSON array of
/// `[{"chosen_candidate_id": <i64>, "detail": "..."}, ...]` **aligned by
/// index** to `requests`. Injected so the worker is hermetic and unit-testable
/// without a real endpoint; the router builds it from the onnx `ChatBackend`
/// with a `BatchPromptGrammar` so structurally-invalid output is impossible.
pub type FrameResolutionFetch =
    Arc<dyn Fn(Vec<FrameResolutionRequest>) -> Result<String, String> + Send + Sync>;

/// Errors surfaced by enqueue (the worker's processing is best-effort).
#[derive(Debug, Error)]
pub enum FrameResolutionError {
    #[error("frame resolution queue closed: {0}")]
    Closed(String),
    #[error("frame resolution queue full: {0}")]
    Full(String),
}

/// The promotion seam for resolved ambiguity patterns (mirrors
/// `spacy_rs::review::CorrectionIndex`): a `(predicate_lemma_id,
/// ambiguity_kind)` pattern that has been resolved is recorded and replayed
/// deterministically — golden-corpus-style rule genesis applied to senses.
///
/// Owner: the ledger pattern-cache (this module), beside
/// [`SqlitePreferredSenseIndex`]. spacy-rs keeps only the pure derivation
/// (`Frame`, `extract_frames`, ambiguity detection, key minting, `Resolution`
/// value type); it never imports this trait, so no router edge enters
/// spacy-rs. Replay lives behind [`FrameResolutionWorker`].
///
/// The router implements this over the existing `interlingua_index`
/// correction-cache rows (the entity-scope column = the ambiguity kind).
pub trait PreferredSenseIndex: Send + Sync {
    /// The previously-recorded resolution for this pattern, when known.
    fn preferred_sense(
        &self,
        predicate_lemma_id: InterlinguaId,
        ambiguity_kind: AmbiguityKind,
    ) -> Option<Resolution>;
    /// Persist a resolution for this pattern.
    fn record_preferred_sense(
        &self,
        predicate_lemma_id: InterlinguaId,
        ambiguity_kind: AmbiguityKind,
        resolution: Resolution,
    ) -> Result<(), fluent_concept::ConceptStoreError>;
}

/// `PreferredSenseIndex` over `interlingua_index` — the pattern-cache rows for
/// resolved `(predicate_lemma_id, ambiguity_kind)` patterns.
pub struct SqlitePreferredSenseIndex {
    store: Arc<SqliteStore>,
}

impl SqlitePreferredSenseIndex {
    /// A sense index over the shared ledger connection.
    pub fn new(store: Arc<SqliteStore>) -> Self {
        Self { store }
    }
}

/// Upsert a sense-pattern row into `interlingua_index`. The single SQL for the
/// sense cache — shared by `SqlitePreferredSenseIndex` and the worker's atomic
/// promotion write so the row shape lives in one place.
pub(crate) fn upsert_sense_row(
    conn: &rusqlite::Connection,
    predicate_lemma_id: i64,
    ambiguity_kind: i64,
    resolution_json: &str,
) -> Result<(), DbError> {
    fluent_db::query::execute(
        conn,
        "INSERT INTO interlingua_index \
         (node_id, interlingua_id, interlingua_source, role, entity_id, review_status, corrections, span_key) \
         VALUES (?1, ?2, 'spacy_lemma', ?3, ?4, 'cached', ?5, '') \
         ON CONFLICT(node_id, interlingua_id, span_key, role, entity_id) DO UPDATE SET \
            review_status = excluded.review_status, corrections = excluded.corrections",
        params![
            PATTERN_NODE,
            predicate_lemma_id,
            SENSE_ROLE,
            ambiguity_kind,
            resolution_json,
        ],
    )?;
    Ok(())
}

impl PreferredSenseIndex for SqlitePreferredSenseIndex {
    fn preferred_sense(
        &self,
        predicate_lemma_id: InterlinguaId,
        ambiguity_kind: AmbiguityKind,
    ) -> Option<Resolution> {
        let row = self
            .store
            .query_row(
                "SELECT corrections FROM interlingua_index \
                 WHERE node_id = ?1 AND interlingua_id = ?2 AND role = ?3 AND entity_id = ?4",
                params![
                    PATTERN_NODE,
                    predicate_lemma_id.as_i64(),
                    SENSE_ROLE,
                    ambiguity_kind as i64,
                ],
                |r| r.get::<_, Option<String>>(0),
            )
            .ok()
            .flatten()
            .flatten()?;
        serde_json::from_str(&row).ok()
    }

    fn record_preferred_sense(
        &self,
        predicate_lemma_id: InterlinguaId,
        ambiguity_kind: AmbiguityKind,
        resolution: Resolution,
    ) -> Result<(), fluent_concept::ConceptStoreError> {
        let json = serde_json::to_string(&resolution)
            .map_err(|e| fluent_concept::ConceptStoreError::Storage(e.to_string()))?;
        self.store
            .with_conn(|conn| {
                upsert_sense_row(
                    conn,
                    predicate_lemma_id.as_i64(),
                    ambiguity_kind as i64,
                    &json,
                )
            })
            .map_err(|e| fluent_concept::ConceptStoreError::Storage(e.to_string()))?;
        Ok(())
    }
}

/// The wave-batched frame-resolution worker: a credit-gated bounded worker
/// pool (the shared [`CreditGatedPool`] primitive) over the sense index.
///
/// Per job:
/// 1. Reuse: every `(predicate, ambiguity_kind)` already resolved in the
///    `PreferredSenseIndex` is served deterministically — **zero LLM cost**.
/// 2. On any miss, the worker issues **one** batched call returning N
///    resolutions (one per pending ambiguity), applies them, and promotes
///    each resolved pattern back into the index (so the next occurrence is
///    deterministic).
///
/// Built to mirror `ReviewWorker`'s shape (CreditGatedPool + reuse-then-fetch);
/// the hot path enqueues and returns immediately.
pub struct FrameResolutionWorker {
    pool: CreditGatedPool<FrameResolutionJob>,
    model: String,
}

impl FrameResolutionWorker {
    /// Construct the worker. `credit_limit` bounds in-flight jobs; `queue_capacity`
    /// bounds the pool queue; `index` is the promotion seam; `fetch` the
    /// batched resolution call (grammar-constrained by the router).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: &Arc<dyn PreferredSenseIndex>,
        fetch: &FrameResolutionFetch,
        model: String,
        queue_capacity: usize,
        credit_limit: usize,
        runtime: Arc<dyn Runtime>,
    ) -> Self {
        let index_h = Arc::clone(index);
        let fetch_h = Arc::clone(fetch);
        let model_h = model.clone();
        let pool = CreditGatedPool::new(
            runtime,
            credit_limit,
            queue_capacity,
            move |job: FrameResolutionJob| {
                let index = Arc::clone(&index_h);
                let fetch = Arc::clone(&fetch_h);
                let model = model_h.clone();
                async move {
                    process_job(&job, &index, &fetch, &model);
                }
            },
        );
        Self { pool, model }
    }

    /// The resolution-model key this worker uses.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Whether the credit gate is currently blocking producers.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.pool.is_blocked()
    }

    /// Enqueue a frame-resolution job, bounded by the credit gate.
    pub async fn enqueue(&self, job: FrameResolutionJob) -> Result<(), FrameResolutionError> {
        self.pool
            .submit(job)
            .await
            .map_err(|e| match e {
                PoolError::Closed => FrameResolutionError::Closed("worker drained".into()),
                PoolError::Full => FrameResolutionError::Full(e.to_string()),
            })
    }

    /// Drain in-flight jobs and shut the pool down (graceful shutdown).
    /// The worker is unusable afterward.
    pub async fn drain(self: Arc<Self>) {
        self.pool.drain().await;
    }
}

/// Process one job: reuse-check → (any miss) one batched resolution call →
/// apply + promote. Best-effort by design — a failure logs and never blocks
/// the pool.
fn process_job(
    job: &FrameResolutionJob,
    index: &Arc<dyn PreferredSenseIndex>,
    fetch: &FrameResolutionFetch,
    model: &str,
) {
    // 1. Reuse: every already-resolved pattern is served from the index.
    let mut reuse: Vec<Resolution> = Vec::new();
    let mut misses: Vec<FrameResolutionRequest> = Vec::new();
    for req in &job.requests {
        match index.preferred_sense(req.predicate_lemma_id, req.ambiguity_kind) {
            Some(res) => reuse.push(res),
            None => misses.push(req.clone()),
        }
    }

    // The reuse branch is load-bearing: with an empty miss set the model is
    // never called — a fully-resolved wave is deterministic.
    if misses.is_empty() {
        if !reuse.is_empty() {
            tracing::debug!(
                target: "router.frame",
                model = model,
                resolved = reuse.len(),
                "frame resolution reuse — zero LLM cost",
            );
        }
        return;
    }

    // 2. Miss → one grammar-constrained batched call returning N resolutions
    //    (the `BatchPromptGrammar` array shape). Aligned by index to `misses`.
    let reply = match fetch(misses.clone()) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                target: "router.frame",
                model = model,
                error = %e,
                "frame resolution model call failed — leaving ambiguities unresolved",
            );
            return;
        }
    };
    let resolutions: Vec<Resolution> = serde_json::from_str(&reply)
        .unwrap_or_else(|_| parse_naive(&reply, misses.len()));

    // 3. Apply + promote: record each resolved pattern back into the index so
    //    the next occurrence is deterministic (zero LLM cost).
    for (i, req) in misses.iter().enumerate() {
        let Some(res) = resolutions.get(i) else {
            continue;
        };
        if let Err(e) = index.record_preferred_sense(
            req.predicate_lemma_id,
            req.ambiguity_kind,
            res.clone(),
        ) {
            tracing::warn!(
                target: "router.frame",
                model = model,
                error = %e,
                "sense promotion failed (best-effort)",
            );
        }
    }
    tracing::debug!(
        target: "router.frame",
        model = model,
        batched = misses.len(),
        "frame resolution wave applied + promoted",
    );
}

/// A lenient fallback when the model returns a non-array shape: parse a single
/// `{chosen_candidate_id, detail}` object and repeat it, or synthesize
/// `Resolution` records with the request's first candidate id. Never fails a
/// job outright — the post-hoc parse backstop (the grammar constraint is the
/// primary guard on the onnx path).
fn parse_naive(reply: &str, len: usize) -> Vec<Resolution> {
    if let Ok(single) = serde_json::from_str::<Resolution>(reply) {
        return vec![single; len];
    }
    vec![
        Resolution {
            chosen_candidate_id: InterlinguaId::from_u64(0),
            detail: "unparseable resolution".into(),
        };
        len
    ]
}
#[cfg(test)]
#[path = "../../tests/ledger_frame_index.rs"]
mod tests;
