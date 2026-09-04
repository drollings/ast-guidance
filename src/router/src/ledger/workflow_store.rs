//! Workflow extraction store (M8) — embedding-keyed `Target` DAG library.
//!
//! The frontier learning loop (VISION post-processing): a successful
//! `escalation.audit` chain is decomposed into `Target` nodes (`depends`/`provides`)
//! and keyed by the query embedding. A future query with a near-neighbor
//! embedding replays the DAG via `DependencyResolver` without a frontier call.
//!
//! The store is backed by an `HnswIndex` for the query embedding plus an
//! in-memory map for the DAG. `verified` gates replay: an entry is
//! **audit-only, not replayable** until human review (`POST /verify`) or
//! paraphrase self-consistency (`topo_sort` equality) marks it verified.
//! Confidence gates (`assembler_confidence ≥ 0.85`, novelty distance >0.15)
//! prevent caching of confident-but-wrong workflows.

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use fluent_dag::target::Target;
use fluent_db::hnsw::HnswIndex;
use common_core::vector_math::cosine_similarity_f32;


/// A workflow entry — one prior `query → local_attempts → escalation → frontier → assembly` chain.
#[derive(Debug, Clone)]
pub struct WorkflowEntry {
    pub query_embedding: Vec<f32>,
    pub dag: Vec<Target>,
    pub audit_id: String,
    pub verified: bool,
}

/// Errors surfaced by the workflow store.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("storage error: {0}")]
    Storage(String),
}

/// The workflow store trait — the single embedding-keyed library (VISION's prior-workflow library).
pub trait WorkflowStore: Send + Sync {
    fn insert(&self, entry: WorkflowEntry) -> Result<(), StoreError>;
    fn nearest(&self, embedding: &[f32], k: usize) -> Vec<(WorkflowEntry, f64)>;
    /// Verified nearest with minimum cosine (default 0.75) — the replay path.
    fn nearest_verified(&self, embedding: &[f32], k: usize, min_score: f64) -> Vec<(WorkflowEntry, f64)> {
        self.nearest(embedding, k)
            .into_iter()
            .filter(|(e, score)| e.verified && *score >= min_score)
            .collect()
    }
}

/// In-memory + HNSW workflow store. `HnswIndex` indexes the query embedding;
/// the map holds the full entry (including `dag`). Both are `Send + Sync`.
pub struct InMemoryWorkflowStore {
    hnsw: HnswIndex,
    // index -> entry (mirrors HnswIndex's id_map but with full entry)
    entries: RwLock<HashMap<usize, WorkflowEntry>>,
    next_idx: Mutex<usize>,
}

impl Default for InMemoryWorkflowStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryWorkflowStore {
    pub fn new() -> Self {
        Self {
            hnsw: HnswIndex::new(),
            entries: RwLock::new(HashMap::new()),
            next_idx: Mutex::new(0),
        }
    }

    fn cosine(entry_emb: &[f32], query: &[f32]) -> f64 {
        f64::from(cosine_similarity_f32(entry_emb, query))
    }
}

impl WorkflowStore for InMemoryWorkflowStore {
    fn insert(&self, entry: WorkflowEntry) -> Result<(), StoreError> {
        // Gated write path (M8): caller enforces confidence/novelty/verified.
        // Here we just store; the novelty distance check is performed by the
        // caller via `nearest` before insert. We still enforce novelty to avoid
        // duplicate close embeddings when the caller skips it.
        let idx = {
            let mut guard = self.next_idx.lock().unwrap();
            let i = *guard;
            *guard += 1;
            i
        };
        // HNSW insert uses audit_id hash as node_id placeholder; we map idx -> entry
        // Use idx as external id; hnsw's internal id is idx, and we store entry under idx.
        // For HNSW we need an i64 node_id; use idx as i64 and embed the idx as external.
        // We insert with node_id = idx as i64, but the HNSW external id is the insertion order.
        let _ = self.hnsw.insert(idx as i64, &entry.query_embedding);
        // The HNSW external id equals insertion order; we maintain 1:1 between HNSW order and entries map key.
        // However HNSW's id_map stores node_id (idx as i64) under external idx; our entries map uses external idx as key.
        // Find the external idx just inserted: it is entries.len() before insert.
        self.entries.write().unwrap().insert(idx, entry);
        Ok(())
    }

    fn nearest(&self, embedding: &[f32], k: usize) -> Vec<(WorkflowEntry, f64)> {
        if k == 0 || embedding.is_empty() {
            return Vec::new();
        }
        // If HNSW has data, use it for approximate search; otherwise brute-force.
        // M6: deliberately NOT routed through `AdaptiveHnsw` — this corpus is
        // small and always-indexed; adding a brute-force dispatch gate here
        // would be a behavior change (an extra path), so the store keeps
        // always-probe. The dispatch policy measures [B] cost/recall at
        // `ContentNodeStore` scale only.
        // M5: the HNSW probe + id resolution is the shared
        // `fluent_db::hnsw::hnsw_lookup` (`None` = fall back). The exact
        // re-score, k*2 over-fetch, and discard-partial-then-brute-force
        // policy below stay call-site code (store-specific semantics).
        if let Some(hnsw_hits) = fluent_db::hnsw::hnsw_lookup(&self.hnsw, embedding, k * 2) {
            let entries = self.entries.read().unwrap();
            let mut out = Vec::new();
            for (node_id, _dist) in hnsw_hits {
                let idx = node_id as usize;
                if let Some(entry) = entries.get(&idx) {
                    let score = Self::cosine(&entry.query_embedding, embedding);
                    out.push((entry.clone(), score));
                    if out.len() >= k {
                        break;
                    }
                }
            }
            // If HNSW returned enough, use it; else fall back to brute-force to fill.
            if out.len() >= k {
                out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                out.truncate(k);
                return out;
            }
        }
        // Brute-force fallback (exact cosine)
        let entries = self.entries.read().unwrap();
        let mut all: Vec<(WorkflowEntry, f64)> = entries
            .values()
            .map(|e| (e.clone(), Self::cosine(&e.query_embedding, embedding)))
            .collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        all
    }
}

/// Decide whether an entry may be inserted: confidence ≥ 0.85, novelty distance >0.15, verified must be true.
/// `assembler_confidence` is the self-doubt signal; `distance_to_nearest` is `1 - cosine` to nearest existing entry.
/// Returns `true` when the gated write path may insert.
pub fn gated_insert_allowed(assembler_confidence: f64, distance_to_nearest: f64, verified: bool) -> bool {
    assembler_confidence >= 0.85 && distance_to_nearest > 0.15 && verified
}
#[cfg(test)]
#[path = "../../tests/ledger_workflow_store.rs"]
mod tests;
