//! ContentNodeStore — the shared, reference-counted, interned, durable ContentNode
//! store.
//!
//! This is the surgical successor to `ContentNodeLedger`'s per-process
//! `SqliteStore`+`Mutex<i64>` storage: nodes live once, behind
//! `Arc<RwLock<ContentNode>>`, so every holder (parallel/filtered ledger
//! views in later milestones) shares the same node object and a lazily-derived
//! LOD tier is computed **at most once** and visible to all holders (VISION).
//! Session and role indices are keyed by `ArcIntern<str>` so sharing a node
//! across N views costs a refcount bump, not a string copy.
//!
//! Durability is preserved: every mutation writes the canonical `content_json`
//! column in the existing `ledger` table (schema + migrations owned by
//! `crate::ledger`), and the maps are **hydrated on open**. Reads prefer the
//! in-memory maps.
//!
//! The LOD lifecycle (LOD0/LOD5 eager, LOD1–LOD4 lazy from LOD0 only via the
//! `Summarizer`, recency compaction) is ported here unchanged from the ledger.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use common_core::sync::{lock, lock_read, lock_write};
use fluent_db::error::DbError;
use fluent_db::hnsw::AdaptiveHnsw;
use fluent_db::migrate::migrate;
use fluent_db::store::SqliteStore;
use fluent_db::vector::knn_brute_force;
use fluent_llm::client::ChatBackend;
use fluent_llm::EmbeddingProvider;
use fluent_types::{AnnotationClaim, ClaimStatus, ContentNode, KnnHit, NodeId, OriginRole, OverlayKind, OverlayStatus};
use fluent_wvr::ArcIntern;
use rusqlite::Connection;

use crate::ledger::correction_index::{upsert_correction_row, CorrectionRow};
use crate::ledger::annotations::AnnotationStore;
use crate::ledger::{
    CompactionStrategy, LedgerEntry, LedgerError, RecencyCompaction, LAZY_LOD_RANGE,
    LOD0_FULL_TEXT, LOD5_LABEL,
};
use crate::ledger_guard::scrub_for_ledger;
use crate::summarization::Summarizer;

/// The shared ContentNode store: refcounted nodes + interned indices + durable
/// backing.
pub struct ContentNodeStore {
    /// Node id → shared node. The `Arc<RwLock<ContentNode>>` is the sharing
    /// primitive: LOD derivation mutates the shared node, so every holder
    /// observes it.
    nodes: RwLock<HashMap<NodeId, Arc<RwLock<ContentNode>>>>,
    /// session_id (interned) → node ids in insertion order.
    by_session: RwLock<HashMap<ArcIntern<str>, Vec<NodeId>>>,
    /// role (interned) → node ids in insertion order.
    by_role: RwLock<HashMap<ArcIntern<str>, Vec<NodeId>>>,
    /// Monotonic id allocator. Seeded from `MAX(node_id) + 1` at hydration so
    /// a restart never re-issues ids that collide with persisted rows.
    next_id: AtomicI64,
    /// The existing `ledger` table. `None` for a pure in-memory store that
    /// skips durability.
    ///
    /// NOTE (M9 decision): this stays a single-connection `SqliteStore`,
    /// not a `SqlitePool`. The ledger sub-stores (annotations, correction
    /// index, concept store) share this one `Arc<SqliteStore>` for
    /// single-writer semantics and atomic cross-view updates; the store API
    /// is sync (served off async paths via `spawn_blocking`); and pool
    /// checkout would require a `DbCapability` token the serving path does
    /// not install. Statements compose the canonical `SqliteStore`
    /// typed helpers (`query_row`/`query_rows`/`with_conn`).
    durable: Option<Arc<SqliteStore>>,
    /// Lazy LOD derivation. `Mutex` for interior mutability so the summarizer
    /// can be attached after the store is `Arc`-shared
    /// (`ContentNodeLedger::with_summarizer`).
    summarizer: Mutex<Option<Summarizer>>,
    /// Optional tier-event feed: a sender the background
    /// `LedgerTierWorker` drains to fill LOD4/LOD5. `None` (the default) leaves
    /// today's behavior — a store with no attached worker is byte-identical to
    /// before. `Mutex` for interior mutability so it can be attached after the
    /// store is `Arc`-shared. The bounded channel (not an unbounded one) is the
    /// memory bound for a burst of writes faster than the worker can derive.
    tier_events: Mutex<Option<tokio::sync::mpsc::Sender<NodeId>>>,
    /// Optional spacy pipeline for the spacy overlay (`annotation`). `None`
    /// (the default) is fail-open: `annotation_for` returns `Ok(None)`.
    overlay_pipeline: Mutex<Option<Arc<spacy_rs::NlpPipeline>>>,
    /// Optional embedder for the embedding overlay (`embedding`). `None`
    /// (the default) is fail-open: `embedding_for` returns `Ok(None)`.
    overlay_embedder: Mutex<Option<Arc<dyn EmbeddingProvider>>>,
    /// Optional LLM backend for the LLM overlay (`metadata["llm_overlay"]`).
    /// `None` (the default) is fail-open: `llm_overlay_for` returns `Ok(None)`.
    overlay_llm: Mutex<Option<Arc<dyn ChatBackend>>>,
    /// Optional overlay-event feed: a sender the background
    /// `OverlayWorker` drains to derive the three overlays in parallel. `None`
    /// (the default) leaves today's behavior.
    overlay_events: Mutex<Option<tokio::sync::mpsc::Sender<NodeId>>>,
    /// Nodes still missing at least one overlay — maintained incrementally
    /// (insert on write, remove on successful derive) so boot backfill and
    /// enqueue checks are O(pending) not O(N) (Plan B F1).
    overlay_pending: RwLock<HashSet<NodeId>>,
    /// Optional HNSW index for adaptive dispatch (M5). `None` until
    /// `|store| > hnsw_threshold` (default 512, `DEFAULT_HNSW_THRESHOLD`).
    /// When `Some`, `knn_search` routes through HNSW (approximate,
    /// recall≥0.95); otherwise it uses brute-force (`knn_brute_force`).
    hnsw: RwLock<Option<Arc<fluent_db::hnsw::HnswIndex>>>,
    /// Adaptive HNSW-vs-brute-force dispatch policy (M6) — the single
    /// threshold source (`DEFAULT_HNSW_THRESHOLD`). [B] cost/recall only.
    hnsw_policy: AdaptiveHnsw,
}

/// The `metadata` key under which the LLM enrichment overlay lives.
pub const LLM_OVERLAY_META_KEY: &str = "llm_overlay";

/// The system prompt for the one-call LLM enrichment overlay.
const LLM_OVERLAY_SYSTEM_PROMPT: &str =
    "Write one concise sentence summarizing the user's message for later retrieval. \
     Reply with only the summary text, no labels.";

impl ContentNodeStore {
    /// Open (or create) the durable store at `path`, run the ledger schema
    /// migrations, and hydrate the in-memory maps from every row.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, LedgerError> {
        let db_path = path.into();
        let store = Arc::new(SqliteStore::open(&db_path).map_err(|e| LedgerError::Db(e.to_string()))?);
        Self::open_with_store(Some(store))
    }

    /// Open an in-memory (non-durable-across-processes) store. Used by tests
    /// and ephemeral ledgers.
    pub fn open_in_memory() -> Result<Self, LedgerError> {
        let store = Arc::new(
            SqliteStore::open_in_memory().map_err(|e| LedgerError::Db(e.to_string()))?,
        );
        Self::open_with_store(Some(store))
    }

    /// A store with no durable backing at all (pure in-memory, no ledger
    /// table). Writes are memory-only.
    pub fn ephemeral() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            by_session: RwLock::new(HashMap::new()),
            by_role: RwLock::new(HashMap::new()),
            next_id: AtomicI64::new(1),
            durable: None,
            summarizer: Mutex::new(None),
            tier_events: Mutex::new(None),
            overlay_pipeline: Mutex::new(None),
            overlay_embedder: Mutex::new(None),
            overlay_llm: Mutex::new(None),
            overlay_events: Mutex::new(None),
            overlay_pending: RwLock::new(HashSet::new()),
            hnsw: RwLock::new(None),
            hnsw_policy: AdaptiveHnsw::default(),
        }
    }

    fn open_with_store(durable: Option<Arc<SqliteStore>>) -> Result<Self, LedgerError> {
        if let Some(ref store) = durable {
            let migrations = crate::ledger::ledger_migrations();
            store
                .with_conn(|conn| migrate(conn, &migrations))
                .map_err(|e| LedgerError::Db(e.to_string()))?;
        }
        let store_obj = Self {
            nodes: RwLock::new(HashMap::new()),
            by_session: RwLock::new(HashMap::new()),
            by_role: RwLock::new(HashMap::new()),
            next_id: AtomicI64::new(1),
            durable,
            summarizer: Mutex::new(None),
            tier_events: Mutex::new(None),
            overlay_pipeline: Mutex::new(None),
            overlay_embedder: Mutex::new(None),
            overlay_llm: Mutex::new(None),
            overlay_events: Mutex::new(None),
            overlay_pending: RwLock::new(HashSet::new()),
            hnsw: RwLock::new(None),
            hnsw_policy: AdaptiveHnsw::default(),
        };
        store_obj.hydrate()?;
        Ok(store_obj)
    }

    /// Attach a `Summarizer` so LOD1–LOD4 can be derived lazily from LOD0.
    /// Without one, `ensure_lod` returns `LedgerError::NoSummarizer`. Interior
    /// mutable so it can be attached after the store is `Arc`-shared.
    #[must_use]
    pub fn with_summarizer(self, summarizer: Summarizer) -> Self {
        self.set_summarizer(summarizer);
        self
    }

    /// Set the summarizer on an already-shared store (the facade's
    /// `with_summarizer` uses this through the `Arc`).
    pub fn set_summarizer(&self, summarizer: Summarizer) {
        *lock(&self.summarizer) = Some(summarizer);
    }

    /// Attach a tier-event sender. When set, the canonical write paths
    /// (`insert_node`, `record_result`) enqueue any node whose LOD4/LOD5 is
    /// empty so the background `LedgerTierWorker` can fill them. A store with
    /// no sender keeps today's behavior (opt-in).
    pub fn set_tier_events(&self, sender: tokio::sync::mpsc::Sender<NodeId>) {
        *lock(&self.tier_events) = Some(sender);
    }

    /// Attach the spacy pipeline seam for the spacy overlay. `None` (the
    /// default) is fail-open: `annotation_for` returns `Ok(None)`.
    pub fn set_overlay_pipeline(&self, pipeline: Arc<spacy_rs::NlpPipeline>) {
        *lock(&self.overlay_pipeline) = Some(pipeline);
    }

    /// Attach the embedder seam for the embedding overlay. `None` (the
    /// default) is fail-open: `embedding_for` returns `Ok(None)`.
    pub fn set_overlay_embedder(&self, embedder: Arc<dyn EmbeddingProvider>) {
        *lock(&self.overlay_embedder) = Some(embedder);
    }

    /// Attach the LLM backend seam for the LLM enrichment overlay. `None` (the
    /// default) is fail-open: `llm_overlay_for` returns `Ok(None)`.
    pub fn set_overlay_llm(&self, backend: Arc<dyn ChatBackend>) {
        *lock(&self.overlay_llm) = Some(backend);
    }

    /// Attach an overlay-event sender. When set, the canonical write paths
    /// enqueue any node missing an overlay so the background `OverlayWorker`
    /// can derive the three overlays in parallel. A store with no sender keeps
    /// today's behavior (opt-in).
    pub fn set_overlay_events(&self, sender: tokio::sync::mpsc::Sender<NodeId>) {
        *lock(&self.overlay_events) = Some(sender);
    }

    /// Whether a node still needs any overlay derived (no annotation, no
    /// embedding, and no `llm_overlay`).
    pub fn needs_overlay(&self, node_id: NodeId) -> bool {
        self.get_node(node_id).is_some_and(|arc| {
            let guard = lock_read(&arc);
            guard.annotation.is_none()
                || guard.embedding.is_none()
                || guard
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get(LLM_OVERLAY_META_KEY))
                    .is_none()
        })
    }

    /// Enqueue `node_id` on the overlay-event feed if it is missing any overlay
    /// and a sender is attached. No-op when no sender is attached; a full feed
    /// skips the enqueue (the boot backfill catches stragglers). Maintains
    /// `overlay_pending` so the backfill stays O(pending) (Plan B F1).
    fn enqueue_if_needs_overlay(&self, node_id: NodeId) {
        let sender = lock(&self.overlay_events).clone();
        if let Some(sender) = sender {
            if self.needs_overlay(node_id) {
                // Track pending before the try_send so a full channel still
                // leaves the node in the pending set for the next backfill.
                lock_write(&self.overlay_pending).insert(node_id);
                match sender.try_send(node_id) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                        tracing::debug!(
                            target: "router.ledger.overlay",
                            node_id = node_id.as_int(),
                            "overlay feed full - skipping sync enqueue",
                        );
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::warn!(
                            target: "router.ledger.overlay",
                            node_id = node_id.as_int(),
                            "overlay feed closed - skipping sync enqueue",
                        );
                    }
                }
            }
        }
    }

    /// All node ids missing at least one overlay (boot backfill). Returns the
    /// incrementally-maintained `overlay_pending` snapshot — O(pending), not
    /// O(N) (Plan B F1). Falls back to a scan only when the pending set is
    /// empty but nodes exist (e.g. ephemeral store without hydrate).
    pub fn node_ids_needing_overlays(&self) -> Vec<NodeId> {
        let pending: Vec<NodeId> = lock_read(&self.overlay_pending).iter().copied().collect();
        if !pending.is_empty() {
            // Filter stale entries (nodes whose overlays have since become ready/failed)
            // and prune the set so a second backfill is idempotent.
            let original_len = pending.len();
            let filtered: Vec<NodeId> = pending.into_iter().filter(|id| self.needs_overlay(*id)).collect();
            // Prune stale entries from the live set when any were filtered
            if filtered.len() != original_len {
                let mut guard = lock_write(&self.overlay_pending);
                guard.retain(|id| self.needs_overlay(*id));
            }
            if !filtered.is_empty() {
                return filtered;
            }
            // All pending entries are now satisfied — fall through to empty (idempotent)
            return Vec::new();
        }
        // Fallback for stores that never hydrated (e.g. ephemeral in tests):
        // scan once and populate pending.
        let ids: Vec<NodeId> = lock_read(&self.by_session)
            .values()
            .flatten()
            .copied()
            .collect();
        let mut out: Vec<NodeId> = Vec::new();
        for id in ids {
            if self.needs_overlay(id) {
                out.push(id);
            }
        }
        if !out.is_empty() {
            lock_write(&self.overlay_pending).extend(out.iter().copied());
        }
        out
    }

    /// The shared at-most-once overlay derivation (OVERLAYS §6), used by
    /// [`Self::annotation_for`] / [`Self::embedding_for`] / [`Self::llm_overlay_for`]:
    ///
    /// 1. Read-guard fast path: a cached value (or a **permanent** `failed`
    ///    status — never retried) returns immediately.
    /// 2. Snapshot LOD0 (the sole derivation source — never another overlay)
    ///    and mark `pending` (advisory).
    /// 3. Derive **off-node** — no guard is held across the model call / parse.
    /// 4. On a permanent derivation error: mark `status: failed` and return
    ///    `Ok(None)` (fail-open, no retry loop).
    /// 5. Re-acquire the write lock, **re-check** (a concurrent worker may have
    ///    won), install the winner and mark `ready` — the re-check is what
    ///    makes installation at-most-once under concurrency.
    ///
    /// `cached` reads the overlay's value slot, `derive` produces it from the
    /// node's LOD0 text, `install` writes it back.
    fn derive_overlay<T>(
        &self,
        node_id: NodeId,
        kind: OverlayKind,
        source: &str,
        cached: impl Fn(&ContentNode) -> Option<T>,
        derive: impl FnOnce(&str) -> Result<T, String>,
        install: impl FnOnce(&mut ContentNode, T),
    ) -> Result<Option<T>, LedgerError> {
        let arc = self.get_node(node_id).ok_or(LedgerError::NotFound(node_id))?;

        // 1. Read-guard fast path.
        {
            let guard = lock_read(&arc);
            if let Some(v) = cached(&guard) {
                return Ok(Some(v));
            }
            if guard.overlay(kind).status == OverlayStatus::Failed {
                return Ok(None);
            }
        }

        // 2. Snapshot LOD0 and mark `pending` (advisory, best-effort).
        let text = {
            let guard = lock_read(&arc);
            guard.lod.first().cloned().unwrap_or_default()
        };
        if let Some(arc) = self.get_node(node_id) {
            let mut guard = lock_write(&arc);
            let _ = guard.transition_overlay(kind, OverlayStatus::Pending, source, None);
        }

        // 3. Derive off-node — no guard held across the model call / parse.
        let derived = match derive(&text) {
            Ok(v) => v,
            Err(error) => {
                // 4. Permanent failure: mark `failed`, fail-open, no retry.
                tracing::warn!(
                    target: "router.ledger.overlay",
                    node_id = node_id.as_int(),
                    kind = ?kind,
                    %error,
                    "overlay derivation failed - marking failed (fail-open)"
                );
                let _ = self.with_node_mut(node_id, |node| {
                    let _ = node.transition_overlay(
                        kind,
                        OverlayStatus::Failed,
                        source,
                        Some(common_core::now_secs()),
                    );
                });
                if !self.needs_overlay(node_id) {
                    lock_write(&self.overlay_pending).remove(&node_id);
                }
                return Ok(None);
            }
        };

        // 5. At-most-once install: re-check under the write lock, then install.
        let mut installed: Option<T> = None;
        self.with_node_mut(node_id, |node| {
            if let Some(v) = cached(node) {
                installed = Some(v);
                return;
            }
            install(node, derived);
            let _ = node.transition_overlay(
                kind,
                OverlayStatus::Ready,
                source,
                Some(common_core::now_secs()),
            );
            installed = cached(node);
        })?;
        if !self.needs_overlay(node_id) {
            lock_write(&self.overlay_pending).remove(&node_id);
        }
        Ok(installed)
    }

    /// Get the shared `NodeAnnotation`, computing it lazily (**at most
    /// once**) if absent and a pipeline is wired. `Ok(None)` when no pipeline is
    /// attached (fail-open), the node's overlay is permanently `failed` (no
    /// retry), or derivation failed. Locking: off-node derivation, then an
    /// at-most-once write-lock install (OVERLAYS §6).
    pub fn annotation_for(
        &self,
        node_id: NodeId,
    ) -> Result<Option<Arc<crate::ledger::node_annotation::NodeAnnotation>>, LedgerError> {
        let Some(pipeline) = lock(&self.overlay_pipeline).clone() else {
            return Ok(None); // fail-open: no pipeline wired
        };
        self.derive_overlay(
            node_id,
            OverlayKind::Spacy,
            "spacy",
            |node| {
                node.annotation.clone().and_then(
                    <dyn fluent_types::NodeOverlay>::downcast_arc::<
                        crate::ledger::node_annotation::NodeAnnotation,
                    >,
                )
            },
            move |text| {
                let (doc, result) = pipeline
                    .process_sync_with_confidence(text, None, None, spacy_rs::RefinePolicy::default())
                    .map_err(|e| e.to_string())?;
                Ok(Arc::new(crate::ledger::node_annotation::node_annotation(&doc, &result)))
            },
            |node, ann| {
                node.annotation = Some(ann);
            },
        )
    }

    /// Get the node's dense embedding, computing it lazily (**at most once**)
    /// if absent and an embedder is wired. `Ok(None)` when no embedder is
    /// attached (fail-open), the overlay is permanently `failed` (no retry), or
    /// derivation failed.
    pub fn embedding_for(&self, node_id: NodeId) -> Result<Option<Vec<f32>>, LedgerError> {
        let Some(embedder) = lock(&self.overlay_embedder).clone() else {
            return Ok(None); // fail-open: no embedder wired
        };
        let name = embedder.name().to_string();
        let result = self.derive_overlay(
            node_id,
            OverlayKind::Embedding,
            &name,
            |node| node.embedding.clone(),
            move |text| embedder.embed(text).map_err(|e| e.to_string()),
            |node, emb| {
                node.embedding = Some(emb);
            },
        )?;
        if let Some(ref emb) = result {
            self.sync_hnsw_insert(node_id, emb);
        }
        Ok(result)
    }

    /// Get the node's LLM enrichment overlay (`metadata["llm_overlay"]`),
    /// computing it lazily (**at most once**) if absent and a backend is wired.
    /// `Ok(None)` when no backend is attached (fail-open), the overlay is
    /// permanently `failed` (no retry), or derivation failed. The one-call
    /// summary is scrubbed through the same `scrub_for_ledger` gate as every
    /// other ledger write before install (OVERLAYS §9).
    pub fn llm_overlay_for(&self, node_id: NodeId) -> Result<Option<serde_json::Value>, LedgerError> {
        let Some(backend) = lock(&self.overlay_llm).clone() else {
            return Ok(None); // fail-open: no backend wired
        };
        self.derive_overlay(
            node_id,
            OverlayKind::Llm,
            "llm",
            |node| {
                node.metadata
                    .as_ref()
                    .and_then(|m| m.get(LLM_OVERLAY_META_KEY).cloned())
            },
            move |text| {
                let reply = backend
                    .chat_complete(&[
                        fluent_llm::ChatMessage {
                            role: "system".into(),
                            content: LLM_OVERLAY_SYSTEM_PROMPT.into(),
                        },
                        fluent_llm::ChatMessage {
                            role: "user".into(),
                            content: text.to_string(),
                        },
                    ])
                    .map_err(|e| e.to_string())?;
                let scrubbed = scrub_for_ledger(&reply).text;
                Ok(serde_json::json!(scrubbed))
            },
            |node, value| {
                let meta = node.metadata.get_or_insert_with(|| serde_json::json!({}));
                meta[LLM_OVERLAY_META_KEY] = value;
            },
        )
    }

    /// Enqueue `node_id` on the tier-event feed if its LOD4 or LOD5 is empty
    /// and a sender is attached. No-op when no sender is attached. The
    /// synchronous write path cannot await a credit bump, so it uses the
    /// bounded channel's non-blocking `try_send`; a full feed skips the
    /// enqueue (agent nodes are still covered by `run_agent`'s credit-gated
    /// `enqueue_with_credit`, and the next boot backfill catches stragglers).
    fn enqueue_if_needs_tier(&self, node_id: NodeId) {
        let sender = lock(&self.tier_events).clone();
        if let Some(sender) = sender {
            if self.needs_tier(node_id) {
                match sender.try_send(node_id) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                        tracing::debug!(
                            target: "router.ledger.tier",
                            node_id = node_id.as_int(),
                            "tier feed full - skipping sync enqueue",
                        );
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        tracing::warn!(
                            target: "router.ledger.tier",
                            node_id = node_id.as_int(),
                            "tier feed closed - skipping sync enqueue",
                        );
                    }
                }
            }
        }
    }

    /// Whether a node still needs background LOD4/LOD5 derivation (its LOD4 or
    /// LOD5 tier is empty).
    pub fn needs_tier(&self, node_id: NodeId) -> bool {
        self.get_node(node_id).is_some_and(|arc| {
            let guard = lock_read(&arc);
            let lod4_empty = guard.lod.get(4).is_none_or(String::is_empty);
            let lod5_empty = guard.lod.get(5).is_none_or(String::is_empty);
            lod4_empty || lod5_empty
        })
    }

    /// All node ids whose given tiers are empty (boot backfill / worker).
    /// Iterates the interned session index for the id list (no full node-scan
    /// Arc clones), then checks each node's tiers under a short read guard.
    pub fn node_ids_needing_tier(&self, levels: &[u8]) -> Vec<NodeId> {
        let ids: Vec<NodeId> = lock_read(&self.by_session)
            .values()
            .flatten()
            .copied()
            .collect();
        let mut out: Vec<NodeId> = Vec::new();
        for id in ids {
            let needs = self.get_node(id).is_some_and(|arc| {
                let guard = lock_read(&arc);
                levels
                    .iter()
                    .any(|l| guard.lod.get(*l as usize).is_none_or(String::is_empty))
            });
            if needs {
                out.push(id);
            }
        }
        out
    }

/// Access the durable backing (for the facade's flat view, the poison test
/// helper, and the shared `SqliteConceptStore`/`SqliteCorrectionIndex` — all
/// three share one connection). `None` for an ephemeral store.
pub(crate) fn durable(&self) -> Option<&Arc<SqliteStore>> {
    self.durable.as_ref()
}

/// A clone of the shared durable connection, so the `SqliteConceptStore` and
/// `SqliteCorrectionIndex` operate over the *same* connection as the ledger
/// (one connection, many typed views — atomic corrections, §12.6). The binary
/// boot composes these at startup.
pub fn shared_sqlite(&self) -> Option<Arc<SqliteStore>> {
    self.durable().cloned()
}

/// Latest `nlp_parse` node id for a session (M9: the single store-owned
/// spelling of the parse-lookup query; the HTTP handler calls this instead
/// of inline SQL).
pub fn latest_parse_node_id(&self, session_id: &str) -> Option<NodeId> {
    let store = self.shared_sqlite()?;
    let row = store
        .query_row(
            "SELECT node_id FROM ledger \
             WHERE session_id = ?1 AND json_extract(metadata, '$.kind') = 'nlp_parse' \
             ORDER BY node_id DESC LIMIT 1",
            rusqlite::params![session_id],
            |r| r.get::<_, i64>(0),
        )
        .ok()??;
    Some(NodeId::from_int(row))
}

    /// Load every persisted row into the maps, seeding `next_id` from
    /// `MAX(node_id) + 1` (pre-existing restart-collision bug fix).
    fn hydrate(&self) -> Result<(), LedgerError> {
        let Some(ref store) = self.durable else {
            return Ok(());
        };
        let rows: Vec<(i64, String)> = store
            .query_rows("SELECT node_id, content_json FROM ledger", &[], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .map_err(|e| LedgerError::Db(e.to_string()))?;

        let mut max_id = 0i64;
        {
            let mut nodes = lock_write(&self.nodes);
            let mut by_session = lock_write(&self.by_session);
            let mut by_role = lock_write(&self.by_role);
            for (id, json) in rows {
                let Ok(node) = serde_json::from_str::<ContentNode>(&json) else {
                    continue;
                };
                let node_id = NodeId::from_int(id);
                let mut node = node;
                // Rows written before the `content_hash` field default to 0;
                // stamp the hash from LOD0 so the annotation keying domain is
                // correct even for pre-M4 rows (ROADMAP M4).
                node.content_hash = content_hash_of(node.lod.get(LOD0_FULL_TEXT as usize).map(String::as_str).unwrap_or_default());
                let session_key = node.session_id.as_deref().map(ArcIntern::from);
                let role_key = node.role.as_ref().map(|r| ArcIntern::from(r.as_str()));
                max_id = max_id.max(id);
                nodes.insert(node_id, Arc::new(RwLock::new(node)));
                if let Some(key) = session_key {
                    by_session.entry(key).or_default().push(node_id);
                }
                if let Some(key) = role_key {
                    by_role.entry(key).or_default().push(node_id);
                }
            }
        }
        self.next_id.store(max_id + 1, Ordering::SeqCst);
        // Populate overlay_pending incrementally — O(N) once at boot, then
        // O(pending) for backfill and O(1) for enqueue checks (Plan B F1).
        let pending: HashSet<NodeId> = {
            let nodes = lock_read(&self.nodes);
            nodes
                .iter()
                .filter_map(|(&id, arc)| {
                    let guard = lock_read(arc);
                    let needs = guard.annotation.is_none()
                        || guard.embedding.is_none()
                        || guard
                            .metadata
                            .as_ref()
                            .and_then(|m| m.get(LLM_OVERLAY_META_KEY))
                            .is_none();
                    // Nodes already marked failed are not pending — derive_overlay
                    // short-circuits them and they should not be backfilled.
                    let failed = [
                        OverlayKind::Spacy,
                        OverlayKind::Embedding,
                        OverlayKind::Llm,
                    ]
                    .iter()
                    .all(|k| guard.overlay(*k).status == OverlayStatus::Failed);
                    if needs && !failed { Some(id) } else { None }
                })
                .collect()
        };
        *lock_write(&self.overlay_pending) = pending;
        // M5: lazily build HNSW if the hydrated store already exceeds the
        // threshold (so a restart with a large ledger does not stay brute-force).
        self.maybe_init_hnsw();
        Ok(())
    }

    /// Fetch the shared node handle (zero-copy internal path). Every holder of
    /// the store shares the same `Arc`, so LOD derivation on one holder is
    /// visible to all.
    pub fn get_node(&self, node_id: NodeId) -> Option<Arc<RwLock<ContentNode>>> {
        lock_read(&self.nodes).get(&node_id).cloned()
    }

    /// Clone a node for serde/persistence (the `snapshot` helper).
    pub fn snapshot(&self, node_id: NodeId) -> Option<ContentNode> {
        self.get_node(node_id).map(|arc| lock_read(&arc).clone())
    }

    /// Record a user request as a new node. LOD0 and LOD5 are written eagerly;
    /// LOD1–LOD4 stay empty until derived lazily. Scrub is non-bypassable — every
    /// write path payload is scrubbed through `scrub_for_ledger` + `emit_write_audit` (D1).
    pub fn record_request(
        &self,
        session_id: &str,
        request_id: &str,
        content: &str,
    ) -> Result<NodeId, LedgerError> {
        let scrubbed = crate::ledger_guard::scrub_for_ledger(content);
        crate::ledger_guard::emit_write_audit(&scrubbed);
        let id = NodeId::from_int(self.next_id.fetch_add(1, Ordering::SeqCst));
        let node = new_node(id, session_id, request_id, "user", &scrubbed.text, None);
        self.insert_node(&node)?;
        Ok(id)
    }

    /// Update the result of a previously recorded request node: acceptance,
    /// score, and final content. Keeps the shared node and the `content_json`
    /// column in sync (LOD0/LOD5 recomputed eagerly).
    pub fn record_result(
        &self,
        node_id: NodeId,
        accepted: bool,
        score: Option<f64>,
        content: &str,
    ) -> Result<(), LedgerError> {
        let scrubbed = crate::ledger_guard::scrub_for_ledger(content);
        crate::ledger_guard::emit_write_audit(&scrubbed);
        let scrubbed_text = scrubbed.text.clone();
        self.with_node_mut(node_id, |node| {
            node.accepted = Some(accepted);
            node.acceptance_score = score;
            if !scrubbed_text.is_empty() {
                node.lod[LOD0_FULL_TEXT as usize].clone_from(&scrubbed_text);
                node.lod[LOD5_LABEL as usize] =
                    derive_label(node.role.as_ref().map(OriginRole::as_str).unwrap_or_default(), &scrubbed_text);
                node.active_lod = Some(LOD0_FULL_TEXT);
            }
        })?;
        self.enqueue_if_needs_tier(node_id);
        self.enqueue_if_needs_overlay(node_id);
        Ok(())
    }

    /// Persist an arbitrary origin-typed `ContentNode`. LOD0/LOD5 are
    /// guaranteed present (derived from the node's text when missing). An id
    /// is allocated when `node.id` is `None`.
    pub fn record_content_node(&self, node: &ContentNode) -> Result<NodeId, LedgerError> {
        let mut node = node.clone();
        // Scrub LOD0 before persistence — irreversible, always on.
        if !node.lod.is_empty() {
            let scrubbed = crate::ledger_guard::scrub_for_ledger(&node.lod[0]);
            crate::ledger_guard::emit_write_audit(&scrubbed);
            node.lod[0].clone_from(&scrubbed.text);
        }
        let id = if let Some(id) = node.id {
            id
        } else {
            let id = NodeId::from_int(self.next_id.fetch_add(1, Ordering::SeqCst));
            node.id = Some(id);
            id
        };
        ensure_lod_eager(&mut node);
        self.insert_node(&node)?;
        Ok(id)
    }

    /// Write a tiered annotation claim against a node's **current content
    /// hash** (ROADMAP M4). This is the single wiring point for annotation
    /// writers: it resolves the node's `content_hash` and routes the claim to
    /// the `AnnotationStore` over the shared ledger connection. A mutation of
    /// LOD0 (a new hash) makes old claims unreachable — invalidation is a
    /// consequence of keying, never a scheduler.
    ///
    /// `Ok(None)` when the store has no durable backing (pure in-memory) —
    /// fail-open, mirroring the store's other best-effort persistence.
    pub fn write_annotation(
        &self,
        node_id: NodeId,
        claim: &AnnotationClaim,
    ) -> Result<Option<ClaimStatus>, LedgerError> {
        let Some(store) = self.shared_sqlite() else {
            return Ok(None);
        };
        let node = self.snapshot(node_id).ok_or(LedgerError::NotFound(node_id))?;
        let status = AnnotationStore::new(store)
            .write(node.content_hash, claim)
            .map_err(|e| LedgerError::Db(e.to_string()))?;
        Ok(Some(status))
    }
    pub fn insert_node(&self, node: &ContentNode) -> Result<NodeId, LedgerError> {
        let mut node = node.clone();
        // Scrub LOD0 — non-bypassable write-path guard (D1).
        if !node.lod.is_empty() && !node.lod[0].is_empty() {
            let scrubbed = crate::ledger_guard::scrub_for_ledger(&node.lod[0]);
            crate::ledger_guard::emit_write_audit(&scrubbed);
            node.lod[0].clone_from(&scrubbed.text);
        }
        let node_id = if let Some(id) = node.id {
            id
        } else {
            let id = NodeId::from_int(self.next_id.fetch_add(1, Ordering::SeqCst));
            node.id = Some(id);
            id
        };
        ensure_lod_eager(&mut node);
        let embedding_for_hnsw = node.embedding.clone();
        let arc = Arc::new(RwLock::new(node));
        lock_write(&self.nodes).insert(node_id, Arc::clone(&arc));
        self.index_node(node_id);
        self.persist_insert(&arc)?;
        self.enqueue_if_needs_tier(node_id);
        self.enqueue_if_needs_overlay(node_id);
        if let Some(emb) = embedding_for_hnsw {
            self.sync_hnsw_insert(node_id, &emb);
        } else {
            self.maybe_init_hnsw();
        }
        Ok(node_id)
    }

    /// Node count threshold for HNSW activation (single source: the policy's
    /// `DEFAULT_HNSW_THRESHOLD`-derived threshold — see `AdaptiveHnsw`).
    pub fn hnsw_threshold(&self) -> usize {
        self.hnsw_policy.threshold
    }

    /// Whether the HNSW index has been built (adaptive dispatch is active).
    pub fn is_hnsw_built(&self) -> bool {
        lock_read(&self.hnsw)
            .as_ref()
            .is_some_and(|h| h.is_built())
    }

    /// Ensure the HNSW index exists when `|store| > threshold` by bulk-building
    /// from all current embeddings. No-op when already built or below threshold.
    fn maybe_init_hnsw(&self) {
        let len = lock_read(&self.nodes).len();
        if !self.hnsw_policy.should_use_built(len) {
            return;
        }
        if self.is_hnsw_built() {
            return;
        }
        // Collect all embeddings under a short read lock, then build.
        let candidates: Vec<(NodeId, Vec<f32>)> = {
            let nodes = lock_read(&self.nodes);
            nodes
                .iter()
                .filter_map(|(&id, arc)| {
                    let g = lock_read(arc);
                    g.embedding.clone().map(|e| (id, e))
                })
                .collect()
        };
        if candidates.is_empty() {
            return;
        }
        let hnsw = Arc::new(fluent_db::hnsw::HnswIndex::new());
        for (id, emb) in &candidates {
            hnsw.insert(id.as_int(), emb);
        }
        *lock_write(&self.hnsw) = Some(hnsw);
    }

    /// Insert a single node's embedding into the HNSW index, lazily
    /// initializing the index if the threshold has been crossed.
    fn sync_hnsw_insert(&self, node_id: NodeId, embedding: &[f32]) {
        // If already built, just insert.
        if let Some(hnsw) = lock_read(&self.hnsw).clone() {
            if hnsw.is_built() {
                hnsw.insert(node_id.as_int(), embedding);
                return;
            }
        }
        // Not built — check if we should build now.
        let len = lock_read(&self.nodes).len();
        if self.hnsw_policy.should_use_built(len) {
            // Build from all embeddings (including this one) for consistency.
            self.maybe_init_hnsw();
            // Ensure this node's embedding is present even if maybe_init found
            // it already (idempotent double-insert is okay for correctness
            // because hnsw.insert tolerates duplicates, but we skip to avoid
            // duplicate entries).
            if !self.is_hnsw_built() {
                // Fallback single insert if bulk had no embeddings (race).
                let hnsw = Arc::new(fluent_db::hnsw::HnswIndex::new());
                hnsw.insert(node_id.as_int(), embedding);
                *lock_write(&self.hnsw) = Some(hnsw);
            }
        }
    }

    /// Update an existing node's `metadata` (the review worker's
    /// `review_status` write, §12.6). The shared `Arc<RwLock<ContentNode>>`
    /// is mutated in place (so every ledger view sees it) and the durable
    /// `content_json` column is rewritten. `Err` when the node is absent.
    pub fn update_node_metadata(
        &self,
        node_id: NodeId,
        metadata: serde_json::Value,
    ) -> Result<(), LedgerError> {
        self.with_node_mut(node_id, |node| {
            node.metadata = Some(metadata);
        })
        .map(|_| ())
    }

    /// Atomic review write (ROADMAP §12.6/§12.7, C4): the parse node's
    /// `review_status` metadata update, the `parse_review` node (on a miss),
    /// and the `interlingua_index` correction rows all commit in **one SQLite
    /// transaction** (the ledger's existing connection), so a crash mid-review
    /// never half-applies. The in-memory maps are updated **only on commit**
    /// (durable-first — a durable failure leaves zero in-memory divergence).
    ///
    /// - `parse_node_id` — the parse node whose metadata is overlaid with
    ///   `parse_metadata` (the `review_status` write).
    /// - `review_node` — the `parse_review` `ContentNode` to persist (only on
    ///   a review **miss**). Its id is allocated here; `None` skips the node.
    /// - `correction_rows` — the correction-pattern cache rows to upsert.
    ///
    /// Returns the allocated `parse_review` node id when one was written.
    pub(crate) fn apply_review(
        &self,
        parse_node_id: NodeId,
        parse_metadata: serde_json::Value,
        review_node: Option<&ContentNode>,
        correction_rows: &[CorrectionRow],
    ) -> Result<Option<NodeId>, LedgerError> {
        // 1. Prepare — no shared-state mutation. Snapshot the parse node and
        //    overlay the new metadata; allocate the review node's id and
        //    ensure its eager LODs. Nothing touches the maps yet, so a failure
        //    below cannot leave in-memory/durable divergence (M1).
        let parse_arc = self
            .get_node(parse_node_id)
            .ok_or(LedgerError::NotFound(parse_node_id))?;
        let mut parse_node = lock_read(&parse_arc).clone();
        drop(parse_arc);
        parse_node.metadata = Some(parse_metadata);

        let review = match review_node {
            Some(rn) => {
                let mut node = rn.clone();
                let id = NodeId::from_int(self.next_id.fetch_add(1, Ordering::SeqCst));
                node.id = Some(id);
                ensure_lod_eager(&mut node);
                Some((id, node))
            }
            None => None,
        };

        // 2. Durable transaction FIRST. On `Err` we return with zero in-memory
        //    mutation (an id consumed here is a harmless monotonic gap —
        //    hydration seeds `next_id` from durable `MAX`).
        let Some(store) = self.durable.clone() else {
            // No durable backing: commit the prepared state in memory only.
            return Ok(self.commit_review_in_memory(parse_node_id, parse_node, review));
        };
        store
            .transaction(|tx| {
                update_node_row(tx, &parse_node)?;
                if let Some((_, node)) = &review {
                    insert_node_row(tx, node)?;
                }
                for row in correction_rows {
                    upsert_correction_row(tx, row)?;
                }
                Ok(())
            })
            .map_err(|e| LedgerError::Db(e.to_string()))?;

        // 3. In-memory only on success: overwrite the parse node's shared
        //    metadata and insert + index the review node.
        Ok(self.commit_review_in_memory(parse_node_id, parse_node, review))
    }

    /// The in-memory half of [`Self::apply_review`]: overwrite the parse node's
    /// shared metadata and (when present) insert + index the review node. Only
    /// ever called after the durable side committed (or for a store with no
    /// durable backing). Infallible — the shared parse node is guaranteed to
    /// still exist.
    fn commit_review_in_memory(
        &self,
        parse_node_id: NodeId,
        parse_node: ContentNode,
        review: Option<(NodeId, ContentNode)>,
    ) -> Option<NodeId> {
        if let Some(parse_arc) = self.get_node(parse_node_id) {
            lock_write(&parse_arc).metadata = parse_node.metadata;
        }
        if let Some((id, node)) = &review {
            let arc = Arc::new(RwLock::new(node.clone()));
            lock_write(&self.nodes).insert(*id, Arc::clone(&arc));
            self.index_node(*id);
            self.enqueue_if_needs_tier(*id);
        }
        review.map(|(id, _)| id)
    }

    /// Maintain the interned session/role indices for a node.
    fn index_node(&self, node_id: NodeId) {
        let (session, role) = match self.get_node(node_id) {
            Some(arc) => {
                let guard = lock_read(&arc);
                (guard.session_id.clone(), guard.role.clone())
            }
            None => (None, None),
        };
        if let Some(session) = session {
            let key = ArcIntern::from(session.as_str());
            lock_write(&self.by_session)
                .entry(key)
                .or_default()
                .push(node_id);
        }
        if let Some(role) = role {
            let key = ArcIntern::from(role.as_str());
            lock_write(&self.by_role)
                .entry(key)
                .or_default()
                .push(node_id);
        }
    }

    /// Write the node's canonical row (flat projection + `content_json`).
    fn persist_insert(&self, node: &Arc<RwLock<ContentNode>>) -> Result<(), LedgerError> {
        let Some(ref store) = self.durable else {
            return Ok(());
        };
        let node = lock_read(node).clone();
        store
            .with_conn(|conn| insert_node_row(conn, &node))
            .map_err(|e| LedgerError::Db(e.to_string()))
    }

    /// Persist an updated node (flat projection + `content_json`).
    fn persist_update(&self, node: &ContentNode) -> Result<(), LedgerError> {
        let Some(ref store) = self.durable else {
            return Ok(());
        };
        store
            .with_conn(|conn| update_node_row(conn, node))
            .map_err(|e| LedgerError::Db(e.to_string()))
    }

    /// Apply a mutation to the shared node and persist both the shared node and
    /// the `content_json` column. The single place that keeps the views in
    /// sync. `pub(crate)` so the background `LedgerTierWorker` writes
    /// derived tiers through the same canonical path.
    pub(crate) fn with_node_mut<F>(&self, node_id: NodeId, f: F) -> Result<ContentNode, LedgerError>
    where
        F: FnOnce(&mut ContentNode),
    {
        let arc = self
            .get_node(node_id)
            .ok_or(LedgerError::NotFound(node_id))?;
        let mut guard = lock_write(&arc);
        f(&mut guard);
        ensure_lod_eager(&mut guard);
        let node = guard.clone();
        drop(guard);
        self.persist_update(&node)?;
        Ok(node)
    }

    /// Collapse a node: replace LOD0's content with `summary` and mark it
    /// `active_lod = lod`. Used by compaction.
    pub fn collapse_node(
        &self,
        node_id: NodeId,
        summary: &str,
        lod: u8,
    ) -> Result<(), LedgerError> {
        self.with_node_mut(node_id, |node| {
            node.lod[LOD0_FULL_TEXT as usize] = summary.to_string();
            node.lod[LOD5_LABEL as usize] =
                derive_label(node.role.as_ref().map(OriginRole::as_str).unwrap_or_default(), summary);
            node.active_lod = Some(lod);
        })?;
        Ok(())
    }

    /// Derive (or return the cached) LOD level for a node, **from LOD0 only**
    /// via the `Summarizer` — never chained from a lower tier.
    ///
    /// Only LOD1–LOD4 are lazy; LOD0/LOD5 are eager at creation. The derived
    /// tier is cached on the shared node, so a second request from any holder
    /// hits the cache, not the LLM.
    pub fn ensure_lod(&self, node_id: NodeId, level: u8) -> Result<ContentNode, LedgerError> {
        self.ensure_tier(node_id, level)?;
        self.snapshot(node_id).ok_or(LedgerError::NotFound(node_id))
    }

    /// Derive a lazy LOD tier (1..=4) for a node if it is not already cached,
    /// from LOD0 only via the `Summarizer`. Returns `()` — the tier text is
    /// read back by the caller. The **snapshot-then-derive** shape: hold a
    /// read guard only long enough to copy LOD0 + the cached tier, drop it,
    /// derive via the `Summarizer`, then write-cache under a fresh guard. No
    /// guard is held across an LLM call (R7).
    fn ensure_tier(&self, node_id: NodeId, level: u8) -> Result<(), LedgerError> {
        if !LAZY_LOD_RANGE.contains(&level) {
            return Err(LedgerError::InvalidLod(level));
        }
        let arc = self
            .get_node(node_id)
            .ok_or(LedgerError::NotFound(node_id))?;

        let (full_text, cached) = {
            let guard = lock_read(&arc);
            let full_text = guard
                .lod
                .first()
                .cloned()
                .ok_or(LedgerError::NotFound(node_id))?;
            let cached = guard
                .lod
                .get(level as usize)
                .map_or("", String::as_str)
                .to_string();
            (full_text, cached)
        };
        if !cached.is_empty() {
            return Ok(());
        }

        let derived = {
            let summarizer = lock(&self.summarizer);
            let summarizer = summarizer.as_ref().ok_or(LedgerError::NoSummarizer)?;
            summarizer
                .summarize_text(&full_text)
                .map_err(|e| LedgerError::Summary(e.to_string()))?
        };

        self.with_node_mut(node_id, |node| {
            while node.lod.len() <= level as usize {
                node.lod.push(String::new());
            }
            node.lod[level as usize] = derived;
        })?;
        Ok(())
    }

    /// Read a single LOD tier's text — the **only** method through which a
    /// view's text leaves the store. Eager tiers (LOD0/LOD5) are
    /// returned directly; lazy tiers (LOD1–LOD4) are derived on demand via
    /// `ensure_tier` and then re-read. A read guard is held only long enough
    /// to copy the string out.
    pub fn lod_text(&self, node_id: NodeId, level: u8) -> Result<String, LedgerError> {
        if level == LOD0_FULL_TEXT || level == LOD5_LABEL {
            let arc = self
                .get_node(node_id)
                .ok_or(LedgerError::NotFound(node_id))?;
            let text = lock_read(&arc)
                .lod
                .get(level as usize)
                .cloned()
                .unwrap_or_default();
            return Ok(text);
        }
        if !LAZY_LOD_RANGE.contains(&level) {
            return Err(LedgerError::InvalidLod(level));
        }
        self.ensure_tier(node_id, level)?;
        let arc = self
            .get_node(node_id)
            .ok_or(LedgerError::NotFound(node_id))?;
        let text = lock_read(&arc)
            .lod
            .get(level as usize)
            .cloned()
            .unwrap_or_default();
        Ok(text)
    }

    /// All node ids for a session (interned index, insertion order). The
    /// zero-copy list path for views: no `ContentNode` clones, unlike
    /// `get_session_nodes`.
    pub fn session_node_ids(&self, session_id: &str) -> Vec<NodeId> {
        let key = ArcIntern::from(session_id);
        lock_read(&self.by_session)
            .get(&key)
            .cloned()
            .unwrap_or_default()
    }

    /// Compact a session: demote older nodes to higher LOD levels via the
    /// `RecencyCompaction` policy. Demotion sets `active_lod`; the actual LOD
    /// text is filled in lazily from LOD0 (never chained). Returns the IDs of
    /// demoted nodes.
    pub fn compact_session(
        &self,
        session_id: &str,
        max_nodes: usize,
    ) -> Result<Vec<NodeId>, LedgerError> {
        let mut nodes = self.get_session_nodes(session_id, usize::MAX)?;
        // `select_lod` expects chronological (oldest-first) order; the store
        // returns newest-first.
        nodes.reverse();
        let lods = RecencyCompaction.select_lod(&nodes, max_nodes);
        let mut demoted = Vec::new();
        for (node, lod) in nodes.iter().zip(lods) {
            let current = node.active_lod.unwrap_or(LOD0_FULL_TEXT);
            if lod > current {
                if let Some(id) = node.id {
                    self.set_active_lod(id, lod)?;
                    demoted.push(id);
                }
            }
        }
        Ok(demoted)
    }

    /// Set a node's active LOD level (demotion), keeping `content_json` in
    /// sync. The demoted text stays lazy — derived from LOD0 on demand.
    fn set_active_lod(&self, node_id: NodeId, active_lod: u8) -> Result<(), LedgerError> {
        self.with_node_mut(node_id, |node| {
            node.active_lod = Some(active_lod);
        })?;
        Ok(())
    }

    /// All nodes for a session (canonical `ContentNode`s), most recent first,
    /// capped at `limit`.
    pub fn get_session_nodes(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<ContentNode>, LedgerError> {
        let key = ArcIntern::from(session_id);
        let ids: Vec<NodeId> = lock_read(&self.by_session)
            .get(&key)
            .cloned()
            .unwrap_or_default();
        let mut nodes = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(node) = self.snapshot(id) {
                nodes.push(node);
            }
        }
        // Most recent first, mirroring the flat store's `ORDER BY turn_index
        // DESC`.
        nodes.sort_by(|a, b| {
            let ta = a
                .turn_index
                .or_else(|| a.id.map(|i| i.as_int() as u64))
                .unwrap_or(0);
            let tb = b
                .turn_index
                .or_else(|| b.id.map(|i| i.as_int() as u64))
                .unwrap_or(0);
            tb.cmp(&ta)
        });
        nodes.truncate(limit);
        Ok(nodes)
    }

    /// The flat `LedgerEntry` audit projection for a session, read from the
    /// durable table (most recent first, capped at `limit`).
    pub fn get_session_entries(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<LedgerEntry>, LedgerError> {
        let Some(ref store) = self.durable else {
            return Ok(Vec::new());
        };
        store
            .query_rows(
                "SELECT node_id, session_id, request_id, role, content,
                        turn_index, accepted, acceptance_score, active_lod,
                        parent_id, step_id, metadata, created_at
                 FROM ledger WHERE session_id = ?1
                 ORDER BY turn_index DESC LIMIT ?2",
                rusqlite::params![session_id, limit as i64],
                |row| {
                    Ok(LedgerEntry {
                        node_id: NodeId::from_int(row.get(0)?),
                        session_id: row.get(1)?,
                        request_id: row.get(2)?,
                        role: row.get(3)?,
                        content: row.get(4)?,
                        turn_index: row.get(5)?,
                        accepted: row.get(6)?,
                        acceptance_score: row.get(7)?,
                        active_lod: row.get(8)?,
                        parent_id: row.get::<_, Option<i64>>(9)?.map(NodeId::from_int),
                        step_id: row.get(10)?,
                        metadata: serde_json::from_str(row.get::<_, String>(11)?.as_str())
                            .unwrap_or_default(),
                        created_at: row.get(12)?,
                    })
                },
            )
            .map_err(|e| LedgerError::Db(e.to_string()))
    }

    /// All node ids for a role (interned index), in insertion order.
    pub fn nodes_for_role(&self, role: &str) -> Vec<NodeId> {
        let key = ArcIntern::from(role);
        lock_read(&self.by_role)
            .get(&key)
            .cloned()
            .unwrap_or_default()
    }

    /// Cosine KNN search over node embeddings.
    ///
    /// For `|store| <= hnsw_threshold` (default 512, `DEFAULT_HNSW_THRESHOLD`)
    /// this is exact brute-force (`knn_brute_force`). For `|store| > threshold`
    /// it routes through the HNSW index (approximate, recall≥0.95 vs brute-force
    /// at N=512..2048). When the HNSW index is not yet built or the query
    /// embedding is malformed, it falls back to brute-force.
    pub fn knn_search(&self, embedding: &[f32], k: usize) -> Vec<KnnHit> {
        // M6: single adaptive-dispatch policy ([B] cost/recall only — never a
        // confidence/verification gate). Probe HNSW iff built and above
        // threshold; the built-implies-above-threshold invariant (nodes never
        // shrink; build decisions go through the same policy) makes this
        // exactly the previous probe-if-built behavior.
        // M5: the HNSW probe + id resolution is the shared
        // `fluent_db::hnsw::hnsw_lookup` (`None` = fall back). Name lookup
        // and the raw-distance `KnnHit` shape stay call-site code.
        let dispatch_hnsw = {
            let len = lock_read(&self.nodes).len();
            self.hnsw_policy.dispatch(self.is_hnsw_built(), len)
        };
        if dispatch_hnsw {
            let hnsw = lock_read(&self.hnsw).clone().expect("dispatched ⇒ built");
            if let Some(neighbours) = fluent_db::hnsw::hnsw_lookup(&hnsw, embedding, k) {
                let mut results = Vec::with_capacity(neighbours.len());
                for (raw, distance) in neighbours {
                    let node_id = NodeId::from_int(raw);
                    let name = self.snapshot(node_id).map(|n| n.name).unwrap_or_default();
                    results.push(KnnHit {
                        node_id,
                        distance,
                        name,
                    });
                }
                if !results.is_empty() {
                    return results;
                }
            }
        }
        // Fallback: exact brute-force.
        let candidates: Vec<(NodeId, Vec<f32>)> = {
            let nodes = lock_read(&self.nodes);
            nodes
                .iter()
                .filter_map(|(&node_id, arc)| {
                    let guard = lock_read(arc);
                    guard.embedding.clone().map(|emb| (node_id, emb))
                })
                .collect()
        };
        let hits = knn_brute_force(
            embedding,
            candidates.iter().map(|(id, emb)| (*id, emb.as_slice())),
            k,
        );
        let mut results = Vec::with_capacity(hits.len());
        for (node_id, distance) in hits {
            let name = self.snapshot(node_id).map(|n| n.name).unwrap_or_default();
            results.push(KnnHit {
                node_id,
                distance,
                name,
            });
        }
        results
    }

    /// Panic while holding the durable connection mutex (test-only): exercises
    /// the poison-recovery path in `SqliteStore`'s `common_core::sync::lock`.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn poison_conn(&self) {
        if let Some(ref store) = self.durable {
            let _ = store.with_conn(|_| -> Result<(), DbError> {
                panic!("simulated panic while holding db mutex")
            });
        }
    }
}

/// The `ledger` table INSERT — the canonical row shape shared by
/// [`ContentNodeStore::persist_insert`] and the atomic review transaction
/// ([`ContentNodeStore::apply_review`]). `DbError` so the caller owns the
/// wrapper error type.
fn insert_node_row(conn: &Connection, node: &ContentNode) -> Result<(), DbError> {
    let metadata = node
        .metadata
        .as_ref()
        .unwrap_or(&serde_json::json!({}))
        .to_string();
    let lod_json = serde_json::to_string(&node.lod).map_err(|e| DbError::Other(e.to_string()))?;
    let content_json = serde_json::to_string(node).map_err(|e| DbError::Other(e.to_string()))?;
    let content = node.lod.first().map_or("", String::as_str);
    let label = node
        .lod
        .get(LOD5_LABEL as usize)
        .cloned()
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| derive_label(node.role.as_ref().map(OriginRole::as_str).unwrap_or_default(), content));

    fluent_db::query::execute(
        conn,
        "INSERT INTO ledger (node_id, session_id, request_id, role, content, turn_index,
                             accepted, acceptance_score, active_lod, parent_id, step_id,
                             metadata, created_at, label, lod, content_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        rusqlite::params![
            node.id.map(NodeId::as_int),
            node.session_id.clone().unwrap_or_default(),
            node.request_id.clone().unwrap_or_default(),
            node.role.as_ref().map(|r| r.as_str().to_string()).unwrap_or_default(),
            content,
            node.turn_index.unwrap_or(0) as i64,
            node.accepted.unwrap_or(true),
            node.acceptance_score,
            i64::from(node.active_lod.unwrap_or(LOD0_FULL_TEXT)),
            node.parent_id.map(NodeId::as_int),
            node.step_id.as_deref(),
            metadata,
            node.created_at.unwrap_or_else(common_core::now_secs) as i64,
            label,
            lod_json,
            content_json,
        ],
    )?;
    Ok(())
}

/// The `ledger` table UPDATE — shared by [`ContentNodeStore::persist_update`]
/// and the atomic review transaction.
fn update_node_row(conn: &Connection, node: &ContentNode) -> Result<(), DbError> {
    let metadata = node
        .metadata
        .as_ref()
        .unwrap_or(&serde_json::json!({}))
        .to_string();
    let lod_json = serde_json::to_string(&node.lod).map_err(|e| DbError::Other(e.to_string()))?;
    let content_json = serde_json::to_string(node).map_err(|e| DbError::Other(e.to_string()))?;
    let content = node.lod.first().map_or("", String::as_str);
    let label = node
        .lod
        .get(LOD5_LABEL as usize)
        .cloned()
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| derive_label(node.role.as_ref().map(OriginRole::as_str).unwrap_or_default(), content));

    fluent_db::query::execute(
        conn,
        "UPDATE ledger SET session_id = ?1, request_id = ?2, role = ?3, content = ?4,
                           turn_index = ?5, accepted = ?6, acceptance_score = ?7,
                           active_lod = ?8, parent_id = ?9, step_id = ?10, metadata = ?11,
                           created_at = ?12, label = ?13, lod = ?14, content_json = ?15
         WHERE node_id = ?16",
        rusqlite::params![
            node.session_id.clone().unwrap_or_default(),
            node.request_id.clone().unwrap_or_default(),
            node.role.as_ref().map(|r| r.as_str().to_string()).unwrap_or_default(),
            content,
            node.turn_index.unwrap_or(0) as i64,
            node.accepted.unwrap_or(true),
            node.acceptance_score,
            i64::from(node.active_lod.unwrap_or(LOD0_FULL_TEXT)),
            node.parent_id.map(NodeId::as_int),
            node.step_id.as_deref(),
            metadata,
            node.created_at.unwrap_or_else(common_core::now_secs) as i64,
            label,
            lod_json,
            content_json,
            node.id.map(NodeId::as_int),
        ],
    )?;
    Ok(())
}

/// Build a fresh `ContentNode` with LOD0 (full text) and LOD5 (label) eager.
/// Re-exported from `crate::ledger` so the facade's tests keep compiling
/// unchanged.
pub(crate) fn new_node(
    id: NodeId,
    session_id: &str,
    request_id: &str,
    role: &str,
    content: &str,
    accepted: Option<bool>,
) -> ContentNode {
    let mut node = ContentNode {
        id: Some(id),
        name: format!("{role}-msg-{}", id.as_int()).into(),
        source: "session".into(),
        content_hash: 0,
        lod: vec![
            content.to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ],
        embedding: None,
        capabilities: None,
        session_id: Some(session_id.to_string()),
        request_id: Some(request_id.to_string()),
        role: Some(match role.to_ascii_lowercase().as_str() {
            "user" => fluent_types::OriginRole::User,
            "system" => fluent_types::OriginRole::System,
            "assistant" => fluent_types::OriginRole::Assistant,
            "tool" => fluent_types::OriginRole::Tool,
            "subagent" => fluent_types::OriginRole::Subagent,
            "self" => fluent_types::OriginRole::SelfOrigin,
            _ => fluent_types::OriginRole::Other(role.to_string()),
        }),
        // Monotonic per allocation → stable `ORDER BY turn_index DESC` within
        // a session even before a real turn counter exists.
        turn_index: Some(id.as_int() as u64),
        accepted,
        acceptance_score: None,
        active_lod: Some(LOD0_FULL_TEXT),
        parent_id: None,
        step_id: None,
        step_status: None,
        metadata: None,
        created_at: Some(common_core::now_secs()),
        annotation: None,
    };
    ensure_lod_eager(&mut node);
    node
}

/// Deterministic, LLM-free LOD5 label (short descriptor), derived eagerly at
/// node creation. Falls back to the role when no content survives truncation.
/// `pub(crate)` so the background `LedgerTierWorker` can use it as the
/// no-model fallback for LOD5.
pub(crate) fn derive_label(role: &str, content: &str) -> String {
    let sentence = common_core::string::first_sentence(content);
    let snippet = if sentence.is_empty() {
        common_core::string::truncate_utf8(content, 64)
    } else {
        common_core::string::truncate_utf8(&sentence, 64)
    };
    if snippet.is_empty() {
        role.to_string()
    } else {
        snippet
    }
}

/// Guarantee LOD0 (full text) and LOD5 (label) are present on a node.
/// The stable content hash for a node's LOD0 — `0` for empty content.
fn content_hash_of(content: &str) -> u64 {
    if content.is_empty() {
        0
    } else {
        common_core::hash::fnv1a64(content.as_bytes())
    }
}

fn ensure_lod_eager(node: &mut ContentNode) {
    while node.lod.len() < LOD5_LABEL as usize + 1 {
        node.lod.push(String::new());
    }
    let content = node.lod[LOD0_FULL_TEXT as usize].clone();
    // The content hash is the annotation keying domain (ROADMAP M4): it tracks
    // LOD0 exactly, recomputed in this single write funnel so a mutation that
    // changes LOD0 changes the hash and thereby invalidates cached annotations
    // — no staleness scheduler. Empty content maps to the stable `0` sentinel.
    node.content_hash = content_hash_of(&content);
    if content.is_empty() {
        // Nothing to derive LOD0 from — nothing to do.
        return;
    }
    if node.lod[LOD5_LABEL as usize].is_empty() {
        let role = node.role.as_ref().map(OriginRole::as_str).unwrap_or_default().to_string();
        node.lod[LOD5_LABEL as usize] = derive_label(&role, &content);
    }
    if node.active_lod.is_none() {
        node.active_lod = Some(LOD0_FULL_TEXT);
    }
}

#[cfg(test)]
#[path = "../tests/node_store.rs"]
mod tests;
