//! `OverlayWorker` — continuous background derivation of the three arc_ready
//! annotation overlays (spacy parse / LLM enrichment / embedding).
//!
//! A background task drains a feed of pending node ids (attached to the
//! `ContentNodeStore` via `set_overlay_events`) and, for each node, derives the
//! three overlays **in parallel** — spacy annotation, LLM enrichment, and
//! embedding — each **from LOD0 only**, cached on the shared node at-most-once.
//! It never blocks a request and never recomputes a filled overlay.
//!
//! The feed mechanics — the bounded `mpsc` of `NodeId`s, the `CreditFlow` gate
//! on the async producer path, the `Limiter` bounding concurrent derivations,
//! the drain loop, and the boot backfill — are composed from the shared
//! [`CreditedFeedWorker`] primitive (`fluent_concurrency::feed_worker`) rather
//! than hand-rolled. The worker supplies only the per-node derivation handler
//! and its audit policy. This is the **second consumer** of the primitive after
//! `LedgerTierWorker` — the overlay worker deliberately does **not**
//! reimplement that worker's shape.
//!
//! The three overlays have no dependencies on each other, so a slow LLM
//! enrichment never delays the spacy parse or the embedding for the same node:
//! the per-node handler spawns three independent derivation jobs on the
//! injected `Runtime` (no ambient `tokio::spawn`) and joins them, so a blocking
//! LLM/embedder call occupies its own executor thread while the others proceed.
//!
//! It reuses the store's already-wired seams — the shared `NlpPipeline`,
//! `EmbeddingProvider`, and `ChatBackend` attached via
//! `set_overlay_pipeline` / `set_overlay_embedder` / `set_overlay_llm` — **no
//! new transport**. Each derivation goes through the store's at-most-once,
//! fail-open `annotation_for` / `embedding_for` / `llm_overlay_for` entry
//! points, so a missing or down seam leaves the other overlays intact.
//!
//! It emits a `kind = "overlay"` audit record per derived overlay (success or
//! a permanent `status: failed`), mirroring the tier worker's `kind = "tier"`
//! records.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use fluent_concurrency::feed_worker::{CreditedFeedWorker, FeedConfig};
use fluent_types::{NodeId, OverlayKind, OverlayStatus};
use fluent_wvr::Runtime;
use tokio::task::JoinHandle;

use crate::node_store::ContentNodeStore;
use common_core::sync::lock_read;

/// Configuration for the background overlay worker. The feed knobs mirror
/// `LedgerTierWorker`'s `TierConfig` (and thus `FeedConfig`'s load-bearing
/// defaults) — do not drift.
#[derive(Debug, Clone, Copy)]
pub struct OverlayWorkerConfig {
    /// Max node ids drained per batch.
    pub batch_size: usize,
    /// Poll interval (ms) before giving up on filling a batch.
    pub poll_interval_ms: u64,
    /// Capacity of the pending-id feed.
    pub queue_capacity: usize,
    /// Max concurrent node-derivations (the `Limiter` cap). Each node fans out
    /// three independent sub-jobs, so this bounds *nodes in flight*, not the
    /// per-node overlay fan-out.
    pub max_concurrent: usize,
    /// Credit granted to the feed's producer up front: the max outstanding
    /// `NodeId`s the async (credit-gated) enqueue path may have in flight
    /// before it blocks.
    pub credit_limit: usize,
    /// How many processed nodes the consumer waits for before bumping credit
    /// back to the producer (`flow::CreditSpec.more_after`).
    pub credit_more_after: usize,
}

impl Default for OverlayWorkerConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            poll_interval_ms: 100,
            queue_capacity: 1024,
            max_concurrent: 8,
            credit_limit: 256,
            credit_more_after: 8,
        }
    }
}

impl OverlayWorkerConfig {
    /// Map an `arc_ready` config block onto the worker's configuration. The
    /// numeric knobs come straight from the config (which defaults them to the
    /// same `CreditedFeedWorker` load-bearing values as [`OverlayWorkerConfig`]);
    /// `batch_size`/`poll_interval_ms` stay at the worker defaults (they have no
    /// config surface).
    pub fn from_arc_ready(cfg: &crate::config::ArcReadyConfig) -> Self {
        Self {
            queue_capacity: cfg.queue_capacity,
            credit_limit: cfg.credit_limit,
            credit_more_after: cfg.credit_more_after,
            max_concurrent: cfg.max_concurrent,
            ..Default::default()
        }
    }
}

/// The background overlay derivation worker — the second consumer of
/// [`CreditedFeedWorker`], over the store's already-wired seams.
pub struct OverlayWorker {
    store: Arc<ContentNodeStore>,
    /// Injected runtime for the per-node sub-job fan-out. Cloned so the worker
    /// can spawn the three independent derivation jobs without ambient
    /// `tokio::spawn`.
    runtime: Arc<dyn Runtime>,
    /// The shared feed primitive owning the bounded `mpsc`, the `CreditFlow`
    /// gate, the `Limiter`, and the drain loop.
    feed: Arc<CreditedFeedWorker<NodeId>>,
    /// The background loop's join handle, stored by [`Self::start`] so
    /// [`Self::drain`] / [`Self::abort`] can own and settle it (the server
    /// holds only the worker `Arc`).
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl OverlayWorker {
    /// Construct the worker over a shared store whose overlay seams are already
    /// attached (`set_overlay_pipeline` / `set_overlay_embedder` /
    /// `set_overlay_llm`). The `sender` is available via
    /// [`OverlayWorker::sender`]; attach it to the store with
    /// `store.set_overlay_events(worker.sender())`.
    pub fn new(
        store: Arc<ContentNodeStore>,
        config: OverlayWorkerConfig,
        runtime: Arc<dyn Runtime>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|weak: &std::sync::Weak<Self>| {
            let feed = Arc::new(CreditedFeedWorker::new(
                Arc::clone(&runtime),
                FeedConfig {
                    queue_capacity: config.queue_capacity,
                    credit_limit: config.credit_limit,
                    credit_more_after: config.credit_more_after,
                    max_concurrent: config.max_concurrent,
                    batch_size: config.batch_size,
                    poll_interval_ms: config.poll_interval_ms,
                },
                {
                    let weak = weak.clone();
                    move |id: NodeId| {
                        let weak = weak.clone();
                        async move {
                            let Some(this) = weak.upgrade() else {
                                return;
                            };
                            this.handle_item(id).await;
                        }
                    }
                },
            ));
            Self {
                store,
                runtime,
                feed,
                handle: Mutex::new(None),
            }
        })
    }

    /// The pending-node feed sender; attach to the store via
    /// `store.set_overlay_events(worker.sender())`.
    pub fn sender(&self) -> tokio::sync::mpsc::Sender<NodeId> {
        self.feed.sender()
    }

    /// Enqueue a node for background overlay derivation. Non-blocking: on a
    /// full feed it skips with a debug log — the credit-gated
    /// `enqueue_with_credit` path and the next boot backfill cover the
    /// stragglers.
    pub fn enqueue(&self, id: NodeId) {
        self.feed.enqueue(id);
    }

    /// Enqueue a node with chain backpressure (credit-gated). `Err(FeedClosed)`
    /// when the feed is closed (the worker is torn down).
    pub async fn enqueue_with_credit(&self, id: NodeId) -> Result<(), fluent_concurrency::feed_worker::FeedError> {
        self.feed.enqueue_with_credit(id).await
    }

    /// Whether the credit-gated producer is currently blocked.
    pub fn producer_blocked(&self) -> bool {
        self.feed.is_blocked()
    }

    /// Start the background loop. Boot backfill first enqueues every node
    /// already missing any overlay, then the feed's drain loop takes over. The
    /// join handle is stored on the worker so [`Self::drain`] (graceful
    /// shutdown) and [`Self::abort`] (tests) can settle it. Spawned through the
    /// injected `Runtime` — no ambient `tokio::spawn`.
    pub fn start(self: &Arc<Self>) {
        let store = Arc::clone(&self.store);
        self.feed
            .backfill(move || store.node_ids_needing_overlays());
        let handle = self.feed.start();
        *self.handle.lock().unwrap() = Some(handle);
    }

    /// Graceful shutdown: stop accepting new items and let queued items
    /// complete, then await the background loop so it exits cleanly. Bounded by
    /// a short timeout so a hung derivation (e.g. an unreachable backend) never
    /// blocks shutdown indefinitely.
    pub async fn drain(self: Arc<Self>) {
        self.feed.drain();
        let handle = self.handle.lock().unwrap().take();
        if let Some(handle) = handle {
            let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
        }
    }

    /// Abort the background loop immediately (tests teardown). After the loop
    /// is settled this is a no-op.
    pub fn abort(&self) {
        if let Some(handle) = self.handle.lock().unwrap().take() {
            handle.abort();
        }
    }

    /// Per-node derivation: fan out the three independent overlays in parallel
    /// (each from LOD0, at-most-once, fail-open via the store), then emit a
    /// `kind = "overlay"` audit record for every overlay that resolved to
    /// `ready` or a permanent `failed`. Runs under the feed's shared `Limiter`;
    /// its completion releases the producer's credit token.
    async fn handle_item(&self, id: NodeId) {
        let outcomes = self.derive_all(id).await;
        for (kind, status) in outcomes {
            if matches!(status, OverlayStatus::Ready | OverlayStatus::Failed) {
                crate::audit::emit(
                    "overlay",
                    serde_json::json!({
                        "node_id": id.as_int(),
                        "kind": overlay_kind_name(kind),
                        "status": overlay_status_name(status),
                    }),
                );
            }
        }
    }

    /// Derive all three overlays for `id` concurrently. Each overlay is a
    /// separate spawned job on the injected runtime so a blocking LLM or
    /// embedder call occupies its own executor thread and never delays the
    /// others. Returns each overlay's resulting `OverlayStatus` (read from the
    /// shared node, which the store's at-most-once install / fail-open marked).
    async fn derive_all(&self, id: NodeId) -> Vec<(OverlayKind, OverlayStatus)> {
        let store = Arc::clone(&self.store);
        let results: Arc<Mutex<Vec<(OverlayKind, OverlayStatus)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(3)));

        let mut handles = Vec::with_capacity(3);
        for kind in [
            OverlayKind::Spacy,
            OverlayKind::Llm,
            OverlayKind::Embedding,
        ] {
            let store = Arc::clone(&store);
            let results = Arc::clone(&results);
            handles.push(self.runtime.spawn(Box::pin(async move {
                run_overlay(&store, id, kind);
                results
                    .lock()
                    .unwrap()
                    .push((kind, overlay_status(&store, id, kind)));
            })));
        }

        // Await each sub-job. A panicked sub-job returns `Err` here — the
        // others have already pushed their outcomes — so a single overlay's
        // panic is contained and the handler still proceeds.
        for handle in handles {
            let _ = handle.await;
        }

        let taken = std::mem::take(&mut *results.lock().unwrap());
        taken
    }
}

/// Run the store's at-most-once derivation for one overlay kind. All three are
/// fail-open: `Ok(None)` (missing seam / permanent failure) is not an error.
fn run_overlay(store: &ContentNodeStore, id: NodeId, kind: OverlayKind) {
    match kind {
        OverlayKind::Spacy => {
            let _ = store.annotation_for(id);
        }
        OverlayKind::Llm => {
            let _ = store.llm_overlay_for(id);
        }
        OverlayKind::Embedding => {
            let _ = store.embedding_for(id);
        }
    }
}

/// Read the node's stored overlay status for `kind` (the store's authoritative
/// record — `ready`, `failed`, `pending`, or `absent` for a node that never
/// started / was fail-open without a seam).
fn overlay_status(store: &ContentNodeStore, id: NodeId, kind: OverlayKind) -> OverlayStatus {
    store
        .get_node(id)
        .map_or(OverlayStatus::Absent, |arc| {
            lock_read(&arc).overlay(kind).status
        })
}

/// The snake_case wire name of an overlay kind (matches the serde rename).
fn overlay_kind_name(kind: OverlayKind) -> &'static str {
    match kind {
        OverlayKind::Spacy => "spacy",
        OverlayKind::Llm => "llm",
        OverlayKind::Embedding => "embedding",
    }
}

/// The snake_case wire name of an overlay status (matches the serde rename).
fn overlay_status_name(status: OverlayStatus) -> &'static str {
    match status {
        OverlayStatus::Absent => "absent",
        OverlayStatus::Pending => "pending",
        OverlayStatus::Ready => "ready",
        OverlayStatus::Failed => "failed",
    }
}
#[cfg(test)]
#[path = "../../tests/ledger_overlay_worker.rs"]
mod tests;
