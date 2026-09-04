//! `LedgerTierWorker` — continuous background LOD4/LOD5 generation.
//!
//! A background task drains a feed of pending node ids (attached to the
//! `ContentNodeStore` via `set_tier_events`) and derives LOD4 (short summary) and
//! LOD5 (LLM description) for each node **from LOD0 only**, caching them on the
//! shared node. It never blocks a request and never recomputes a filled tier
//! (at-most-once, checked-and-set under the write lock).
//!
//! The feed mechanics — the bounded `mpsc` of `NodeId`s, the `CreditFlow`
//! gate on the async producer path, the `Limiter` bounding concurrent
//! derivations, the drain loop, and the boot backfill — are composed from the
//! shared [`CreditedFeedWorker`] primitive (`fluent_concurrency::feed_worker`)
//! rather than hand-rolled. The worker supplies only the per-node derivation
//! handler and its retry/bookkeeping policy.
//!
//! It reuses the shared `ChatBackend` — **no second HTTP client** — and is
//! spawned through the injected `Runtime` (no ambient `tokio::spawn`). It
//! emits a `kind = "tier"` audit record per derived node.
//!
//! Degradation rules: when the backend is unavailable, LOD5 falls back
//! to the deterministic `derive_label` and LOD4 is left empty — never a crash.
//! Transient backend errors are re-enqueued with bounded retry (no infinite
//! loop).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fluent_concurrency::feed_worker::{CreditedFeedWorker, FeedConfig};
use fluent_llm::client::ChatBackend;
use fluent_llm::{ChatMessage, LlmError};
use fluent_types::NodeId;
use fluent_wvr::Runtime;

use crate::node_store::{derive_label, ContentNodeStore};

use super::LOD5_LABEL;

/// Configuration for the background tier worker (defaults honor §0.3).
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Max characters for LOD4 (short summary). Default 240.
    pub lod4_max_chars: usize,
    /// Max characters for LOD5 (description). Default 80.
    pub lod5_max_chars: usize,
    /// Max node ids drained per batch.
    pub batch_size: usize,
    /// Poll interval (ms) before giving up on filling a batch.
    pub poll_interval_ms: u64,
    /// Capacity of the pending-id feed.
    pub queue_capacity: usize,
    /// Max concurrent LLM derivations (the `Limiter` cap).
    pub max_concurrent: usize,
    /// Credit granted to the feed's producer up front: the max outstanding
    /// `NodeId`s the async (credit-gated) enqueue path may have in flight
    /// before it blocks. The `CreditSender`'s `is_blocked()` reflects
    /// exhaustion. Default 256.
    pub credit_limit: usize,
    /// How many processed nodes the consumer waits for before bumping credit
    /// back to the producer (`flow::CreditSpec.more_after`). Default 8.
    pub credit_more_after: usize,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            lod4_max_chars: 240,
            lod5_max_chars: 80,
            batch_size: 8,
            poll_interval_ms: 100,
            queue_capacity: 1024,
            max_concurrent: 8,
            credit_limit: 256,
            credit_more_after: 8,
        }
    }
}

/// Errors produced by the tier worker.
#[derive(Debug, thiserror::Error)]
pub enum TierError {
    /// The summarizer/backend is unavailable for lazy derivation — degrade
    /// LOD5 to `derive_label`, leave LOD4 empty (never a crash, no retry).
    #[error("no summarizer backend for tier derivation")]
    NoSummarizer,
    /// A transient backend error — re-enqueue with bounded retry.
    #[error("backend error: {0}")]
    Backend(#[from] LlmError),
    /// The pending-node feed is closed (the worker has been torn down).
    #[error("tier feed closed")]
    FeedClosed,
}

/// A fully derived LOD4 + LOD5 pair for one node.
type DerivedTiers = (Option<String>, Option<String>);

/// The background LOD4/LOD5 derivation worker.
pub struct LedgerTierWorker {
    store: Arc<ContentNodeStore>,
    /// The injected backend — the same transport the `Summarizer` uses; no
    /// second HTTP client.
    backend: Arc<dyn ChatBackend>,
    target_levels: Vec<u8>,
    config: TierConfig,
    /// Bounded re-enqueue attempt counts (avoid infinite retry loops).
    retries: Mutex<HashMap<NodeId, u32>>,
    max_retries: u32,
    /// Latency histogram over the backend derive step (observability).
    metrics: Arc<common_core::metrics::LatencyHistogram>,
    /// The shared feed primitive owning the bounded `mpsc`, the `CreditFlow`
    /// gate, the `Limiter`, and the drain loop.
    feed: Arc<CreditedFeedWorker<NodeId>>,
}

impl LedgerTierWorker {
    /// Construct the worker over a shared store + injected backend. The
    /// `sender` is available via [`LedgerTierWorker::sender`]; attach it to the
    /// store with `store.set_tier_events(worker.sender())`.
    pub fn new(
        store: Arc<ContentNodeStore>,
        backend: Arc<dyn ChatBackend>,
        target_levels: Vec<u8>,
        config: TierConfig,
        runtime: Arc<dyn Runtime>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|weak: &std::sync::Weak<Self>| {
            let feed = Arc::new(CreditedFeedWorker::new(
                runtime,
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
                            this.handle_item(id);
                        }
                    }
                },
            ));
            Self {
                store,
                backend,
                target_levels,
                config,
                retries: Mutex::new(HashMap::new()),
                max_retries: 3,
                metrics: Arc::new(common_core::metrics::LatencyHistogram::new()),
                feed,
            }
        })
    }

    /// The shared latency histogram over tier derivation. Exposed so the
    /// server's metrics surface can aggregate it.
    pub fn metrics(&self) -> Arc<common_core::metrics::LatencyHistogram> {
        Arc::clone(&self.metrics)
    }

    /// The pending-node feed sender; attach to the store via
    /// `store.set_tier_events(worker.sender())`.
    pub fn sender(&self) -> tokio::sync::mpsc::Sender<NodeId> {
        self.feed.sender()
    }

    /// Enqueue a node for background derivation (also used for bounded
    /// re-enqueue of transient failures). Non-blocking: on a full feed it
    /// skips with a debug log — the credit-gated `enqueue_with_credit` path
    /// (agent turns) and the next boot backfill cover the stragglers.
    pub fn enqueue(&self, id: NodeId) {
        self.feed.enqueue(id);
    }

    /// Enqueue a node with chain backpressure: acquires a credit token
    /// (blocking while exhausted) before forwarding the `NodeId`, so a burst
    /// of agent turns cannot grow the feed without bound. The consumer
    /// releases the token via `recv()` after processing each node.
    ///
    /// `Err(FeedClosed)` when the feed is closed (the worker is torn down) —
    /// the caller logs and continues; the node's tiers are filled by the next
    /// boot backfill.
    pub async fn enqueue_with_credit(&self, id: NodeId) -> Result<(), TierError> {
        self.feed
            .enqueue_with_credit(id)
            .await
            .map_err(|_| TierError::FeedClosed)
    }

    /// Whether the credit-gated producer is currently blocked waiting for a
    /// credit bump (i.e. the feed is saturated and the consumer has not yet
    /// processed enough nodes).
    pub fn producer_blocked(&self) -> bool {
        self.feed.is_blocked()
    }

    /// The target LOD levels this worker derives.
    pub fn target_levels(&self) -> &[u8] {
        &self.target_levels
    }

    /// Start the background loop, returning its join handle (the caller holds
    /// it so the worker lives for the process lifetime). Boot backfill first
    /// enqueues every node already missing its target tiers, then the feed's
    /// drain loop takes over. Spawned through the injected `Runtime` — no
    /// ambient `tokio::spawn`.
    pub fn start(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let store = Arc::clone(&self.store);
        let target_levels = self.target_levels.clone();
        self.feed
            .backfill(move || store.node_ids_needing_tier(&target_levels));
        self.feed.start()
    }

    /// Per-node derivation: derive LOD4/LOD5, write them at-most-once, emit
    /// the audit record on full success, degrade on backend failure, and
    /// re-enqueue (bounded) anything still missing tiers. Runs under the
    /// feed's shared `Limiter`; its completion releases the producer's credit
    /// token.
    fn handle_item(&self, id: NodeId) {
        let start = std::time::Instant::now();
        let outcome = self.derive_one(id);
        self.metrics.observe_duration(start);
        match outcome {
            Ok((lod4, lod5)) => {
                self.write_tiers(id, lod4, lod5);
                if self.store.needs_tier(id) {
                    // Partial failure (e.g. LOD4 still empty): bounded retry.
                    self.reenqueue_bounded(id);
                } else {
                    self.retries.lock().unwrap().remove(&id);
                    crate::audit::emit(
                        "tier",
                        serde_json::json!({
                            "node_id": id.as_int(),
                            "levels": self.target_levels,
                        }),
                    );
                }
            }
            Err(e) => {
                // Backend unavailable: degrade LOD5 to derive_label, leave
                // LOD4 empty (never a crash), and re-enqueue with backoff.
                tracing::warn!(
                    target: "router.ledger.tier",
                    node_id = id.as_int(),
                    error = %e,
                    "tier derivation backend error - degrading LOD5, leaving LOD4 empty",
                );
                self.degrade_lod5(id);
                self.reenqueue_bounded(id);
            }
        }
    }

    /// Bounded re-enqueue for a node still missing tiers (up to `max_retries`,
    /// then give up).
    fn reenqueue_bounded(&self, id: NodeId) {
        let mut retries = self.retries.lock().unwrap();
        let count = retries.entry(id).or_insert(0);
        if *count < self.max_retries {
            *count += 1;
            self.feed.enqueue(id);
        } else {
            retries.remove(&id);
            tracing::warn!(
                target: "router.ledger.tier",
                node_id = id.as_int(),
                "tier derivation gave up after max retries",
            );
        }
    }

    /// Derive LOD4 + LOD5 for a node in a **single** backend call (from LOD0
    /// only), so a node is derived at most once (at-most-once).
    fn derive_one(&self, id: NodeId) -> Result<DerivedTiers, TierError> {
        // Snapshot LOD0 + which target tiers still need filling (short read
        // guard — never held across the LLM call).
        let Some(arc) = self.store.get_node(id) else {
            return Ok((None, None));
        };
        let (lod0, need_lod4, need_lod5) = {
            let guard = common_core::sync::lock_read(&arc);
            let lod0 = guard.lod.first().cloned().unwrap_or_default();
            let role = guard.role.as_ref().map(|r| r.as_str().to_string()).unwrap_or_default();
            let lod4_empty = guard.lod.get(4).is_none_or(String::is_empty);
            // LOD5 is eager (deterministic label); it "needs" derivation only
            // when empty or still holding the deterministic placeholder (i.e.
            // not yet upgraded to an LLM description)
            let current_l5 = guard.lod.get(5).cloned().unwrap_or_default();
            let lod5_needs_upgrade = current_l5.is_empty() || current_l5 == derive_label(&role, &lod0);
            (
                lod0,
                self.target_levels.contains(&4) && lod4_empty,
                self.target_levels.contains(&5) && lod5_needs_upgrade,
            )
        };
        if lod0.is_empty() || (!need_lod4 && !need_lod5) {
            return Ok((None, None));
        }

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: format!(
                    "Condense the following content. Return exactly two lines:\n\
                     SUMMARY: <one-line summary under {} characters>\n\
                     DESCRIPTION: <one-phrase description under {} characters>\n\
                     No preamble, no other text.",
                    self.config.lod4_max_chars, self.config.lod5_max_chars
                ),
            },
            ChatMessage {
                role: "user".into(),
                content: lod0,
            },
        ];
        let text = self.backend.chat_complete(&messages).map_err(TierError::Backend)?;
        let (summary, desc) = parse_tiers(&text);
        let lod4 = if need_lod4 {
            summary.map(|s| truncate_chars(&s, self.config.lod4_max_chars))
        } else {
            None
        };
        let lod5 = if need_lod5 {
            desc.map(|s| truncate_chars(&s, self.config.lod5_max_chars))
        } else {
            None
        };
        Ok((lod4, lod5))
    }

    /// Write derived tiers to the shared node via the normal `with_node_mut`
    /// path — **at-most-once**: re-check each tier under the write lock before
    /// writing. LOD5 is upgraded from its deterministic label to the LLM
    /// description exactly once.
    fn write_tiers(&self, id: NodeId, lod4: Option<String>, lod5: Option<String>) {
        if lod4.is_none() && lod5.is_none() {
            return;
        }
        let role = self
            .store
            .snapshot(id)
            .and_then(|n| n.role.clone())
            .map(|r| r.as_str().to_string())
            .unwrap_or_default();
        let lod0 = self
            .store
            .snapshot(id)
            .and_then(|n| n.lod.first().cloned())
            .unwrap_or_default();
        let _ = self.store.with_node_mut(id, |node| {
            while node.lod.len() < LOD5_LABEL as usize + 1 {
                node.lod.push(String::new());
            }
            if let Some(text) = lod4 {
                if node.lod[4].is_empty() {
                    node.lod[4] = text;
                }
            }
            if let Some(text) = lod5 {
                let label = derive_label(&role, &lod0);
                // Upgrade only the deterministic placeholder, exactly once.
                if node.lod[5].is_empty() || node.lod[5] == label {
                    node.lod[5] = text;
                }
            }
        });
    }

    /// No-backend degradation: LOD5 falls back to the deterministic
    /// `derive_label`, LOD4 is left empty (never a crash).
    fn degrade_lod5(&self, id: NodeId) {
        let role = self
            .store
            .snapshot(id)
            .and_then(|n| n.role.clone())
            .map(|r| r.as_str().to_string())
            .unwrap_or_default();
        let lod0 = self
            .store
            .snapshot(id)
            .and_then(|n| n.lod.first().cloned())
            .unwrap_or_default();
        let label = derive_label(&role, &lod0);
        let _ = self.store.with_node_mut(id, |node| {
            while node.lod.len() < LOD5_LABEL as usize + 1 {
                node.lod.push(String::new());
            }
            if node.lod[5].is_empty() {
                node.lod[5] = label;
            }
        });
    }
}

/// Truncate to at most `max_chars` **characters** at a UTF-8 char boundary
/// (the LOD4/LOD5 caps are character counts, not bytes). Never exceeds the cap
/// and never splits a multi-byte char. Builds on `truncate_utf8`'s char-boundary
/// discipline.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let byte_end = s.char_indices().nth(max_chars).map_or(s.len(), |(i, _)| i);
    s[..byte_end].to_string()
}

/// Parse the two-line `SUMMARY:`/`DESCRIPTION:` model output into a
/// `(summary, description)` pair. Falls back to the raw text for the summary
/// when the delimiters are absent (never loses the derivation).
fn parse_tiers(text: &str) -> (Option<String>, Option<String>) {
    let mut summary: Option<String> = None;
    let mut desc: Option<String> = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("SUMMARY:") {
            summary = Some(rest.trim().to_string());
        } else if let Some(rest) = t.strip_prefix("DESCRIPTION:") {
            desc = Some(rest.trim().to_string());
        }
    }
    let summary = summary.or_else(|| Some(text.trim().to_string()));
    (summary, desc)
}
#[cfg(test)]
#[path = "../../tests/ledger_tiering.rs"]
mod tests;
