//! `LedgerTierWorker` — continuous background LOD4/LOD5 generation.
//!
//! A background task drains a feed of pending node ids (attached to the
//! `ContentNodeStore` via `set_tier_events`) and derives LOD4 (short summary) and
//! LOD5 (LLM description) for each node **from LOD0 only**, caching them on the
//! shared node. It never blocks a request and never recomputes a filled tier
//! (at-most-once, checked-and-set under the write lock).
//!
//! It reuses the shared `ChatBackend` — **no second HTTP client** — and is
//! spawned through the injected `Runtime` (no ambient `tokio::spawn`). It
//! bounds concurrent derivations with a `Limiter` and emits a `kind = "tier"`
//! audit record per derived node.
//!
//! Degradation rules: when the backend is unavailable, LOD5 falls back
//! to the deterministic `derive_label` and LOD4 is left empty — never a crash.
//! Transient backend errors are re-enqueued with bounded retry (no infinite
//! loop).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fluent_llm::client::ChatBackend;
use fluent_llm::{ChatMessage, LlmError};
use fluent_types::NodeId;
use fluent_wvr::Runtime;

use crate::node_store::{derive_label, ContentNodeStore};

use super::{LOD5_LABEL};

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
    runtime: Arc<dyn Runtime>,
    /// The pending-node feed sender (exposed to the store via `set_tier_events`,
    /// and reused for bounded re-enqueue).
    sender: tokio::sync::mpsc::UnboundedSender<NodeId>,
    /// The feed receiver, taken once by the background task at `start`.
    receiver: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<NodeId>>>,
    limiter: Arc<fluent_concurrency::pool::Limiter>,
    /// Bounded re-enqueue attempt counts (avoid infinite retry loops).
    retries: Mutex<HashMap<NodeId, u32>>,
    max_retries: u32,
    /// Latency histogram over the backend derive step (observability).
    metrics: Arc<common_core::metrics::LatencyHistogram>,
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
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let limiter = Arc::new(fluent_concurrency::pool::Limiter::new(
            config.max_concurrent.max(1),
        ));
        Arc::new(Self {
            store,
            backend,
            target_levels,
            config,
            runtime,
            sender,
            receiver: Mutex::new(Some(receiver)),
            limiter,
            retries: Mutex::new(HashMap::new()),
            max_retries: 3,
            metrics: Arc::new(common_core::metrics::LatencyHistogram::new()),
        })
    }

    /// The shared latency histogram over tier derivation. Exposed so the
    /// server's metrics surface can aggregate it.
    pub fn metrics(&self) -> Arc<common_core::metrics::LatencyHistogram> {
        Arc::clone(&self.metrics)
    }

    /// The pending-node feed sender; attach to the store via
    /// `store.set_tier_events(worker.sender())`.
    pub fn sender(&self) -> tokio::sync::mpsc::UnboundedSender<NodeId> {
        self.sender.clone()
    }

    /// Enqueue a node for background derivation (also used for bounded
    /// re-enqueue of transient failures).
    pub fn enqueue(&self, id: NodeId) {
        let _ = self.sender.send(id);
    }

    /// The target LOD levels this worker derives.
    pub fn target_levels(&self) -> &[u8] {
        &self.target_levels
    }

    /// Start the background loop, returning its join handle (the caller holds
    /// it so the worker lives for the process lifetime). Spawned through the
    /// injected `Runtime` — no ambient `tokio::spawn`.
    pub fn start(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let receiver = self
            .receiver
            .lock()
            .unwrap()
            .take()
            .expect("LedgerTierWorker::start must be called at most once");
        let this = Arc::clone(self);
        let runtime = Arc::clone(&this.runtime);
        runtime.spawn(Box::pin(async move {
            this.run(receiver).await;
        }))
    }

    async fn run(self: Arc<Self>, mut receiver: tokio::sync::mpsc::UnboundedReceiver<NodeId>) {
        // Boot backfill: any existing node missing its target tiers.
        for id in self.store.node_ids_needing_tier(&self.target_levels) {
            self.enqueue(id);
        }

        loop {
            // Wait for the first item, then drain up to batch_size.
            let Some(first) = receiver.recv().await else {
                break; // channel closed
            };
            let mut batch = vec![first];
            let mut timer = self
                .runtime
                .sleep(Duration::from_millis(self.config.poll_interval_ms));
            while batch.len() < self.config.batch_size {
                tokio::select! {
                    item = receiver.recv() => match item {
                        Some(id) => batch.push(id),
                        None => break,
                    },
                    () = &mut timer => break,
                }
            }
            self.process_batch(batch).await;
        }
    }

    async fn process_batch(&self, batch: Vec<NodeId>) {
        let mut still_needs: Vec<NodeId> = Vec::new();
        for id in batch {
            let start = std::time::Instant::now();
            let outcome = self
                .limiter
                .run(|| async move { self.derive_one(id) })
                .await;
            self.metrics.observe_duration(start);
            match outcome {
                Ok((lod4, lod5)) => {
                    self.write_tiers(id, lod4, lod5);
                    if self.store.needs_tier(id) {
                        // Partial failure (e.g. LOD4 still empty): bounded retry.
                        still_needs.push(id);
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
                    still_needs.push(id);
                }
            }
        }

        // Bounded re-enqueue for nodes still missing tiers.
        for id in still_needs {
            let mut retries = self.retries.lock().unwrap();
            let count = retries.entry(id).or_insert(0);
            if *count < self.max_retries {
                *count += 1;
                self.enqueue(id);
            } else {
                retries.remove(&id);
                tracing::warn!(
                    target: "router.ledger.tier",
                    node_id = id.as_int(),
                    "tier derivation gave up after max retries",
                );
            }
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
            let role = guard.role.clone().unwrap_or_default();
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
mod tests {
    use super::*;
    use crate::test_stubs::{CountingBackend, StubChatBackend};
    use crate::views::ParallelLedger;

    fn temp_store() -> Arc<ContentNodeStore> {
        let dir = std::env::temp_dir().join(format!(
            "coral-router-tiering-{}",
            common_core::hash::uuid_v4()
        ));
        let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
        let _ = std::fs::remove_file(&dir);
        store
    }

    fn config() -> TierConfig {
        TierConfig {
            poll_interval_ms: 5,
            batch_size: 4,
            ..Default::default()
        }
    }

    /// A stub backend with enough copies of a response to serve every call in a
    /// test (the boot-backfill + create-enqueue paths can both enqueue nodes).
    fn repeating(response: &str, copies: usize) -> Arc<StubChatBackend> {
        Arc::new(StubChatBackend::new(vec![response.to_string(); copies]))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn background_fill_observable_without_render() {
        let store = temp_store();
        let backend: Arc<dyn ChatBackend> = repeating(
            "SUMMARY: short summary here\nDESCRIPTION: a description",
            8,
        );
        let worker = LedgerTierWorker::new(
            Arc::clone(&store),
            backend,
            vec![4, 5],
            config(),
            fluent_concurrency::tokio_runtime(),
        );
        store.set_tier_events(worker.sender());
        let handle = worker.start();

        // Create a node; LOD4/LOD5 must fill in the background WITHOUT any
        // `render()`/`lod_text` lazy call being made first.
        let id = store
            .record_request("sess", "r1", "The full text to derive tiers from.")
            .unwrap();

        // Poll until the background worker fills LOD4/LOD5.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            let node = store.snapshot(id).unwrap();
            if !node.lod[4].is_empty() && node.lod[5].contains("a description") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let node = store.snapshot(id).unwrap();
        assert_eq!(node.lod[4], "short summary here", "LOD4 filled in background");
        assert_eq!(node.lod[5], "a description", "LOD5 upgraded to LLM description");

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn at_most_once_under_concurrent_views() {
        let store = temp_store();
        let backend = Arc::new(CountingBackend::new(
            "SUMMARY: once\nDESCRIPTION: desc once",
        ));
        let worker = LedgerTierWorker::new(
            Arc::clone(&store),
            backend.clone(),
            vec![4, 5],
            config(),
            fluent_concurrency::tokio_runtime(),
        );
        store.set_tier_events(worker.sender());
        let handle = worker.start();

        let id = store
            .record_request("sess", "r1", "derive exactly once")
            .unwrap();

        // Two "concurrent views" sharing the store observe the same node.
        let _v1 = ParallelLedger::for_session(Arc::clone(&store), "sess");
        let _v2 = ParallelLedger::for_session(Arc::clone(&store), "sess");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            let node = store.snapshot(id).unwrap();
            if !node.lod[4].is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let node = store.snapshot(id).unwrap();
        assert!(!node.lod[4].is_empty());
        assert_eq!(
            backend.calls(),
            1,
            "a node is derived exactly once (at-most-once), got {}",
            backend.calls()
        );

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn enqueue_on_create_and_backfill_on_boot() {
        let store = temp_store();
        // A pre-existing node (recorded before the worker attaches) missing LOD4.
        let preexisting = store
            .record_request("sess", "r0", "backfill me")
            .unwrap();

        let backend: Arc<dyn ChatBackend> =
            repeating("SUMMARY: backfilled\nDESCRIPTION: desc", 8);
        let worker = LedgerTierWorker::new(
            Arc::clone(&store),
            backend,
            vec![4, 5],
            config(),
            fluent_concurrency::tokio_runtime(),
        );
        store.set_tier_events(worker.sender());
        let handle = worker.start();

        // A node created after attach is enqueued on create.
        let created = store
            .record_request("sess", "r1", "create me")
            .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            let a = store.snapshot(preexisting).unwrap();
            let b = store.snapshot(created).unwrap();
            if !a.lod[4].is_empty() && !b.lod[4].is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(!store.snapshot(preexisting).unwrap().lod[4].is_empty(), "boot backfill filled LOD4");
        assert!(!store.snapshot(created).unwrap().lod[4].is_empty(), "enqueue-on-create filled LOD4");

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_summarizer_degrades_lod5_and_leaves_lod4_empty() {
        let store = temp_store();
        // A backend that always fails mimics "no summarizer".
        let failing = Arc::new(StubChatBackend::new(vec![])); // empty -> NoResponse
        let worker = LedgerTierWorker::new(
            Arc::clone(&store),
            failing,
            vec![4, 5],
            config(),
            fluent_concurrency::tokio_runtime(),
        );
        store.set_tier_events(worker.sender());
        let handle = worker.start();

        let id = store
            .record_request("sess", "r1", "Some content to label deterministically.")
            .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            let node = store.snapshot(id).unwrap();
            if !node.lod[5].is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let node = store.snapshot(id).unwrap();
        // LOD5 falls back to the deterministic label; LOD4 stays empty.
        assert!(!node.lod[5].is_empty(), "LOD5 degraded to derive_label");
        assert!(
            node.lod[4].is_empty(),
            "LOD4 left empty on backend failure (no crash)"
        );
        assert!(
            store.snapshot(id).is_some(),
            "node still present after degradation"
        );

        handle.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn budget_enforced_via_truncation() {
        let store = temp_store();
        let backend: Arc<dyn ChatBackend> = repeating(
            &format!(
                "SUMMARY: {}\nDESCRIPTION: {}",
                "x".repeat(500),
                "y".repeat(300),
            ),
            8,
        );
        let cfg = TierConfig {
            lod4_max_chars: 240,
            lod5_max_chars: 80,
            poll_interval_ms: 5,
            ..Default::default()
        };
        let worker = LedgerTierWorker::new(
            Arc::clone(&store),
            backend,
            vec![4, 5],
            cfg,
            fluent_concurrency::tokio_runtime(),
        );
        store.set_tier_events(worker.sender());
        let handle = worker.start();

        let id = store
            .record_request("sess", "r1", "truncation test content")
            .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            let node = store.snapshot(id).unwrap();
            if !node.lod[4].is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let node = store.snapshot(id).unwrap();
        assert!(node.lod[4].len() <= 240, "LOD4 truncated to <= 240, got {}", node.lod[4].len());
        assert!(node.lod[5].len() <= 80, "LOD5 truncated to <= 80, got {}", node.lod[5].len());

        handle.abort();
    }

    #[test]
    fn parse_tiers_parses_summary_and_description() {
        let (s, d) = parse_tiers("SUMMARY: hi\nDESCRIPTION: yo");
        assert_eq!(s.as_deref(), Some("hi"));
        assert_eq!(d.as_deref(), Some("yo"));
    }

    #[test]
    fn truncate_chars_never_exceeds_char_cap() {
        assert_eq!(truncate_chars("hello", 5), "hello");
        assert_eq!(truncate_chars("hello", 3), "hel");
        assert_eq!(truncate_chars("hello", 0), "");
        // Multi-byte: never splits a char.
        let s = "héllo";
        assert_eq!(truncate_chars(s, 4), "héll");
        assert!(truncate_chars(s, 4).chars().count() <= 4);
    }

    #[test]
    fn parse_tiers_falls_back_to_full_text_summary() {
        let (s, d) = parse_tiers("plain text no delimiters");
        assert_eq!(s.as_deref(), Some("plain text no delimiters"));
        assert_eq!(d, None);
    }

    #[test]
    fn node_ids_needing_tier_returns_only_unfilled() {
        let store = temp_store();
        let id = store.record_request("sess", "r1", "text").unwrap();
        let ids = store.node_ids_needing_tier(&[4]);
        assert_eq!(ids, vec![id], "LOD4 empty -> needs tier");
        let none = store.node_ids_needing_tier(&[5]);
        assert!(none.is_empty(), "LOD5 eager -> no tier needed");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn derive_records_latency_metric() {
        // Each backend derive step records into the shared histogram.
        let store = temp_store();
        let backend: Arc<dyn ChatBackend> = repeating(
            "SUMMARY: s\nDESCRIPTION: d",
            8,
        );
        let worker = LedgerTierWorker::new(
            Arc::clone(&store),
            backend,
            vec![4, 5],
            config(),
            fluent_concurrency::tokio_runtime(),
        );
        store.set_tier_events(worker.sender());
        let handle = worker.start();
        let id = store.record_request("sess", "r1", "metric text").unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if !store.snapshot(id).unwrap().lod[4].is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            worker.metrics().count() > 0,
            "tier derivation must record a latency observation"
        );
        handle.abort();
    }
}
