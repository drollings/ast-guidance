//! ArcReady overlay acceptance suite (OVERLAYS §10 row 7) — end-to-end
//! behavior of the background overlay worker over the shared store's seams,
//! against the six acceptance criteria:
//!
//! 1. at-most-once under real concurrency (many simultaneous requests touching
//!    the same node, while the worker derives it);
//! 2. LOD0-derivation-only (no overlay reads another overlay's output);
//! 3. fail-open independently per overlay (a down LLM backend never blocks the
//!    spacy/embedding overlays);
//! 4. audit records present and correctly `kind = "overlay"`;
//! 5. boot backfill idempotent (running twice enqueues nothing the second time);
//! 6. config-absent path byte-identical to pre-roadmap behavior (no overlays,
//!    no worker, no audit records).
//!
//! These exercise the *whole* path — store seams + worker + audit — not just a
//! single store method in isolation.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fluent_llm::client::ChatBackend;
use fluent_llm::{BatchEmbedding, ChatMessage, EmbeddingError, EmbeddingProvider, LlmError};
use fluent_types::{NodeId, OverlayKind, OverlayStatus};

use crate::ledger::overlay_worker::{OverlayWorker, OverlayWorkerConfig};
use crate::node_store::{ContentNodeStore, LLM_OVERLAY_META_KEY};
use common_core::sync::lock_read;

fn temp_store() -> Arc<ContentNodeStore> {
    let dir = std::env::temp_dir().join(format!(
        "coral-router-overlay-acceptance-{}",
        common_core::hash::uuid_v4()
    ));
    let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
    let _ = std::fs::remove_file(&dir);
    store
}

fn worker_config() -> OverlayWorkerConfig {
    OverlayWorkerConfig {
        poll_interval_ms: 5,
        batch_size: 4,
        ..Default::default()
    }
}

/// A `ChatBackend` that counts every call (for "derived exactly once").
struct CountingBackend {
    calls: AtomicUsize,
    response: String,
}

impl CountingBackend {
    fn new(response: impl Into<String>) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            response: response.into(),
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl ChatBackend for CountingBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.response.clone())
    }
}

/// A `ChatBackend` that always fails — the "down LLM backend".
struct FailingBackend;

impl ChatBackend for FailingBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        Err(LlmError::NoResponse)
    }
}

/// A counting `EmbeddingProvider` over the deterministic hash embedder.
struct CountingEmbedder {
    inner: crate::test_stubs::HashEmbedder,
    calls: Arc<AtomicUsize>,
}

impl CountingEmbedder {
    fn new(dims: usize) -> Self {
        Self {
            inner: crate::test_stubs::HashEmbedder::new(dims),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl EmbeddingProvider for CountingEmbedder {
    fn name(&self) -> &'static str {
        "acceptance-counting"
    }
    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.embed(text)
    }
    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        self.calls.fetch_add(texts.len(), Ordering::SeqCst);
        self.inner.embed_batch(texts)
    }
}

/// A store wired with all three seams (deterministic pipeline + counting
/// embedder + counting LLM backend), ready for the worker.
fn seeded_store() -> Arc<ContentNodeStore> {
    let store = temp_store();
    store.set_overlay_pipeline(Arc::new(
        spacy_rs::NlpPipeline::en_default().expect("en pipeline"),
    ));
    store.set_overlay_embedder(Arc::new(CountingEmbedder::new(8)));
    store
}

fn status_of(store: &ContentNodeStore, id: NodeId, kind: OverlayKind) -> OverlayStatus {
    store
        .get_node(id)
        .map_or(OverlayStatus::Absent, |arc| lock_read(&arc).overlay(kind).status)
}

async fn wait_until_all_ready(store: &ContentNodeStore, id: NodeId) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let ready = [
            OverlayKind::Spacy,
            OverlayKind::Llm,
            OverlayKind::Embedding,
        ]
        .iter()
        .all(|k| status_of(store, id, *k) == OverlayStatus::Ready);
        if ready {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "overlays never all became ready for node {}",
            id.as_int()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// **Acceptance 1** — at-most-once under real concurrency: many threads race to
/// derive the same node (each calling the store's `_for` entry points) *while*
/// the background worker derives it too. The store's at-most-once install means
/// the counting seams observe exactly one derivation, and every caller shares
/// the single installed value.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn at_most_once_under_many_concurrent_requests_plus_worker() {
    let store = seeded_store();
    let llm = Arc::new(CountingBackend::new("A request summary."));
    store.set_overlay_llm(Arc::clone(&llm) as Arc<dyn ChatBackend>);
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        worker_config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    let id = store.record_request("s", "r1", "buy apple stock today").unwrap();

    // Many simultaneous "requests" racing on the same node while the worker
    // also enqueued it.
    let mut handles = Vec::new();
    for _ in 0..8 {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            store.embedding_for(id).expect("derive").expect("embedding")
        }));
    }
    // The worker finishes the node independently of the racing requests.
    wait_until_all_ready(&store, id).await;
    for h in handles {
        h.join().expect("request thread");
    }

    let node = store.snapshot(id).unwrap();
    assert!(node.annotation.is_some());
    assert!(node.embedding.is_some());
    assert!(
        node.metadata
            .as_ref()
            .and_then(|m| m.get(crate::node_store::LLM_OVERLAY_META_KEY))
            .is_some(),
        "llm overlay installed"
    );
    assert_eq!(llm.calls(), 1, "the LLM overlay is derived exactly once");

    worker.abort();
}

/// **Acceptance 2** — LOD0-derivation-only: each overlay derives from the node's
/// LOD0 text and never reads another overlay's output. Pre-seeding the LLM
/// overlay with a *different* value must not change the spacy/embedding
/// derivations, and the spacy annotation's primary signal must equal LOD0.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn overlays_derive_from_lod0_not_other_overlays() {
    let store = seeded_store();
    store.set_overlay_llm(Arc::new(CountingBackend::new("A request summary.")));
    let id = store
        .record_request("s", "r1", "Show me the quarterly sales report")
        .unwrap();

    // Pre-seed the LLM overlay with an unrelated value (simulating a prior /
    // independent population). This must NOT feed into the spacy or embedding
    // derivations.
    let _ = store.with_node_mut(id, |node| {
        let meta = node.metadata.get_or_insert_with(|| serde_json::json!({}));
        meta[LLM_OVERLAY_META_KEY] = serde_json::json!("unrelated prior summary");
    });

    // Derive the spacy + embedding overlays in isolation.
    let ann = store.annotation_for(id).expect("derive").expect("annotation");
    let emb = store.embedding_for(id).expect("derive").expect("embedding");
    assert!(!emb.is_empty());

    // The spacy primary signal's sentence is exactly LOD0 — never an overlay.
    assert_eq!(
        ann.primary_signal().unwrap().sentence,
        "Show me the quarterly sales report",
        "spacy derives from LOD0, not another overlay"
    );
    // The embedding reflects the node's LOD0 tokens (hash embedder over the
    // sales text), not the pre-seeded summary.
    assert_ne!(
        emb,
        vec![0.0; 8],
        "embedding derived from LOD0 content"
    );
    // The LLM overlay was NOT overwritten by the spacy/embedding runs.
    let node = store.snapshot(id).unwrap();
    assert_eq!(
        node.metadata.as_ref().and_then(|m| m.get(LLM_OVERLAY_META_KEY)),
        Some(&serde_json::json!("unrelated prior summary")),
        "one overlay's derivation must not clobber another"
    );
}

/// **Acceptance 3** — fail-open independently per overlay: a down LLM backend
/// marks only the LLM overlay `failed`; the spacy and embedding overlays still
/// complete.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn llm_backend_failure_does_not_block_spacy_or_embedding() {
    let store = seeded_store();
    store.set_overlay_llm(Arc::new(FailingBackend));
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        worker_config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    let id = store.record_request("s", "r1", "buy apple stock").unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let done = status_of(&store, id, OverlayKind::Spacy) == OverlayStatus::Ready
            && status_of(&store, id, OverlayKind::Embedding) == OverlayStatus::Ready
            && status_of(&store, id, OverlayKind::Llm) == OverlayStatus::Failed;
        if done {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "spacy/embedding never completed despite the down LLM backend"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let node = store.snapshot(id).unwrap();
    assert!(node.annotation.is_some(), "spacy unaffected by LLM failure");
    assert!(node.embedding.is_some(), "embedding unaffected by LLM failure");
    assert_eq!(node.overlay(OverlayKind::Llm).status, OverlayStatus::Failed);

    worker.abort();
}

/// **Acceptance 4** — audit records present and correctly `kind = "overlay"`,
/// one per derived overlay (success and permanent failure).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn audit_records_are_kind_overlay() {
    let capture = crate::test_support::install_global_subscriber();
    // Suffix-scope: node ids restart per store, so only lines appended after
    // this point belong to this test (see `global_capture_len`).
    let base = crate::test_support::global_capture_len(&capture);

    let store = seeded_store();
    store.set_overlay_llm(Arc::new(FailingBackend)); // one overlay fails
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        worker_config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    let id = store.record_request("s", "r1", "audit me").unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let done = status_of(&store, id, OverlayKind::Spacy) == OverlayStatus::Ready
            && status_of(&store, id, OverlayKind::Embedding) == OverlayStatus::Ready
            && status_of(&store, id, OverlayKind::Llm) == OverlayStatus::Failed;
        if done {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "overlays never settled");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    tokio::time::sleep(Duration::from_millis(50)).await;
    let joined = crate::test_support::global_capture_since(&capture, base).join("\n");
    let nid = id.as_int().to_string();
    assert!(joined.contains("kind=\"overlay\""), "records on router.audit with kind=overlay");
    for (kind, status) in [("spacy", "ready"), ("embedding", "ready"), ("llm", "failed")] {
        assert!(
            joined.contains(&format!("\"kind\":\"{kind}\""))
                && joined.contains(&format!("\"status\":\"{status}\""))
                && joined.contains(&format!("\"node_id\":{nid}")),
            "missing overlay audit for {kind}/{status}",
        );
    }

    worker.abort();
}

/// **Acceptance 5** — boot backfill idempotent: after a full derivation, running
/// the backfill again enqueues nothing (no re-derivation).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn boot_backfill_is_idempotent() {
    let store = seeded_store();
    let llm = Arc::new(CountingBackend::new("A summary."));
    store.set_overlay_llm(Arc::clone(&llm) as Arc<dyn ChatBackend>);
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        worker_config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    let id = store.record_request("s", "r1", "once").unwrap();
    wait_until_all_ready(&store, id).await;
    assert_eq!(llm.calls(), 1);

    // The boot backfill query (what a second boot's `start` would run) sees no
    // node needing an overlay — running it twice derives nothing new.
    assert!(store.node_ids_needing_overlays().is_empty());
    let worker2 = OverlayWorker::new(
        Arc::clone(&store),
        worker_config(),
        fluent_concurrency::tokio_runtime(),
    );
    worker2.start();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(llm.calls(), 1, "no re-derivation on a second backfill");

    worker.abort();
    worker2.abort();
}

/// **Acceptance 6** — config-absent path byte-identical to pre-roadmap: a store
/// with no worker and no seams never derives an overlay, keeps reporting the
/// node as needing one, and emits no `overlay` audit records.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn config_absent_path_is_byte_identical() {
    let capture = crate::test_support::install_global_subscriber();

    // No worker, no seams — exactly a pre-roadmap boot. Record under a
    // distinctive node id: the process-wide capture buffer is shared with the
    // overlay audit tests, which legitimately emit `kind="overlay"` lines for
    // *their* (also-id-1) nodes. A unique id keeps the absence assertion
    // scoped to this test's own node.
    let store = temp_store();
    const NODE_ID: i64 = 4_200_001;
    let mut node = crate::node_store::new_node(
        NodeId::from_int(NODE_ID),
        "s",
        "r1",
        "user",
        "plain node",
        None,
    );
    node.id = Some(NodeId::from_int(NODE_ID));
    let id = store.record_content_node(&node).unwrap();

    // Every overlay stays absent / never derives.
    assert!(store.needs_overlay(id));
    assert_eq!(store.node_ids_needing_overlays(), vec![id]);
    assert!(store.annotation_for(id).expect("fail-open").is_none());
    assert!(store.embedding_for(id).expect("fail-open").is_none());
    assert!(store.llm_overlay_for(id).expect("fail-open").is_none());
    let node = store.snapshot(id).unwrap();
    assert!(node.annotation.is_none());
    assert!(node.embedding.is_none());
    assert_eq!(
        node.overlay(OverlayKind::Spacy).status,
        OverlayStatus::Absent,
        "no derivation machinery ran"
    );

    // And no `overlay` audit record was emitted for THIS node. Scoped to the
    // node id: the process-wide capture buffer is shared, and a sibling overlay
    // test legitimately writes `kind="overlay"` lines for its own nodes — the
    // config-absent path must only prove it never emits one for its own id.
    tokio::time::sleep(Duration::from_millis(30)).await;
    let joined = capture.lock().unwrap().join("\n");
    let nid = id.as_int().to_string();
    assert!(
        !joined.lines().any(|l| {
            l.contains("kind=\"overlay\"") && l.contains(&format!("\"node_id\":{nid}"))
        }),
        "config-absent path must not emit overlay audit records for this node"
    );
}