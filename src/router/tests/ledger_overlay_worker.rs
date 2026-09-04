use super::*;
use std::time::Duration;

use crate::node_store::LLM_OVERLAY_META_KEY;
use fluent_llm::client::ChatBackend;
use fluent_llm::{ChatMessage, LlmError};

fn temp_store() -> Arc<ContentNodeStore> {
    let dir = std::env::temp_dir().join(format!(
        "coral-router-overlay-worker-{}",
        common_core::hash::uuid_v4()
    ));
    let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
    let _ = std::fs::remove_file(&dir);
    store
}

fn config() -> OverlayWorkerConfig {
    OverlayWorkerConfig {
        poll_interval_ms: 5,
        batch_size: 4,
        ..Default::default()
    }
}

/// A store wired with all three seams (deterministic pipeline, hash
/// embedder, and a canned LLM backend).
fn seeded_store() -> Arc<ContentNodeStore> {
    let store = temp_store();
    store.set_overlay_pipeline(Arc::new(
        spacy_rs::NlpPipeline::en_default().expect("en pipeline"),
    ));
    store.set_overlay_embedder(Arc::new(crate::test_stubs::HashEmbedder::new(8)));
    store
}

/// A `ChatBackend` that sleeps before returning — the "artificially slow
/// LLM" that must never delay the spacy/embedding overlays for the same
/// node.
struct SlowBackend {
    millis: u64,
}

impl ChatBackend for SlowBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        std::thread::sleep(Duration::from_millis(self.millis));
        Ok("A slow request summary.".to_string())
    }
}

/// A `ChatBackend` that always fails — the "down LLM backend" used to prove
/// the LLM overlay's failure is contained.
struct FailingBackend;

impl ChatBackend for FailingBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        Err(LlmError::NoResponse)
    }
}

fn status_of(store: &ContentNodeStore, id: NodeId, kind: OverlayKind) -> OverlayStatus {
    store
        .get_node(id)
        .map_or(OverlayStatus::Absent, |arc| {
            lock_read(&arc).overlay(kind).status
        })
}

async fn wait_until(
    store: &ContentNodeStore,
    id: NodeId,
    pred: impl Fn(&ContentNodeStore, NodeId) -> bool,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if pred(store, id) {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "condition never satisfied for node {}",
            id.as_int()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Poll the shared capture buffer until an overlay audit line for `nid`
/// appears. The audit emission is async and schedules independently of the
/// status flip — a single sleep races the parallel runtime, so poll with a
/// deadline instead. Only lines appended after `base` are scanned: node ids
/// restart in every fresh store, so a whole-buffer scan could match a
/// sibling test's stale lines and return before this test's own worker has
/// emitted anything (flaky both ways — see `global_capture_len`).
async fn wait_for_audit(capture: &Arc<Mutex<Vec<String>>>, base: usize, nid: &str) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let fresh = crate::test_support::global_capture_since(capture, base);
        let joined = fresh.join("\n");
        if joined
            .lines()
            .any(|l| l.contains("kind=\"overlay\"") && l.contains(&format!("\"node_id\":{nid}")))
        {
            return;
        }
        if std::time::Instant::now() >= deadline {
            match capture.lock() {
                Ok(all) => {
                    let joined = all.join("\n");
                    panic!(
                        "overlay audit never flushed for node {nid} (fresh lines since base={base}: {}; \
                         whole-buffer lines: {}, audit lines: {})",
                        fresh.len(),
                        all.len(),
                        joined.lines().filter(|l| l.contains("audit")).count(),
                    );
                }
                Err(_) => panic!(
                    "overlay audit never flushed for node {nid}: GLOBAL_CAPTURE mutex POISONED \
                     (the LogCapture writer silently drops on poison)"
                ),
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// M7.5 — parallel independence: a slow LLM enrichment for a node never
/// delays that same node's spacy/embedding overlays completing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_llm_never_delays_spacy_or_embedding_for_same_node() {
    let store = seeded_store();
    store.set_overlay_llm(Arc::new(SlowBackend { millis: 1_000 }));
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    let id = store.record_request("s", "r1", "Show me the sales report").unwrap();

    // The spacy and embedding overlays must reach `ready` while the LLM is
    // still sleeping (not yet `ready`). Poll on a short window so a slow LLM
    // would fail the test if it blocked the fan-out.
    let deadline = std::time::Instant::now() + Duration::from_millis(800);
    let (ann_ready, emb_ready) = loop {
        let ann = status_of(&store, id, OverlayKind::Spacy) == OverlayStatus::Ready;
        let emb = status_of(&store, id, OverlayKind::Embedding) == OverlayStatus::Ready;
        if ann && emb {
            break (ann, emb);
        }
        assert!(
            std::time::Instant::now() < deadline,
            "spacy/embedding delayed behind the slow LLM"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert!(ann_ready && emb_ready);
    assert_ne!(
        status_of(&store, id, OverlayKind::Llm),
        OverlayStatus::Ready,
        "the LLM overlay is still in flight (its slow stub has not returned)"
    );

    // Eventually the LLM overlay also resolves.
    wait_until(&store, id, |s, n| {
        status_of(s, n, OverlayKind::Llm) == OverlayStatus::Ready
    })
    .await;
    let node = store.snapshot(id).unwrap();
    assert!(node.annotation.is_some());
    assert!(node.embedding.is_some());

    worker.abort();
}

/// M7.5 — failure containment: a down LLM backend marks only the LLM
/// overlay `failed`; the spacy and embedding overlays still complete.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_overlay_failure_is_contained() {
    let store = seeded_store();
    store.set_overlay_llm(Arc::new(FailingBackend));
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    let id = store.record_request("s", "r1", "buy apple stock").unwrap();

    wait_until(&store, id, |s, n| {
        status_of(s, n, OverlayKind::Spacy) == OverlayStatus::Ready
            && status_of(s, n, OverlayKind::Embedding) == OverlayStatus::Ready
            && status_of(s, n, OverlayKind::Llm) == OverlayStatus::Failed
    })
    .await;

    let node = store.snapshot(id).unwrap();
    assert!(node.annotation.is_some(), "spacy overlay unaffected");
    assert!(node.embedding.is_some(), "embedding overlay unaffected");
    assert_eq!(
        node.overlay(OverlayKind::Llm).status,
        OverlayStatus::Failed,
        "only the LLM overlay is failed"
    );
    assert_eq!(node.metadata.as_ref().and_then(|m| m.get(LLM_OVERLAY_META_KEY)), None);

    worker.abort();
}

/// M7.5 — boot backfill: nodes recorded before the worker attaches are
/// enqueued by `start`'s backfill and get their overlays derived.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn boot_backfill_derives_preexisting_nodes() {
    let store = seeded_store();
    store.set_overlay_llm(Arc::new(crate::test_stubs::CountingBackend::new(
        "A backfilled summary.",
    )));
    let preexisting = store.record_request("s", "r0", "backfill me").unwrap();

    let worker = OverlayWorker::new(
        Arc::clone(&store),
        config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    // A node created after attach is enqueued on create.
    let created = store.record_request("s", "r1", "create me").unwrap();

    for id in [preexisting, created] {
        wait_until(&store, id, |s, n| {
            status_of(s, n, OverlayKind::Spacy) == OverlayStatus::Ready
                && status_of(s, n, OverlayKind::Embedding) == OverlayStatus::Ready
                && status_of(s, n, OverlayKind::Llm) == OverlayStatus::Ready
        })
        .await;
    }
    assert!(
        store.snapshot(preexisting).unwrap().annotation.is_some(),
        "boot backfill derived the pre-existing node"
    );
    assert!(
        store.snapshot(created).unwrap().annotation.is_some(),
        "enqueue-on-create derived the created node"
    );

    worker.abort();
}

/// M7.5 — clean drain: after `drain`, the worker processes what remains in
/// the feed and exits.
///
/// The node is enqueued manually (no store-attached sender clone), so the
/// worker's own sender is the only one held and `drain` closes the channel
/// — the same post-teardown condition a real shutdown reaches once the
/// store is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drain_completes_cleanly() {
    let store = seeded_store();
    store.set_overlay_llm(Arc::new(crate::test_stubs::CountingBackend::new(
        "A drained summary.",
    )));
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        config(),
        fluent_concurrency::tokio_runtime(),
    );
    let id = store.record_request("s", "r1", "drain me").unwrap();
    worker.enqueue(id);
    worker.start();

    wait_until(&store, id, |s, n| {
        status_of(s, n, OverlayKind::Spacy) == OverlayStatus::Ready
            && status_of(s, n, OverlayKind::Llm) == OverlayStatus::Ready
            && status_of(s, n, OverlayKind::Embedding) == OverlayStatus::Ready
    })
    .await;

    worker.drain().await;
}

/// M7.4 — a `kind = "overlay"` audit record is emitted per derived overlay
/// (success and permanent failure), carrying the node id, kind, and status.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn emits_overlay_audit_records() {
    let capture = crate::test_support::install_global_subscriber();
    let base = crate::test_support::global_capture_len(&capture);

    let store = seeded_store();
    store.set_overlay_llm(Arc::new(FailingBackend));
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    let id = store.record_request("s", "r1", "audit me").unwrap();

    wait_until(&store, id, |s, n| {
        status_of(s, n, OverlayKind::Spacy) == OverlayStatus::Ready
            && status_of(s, n, OverlayKind::Embedding) == OverlayStatus::Ready
            && status_of(s, n, OverlayKind::Llm) == OverlayStatus::Failed
    })
    .await;

    // Give the audit emissions a beat to flush, then scan the shared
    // process-wide buffer for the overlay records for this node. Polled
    // (not a single sleep) so parallel-runtime scheduling never races the
    // worker's async audit emission.
    let nid = id.as_int().to_string();
    wait_for_audit(&capture, base, &nid).await;
    let lines = crate::test_support::global_capture_since(&capture, base);
    let joined = lines.join("\n");
    assert!(
        joined.contains("router.audit"),
        "audit records must land on the router.audit target"
    );
    // The two successful overlays.
for (kind, status) in [("spacy", "ready"), ("embedding", "ready")] {
        assert!(
            joined.contains("kind=\"overlay\"")
                && joined.contains(&format!("\"node_id\":{nid}"))
                && joined.contains(&format!("\"kind\":\"{kind}\""))
                && joined.contains(&format!("\"status\":\"{status}\"")),
            "missing overlay audit for {kind}/{status}, got:\n{joined}"
        );
    }
    // The failed overlay is audited as `failed`, not silently dropped.
    assert!(
        joined.contains("\"kind\":\"llm\"")
            && joined.contains("\"status\":\"failed\"")
            && joined.contains(&format!("\"node_id\":{nid}")),
        "the failed LLM overlay must still emit an audit record"
    );

    worker.abort();
}

/// M7.2/7.5 — the boot backfill is idempotent: running it again enqueues
/// nothing the second time (a derived node needs no overlay).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn backfill_is_idempotent() {
    let store = seeded_store();
    store.set_overlay_llm(Arc::new(crate::test_stubs::CountingBackend::new(
        "A summary.",
    )));
    let worker = OverlayWorker::new(
        Arc::clone(&store),
        config(),
        fluent_concurrency::tokio_runtime(),
    );
    store.set_overlay_events(worker.sender());
    worker.start();

    let id = store.record_request("s", "r1", "once").unwrap();
    wait_until(&store, id, |s, n| {
        status_of(s, n, OverlayKind::Spacy) == OverlayStatus::Ready
            && status_of(s, n, OverlayKind::Llm) == OverlayStatus::Ready
            && status_of(s, n, OverlayKind::Embedding) == OverlayStatus::Ready
    })
    .await;

    // A second backfill (as a fresh boot would run) sees no node needing an
    // overlay.
    assert!(
        store.node_ids_needing_overlays().is_empty(),
        "a fully derived node is not re-enqueued"
    );

    worker.abort();
}
