use super::*;
use std::time::Duration;
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

/// Chain backpressure: with `credit_limit=1` the credit-gated producer
/// (`enqueue_with_credit`) blocks once the token is consumed, and the
/// consumer's `recv()` after processing a node releases it. The store's
/// sync enqueue feed is deliberately NOT attached so the credit accounting
/// is isolated from it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enqueue_with_credit_blocks_then_releases() {
    let store = temp_store();
    let backend: Arc<dyn ChatBackend> = repeating(
        "SUMMARY: short summary here\nDESCRIPTION: a description",
        32,
    );
    let cfg = TierConfig {
        credit_limit: 1,
        credit_more_after: 1,
        poll_interval_ms: 5,
        batch_size: 1,
        ..Default::default()
    };
    let worker = LedgerTierWorker::new(
        Arc::clone(&store),
        backend,
        vec![4, 5],
        cfg,
        fluent_concurrency::tokio_runtime(),
    );
    let id1 = store
        .record_request("sess", "r1", "first node to derive")
        .unwrap();
    let id2 = store
        .record_request("sess", "r2", "second node to derive")
        .unwrap();

    // First enqueue consumes the single credit token.
    worker.enqueue_with_credit(id1).await.unwrap();
    assert!(!worker.producer_blocked(), "credit available -> not blocked");

    // Second enqueue must block: credit exhausted and the worker (not yet
    // started) cannot process anything to release it.
    let w = Arc::clone(&worker);
    let blocked = tokio::spawn(async move { w.enqueue_with_credit(id2).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        !blocked.is_finished(),
        "producer must block while credit is exhausted"
    );
    assert!(
        worker.producer_blocked(),
        "is_blocked() reflects the exhausted credit"
    );

    // Starting the worker lets it process id1; the consumer's recv()
    // releases the token, the blocked enqueue forwards id2, and the worker
    // derives it.
    let handle = worker.start();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if store.snapshot(id2).is_some_and(|n| !n.lod[4].is_empty()) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "id2 never derived after credit release"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    blocked.await.unwrap().unwrap();
    handle.abort();
}
