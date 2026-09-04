use super::*;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

fn runtime() -> Arc<dyn Runtime> {
    crate::tokio_runtime()
}

fn config() -> FeedConfig {
    FeedConfig {
        poll_interval_ms: 5,
        batch_size: 4,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_runs_once_per_item_and_credit_released_after_completion() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let worker = Arc::new(CreditedFeedWorker::new(
        runtime(),
        FeedConfig {
            credit_limit: 1,
            credit_more_after: 1,
            poll_interval_ms: 5,
            ..config()
        },
        move |_: usize| {
            let c = Arc::clone(&c);
            async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                c.fetch_add(1, AtomicOrdering::SeqCst);
            }
        },
    ));

    // Enqueue through the credit-gated path. With credit 1, the second
    // enqueue must wait for the first item's handler to complete.
    worker.enqueue_with_credit(1).await.expect("first enqueue");
    assert!(!worker.is_blocked(), "credit available after first enqueue");
    let w = Arc::clone(&worker);
    let second = tokio::spawn(async move { w.enqueue_with_credit(2).await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(worker.is_blocked(), "credit exhausted -> second enqueue blocked");
    assert!(
        !second.is_finished(),
        "enqueue must block until the handler releases credit"
    );

    // Starting the worker lets it process item 1; the handler's completion
    // (not the enqueue) releases the token and the blocked enqueue proceeds.
    let handle = worker.start();
    second.await.expect("blocked enqueue completes").expect("ok");
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while counter.load(AtomicOrdering::SeqCst) < 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "both items never processed"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(counter.load(AtomicOrdering::SeqCst), 2, "one run per item");
    assert!(!worker.is_blocked(), "credit released after processing");
    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enqueue_skips_on_full_feed() {
    // A tiny feed with a handler that never completes: the non-blocking
    // `enqueue` fills the channel then skips (never blocks, never panics).
    let processed = Arc::new(AtomicUsize::new(0));
    let p = Arc::clone(&processed);
    let worker = Arc::new(CreditedFeedWorker::new(
        runtime(),
        FeedConfig {
            queue_capacity: 2,
            credit_limit: 100,
            poll_interval_ms: 5,
            batch_size: 1,
            ..config()
        },
        move |_: usize| {
            let p = Arc::clone(&p);
            async move {
                p.fetch_add(1, AtomicOrdering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        },
    ));
    for i in 0..5 {
        worker.enqueue(i);
    }
    // The worker has not started: only `queue_capacity` items are buffered.
    assert_eq!(processed.load(AtomicOrdering::SeqCst), 0, "no handler yet");
    let handle = worker.start();
    // Eventually the buffered items process; the skipped ones are lost by
    // design (the credit-gated path / boot backfill cover stragglers).
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while processed.load(AtomicOrdering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "never processed");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backfill_is_bounded_by_feed_capacity() {
    // Backfill with more items than the feed can hold: bounded by the
    // channel capacity (never unbounded growth, never blocks the caller).
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let worker = Arc::new(CreditedFeedWorker::new(
        runtime(),
        FeedConfig {
            queue_capacity: 4,
            credit_limit: 100,
            poll_interval_ms: 5,
            batch_size: 2,
            ..config()
        },
        move |_: usize| {
            let c = Arc::clone(&c);
            async move {
                c.fetch_add(1, AtomicOrdering::SeqCst);
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        },
    ));
    worker.backfill(|| (0..100).collect());
    let handle = worker.start();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while counter.load(AtomicOrdering::SeqCst) < 4 {
        assert!(
            std::time::Instant::now() < deadline,
            "only the buffered items process, got {}",
            counter.load(AtomicOrdering::SeqCst)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        counter.load(AtomicOrdering::SeqCst),
        4,
        "backfill bounded by queue capacity, not the source size"
    );
    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_concurrent_bounds_handler_invocations() {
    // max_concurrent = 1: even with many items queued, at most one handler
    // runs at a time (verified by an in-flight counter).
    let in_flight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let f = Arc::clone(&in_flight);
    let p = Arc::clone(&peak);
    let worker = Arc::new(CreditedFeedWorker::new(
        runtime(),
        FeedConfig {
            queue_capacity: 64,
            credit_limit: 64,
            credit_more_after: 64,
            max_concurrent: 1,
            poll_interval_ms: 5,
            batch_size: 8,
            ..config()
        },
        move |_: usize| {
            let f = Arc::clone(&f);
            let p = Arc::clone(&p);
            async move {
                let cur = f.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                p.fetch_max(cur, AtomicOrdering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                f.fetch_sub(1, AtomicOrdering::SeqCst);
            }
        },
    ));
    for i in 0..16 {
        worker.enqueue(i);
    }
    let handle = worker.start();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while in_flight.load(AtomicOrdering::SeqCst) != 0 || peak.load(AtomicOrdering::SeqCst) < 1 {
        assert!(std::time::Instant::now() < deadline, "never processed");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        peak.load(AtomicOrdering::SeqCst) <= 1,
        "max_concurrent=1 -> peak in-flight {}",
        peak.load(AtomicOrdering::SeqCst)
    );
    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drain_lets_queued_items_complete_then_exits() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let worker = Arc::new(CreditedFeedWorker::new(
        runtime(),
        FeedConfig {
            queue_capacity: 32,
            credit_limit: 32,
            credit_more_after: 32,
            poll_interval_ms: 5,
            batch_size: 4,
            ..config()
        },
        move |_: usize| {
            let c = Arc::clone(&c);
            async move {
                c.fetch_add(1, AtomicOrdering::SeqCst);
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        },
    ));
    for i in 0..8 {
        worker.enqueue(i);
    }
    let handle = worker.start();
    // Wait until some items have been consumed, then drain.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while counter.load(AtomicOrdering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "never started");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    worker.drain();
    // The loop processes whatever remains in the feed, then exits cleanly.
    let finished = tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("drain completes within timeout")
        .expect("no panic");
    assert_eq!(finished, ());
    assert_eq!(
        counter.load(AtomicOrdering::SeqCst),
        8,
        "drain lets all queued items complete"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drained_feed_rejects_new_items() {
    let worker = Arc::new(CreditedFeedWorker::new(
        runtime(),
        FeedConfig {
            queue_capacity: 8,
            credit_limit: 8,
            ..config()
        },
        |_: usize| async {},
    ));
    worker.drain();
    worker.enqueue(1); // no-op, never blocks
    assert!(matches!(
        worker.enqueue_with_credit(2).await,
        Err(FeedError::FeedClosed)
    ));
}
