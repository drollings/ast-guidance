use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn runtime() -> Arc<dyn Runtime> {
    crate::tokio_runtime()
}

#[tokio::test(start_paused = true)]
async fn credit_one_blocks_second_submit_until_processed() {
    let pool = Arc::new(CreditGatedPool::new(
        runtime(),
        1,
        8,
        |_: usize| async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
        },
    ));
    // The first submit consumes the only credit token while the job runs.
    pool.submit(1).await.expect("first submit");
    // The second submit must block until the first job's `recv()` releases it.
    let p = Arc::clone(&pool);
    let second = tokio::spawn(async move { p.submit(2).await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(pool.is_blocked(), "second submit blocked on credit");
    second
        .await
        .expect("second submit completes after credit release")
        .expect("ok");
    assert!(!pool.is_blocked(), "credit released once the job processed");
    pool.drain().await;
}

#[tokio::test(start_paused = true)]
async fn is_blocked_reflects_exhaustion() {
    let pool = Arc::new(CreditGatedPool::new(
        runtime(),
        1,
        8,
        |_: usize| async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
        },
    ));
    assert!(!pool.is_blocked());
    pool.submit(1).await.expect("first submit consumes the credit");
    // A second submit must block on the exhausted credit and flip the flag.
    let p = Arc::clone(&pool);
    let second = tokio::spawn(async move { p.submit(2).await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(pool.is_blocked(), "credit exhausted → second submit blocked");
    second.await.expect("second completes").expect("ok");
    assert!(!pool.is_blocked(), "credit replenished after the job processed");
    pool.drain().await;
}

#[tokio::test(start_paused = true)]
async fn handler_runs_once_and_releases_credit_per_job() {
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let pool = Arc::new(CreditGatedPool::new(
        runtime(),
        1,
        8,
        move |_: usize| {
            let c = Arc::clone(&c);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        },
    ));
    // With credit 1, each submit proceeds only after the previous job's
    // `recv()` has released its token — the handler runs exactly once per
    // job and replenishes credit each time.
    for i in 0..3 {
        pool.submit(i).await.expect("submit");
        for _ in 0..100 {
            if counter.load(Ordering::SeqCst) >= i + 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(counter.load(Ordering::SeqCst), i + 1, "job processed");
        assert!(!pool.is_blocked(), "credit released after each job");
    }
    pool.drain().await;
}

#[tokio::test]
async fn drained_pool_returns_closed() {
    let pool = CreditGatedPool::new(runtime(), 4, 8, |_: usize| async move {});
    pool.drain().await;
    assert!(matches!(pool.submit(1).await, Err(PoolError::Closed)));
}

#[tokio::test(start_paused = true)]
async fn queue_at_capacity_applies_backpressure() {
    // Credit is high so the gate never blocks; a submit that would
    // overflow the tiny queue instead waits for a worker to free a slot.
    // `PoolError::Full` is never returned on the submit path — it is the
    // synchronous `push` fast-path's error (see `pool.rs`).
    let pool = Arc::new(CreditGatedPool::new(
        runtime(),
        100,
        1,
        |_: usize| async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
        },
    ));
    // Three submits: two occupy the workers, one sits in the queue.
    for i in 1..=3 {
        pool.submit(i).await.expect("submit");
    }
    // The fourth submit must wait for a worker slot (never errors).
    let p = Arc::clone(&pool);
    let fourth = tokio::spawn(async move { p.submit(4).await });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!fourth.is_finished(), "submit waits while the queue is full");
    // Once the workers finish, the queued and waiting submits drain.
    tokio::time::sleep(Duration::from_millis(200)).await;
    fourth.await.expect("fourth completes").expect("ok");
    pool.drain().await;
}
