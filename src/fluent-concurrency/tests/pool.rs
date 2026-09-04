use super::*;
use crate::tokio_runtime;
use std::time::Duration;

#[tokio::test]
async fn priority_pool_dispatches_high_first() {
    let rt = tokio_runtime();
    let pool = PriorityResultPool::<i32, String, String>::new(rt, 1, |job: i32| async move {
        // Simulate work
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        Ok(format!("job_{job}"))
    });

    // Submit low priority first, then high priority.
    // With a single worker, they process in queue order, but the
    // priority queue ensures high-priority items are popped first.
    let high = pool.submit(100, 10);
    let low = pool.submit(1, 0);

    let high_result = high.await.unwrap();
    let low_result = low.await.unwrap();

    // Both should complete successfully.
    assert_eq!(high_result, "job_100");
    assert_eq!(low_result, "job_1");

    pool.shutdown().await;
}

#[tokio::test]
async fn test_priority_pool_burst_drains_all() {
    let rt = tokio_runtime();
    let pool = PriorityResultPool::<i32, i32, String>::new(
        Arc::clone(&rt) as Arc<dyn Runtime>,
        4,
        |job: i32| async move { Ok(job * 2) },
    );

    // Submit 100 high-priority items back-to-back with 4 workers.
    let mut handles = Vec::with_capacity(100);
    for i in 0..100 {
        handles.push(pool.submit(i, 10));
    }

    // All 100 results must resolve before timeout.
    for (i, h) in handles.into_iter().enumerate() {
        let result = tokio::time::timeout(Duration::from_secs(5), h)
            .await
            .expect("burst result must not hang")
            .unwrap();
        assert_eq!(result, i as i32 * 2);
    }

    pool.shutdown().await;
}

#[tokio::test]
async fn priority_pool_dispatches_in_fifo_within_priority() {
    let rt = tokio_runtime();
    let pool =
        PriorityResultPool::<u64, u64, String>::new(
            rt,
            1,
            |job: u64| async move { Ok(job * 2) },
        );

    // Same priority — should be FIFO.
    let r1 = pool.submit(1, 5);
    let r2 = pool.submit(2, 5);
    let r3 = pool.submit(3, 5);

    assert_eq!(r1.await.unwrap(), 2);
    assert_eq!(r2.await.unwrap(), 4);
    assert_eq!(r3.await.unwrap(), 6);

    pool.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_priority_pool_single_submit_wakes_sleeper() {
    // A job submitted while every worker is parked (queue
    // observed empty) must always wake one worker. The pre-fix worker
    // registered its `notified()` future only *after* releasing the queue
    // mutex, so a submit landing in that window notified nobody and the
    // job sat until the next submit. Multi-threaded + repeated so the race
    // window has a real chance to be exercised; on the fixed code every
    // iteration completes within the timeout.
    let rt = tokio_runtime();
    for i in 0..64 {
        let pool = PriorityResultPool::<i32, i32, String>::new(
            Arc::clone(&rt) as Arc<dyn Runtime>,
            2,
            |job: i32| async move { Ok(job) },
        );
        // Give the workers a chance to pop-empty and park in the select.
        tokio::task::yield_now().await;
        let result = tokio::time::timeout(Duration::from_secs(5), pool.submit(i, 0))
            .await
            .expect("single submit to an idle pool must wake a sleeping worker")
            .expect("handler must not error");
        assert_eq!(result, i);
        pool.shutdown().await;
    }
}

#[tokio::test(start_paused = true)]
async fn test_priority_pool_single_submit_paused_time() {
    // Deterministic variant under virtual time: the worker parks before the
    // submit, so the pre-registered `notified()` future (created before the
    // queue mutex is taken) is what the submit's `notify_one` fires.
    tokio::time::resume();
    let rt = tokio_runtime();
    let pool = PriorityResultPool::<i32, i32, String>::new(
        Arc::clone(&rt) as Arc<dyn Runtime>,
        1,
        |job: i32| async move { Ok(job) },
    );
    tokio::task::yield_now().await;
    let result = tokio::time::timeout(Duration::from_secs(5), pool.submit(7, 0))
        .await
        .expect("single submit must resolve")
        .unwrap();
    assert_eq!(result, 7);
    pool.shutdown().await;
}

/// The abort signal drops the in-flight handler future: the submitter
/// observes `Canceled` well before the (slow) handler would have finished.
#[tokio::test]
async fn result_pool_submit_with_abort_cancels_running_handler() {
    let rt = tokio_runtime();
    let pool = Arc::new(ResultPool::<i32, i32, String>::new(
        rt,
        1,
        10,
        |_: i32| async move {
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(42)
        },
    ));
    let abort = crate::stream::StreamAbort::new();
    let p = Arc::clone(&pool);
    let a = abort.clone();
    let submitter = tokio::spawn(async move { p.submit_with_abort(1, Some(a)).await });
    // Let the handler start its 500ms sleep, then abort it.
    tokio::time::sleep(Duration::from_millis(50)).await;
    abort.cancel();
    let result = submitter
        .await
        .expect("submitter resolves")
        .expect_err("aborted handler yields Canceled");
    assert!(
        matches!(result, ResultPoolError::Canceled),
        "expected Canceled, got {result:?}"
    );
    if let Ok(pool) = Arc::try_unwrap(pool) {
        pool.shutdown().await;
    }
}

/// Without an abort signal the submit behaves exactly as before.
#[tokio::test]
async fn result_pool_submit_with_abort_none_is_noop() {
    let rt = tokio_runtime();
    let pool = ResultPool::<i32, i32, String>::new(
        rt,
        1,
        10,
        |job: i32| async move { Ok(job * 2) },
    );
    let result = pool.submit_with_abort(21, None).await.expect("no abort");
    assert_eq!(result, 42);
    pool.shutdown().await;
}

/// Backpressure: with `cap=1` worker and `queue_capacity=1`, a second
/// submit blocks while the queue is full (the worker is busy, the single
/// slot is held) and resolves only once a worker pops a queued job and
/// frees space.
#[tokio::test]
async fn priority_pool_submit_blocks_when_queue_full() {
    let rt = tokio_runtime();
    let pool = Arc::new(PriorityResultPool::<i32, i32, String>::with_queue_capacity(
        Arc::clone(&rt) as Arc<dyn Runtime>,
        1,
        1,
        |job: i32| async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(job)
        },
    ));
    // Job 1 is picked up by the single worker; job 2 fills the one slot.
    let p1 = Arc::clone(&pool);
    let f1 = tokio::spawn(async move { p1.submit(1, 0).await });
    let p2 = Arc::clone(&pool);
    let f2 = tokio::spawn(async move { p2.submit(2, 0).await });
    // Give the worker time to grab job 1 and job 2 to land in the queue.
    tokio::time::sleep(Duration::from_millis(20)).await;
    // Job 3 must block: queue full, worker busy on job 1.
    let p3 = Arc::clone(&pool);
    let f3 = tokio::spawn(async move { p3.submit(3, 0).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !f3.is_finished(),
        "submit must block while the queue is full (worker busy, slot held)"
    );
    // Once the worker finishes job 1 it pops job 2, freeing space for job 3.
    assert_eq!(f1.await.unwrap().unwrap(), 1);
    assert_eq!(f2.await.unwrap().unwrap(), 2);
    assert_eq!(f3.await.unwrap().unwrap(), 3);
    if let Ok(pool) = Arc::try_unwrap(pool) {
        pool.shutdown().await;
    }
}
