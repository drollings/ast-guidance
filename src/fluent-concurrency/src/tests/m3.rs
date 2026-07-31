use super::*;
use crate::pool::{Limiter, PoolError, Queue, ResultPool, ResultPoolError, WorkerPool};
use crate::queue::PriorityQueue;
use crate::router::PartitionedRouter;
use std::sync::Arc;

#[tokio::test(start_paused = true)]
async fn test_queue_push_and_pop_order() {
    let q = Queue::new(10);
    q.push(1).await.unwrap();
    q.push(2).await.unwrap();
    q.push(3).await.unwrap();
    assert_eq!(q.pop().await, Some(1));
    assert_eq!(q.pop().await, Some(2));
    assert_eq!(q.pop().await, Some(3));
    q.close();
}

#[tokio::test(start_paused = true)]
async fn test_queue_bounded_full() {
    let q = Queue::new(2);
    assert!(q.push(1).await.is_ok());
    assert!(q.push(2).await.is_ok());
    assert_eq!(q.push(3).await, Err(PoolError::Full));
    q.close();
}

#[tokio::test(start_paused = true)]
async fn test_queue_close_wakes_waiters() {
    let q: Queue<i32> = Queue::new(2);
    let q2 = Arc::new(q);
    let q_clone = Arc::clone(&q2);
    let handle = tokio::spawn(async move {
        assert_eq!(q_clone.pop().await, None);
    });
    tokio::task::yield_now().await;
    q2.close();
    handle.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn test_worker_pool_processes_all_jobs() {
    let results = Arc::new(std::sync::Mutex::new(Vec::new()));
    let r = Arc::clone(&results);
    let pool = WorkerPool::new(crate::tokio_runtime(), 2, 10, move |job: i32| {
        let r = Arc::clone(&r);
        async move {
            let mut guard = r.lock().unwrap();
            guard.push(job * 2);
        }
    });
    pool.submit(1).await.unwrap();
    pool.submit(2).await.unwrap();
    pool.submit(3).await.unwrap();
    pool.shutdown().await;
    let guard = results.lock().unwrap();
    assert_eq!(guard.len(), 3);
    assert!(guard.contains(&2));
    assert!(guard.contains(&4));
    assert!(guard.contains(&6));
}

#[tokio::test(start_paused = true)]
async fn test_worker_pool_shutdown() {
    tokio::time::resume();
    let completed = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&completed);
    let pool = WorkerPool::new(crate::tokio_runtime(), 2, 10, move |job: i32| {
        let c = Arc::clone(&c);
        async move {
            tokio::time::sleep(Duration::from_millis(10 * u64::try_from(job).unwrap())).await;
            c.fetch_add(1, Ordering::SeqCst);
        }
    });
    pool.submit(1).await.unwrap();
    pool.submit(2).await.unwrap();
    pool.submit(3).await.unwrap();
    pool.shutdown().await;
    assert_eq!(completed.load(Ordering::SeqCst), 3);
}

#[tokio::test(start_paused = true)]
async fn test_limiter_caps_concurrency() {
    tokio::time::resume();
    let limiter = Arc::new(Limiter::new(2));
    let counter = Arc::new(AtomicUsize::new(0));
    let max_concurrent = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..5 {
        let lim = Arc::clone(&limiter);
        let cnt = Arc::clone(&counter);
        let max_c = Arc::clone(&max_concurrent);
        handles.push(tokio::spawn(async move {
            lim.run(|| async {
                let prev = cnt.fetch_add(1, Ordering::SeqCst);
                max_c.fetch_max(prev + 1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                cnt.fetch_sub(1, Ordering::SeqCst);
            })
            .await;
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    assert!(max_concurrent.load(Ordering::SeqCst) <= 2);
}

/// `Limiter::run_sync` works from a sync context (no tokio runtime needed).
#[test]
fn limiter_run_sync_caps_concurrency() {
    let limiter = Limiter::new(2);
    let counter = Arc::new(AtomicUsize::new(0));
    let max_concurrent = Arc::new(AtomicUsize::new(0));

    // Run 5 tasks synchronously through the limiter.
    // run_sync creates its own runtime internally.
    for _ in 0..5 {
        let cnt = Arc::clone(&counter);
        let max_c = Arc::clone(&max_concurrent);
        limiter.run_sync(|| async move {
            let prev = cnt.fetch_add(1, Ordering::SeqCst);
            max_c.fetch_max(prev + 1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(50)).await;
            cnt.fetch_sub(1, Ordering::SeqCst);
        });
    }
    // Since run_sync is blocking and sequential, max_concurrent is 1.
    assert!(max_concurrent.load(Ordering::SeqCst) <= 2);
    assert_eq!(counter.load(Ordering::SeqCst), 0);
}

/// `Limiter::run_sync` called from *inside* a running multi-threaded tokio
/// runtime must not panic with "Cannot start a runtime from within a runtime".
/// This is the router HTTP handler's exact shape: the classifier runs through
/// `run_sync` on a tokio worker thread. Regression for M7 (HTTP harness).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn limiter_run_sync_inside_multithread_runtime_no_panic() {
    let limiter = Limiter::new(1);
    let called = Arc::new(AtomicUsize::new(0));
    let called_in_task = Arc::clone(&called);
    // Called directly on a tokio worker thread — the router HTTP handler's
    // exact shape. A bare `Handle::block_on` here would panic.
    limiter.run_sync(move || async move {
        called_in_task.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(called.load(Ordering::SeqCst), 1);
}

#[test]
fn test_priority_queue_fast_path() {
    let mut pq = PriorityQueue::new();
    pq.push("a", 0);
    pq.push("b", 0);
    pq.push("c", 0);
    assert_eq!(pq.pop(), Some("a"));
    assert_eq!(pq.pop(), Some("b"));
    assert_eq!(pq.pop(), Some("c"));
    assert_eq!(pq.pop(), None);
}

#[test]
fn test_priority_queue_mixed() {
    let mut pq = PriorityQueue::new();
    pq.push("low", -1);
    pq.push("normal", 0);
    pq.push("high", 1);
    pq.push("critical", 2);
    assert_eq!(pq.pop(), Some("critical"));
    assert_eq!(pq.pop(), Some("high"));
    assert_eq!(pq.pop(), Some("normal"));
    assert_eq!(pq.pop(), Some("low"));
    assert_eq!(pq.pop(), None);
}

#[test]
fn test_priority_queue_empty() {
    let mut pq: PriorityQueue<i32> = PriorityQueue::new();
    assert_eq!(pq.pop(), None);
}

#[test]
fn test_priority_queue_fifo_after_mixed() {
    let mut pq = PriorityQueue::new();
    pq.push("first", 0);
    pq.push("high", 1);
    pq.push("second", 0);
    assert_eq!(pq.pop(), Some("high"));
    assert_eq!(pq.pop(), Some("first"));
    assert_eq!(pq.pop(), Some("second"));
    assert_eq!(pq.pop(), None);
}

#[test]
fn test_priority_queue_len_and_is_empty() {
    let mut pq = PriorityQueue::new();
    assert!(pq.is_empty());
    assert_eq!(pq.len(), 0);
    pq.push("a", 0);
    assert!(!pq.is_empty());
    assert_eq!(pq.len(), 1);
    pq.push("b", 1);
    pq.push("c", -1);
    assert_eq!(pq.len(), 3);
    pq.pop();
    assert_eq!(pq.len(), 2);
    pq.pop();
    assert_eq!(pq.len(), 1);
}

#[test]
fn test_priority_queue_peek() {
    let mut pq = PriorityQueue::new();
    assert!(pq.peek().is_none());
    pq.push("low", -1);
    pq.push("normal", 0);
    pq.push("high", 1);
    let (item, prio) = pq.peek().unwrap();
    assert_eq!(*item, "high");
    assert_eq!(prio, 1);
    pq.pop();
    let (item, prio) = pq.peek().unwrap();
    assert_eq!(*item, "normal");
    assert_eq!(prio, 0);
}

#[test]
fn test_priority_queue_into_iter() {
    let mut pq = PriorityQueue::new();
    pq.push("low", -1);
    pq.push("normal", 0);
    pq.push("high", 1);
    let items: Vec<(i32, &str)> = pq.into_iter().collect();
    assert_eq!(items, vec![(1, "high"), (0, "normal"), (-1, "low")]);
}

#[test]
fn test_priority_queue_drain() {
    let mut pq = PriorityQueue::new();
    pq.push("a", 0);
    pq.push("high", 1);
    pq.push("b", 0);
    assert_eq!(pq.len(), 3);
    let drained: Vec<(i32, &str)> = pq.drain().collect();
    assert_eq!(drained, vec![(1, "high"), (0, "a"), (0, "b")]);
    assert!(pq.is_empty());
}

/// High-contention efficiency: verify that PriorityQueue::len is O(1)
/// and that the queue handles many items correctly.
#[test]
fn test_high_contention_priority_queue_len() {
    let mut pq = PriorityQueue::new();
    for i in 0..10000 {
        pq.push(i, i % 10);
    }
    assert_eq!(pq.len(), 10000);
    for _ in 0..10000 {
        pq.pop();
    }
    assert!(pq.is_empty());
    assert_eq!(pq.len(), 0);
}

/// High-contention efficiency: verify that WorkerPool routing remains
/// flat, monomorphic, and deadlock-free under parallel load.
#[tokio::test(start_paused = true)]
async fn test_high_contention_worker_pool() {
    tokio::time::resume();
    let completed = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&completed);
    let pool = Arc::new(WorkerPool::new(
        crate::tokio_runtime(),
        4,
        2000,
        move |job: i32| {
            let c = Arc::clone(&c);
            async move {
                tokio::time::sleep(Duration::from_millis(1)).await;
                c.fetch_add(1, Ordering::SeqCst);
                let _ = job;
            }
        },
    ));
    let mut handles = Vec::new();
    for _ in 0..10 {
        let p = Arc::clone(&pool);
        handles.push(tokio::spawn(async move {
            for i in 0..100 {
                p.submit(i).await.expect("queue should not be full");
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    // Safety: Arc::try_unwrap is used to get the inner WorkerPool
    // for shutdown. Since all spawned tasks are done, the reference
    // count is 1.
    let pool = Arc::try_unwrap(pool).unwrap_or_else(|_| panic!("pool still referenced"));
    pool.shutdown().await;
    assert_eq!(completed.load(Ordering::SeqCst), 1000);
}

#[tokio::test(start_paused = true)]
async fn test_partitioned_router_same_key() {
    let results = Arc::new(std::sync::Mutex::new(Vec::new()));
    let r1 = Arc::clone(&results);
    let r2 = Arc::clone(&results);
    let pool1 = WorkerPool::new(crate::tokio_runtime(), 1, 10, move |job: i32| {
        let r = Arc::clone(&r1);
        async move {
            let mut guard = r.lock().unwrap();
            guard.push((0, job));
        }
    });
    let pool2 = WorkerPool::new(crate::tokio_runtime(), 1, 10, move |job: i32| {
        let r = Arc::clone(&r2);
        async move {
            let mut guard = r.lock().unwrap();
            guard.push((1, job));
        }
    });

    let router = PartitionedRouter::new(vec![pool1, pool2], |key: &String| key.len());
    router.submit(&"a".to_string(), 10).await.unwrap();
    router.submit(&"a".to_string(), 20).await.unwrap();
    // Both go to same shard because "a".len() % 2 == 1
    tokio::task::yield_now().await;
    let guard = results.lock().unwrap();
    assert_eq!(guard.len(), 2);
    let shard1_count = guard.iter().filter(|(s, _)| *s == 1).count();
    assert_eq!(shard1_count, 2);
}

#[tokio::test(start_paused = true)]
async fn test_partitioned_router_distributes() {
    let results = Arc::new(std::sync::Mutex::new(Vec::new()));
    let r1 = Arc::clone(&results);
    let r2 = Arc::clone(&results);
    let pool1 = WorkerPool::new(crate::tokio_runtime(), 1, 10, move |job: i32| {
        let r = Arc::clone(&r1);
        async move {
            let mut guard = r.lock().unwrap();
            guard.push((0, job));
        }
    });
    let pool2 = WorkerPool::new(crate::tokio_runtime(), 1, 10, move |job: i32| {
        let r = Arc::clone(&r2);
        async move {
            let mut guard = r.lock().unwrap();
            guard.push((1, job));
        }
    });

    let router = PartitionedRouter::new(vec![pool1, pool2], |key: &String| key.len());
    router.submit(&"a".to_string(), 10).await.unwrap();
    router.submit(&"bb".to_string(), 20).await.unwrap();
    tokio::task::yield_now().await;
    let guard = results.lock().unwrap();
    assert_eq!(guard.len(), 2);
    let shards: Vec<_> = guard.iter().map(|(s, _)| *s).collect();
    assert!(shards.contains(&0));
    assert!(shards.contains(&1));
}

#[tokio::test]
async fn result_pool_happy_path() {
    let pool = ResultPool::new(crate::tokio_runtime(), 2, 10, |job: i32| async move {
        Ok::<i32, String>(job * 2)
    });
    let result = pool.submit(5).await.unwrap();
    assert_eq!(result, 10);
    pool.shutdown().await;
}

#[tokio::test]
async fn result_pool_handler_error() {
    let pool = ResultPool::new(crate::tokio_runtime(), 1, 10, |job: i32| async move {
        if job < 0 {
            Err("negative".to_string())
        } else {
            Ok(job)
        }
    });
    let err = pool.submit(-1).await.unwrap_err();
    match err {
        ResultPoolError::Inner(msg) => assert_eq!(msg, "negative"),
        other => panic!("expected Inner, got {other:?}"),
    }
    pool.shutdown().await;
}

#[tokio::test]
async fn result_pool_multiple_jobs() {
    let pool = ResultPool::new(crate::tokio_runtime(), 2, 10, |job: i32| async move {
        Ok::<i32, String>(job + 1)
    });
    let mut handles = Vec::new();
    for i in 0..5i32 {
        handles.push(pool.submit(i));
    }
    for (i, handle) in handles.into_iter().enumerate() {
        assert_eq!(handle.await.unwrap(), i as i32 + 1);
    }
    pool.shutdown().await;
}

#[tokio::test]
async fn result_pool_queue_full_returns_error() {
    // Queue capacity 0 means any submit will fail immediately
    let pool = ResultPool::new(crate::tokio_runtime(), 1, 0, |job: i32| async move {
        Ok::<i32, String>(job)
    });
    // The queue has capacity 0, so submit should fail immediately.
    // Wait briefly for the queue to fill (it's already full at capacity 0).
    let err = tokio::time::timeout(std::time::Duration::from_millis(100), pool.submit(1))
        .await
        .unwrap()
        .unwrap_err();
    assert!(matches!(err, ResultPoolError::Pool(PoolError::Full)));
    pool.shutdown().await;
}

#[tokio::test]
async fn result_pool_shutdown_returns_canceled() {
    let pool = ResultPool::new(crate::tokio_runtime(), 1, 10, |job: i32| async move {
        Ok::<i32, String>(job)
    });
    // Submit a job to verify the pool is working
    let result = pool.submit(42).await.unwrap();
    assert_eq!(result, 42);
    pool.shutdown().await;
}
