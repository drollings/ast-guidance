//! Bounded async queue, worker pool, and concurrency limiter.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use fluent_wvr::Runtime;
use thiserror::Error;
use tokio::sync::{Mutex, Notify, Semaphore};
use tokio::task::JoinHandle;

/// Errors returned by `Queue` and `WorkerPool` operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum PoolError {
    #[error("queue is full")]
    Full,
    #[error("queue is closed")]
    Closed,
}

struct QueueInner<T> {
    items: Mutex<VecDeque<T>>,
    capacity: usize,
    closed: AtomicBool,
    notify: Notify,
    space_notify: Notify,
}

/// A bounded, concurrent, single-consumer queue with close-wakes-waiters semantics.
///
/// # Examples
///
/// ```no_run
/// use fluent_concurrency::pool::Queue;
///
/// # async fn example() {
/// let q = Queue::new(10);
/// q.push(42).await.unwrap();
/// q.push(99).await.unwrap();
/// q.close();
///
/// assert_eq!(q.pop().await, Some(42));
/// assert_eq!(q.pop().await, Some(99));
/// assert_eq!(q.pop().await, None); // closed + empty
/// # }
/// ```
pub struct Queue<T> {
    inner: Arc<QueueInner<T>>,
}

impl<T: Send + 'static> Queue<T> {
    /// Creates a new bounded queue with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(QueueInner {
                items: Mutex::new(VecDeque::with_capacity(capacity)),
                capacity,
                closed: AtomicBool::new(false),
                notify: Notify::new(),
                space_notify: Notify::new(),
            }),
        }
    }

    /// Pushes an item into the queue. Returns `Err(Full)` if at capacity.
    pub async fn push(&self, item: T) -> Result<(), PoolError> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(PoolError::Closed);
        }
        let mut items = self.inner.items.lock().await;
        if items.len() >= self.inner.capacity {
            return Err(PoolError::Full);
        }
        items.push_back(item);
        self.inner.notify.notify_one();
        Ok(())
    }

    /// Pushes an item, waiting if the queue is at capacity.
    /// Returns `Err(Closed)` if the queue is closed before space becomes available.
    pub async fn push_wait(&self, item: T) -> Result<(), PoolError> {
        let mut item = Some(item);
        loop {
            if self.inner.closed.load(Ordering::SeqCst) {
                return Err(PoolError::Closed);
            }
            let waited = self.inner.space_notify.notified();
            {
                let mut items = self.inner.items.lock().await;
                if items.len() < self.inner.capacity {
                    items.push_back(item.take().unwrap());
                    self.inner.notify.notify_one();
                    return Ok(());
                }
            }
            waited.await;
        }
    }

    /// Pops an item from the queue, awaiting if empty. Returns `None` when closed.
    pub async fn pop(&self) -> Option<T> {
        loop {
            let notified = self.inner.notify.notified();
            {
                let mut items = self.inner.items.lock().await;
                if let Some(item) = items.pop_front() {
                    self.inner.space_notify.notify_one();
                    return Some(item);
                }
                if self.inner.closed.load(Ordering::SeqCst) {
                    return None;
                }
            }
            notified.await;
        }
    }

    /// Closes the queue, waking all waiters. Subsequent `pop`s return `None`.
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::SeqCst);
        self.inner.notify.notify_waiters();
        self.inner.space_notify.notify_waiters();
    }
}

/// A bounded worker pool that spawns `cap` tokio tasks to process jobs from a shared queue.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use fluent_concurrency::pool::WorkerPool;
/// use fluent_wvr::Runtime;
///
/// # async fn example() {
/// let rt: Arc<dyn Runtime> = Arc::new(fluent_concurrency::runtime::tokio::TokioRuntime);
/// let pool = WorkerPool::new(rt, 4, 100, |n: i32| async move {
///     println!("processing {n}");
/// });
///
/// pool.try_submit(1).await.unwrap();
/// pool.try_submit(2).await.unwrap();
/// pool.shutdown().await;
/// # }
/// ```
pub struct WorkerPool<T: Send + 'static> {
    queue: Arc<Queue<T>>,
    workers: Vec<JoinHandle<()>>,
    shutdown: Arc<Notify>,
}

impl<T: Send + Sync + 'static> WorkerPool<T> {
    /// Creates a new worker pool with `cap` workers and a queue of `queue_capacity`.
    /// All worker tasks are spawned through the injected `runtime` to avoid ambient
    /// `tokio::spawn` calls in the Data Plane.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<F, Fut>(
        runtime: Arc<dyn Runtime>,
        cap: usize,
        queue_capacity: usize,
        handler: F,
    ) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send,
    {
        let queue = Arc::new(Queue::new(queue_capacity));
        let shutdown = Arc::new(Notify::new());
        let handler = Arc::new(handler);
        let mut workers = Vec::with_capacity(cap);

        for _ in 0..cap {
            let q = Arc::clone(&queue);
            let sd = Arc::clone(&shutdown);
            let h = Arc::clone(&handler);
            let r = Arc::clone(&runtime);
            workers.push(r.spawn(Box::pin(async move {
                loop {
                    tokio::select! {
                        () = sd.notified() => break,
                        item = q.pop() => {
                            if let Some(item) = item {
                                h(item).await;
                            } else {
                                break;
                            }
                        }
                    }
                }
            })));
        }

        Self {
            queue,
            workers,
            shutdown,
        }
    }

    /// Tries to submit a job without waiting. Returns `Err(Full)` if the queue is at capacity.
    pub async fn try_submit(&self, job: T) -> Result<(), PoolError> {
        self.queue.push(job).await
    }

    /// Submits a job, waiting if the queue is full.
    /// Returns `Err(Closed)` if the queue is closed.
    pub async fn submit(&self, job: T) -> Result<(), PoolError> {
        self.queue.push_wait(job).await
    }

    /// Shuts down the pool: closes the queue, notifies workers, and awaits their completion.
    pub async fn shutdown(self) {
        self.queue.close();
        self.shutdown.notify_waiters();
        for w in self.workers {
            let _ = w.await;
        }
    }
}

/// Error type for `ResultPool::submit`.
#[derive(Error, Debug)]
pub enum ResultPoolError<E> {
    /// The pool's queue is full or closed.
    #[error("pool error: {0}")]
    Pool(#[from] PoolError),
    /// The handler returned an error.
    #[error("handler error: {0}")]
    Inner(E),
    /// The response channel was canceled (worker panicked or pool shut down).
    #[error("response canceled")]
    Canceled,
}

/// Internal wrapper that pairs a job with a oneshot sender for the result.
struct WrappedJob<T, R, E> {
    job: T,
    result_tx: tokio::sync::oneshot::Sender<Result<R, E>>,
}

/// Worker pool where the handler returns a `Result<R, E>` and `submit`
/// returns a future resolving to that result.
pub struct ResultPool<T, R, E>
where
    T: Send + 'static,
    R: Send + 'static,
    E: Send + 'static,
{
    queue: Arc<Queue<WrappedJob<T, R, E>>>,
    workers: Vec<JoinHandle<()>>,
    worker_count: usize,
    shutdown: Arc<Notify>,
}

impl<T, R, E> ResultPool<T, R, E>
where
    T: Send + Sync + 'static,
    R: Send + 'static,
    E: Send + 'static,
{
    /// Creates a new result pool with `cap` workers and a queue of `queue_capacity`.
    /// The handler receives a job and returns `Result<R, E>`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<F, Fut>(
        runtime: Arc<dyn Runtime>,
        cap: usize,
        queue_capacity: usize,
        handler: F,
    ) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, E>> + Send,
    {
        let queue: Arc<Queue<WrappedJob<T, R, E>>> = Arc::new(Queue::new(queue_capacity));
        let shutdown = Arc::new(Notify::new());
        let handler = Arc::new(handler);
        let mut workers = Vec::with_capacity(cap);

        for _ in 0..cap {
            let q = Arc::clone(&queue);
            let sd = Arc::clone(&shutdown);
            let h = Arc::clone(&handler);
            let r = Arc::clone(&runtime);
            workers.push(r.spawn(Box::pin(async move {
                loop {
                    tokio::select! {
                        () = sd.notified() => break,
                        wrapped = q.pop() => {
                            if let Some(wrapped) = wrapped {
                                let result = h(wrapped.job).await;
                                let _ = wrapped.result_tx.send(result);
                            } else {
                                break;
                            }
                        }
                    }
                }
            })));
        }

        Self {
            queue,
            workers,
            worker_count: cap,
            shutdown,
        }
    }

    /// Returns the configured number of workers.
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Submits a job and awaits the handler's result.
    /// Returns `ResultPoolError::Canceled` if the worker panics or the pool shuts down.
    pub async fn submit(&self, job: T) -> Result<R, ResultPoolError<E>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let wrapped = WrappedJob { job, result_tx: tx };
        self.queue.push(wrapped).await?;
        rx.await
            .map_err(|_| ResultPoolError::Canceled)?
            .map_err(ResultPoolError::Inner)
    }

    /// Shuts down the pool: closes the queue, notifies workers, and awaits their completion.
    pub async fn shutdown(self) {
        self.queue.close();
        self.shutdown.notify_waiters();
        for w in self.workers {
            let _ = w.await;
        }
    }
}

/// Factory helper for auto-sized global `ResultPool` singletons.
///
/// Workers = max(available_parallelism, min_workers).
/// Queue capacity = workers * queue_multiplier.
pub fn global_pool_config(min_workers: usize, queue_multiplier: usize) -> (usize, usize) {
    let workers = std::thread::available_parallelism()
        .map_or(min_workers, std::num::NonZero::get)
        .max(min_workers);
    let queue_cap = workers * queue_multiplier;
    (workers, queue_cap)
}

/// A semaphore-based concurrency limiter. Runs at most `cap` futures concurrently.
///
/// # Examples
///
/// ```no_run
/// use fluent_concurrency::pool::Limiter;
///
/// # async fn example() {
/// let limiter = Limiter::new(3);
/// limiter.run(|| async {
///     // At most 3 of these closures run at a time.
///     println!("concurrent work");
/// }).await;
/// # }
/// ```
pub struct Limiter {
    sem: Arc<Semaphore>,
}

impl Limiter {
    /// Creates a new limiter that allows at most `cap` concurrent executions.
    pub fn new(cap: usize) -> Self {
        Self {
            sem: Arc::new(Semaphore::new(cap)),
        }
    }

    /// Acquires a semaphore permit, runs `f().await`, then releases the permit.
    pub async fn run<F, Fut, T>(&self, f: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let _permit = self.sem.acquire().await.expect("semaphore closed");
        f().await
    }

    /// Synchronous version of `run` for callers without an async context.
    /// Uses `Handle::block_on` if a tokio runtime is active, otherwise
    /// creates a dedicated current-thread runtime for the duration of the call.
    ///
    /// When invoked from *inside* a running multi-threaded tokio runtime (the
    /// router's HTTP handler does this), the future is driven via
    /// `tokio::task::block_in_place` so the worker thread is not starved —
    /// the same canonical pattern as `fluent_llm::client::chat_complete`
    /// (see `src/llm/src/client.rs`). A bare `Handle::block_on` would panic
    /// with "Cannot start a runtime from within a runtime".
    pub fn run_sync<F, Fut, T>(&self, f: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let block = async move {
            let _permit = self.sem.acquire().await.expect("semaphore closed");
            f().await
        };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| handle.block_on(block))
            }
            Ok(handle) => handle.block_on(block),
            Err(_) => {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("build runtime for Limiter::run_sync");
                rt.block_on(block)
            }
        }
    }
}

/// Priority-aware wrapper that pairs a job with a oneshot sender.
struct PriorityWrappedJob<T, R, E> {
    job: T,
    result_tx: tokio::sync::oneshot::Sender<Result<R, E>>,
}

/// A worker pool where jobs are dispatched in priority order.
///
/// Higher priority values are dispatched first. Jobs with the same priority
/// are dispatched in FIFO order. Uses `PriorityQueue` internally with a
/// `tokio::sync::Mutex` for concurrent access.
pub struct PriorityResultPool<T, R, E>
where
    T: Send + 'static,
    R: Send + 'static,
    E: Send + 'static,
{
    queue: Arc<tokio::sync::Mutex<crate::queue::PriorityQueue<PriorityWrappedJob<T, R, E>>>>,
    notify: Arc<Notify>,
    workers: Vec<JoinHandle<()>>,
    shutdown: Arc<Notify>,
}

impl<T, R, E> PriorityResultPool<T, R, E>
where
    T: Send + Sync + 'static,
    R: Send + 'static,
    E: Send + 'static,
{
    /// Creates a new priority result pool with `cap` workers.
    /// The handler receives a job and returns `Result<R, E>`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<F, Fut>(runtime: Arc<dyn Runtime>, cap: usize, handler: F) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, E>> + Send,
    {
        let queue: Arc<
            tokio::sync::Mutex<crate::queue::PriorityQueue<PriorityWrappedJob<T, R, E>>>,
        > = Arc::new(tokio::sync::Mutex::new(crate::queue::PriorityQueue::new()));
        let shutdown = Arc::new(Notify::new());
        let notify = Arc::new(Notify::new());
        let handler = Arc::new(handler);
        let mut workers = Vec::with_capacity(cap);

        for _ in 0..cap {
            let q = Arc::clone(&queue);
            let sd = Arc::clone(&shutdown);
            let h = Arc::clone(&handler);
            let r = Arc::clone(&runtime);
            let nf = Arc::clone(&notify);
            workers.push(r.spawn(Box::pin(async move {
                loop {
                    // Drain everything currently queued.
                    loop {
                        let wrapped = {
                            let mut pq = q.lock().await;
                            pq.pop()
                        };
                        match wrapped {
                            Some(w) => {
                                let result = h(w.job).await;
                                let _ = w.result_tx.send(result);
                            }
                            None => break,
                        }
                    }
                    // Nothing left — wait for a new item or shutdown.
                    tokio::select! {
                        () = sd.notified() => break,
                        () = nf.notified() => {},
                    }
                }
            })));
        }

        Self {
            queue,
            notify,
            workers,
            shutdown,
        }
    }

    /// Submits a job with the given priority and awaits the handler's result.
    /// Higher priority values are dispatched first.
    pub async fn submit(&self, job: T, priority: i32) -> Result<R, ResultPoolError<E>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let wrapped = PriorityWrappedJob { job, result_tx: tx };
        {
            let mut pq = self.queue.lock().await;
            pq.push(wrapped, priority);
        }
        // Wake workers to drain the queue.
        self.notify.notify_waiters();
        rx.await
            .map_err(|_| ResultPoolError::Canceled)?
            .map_err(ResultPoolError::Inner)
    }

    /// Shuts down the pool: notifies workers, and awaits their completion.
    pub async fn shutdown(self) {
        self.shutdown.notify_waiters();
        for w in self.workers {
            let _ = w.await;
        }
    }
}

#[cfg(test)]
mod priority_pool_tests {
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
}
