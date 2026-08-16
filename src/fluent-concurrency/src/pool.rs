//! Bounded async queue, worker pool, and concurrency limiter.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use common_core::sync as sync_lock;
use fluent_wvr::Runtime;
use thiserror::Error;
use tokio::sync::{Notify, Semaphore};
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
/// The queue's backing store is a `std::sync::Mutex` — never held across an
/// `.await` (every guard is scoped so it drops before the future yields), so
/// the async mutex's waker bookkeeping would be pure overhead. Poisoned
/// mutexes are recovered via `common_core::sync::lock` rather than panicking
/// the caller. Contrast `CreditSender` in `flow.rs`, whose mutex guards the
/// bump receiver across `recv().await` and therefore must stay a
/// `tokio::sync::Mutex`.
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
    ///
    /// The fast path is synchronous (never blocks), but the signature stays
    /// `async` so callers can switch freely between `push` and `push_wait`
    /// without churn.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn push(&self, item: T) -> Result<(), PoolError> {
        if self.inner.closed.load(Ordering::SeqCst) {
            return Err(PoolError::Closed);
        }
        let mut items = sync_lock::lock(&self.inner.items);
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
                let mut items = sync_lock::lock(&self.inner.items);
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
                let mut items = sync_lock::lock(&self.inner.items);
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
        let workers = spawn_queue_workers(runtime, cap, &queue, &shutdown, move |job: T| {
            let handler = Arc::clone(&handler);
            async move { handler(job).await }
        });

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

/// Internal wrapper pairing a job with a oneshot sender for its result and an
/// optional abort signal. When the signal fires while the handler is running,
/// the handler future is dropped and the sender is dropped without a result —
/// the submitter observes [`ResultPoolError::Canceled`]. Shared by
/// `ResultPool` and `PriorityResultPool`.
struct WrappedJob<T, R, E> {
    job: T,
    result_tx: tokio::sync::oneshot::Sender<Result<R, E>>,
    abort: Option<crate::stream::StreamAbort>,
}

/// Spawn `cap` worker tasks running one shared worker core.
///
/// Each worker loops: `next_job().await` resolves to `Some(job)` when work is
/// available or `None` when the pool is shutting down, and `handle(job).await`
/// processes it. `next_job` encapsulates the pool's queue discipline — the
/// `Queue`-backed pools `select!` between a shutdown notification and `pop`
/// (so shutdown can preempt a pending pop); the priority pool drains its queue
/// and then waits on shutdown/notify (drain-then-wait). `handle` runs the job;
/// result pools also reply through the job's oneshot channel. This is the one
/// worker loop every public pool type parameterizes.
#[allow(clippy::needless_pass_by_value)]
fn spawn_workers<J, F, Fut, H, HFut>(
    runtime: Arc<dyn Runtime>,
    cap: usize,
    next_job: F,
    handle: H,
) -> Vec<JoinHandle<()>>
where
    J: Send + 'static,
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Option<J>> + Send + 'static,
    H: Fn(J) -> HFut + Send + Sync + 'static,
    HFut: Future<Output = ()> + Send + 'static,
{
    let next_job = Arc::new(next_job);
    let handle = Arc::new(handle);
    let mut workers = Vec::with_capacity(cap);
    for _ in 0..cap {
        let n = Arc::clone(&next_job);
        let h = Arc::clone(&handle);
        let r = Arc::clone(&runtime);
        workers.push(r.spawn(Box::pin(async move {
            while let Some(job) = n().await {
                h(job).await;
            }
        })));
    }
    workers
}

/// Run a wrapped job's handler, racing it against the optional abort signal.
///
/// When the abort fires mid-flight, the handler future is dropped (cancelling
/// its transport — an in-flight HTTP call is aborted, a worker slot is freed)
/// and the oneshot sender is dropped without a result, so the submitter
/// observes [`ResultPoolError::Canceled`]. Without an abort signal this is
/// exactly `result_tx.send(handler(job).await)`. Shared by `ResultPool` and
/// `PriorityResultPool`.
async fn run_wrapped<T, R, E, F, Fut>(
    wrapped: WrappedJob<T, R, E>,
    handler: Arc<F>,
) where
    T: Send + 'static,
    R: Send + 'static,
    E: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<R, E>> + Send,
{
    let WrappedJob { job, result_tx, abort } = wrapped;
    let fut = handler(job);
    match abort {
        Some(abort) => {
            let cancelled = abort.cancelled();
            tokio::pin!(fut);
            tokio::pin!(cancelled);
            tokio::select! {
                r = &mut fut => { let _ = result_tx.send(r); }
                () = &mut cancelled => {}
            }
        }
        None => {
            let _ = result_tx.send(fut.await);
        }
    }
}

/// Spawn `cap` workers for a `Queue`-backed pool. Each worker
/// `select!`s between a shutdown notification and `pop`, so shutdown can
/// preempt a pending pop; `handle` runs each job. Shared by `WorkerPool`
/// and `ResultPool`.
fn spawn_queue_workers<J, H, HFut>(
    runtime: Arc<dyn Runtime>,
    cap: usize,
    queue: &Arc<Queue<J>>,
    shutdown: &Arc<Notify>,
    handle: H,
) -> Vec<JoinHandle<()>>
where
    J: Send + 'static,
    H: Fn(J) -> HFut + Send + Sync + 'static,
    HFut: Future<Output = ()> + Send + 'static,
{
    spawn_workers(
        runtime,
        cap,
        {
            let queue = Arc::clone(queue);
            let shutdown = Arc::clone(shutdown);
            move || {
                let queue = Arc::clone(&queue);
                let shutdown = Arc::clone(&shutdown);
                async move {
                    tokio::select! {
                        () = shutdown.notified() => None,
                        item = queue.pop() => item,
                    }
                }
            }
        },
        handle,
    )
}

/// Spawn `cap` workers for `PriorityResultPool`. Each worker pops one
/// job from the shared `BoundedPriorityQueue` (priority desc, FIFO within
/// priority) and, when empty, waits for a submit notification or shutdown.
///
/// Shutdown is signaled through a `watch` channel rather than a `Notify`
/// because `Notify::notify_waiters` does **not** store a permit: a worker that
/// re-registers its shutdown future *after* `notify_waiters` would miss the
/// wake and park forever. `watch::Receiver::changed()` is sticky — a receiver
/// that registers after the send sees it immediately. (The same reasoning
/// means `BoundedPriorityQueue::close` cannot be the only shutdown path for a
/// worker mid-`pop` — the outer sticky watch closes that race.)
fn spawn_priority_workers<T, R, E, H, HFut>(
    runtime: Arc<dyn Runtime>,
    cap: usize,
    queue: &Arc<crate::queue::BoundedPriorityQueue<WrappedJob<T, R, E>>>,
    shutdown: &tokio::sync::watch::Receiver<bool>,
    handle: H,
) -> Vec<JoinHandle<()>>
where
    T: Send + 'static,
    R: Send + 'static,
    E: Send + 'static,
    H: Fn(WrappedJob<T, R, E>) -> HFut + Send + Sync + 'static,
    HFut: Future<Output = ()> + Send + 'static,
{
    spawn_workers(
        runtime,
        cap,
        {
            let queue = Arc::clone(queue);
            let shutdown = shutdown.clone();
            move || {
                let queue = Arc::clone(&queue);
                let mut shutdown = shutdown.clone();
                async move {
                    let shutdown = shutdown.changed();
                    tokio::pin!(shutdown);
                    // `biased` so a shutdown that is already pending (a leftover
                    // job-wake permit, or a close that landed while this worker was
                    // mid-loop) is never shadowed by the job notify. `queue.pop()`
                    // internally registers its notify before checking the queue
                    // (the anti-missed-wakeup discipline from `Queue::pop`) and
                    // parks until an item arrives or the queue is closed.
                    tokio::select! {
                        biased;
                        _ = &mut shutdown => None,
                        item = queue.pop() => item,
                    }
                }
            }
        },
        handle,
    )
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
        let workers = spawn_queue_workers(runtime, cap, &queue, &shutdown, {
            let handler = Arc::clone(&handler);
            move |wrapped: WrappedJob<T, R, E>| {
                let handler = Arc::clone(&handler);
                async move {
                    run_wrapped(wrapped, handler).await;
                }
            }
        });

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
    /// Returns `ResultPoolError::Canceled` if the worker panics, the pool shuts
    /// down, or the submission's abort signal fires.
    pub async fn submit(&self, job: T) -> Result<R, ResultPoolError<E>> {
        self.submit_with_abort(job, None).await
    }

    /// Submits a job with an optional abort signal and awaits the handler's
    /// result. When the signal fires mid-flight the handler future is dropped
    /// and the submitter observes [`ResultPoolError::Canceled`].
    pub async fn submit_with_abort(
        &self,
        job: T,
        abort: Option<crate::stream::StreamAbort>,
    ) -> Result<R, ResultPoolError<E>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let wrapped = WrappedJob {
            job,
            result_tx: tx,
            abort,
        };
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
    ///
    /// **Caveat:** on the multi-thread scheduler `block_in_place` *parks the
    /// calling worker thread* for the duration of the call — the scheduler
    /// spins up a replacement thread and the parked thread is unavailable for
    /// work stealing until the call returns. This is the correct sync↔async
    /// bridge, but it must NOT be used on throughput-critical paths (that is
    /// what the async [`Limiter::run`] is for); bound its use to control-plane
    /// callers that genuinely cannot await.
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

/// A worker pool where jobs are dispatched in priority order.
///
/// Higher priority values are dispatched first. Jobs with the same priority
/// are dispatched in FIFO order. Backed by a
/// [`crate::queue::BoundedPriorityQueue`] — the same close-wakes-waiters
/// discipline as `Queue`, with `PriorityQueue` ordering. `submit`/`submit_with_abort`
/// apply backpressure: they block while the queue is at capacity instead of
/// growing without bound.
pub struct PriorityResultPool<T, R, E>
where
    T: Send + 'static,
    R: Send + 'static,
    E: Send + 'static,
{
    queue: Arc<crate::queue::BoundedPriorityQueue<WrappedJob<T, R, E>>>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    workers: Vec<JoinHandle<()>>,
}

/// Default queue capacity for `PriorityResultPool::new`: `cap * 4` (saturating).
const DEFAULT_PRIORITY_QUEUE_MULTIPLIER: usize = 4;

impl<T, R, E> PriorityResultPool<T, R, E>
where
    T: Send + Sync + 'static,
    R: Send + 'static,
    E: Send + 'static,
{
    /// Creates a new priority result pool with `cap` workers and a queue of
    /// `cap * 4` slots (the documented default; bounded growth).
    /// The handler receives a job and returns `Result<R, E>`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<F, Fut>(runtime: Arc<dyn Runtime>, cap: usize, handler: F) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, E>> + Send,
    {
        Self::with_queue_capacity(
            runtime,
            cap,
            cap.saturating_mul(DEFAULT_PRIORITY_QUEUE_MULTIPLIER),
            handler,
        )
    }

    /// Creates a new priority result pool with `cap` workers and an explicit
    /// queue capacity. `submit`/`submit_with_abort` block while the queue is
    /// at capacity (`PoolError::Full` is never returned on the submit path —
    /// it is the synchronous [`crate::queue::BoundedPriorityQueue::push`]
    /// fast path's error). Returns `Err(Pool(Closed))` from a submit once the
    /// pool has shut down.
    #[allow(clippy::needless_pass_by_value)]
    pub fn with_queue_capacity<F, Fut>(
        runtime: Arc<dyn Runtime>,
        cap: usize,
        queue_capacity: usize,
        handler: F,
    ) -> Self
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R, E>> + Send,
    {
        let queue: Arc<crate::queue::BoundedPriorityQueue<WrappedJob<T, R, E>>> =
            Arc::new(crate::queue::BoundedPriorityQueue::new(queue_capacity));
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let handler = Arc::new(handler);
        let workers = spawn_priority_workers(runtime, cap, &queue, &shutdown_rx, {
            let handler = Arc::clone(&handler);
            move |wrapped: WrappedJob<T, R, E>| {
                let handler = Arc::clone(&handler);
                async move {
                    run_wrapped(wrapped, handler).await;
                }
            }
        });

        Self {
            queue,
            shutdown_tx,
            workers,
        }
    }

    /// Submits a job with the given priority and awaits the handler's result.
    /// Higher priority values are dispatched first. Applies backpressure:
    /// blocks while the queue is at capacity.
    pub async fn submit(&self, job: T, priority: i32) -> Result<R, ResultPoolError<E>> {
        self.submit_with_abort(job, priority, None).await
    }

    /// Submits a job with the given priority and an optional abort signal.
    /// Higher priority values are dispatched first; when the abort fires
    /// mid-flight the handler future is dropped and the submitter observes
    /// [`ResultPoolError::Canceled`]. Applies backpressure: blocks while the
    /// queue is at capacity (`Pool(Closed)` on a submit after shutdown).
    pub async fn submit_with_abort(
        &self,
        job: T,
        priority: i32,
        abort: Option<crate::stream::StreamAbort>,
    ) -> Result<R, ResultPoolError<E>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let wrapped = WrappedJob {
            job,
            result_tx: tx,
            abort,
        };
        self.queue.push_wait(wrapped, priority).await?;
        rx.await
            .map_err(|_| ResultPoolError::Canceled)?
            .map_err(ResultPoolError::Inner)
    }

    /// Shuts down the pool: closes the queue (waking any blocked
    /// `submit`/`submit_with_abort` with `Pool(Closed)` and letting parked
    /// workers drain-and-exit), signals every worker through the sticky watch,
    /// then awaits their completion. The watch sender keeps the receivers
    /// alive; workers that outlive an explicit `shutdown` also exit when the
    /// pool (and thus the sender) is dropped.
    pub async fn shutdown(self) {
        self.queue.close();
        let _ = self.shutdown_tx.send(true);
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
}
