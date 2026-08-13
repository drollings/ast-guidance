---
name: fluent-concurrency
description: Bounded async worker pools, structured concurrency, capability-gated I/O, and runtime abstraction for deterministic-first execution
anchors:
  - WorkerPool
  - Queue
  - Limiter
  - Scope
  - SupervisedBatch
  - Runtime
  - TokioRuntime
  - CapabilitySet
  - PriorityQueue
  - PartitionedRouter
  - CreditSender
---

# Fluent Concurrency

Bounded async worker pools, structured concurrency primitives, capability-gated I/O, and a runtime-agnostic abstraction layer. Designed for deterministic-first execution with backpressure, panic containment, and dependency-aware task orchestration.

## Key files

- `src/fluent-concurrency/src/pool.rs` — `WorkerPool`, `Queue`, `Limiter`
- `src/fluent-concurrency/src/batch.rs` — `SupervisedBatch`, `SupervisedBatchConfig`, `SupervisedBatchSummary`
- `src/fluent-concurrency/src/scope.rs` — `Scope` (structured concurrency)
- `src/fluent-concurrency/src/router.rs` — `PartitionedRouter` (sharded pools)
- `src/fluent-concurrency/src/queue.rs` — `PriorityQueue`
- `src/fluent-concurrency/src/flow.rs` — `CreditSender`, `CreditReceiver` (credit flow)
- `src/fluent-concurrency/src/capability.rs` — `CapabilitySet`, `default_capability_set()`
- `src/fluent-concurrency/src/io/` — `FsCapability`, `NetCapability`, `DbCapability`
- `src/fluent-concurrency/src/runtime/tokio.rs` — `TokioRuntime`
- `src/fluent-wvr/src/lib.rs` — `Runtime` trait, `WorkUnit`, `WorkContext`, `WorkOutput`

## Core Primitives

### WorkerPool<T>

Bounded worker pool with backpressure. Spawns `cap` async worker tasks that pull from a shared `Queue<T>`.

```rust
pub fn new<F, Fut>(
    runtime: Arc<dyn Runtime>,
    cap: usize,
    queue_capacity: usize,
    handler: F,
) -> Self
where
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send;

pub async fn submit(&self, job: T) -> Result<(), PoolError>;
pub async fn shutdown(self);
```

- `PoolError::Full` — queue at capacity (backpressure signal)
- `PoolError::Closed` — queue shut down
- Workers are spawned via `Runtime::spawn`, not ambient `tokio::spawn`
- One task's panic does **not** crash the pool; worker loop catches and continues

### Queue<T>

Bounded, concurrent, single-consumer queue. Used internally by `WorkerPool`.

```rust
pub fn new(capacity: usize) -> Self;
pub async fn push(&self, item: T) -> Result<(), PoolError>;
pub async fn pop(&self) -> Option<T>;  // None when closed
pub fn close(&self);
```

### Limiter

Semaphore-based concurrency limiter. Runs at most `cap` futures concurrently.

```rust
pub fn new(cap: usize) -> Self;
pub async fn run<F, Fut, T>(&self, f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>;
```

## Structured Concurrency

### Scope

All spawned tasks are joined or aborted on drop. **Panics if dropped without calling `close().await`.**

```rust
pub fn new() -> Self;
pub async fn spawn<F>(&mut self, future: F) -> AbortHandle;
pub async fn close(&mut self);
```

### SupervisedBatch

Supervision zone with async retry, dependency cancellation, and timeout. Implements `Future<Output = SupervisedBatchSummary>`.

```rust
pub fn new(runtime: Arc<dyn Runtime>, caps: CapabilitySet) -> Self;
pub fn register(&mut self, unit: Arc<dyn WorkUnit>) -> &mut Self;
```

- Dependencies form a DAG; if unit A depends on B and B fails, A is cancelled
- `SupervisedBatchConfig { poll_budget }` limits per-poll work (default: 64)
- `SupervisedBatchSummary` reports completed, panicked, and cancelled tasks

## Runtime Abstraction

### Runtime trait

```rust
pub trait Runtime: Send + Sync + 'static {
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send>>) -> JoinHandle<()>;
    fn sleep(&self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>>;
    fn now(&self) -> Instant;
}
```

### TokioRuntime

Zero-sized struct that delegates to `tokio::spawn`, `tokio::time::sleep`, and `Instant::now()`. Use when building on tokio.

**Critical**: `WorkerPool::new` calls `Runtime::spawn` during construction. If using a static `WorkerPool`, the `Runtime::spawn` call must happen inside a `Runtime::block_on` context (which sets up the tokio reactor), otherwise `tokio::spawn` panics with "no reactor running".

**Pattern for static pools:**
```rust
static RT: LazyLock<Runtime> = LazyLock::new(|| { /* create runtime */ });
static POOL: LazyLock<WorkerPool<T>> = LazyLock::new(|| {
    RT.block_on(async {
        WorkerPool::new(Arc::new(TokioRuntime), cap, queue_cap, handler)
    })
});
```

## Capability-Gated I/O

All I/O goes through capability structs that enforce access at runtime.

```rust
pub struct CapabilitySet { /* ... */ }

pub fn default_capability_set() -> CapabilitySet;
pub fn capability_set_with_db(path: &str) -> Result<CapabilitySet, ConcurrencyError>;
```

### Available Capabilities

| Capability | Operations | Notes |
|-----------|-----------|-------|
| `FsCapability` | `read`, `write`, `metadata` | Path-sandboxed |
| `NetCapability` | `tcp_connect`, `http_get`, `http_post` | Configurable reqwest client |
| `DbCapability` | `open`, `query`, `execute` | SQLite with WAL, connection-pooled |

Errors: `CapabilityError::Missing { name }` or `CapabilityError::Exhausted { name, detail }`.

## Patterns

### Sync-to-async bridge

For calling async queue methods from sync code, create a lightweight bridge runtime:

```rust
pub fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
    let dq = StaticQueue::get();  // holds &Runtime + &LlmRequestQueue
    let queue = self.queue.clone().unwrap_or_else(|| dq.queue.clone());
    dq.runtime.block_on(queue.submit_async(messages, config))
}
```

`block_on` creates a single-threaded executor on the calling thread. The actual work (HTTP via `spawn_blocking`) runs on the static pool's multi-threaded runtime.

### Async embedding path

Embedding providers have both sync (`embed`) and async (`embed_async`) trait methods. Sync callers use a `LazyLock<Runtime>` + `block_on` bridge. Async callers use `do_http_post_async` directly. **Never call sync `embed()` from `#[tokio::test]` or async contexts** — it will panic ("Cannot start a runtime from within a runtime").

### Sharded pools

`PartitionedRouter<K, J>` hashes a key to select from multiple `WorkerPool` instances, providing per-key ordering:

```rust
let router = PartitionedRouter::new(pools, |key| hash(key) % pools.len());
router.submit(&key, job).await?;
```

## Semantic Deviations

- **No `spawn_blocking` abstraction** — use `tokio::task::spawn_blocking` directly for sync work inside pool handlers
- **`Queue` is single-consumer** — multiple consumers race for `pop()`
- **`Scope` panics on drop** if `close()` wasn't called — use `defer` patterns or ensure scope lives long enough
- **`SupervisedBatch` is a `Future`** — must be `.await`ed or aborted; dropping a running zone cancels all tasks
- **`WorkerPool::new` spawns immediately** — workers start during construction; ensure runtime context exists
- **Credit flow** is for producer-consumer rate limiting, not general concurrency control
