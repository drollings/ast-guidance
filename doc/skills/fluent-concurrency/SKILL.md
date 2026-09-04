# `fluent-concurrency` — Lightweight Async Runtime Framework Specification

## 1. Executive Summary

This document specifies `fluent-concurrency`, a thin, safe, composable extension layer over **Tokio**. It is designed for systems that need the operational resilience of RabbitMQ (bounded worker pools, credit-based backpressure, priority queues, supervision batches) and the minimalism of `smol`, without reimplementing Tokio's scheduler, I/O driver, or timer wheel.

**Core philosophy:**
- **Tokio is the workhorse.** We do not rebuild `async_executor`, `epoll/kqueue/IOCP`, or work-stealing. We *compose* Tokio's primitives.
- **Fluent WVR is the control plane.** Every unit of work presents the same `Component` / `WorkUnit` interface regardless of origin (native struct, WASM plugin, DB config). The orchestrator iterates over uniform handles. (The one sanctioned exception — the router's `PipelineOrchestrator` downcasting to `StageDecisionProducer` for a typed per-request handoff — is documented in the fluent-wvr SKILL §0.)
- **Safety and locality.** 100% safe Rust (`#![forbid(unsafe_code)]`). No procedural macros hiding task boundaries. `dyn Trait` is restricted to the Control Plane; the Data Plane uses concrete types and flat enums.
- **Explicit ownership.** Every task is owned: `Scope`-spawned tasks are awaited when the scope closes, and server-owned background/connection tasks are awaited when the server drains them at graceful shutdown. Effects are capability-gated: DB and knowledge access require tokens, and file/process/network effects are gated wherever a capability set is installed (the serving path) — operator CLI tooling is capability-exempt by design.

## 2. Design Decisions (Resolving Manifest Open Questions)

| Question | Resolution | Rationale |
|----------|------------|-----------|
| **Q1 — Supervision restart** | **Containment-only.** A `SupervisedBatch` catches task panics, emits a typed `SupervisedBatchEvent`, and cancels dependent tasks. It does **not** automatically restart. | Restarting async tasks from arbitrary state is a checkpoint-semantics problem. RabbitMQ's `supervisor2` gets away with it because Erlang processes are stateless on restart. Rust async tasks carry arbitrary stack state; automatic restart is a trap. We add restart only when profiling proves it necessary. |
| **Q2 — Capability granularity** | **Per-scope/batch establishment with task-local inheritance.** Entering a `Scope` or a `SupervisedBatch` installs a `CapabilitySet` into `tokio::task_local! CURRENT_CAPS`. `Scope::spawn` captures the current set and reinstalls it via `CURRENT_CAPS.scope(caps, future)` (`scope.rs:68-76`); `SupervisedBatch::spawn_unit` captures the effective `WorkContext.caps` (the batch caps plus per-unit overrides) and reinstalls it the same way (`batch.rs:206-235`). A task that bypasses these two entry points (raw `tokio::spawn`, `spawn_blocking`, or channel handoff without an explicit `scope`) sees an empty `CURRENT_CAPS` and fails closed with `PermissionDenied` — there is no fallback to ambient/default capabilities. | Per-call `&Capability` at every `spawn` site adds ceremony without meaningful security gain when scope/batch boundaries are already enforced. Effect *entry points* (e.g., `fs::read`, `db::query`) still require an explicit `&Capability` parameter in their signature, and the gating check (`check_capability`) reads `CURRENT_CAPS.try_with` to enforce presence; absence is a hard `Missing` error, not a silent default. |
| **Q3 — Deterministic testing** | **Both, phased.** The `Runtime` trait supports a `TestRuntime` that uses Tokio's `start_paused` virtual time + a seeded `fastrand::Rng` for **record-replay**. For **combinatorial exploration**, the trait is designed to swap in a future `LoomRuntime` backend. The initial stack ships record-replay; loom integration is a future primitive. | A full loom-compatible async executor is a research project. Shipping it now would violate the "no academic abstraction inflation" red flag. The trait boundary is wide enough to add it later without breaking user code. |

## 3. Core Primitives

The crate exports the following modules from `src/lib.rs:3-24`:

```text
affinity  capability  credit_pool  feed_worker  flow  io  ladder  pool  queue  reserve  router  runtime  scope  stream  thread_resource  batch
```

Each is described below. The spec primitives are in §3.1–§3.9; the bonus primitives (`AffinityScheduler`, `ResultPool`, `PriorityResultPool`, `Reserve`, `StreamAbort`, `thread_local_resource!`) are in §3.10; the canonical first-Ok fallback combinator (`first_accept_in_order`) is in §3.11.

> ROADMAP_20260903_LLM M11 removed the `llm_queue` module. The LLM
> protocol types (`ChatMessage`, `LlmConfig`, `LlmError`, `LlmTask`,
> `LlmQueueConfig`) and `LlmRequestQueue` are owned by
> `fluent_llm::protocol` (a thin wrapper over this crate's generic
> `ResultPool`, composed — not re-implemented). Import LLM queue types
> from `fluent_llm`, never from here.

### 3.1 `Capability` — Bounded Resource Access

Every high-overhead effect (file system, database, AI inference endpoint, blocking thread pool) requires a non-cloneable capability token. The `Capability` trait (defined in `fluent-wvr::capability`) and the `CapabilitySet` typed-map container (also in `fluent-wvr`) form the abstract layer. This crate provides the concrete tokens and the gating.

**Concrete capability tokens** (`src/io/`):

- `FsCapability` — gates `tokio::fs::{read, write, metadata}`. Constructed via `FsCapability::new()`.
- `NetCapability` — wraps a `reqwest::Client` configured via `NetConfig { max_idle_per_host, idle_timeout, connect_timeout, request_timeout, user_agent }`. Constructed via `NetCapability::new()` or `with_config(&NetConfig)`. Exposes `http_get`, `http_post`, `http_post_json_stream` (returns a streaming `Stream<Item = Result<Bytes, IoError>>` of response chunks, used by SSE forwarding), `tcp_connect`, and the raw `client()`.
- `DbCapability` — **home is now `fluent-db`** (`fluent_db::capability`, re-exported from `fluent_concurrency::io::db::DbCapability` to keep the module path). Opens a `fluent-db::SqlitePool` (default 5 connections, WAL mode enabled) via `DbCapability::open(path)`. The legacy lossy `query(sql) -> Vec<HashMap<String, String>>` and `execute(sql) -> usize` (all-values-as-strings) are **deprecated*: new code uses the typed `SqlitePool::query_row` / `query_rows` / `execute` helpers. Both still check out a `PooledConnection` (RAII; the connection returns to the pool on `Drop`) and offload the synchronous `rusqlite` work via `spawn_blocking`. Note that `SqlitePool::acquire()` is **gated** like every other effect entry point — it requires a `DbCapability` token in the current task-local and returns `DbError::PermissionDenied` otherwise (no caller behavior change; the token still gates via `fluent-wvr`).

**Helpers** (`capability.rs`):

- `default_capability_set() -> CapabilitySet` — pre-populated with `FsCapability` and `NetCapability`.
- `capability_set_with_db(path) -> Result<CapabilitySet, DbError>` — also adds a `DbCapability` rooted at `path` (error type is `fluent_db::error::DbError`).

**Gating**: `check_capability<C: Capability>(cap: &C)` (canonical home `fluent_wvr::capability::check_capability`, re-exported from `fluent_concurrency::io`) reads `CURRENT_CAPS.try_with(|caps| caps.get::<C>().is_some())`. If absent (no enclosing `Scope`/`SupervisedBatch` or a raw `tokio::spawn` that bypassed them), returns `Err(io::Error::new(PermissionDenied, CapabilityError::Missing { name: cap.name() }))` — fail-closed, no ambient fallback. `check_capability` is never a silent `default()` grant; an un-scoped task is denied, not allowed. The `name()` field on the trait is informational only; `CapabilitySet::get` uses `TypeId::of::<C>()` for the actual lookup. `CURRENT_CAPS` is a `tokio::task_local!` owned by `fluent-wvr` (re-exported from `fluent_concurrency::scope`) so both this crate and `fluent-db` read the same variable without a dependency cycle.

**Error type** (`fluent_wvr::capability::CapabilityError`, re-exported unchanged
from `fluent_concurrency::io`): `CapabilityError::{Missing { name }, Exhausted { name, detail }}`. The `Exhausted` variant is currently only used by `DbCapability`'s pool-empty branch.

This is a lightweight, safe, two-phase effect pipeline. It maps directly to RabbitMQ's `credit_flow` and Tokio's `Semaphore` semantics, but without the lifetime complications of `tokio::sync::SemaphorePermit`.

### 3.2 `Scope` — Structured Concurrency & Region Ownership

A `Scope` is the fundamental owner of tasks. It is **`#[must_use]`** and requires explicit close. The `Scope` wraps a `tokio::task::JoinSet<()>` plus a `closed: bool` flag.

**Close variants** (`scope.rs:67-139`):

- `close(&mut self).await` — async; sets `closed = true`, calls `abort_all()`, and drains the `JoinSet`. Use this from within an async context.
- `close_sync(&mut self)` — synchronous; sets `closed = true`, calls `abort_all()`, and replaces the `JoinSet` with a fresh one. Aborted tasks are guaranteed not to make progress past their next yield point. Use this from `Drop` or anywhere that cannot await.
- `close_graceful(&mut self, timeout: Duration).await` — waits up to `timeout` for tasks to complete naturally, then aborts and drains any stragglers.
- `defer(&mut self) -> ScopeGuard<'_>` — returns an RAII guard whose `Drop` calls `close_sync()`. The canonical ergonomic alternative to manual `close().await`. Note: the guard does **not** spawn a `close` task; it uses `close_sync()` directly to avoid a fire-and-forget task that could itself be aborted if the runtime drops.

**Spawn and propagation** (`scope.rs:68-76`, `batch.rs:206-235`): `Scope::spawn` captures the current `CURRENT_CAPS` (defaulting to `CapabilitySet::default()` outside a task-local) and re-installs it in the child task via `CURRENT_CAPS.scope(caps, future)`. `SupervisedBatch::spawn_unit` does the same — it clones the effective `WorkContext.caps` for the unit (batch caps plus any per-`register_with_context` overrides) and spawns `CURRENT_CAPS.scope(caps_for_task, execute_with_timeout_and_retry(...))` into its `JoinSet`. Both are the load-bearing mechanism for the Q2 design decision; any `tokio::spawn`/`spawn_blocking` that bypasses them sees an empty `CURRENT_CAPS` and is denied at the first gated I/O call (fail-closed).

**Drop semantics** (`scope.rs:148-168`): dropping a `Scope` without closing it panics with a structured-concurrency violation message. The panic is suppressed during a panic unwind to let the original panic propagate; instead, an `error!` is logged and tasks are aborted.

**Why not `async Drop`?** Rust does not have async drop. The RabbitMQ Erlang model achieves this because `supervisor2` runs in its own process and can block on `receive`. In Rust, the only way to *guarantee* a child is awaited before the parent frame exits is to make the parent frame itself a `Future` that ends with `scope.close().await`. The `#[must_use]` + `Drop`-panics pattern enforces this at the API level without unsafe or proc macros.

### 3.3 `SupervisedBatch` — Failure Containment & Supervision

A `SupervisedBatch` mirrors `Scope`'s ownership and capability-propagation discipline but owns its own `JoinSet` plus a dependency graph, a typed event sink, retry-with-backoff, and per-task timeouts (it does not wrap `Scope` — both install `CURRENT_CAPS` the same way). It is **also** a `Future<Output = SupervisedBatchSummary>` (`batch.rs:251-365`), so the canonical use is `let summary: SupervisedBatchSummary = batch.await`.

**Construction** (`batch.rs:147-172`):

- `SupervisedBatch::new(runtime: Arc<dyn Runtime>, caps: CapabilitySet) -> Self` — defaults to `SupervisedBatchConfig::default()`.
- `SupervisedBatch::new_with_config(runtime, caps, config: SupervisedBatchConfig) -> Self`.

**Configuration** (`batch.rs:48-92`): `SupervisedBatchConfig { poll_budget: usize, is_retryable: fn(&WorkError) -> bool, backoff_base_ms: u64, backoff_jitter_ms: u64 }`. `poll_budget` is the maximum ready `JoinSet` entries drained per `SupervisedBatch::poll` invocation (default `64`); only when that budget is exhausted on *ready* completions does `poll` call `cx.waker().wake_by_ref()` and return `Pending` to prevent one batch starving the executor (`batch.rs:343-346`). When the `JoinSet` is `Pending` (all remaining children still I/O-bound), `poll` returns `Pending` without a self-wake — there is no tight spinning on delayed tasks. `backoff_base_ms` (default `100`) and `backoff_jitter_ms` (default `50`) tune the retry cadence: they are forwarded to `common_core::retry::retry_async` as `base_ms` and `jitter_pct` respectively, preserving the pre-configuration `retry_async(..., 100, 50, ...)` schedule exactly.

**Registration** (`batch.rs:189-204`):

- `register(unit: Arc<dyn Component>) -> Result<&mut Self, SupervisedBatchError>` — builds a `WorkContext` via `WorkContext::for_unit_in_batch(&self.runtime, &self.caps, |_| {})` and forwards.
- `register_with_context(unit, ctx: WorkContext) -> Result<&mut Self, SupervisedBatchError>` — explicit context (its `ctx.caps` is the `CURRENT_CAPS` installed for that task, so per-unit overrides are honored).
- `SupervisedBatchError::DuplicateName(ArcIntern<str>)` — duplicate `name()` rejection. The signature is `Result`, not panicking, so callers can decide.

**Dependency tracking** (`batch.rs:130-145, 233-248, 251-365`): each registered unit contributes to a `DependencyGraph<ArcIntern<str>>` composed from `fluent_dag::dep_graph` (`batch.rs:137`), plus two maps for the abort side effects: `task_names: HashMap<task::Id, ArcIntern<str>>` and `abort_handles: HashMap<ArcIntern<str>, AbortHandle>`.

When a unit fails/panics/times out, `cancel_dependents_of(name)` (`batch.rs:189-204`) calls `DependencyGraph::dependents_of(name)` — a cycle-resilient DFS — and aborts each transitive dependent's handle. A back-edge into the DFS active path emits a `tracing::warn!` rather than panicking — the cycle is left in place but the offending dependents are not double-cancelled.

**Retry and timeout** (`batch.rs:364-416`): each registered unit is wrapped in `execute_with_timeout_and_retry` which:

- Yields once before the first attempt so pending abort signals are processed.
- Calls `unit.execute(&ctx)` *inside the installed `CURRENT_CAPS` scope* (`batch.rs:214-232`), so any capability-gated I/O sees the batch's `CapabilitySet`.
- On `Err(WorkError)`, sleeps the shared jittered-exponential backoff via `common_core::retry::retry_async`
- Wraps the whole thing in `tokio::time::timeout(timeout_ms, …)`; on timeout returns `Err(WorkError::Timeout { duration_ms, unit })` and `poll` records it as `Cancelled(Timeout)`.

**Panic propagation**: panics are *not* caught via `catch_unwind`. They propagate through `JoinSet` as `JoinError::Panic`, which `SupervisedBatch::poll` intercepts (`batch.rs:300-330`) and records as `SupervisedBatchEvent::Panicked`. This is deliberate: a `catch_unwind` would lose the panic site and prevent the dependency-aware cancellation graph from firing.

**Event taxonomy** (`batch.rs:20-75`):

```text
SupervisedBatchEvent::Completed   { name, output }
SupervisedBatchEvent::Panicked    { name, info }
SupervisedBatchEvent::Failed      { name, error: WorkError }
SupervisedBatchEvent::Cancelled   { name, reason: CancelReason }

CancelReason::Timeout
CancelReason::DependencyFailed
CancelReason::Aborted
```

`WorkError` and `WorkOutput` are defined in `fluent-wvr::work` with three error variants: `Execution(String)`, `Dependency(String)`, and `Timeout { duration_ms: u64, unit: String }`. `WorkOutput` carries a `serde_json::Value data` field with `typed`/`typed_infallible`/`data_as`/`data_take` accessors for structured-data round-tripping.

**Summary** (`batch.rs:69-75`): `SupervisedBatchSummary { completed, panicked, failed, cancelled: Vec<SupervisedBatchEvent> }`. The batch `await`s to completion (when all `JoinSet` entries are drained and `active_count == 0`) and yields the summary.

**Key properties:**
- A panic in task A does **not** propagate to the parent runtime thread. It is caught as a `JoinError` by the SupervisedBatch.s `poll` loop.
- The SupervisedBatch cancels only the dependents of the failed task; independent tasks continue.
- Neighboring SupervisedBatches are fully isolated because each owns its own `JoinSet`.
- `WorkError::Execution` failures go to `summary.failed`; real panics go to `summary.panicked`.

### 3.4 `WorkerPool` — Bounded Worker Pool

RabbitMQ's `worker_pool` uses a central queue and a fixed set of worker processes that pull jobs. We translate this directly to Tokio tasks. `WorkerPool` is the fire-and-forget variant: each worker calls the handler, the result is discarded.

**API** (`pool.rs:147-220`):

```rust
pub struct WorkerPool<T: Send + 'static> { /* queue, workers, shutdown */ }

impl<T: Send + Sync + 'static> WorkerPool<T> {
    pub fn new<F, Fut>(
        runtime: Arc<dyn Runtime>,
        cap: usize,
        queue_capacity: usize,
        handler: F,
    ) -> Self
    where F: Fn(T) -> Fut + Send + Sync + 'static,
          Fut: Future<Output = ()> + Send;

    pub async fn try_submit(&self, job: T) -> Result<(), PoolError>;  // Err(Full) if at cap
    pub async fn submit(&self, job: T) -> Result<(), PoolError>;      // waits, Err(Closed) if closed
    pub async fn shutdown(self);                                       // close queue, await workers
}
```

Internally it uses a `Queue<T>` (see below) and a `Notify` for close-wakes-waiters. Workers loop on `tokio::select! { shutdown.notified() | queue.pop() }`. If you don't need the result, this is the zero-allocation-per-submit primitive. For the result-returning variant, see `ResultPool` (§3.10). `Queue<T>`'s backing store is a `std::sync::Mutex<VecDeque<T>>` — the lock is never held across an await (every guard is scoped and dropped before the future yields), so the async `Mutex`'s waker bookkeeping would be pure overhead; a `std::sync::Mutex` with a microsecond critical section (`VecDeque::push_back`/`pop_front`) is the Tokio-recommended choice for this shape. Poison is recovered via `common_core::sync::lock` (`common_core::sync::lock`) rather than `unwrap` — the queue never runs user code while locked, so the only panic-while-locked path is allocator OOM (which leaves `VecDeque` in a valid state); wedging the pool on poison would be a worse availability failure than recovering the valid deque. (`CreditSender` in `flow.rs` is the one exception that must stay a `tokio::sync::Mutex` — its guard protects the bump receiver across `recv().await`.)

**Credit-gated variant — `CreditGatedPool<J>`** (`credit_pool.rs`): a `WorkerPool` whose `submit` path acquires a credit token before enqueuing and whose handler releases it once per processed job. Extracted from the router's two structurally-identical workers (`ReviewWorker`, `EntityLinkWorker`); a third credit-gated worker MUST compose it rather than re-copying the `CreditSpec`/cap scaffolding (the worker cap `2` and the `CreditSpec { max(1), max(1) }` formulas are load-bearing). API: `new(runtime, credit_limit, queue_capacity, handler)`, `submit(job) -> Result<(), PoolError>` (blocks only when credit is exhausted), `is_blocked()`, `drain()`, `worker_count()`. **Status: production** (both workers migrated).

**Why not `tokio::sync::Semaphore`?** A `Semaphore` is perfect for a *limiter* (see §3.5), but it does not provide a FIFO queue of jobs or dedicated workers. RabbitMQ's `worker_pool` explicitly wants workers to pull from a queue, allowing prioritization and monitoring of queue depth. Our `WorkerPool` gives exactly that.

### 3.5 `Limiter` — Lightweight Concurrency Cap

For cases where you don't need a dedicated worker pool, just a cap on concurrent executions, the `Limiter` is a `Semaphore`-backed wrapper (`pool.rs:534-597`).

**API**:

```rust
pub struct Limiter { sem: Arc<Semaphore> }

impl Limiter {
    pub fn new(cap: usize) -> Self;
    pub async fn run<F, Fut, T>(&self, f: F) -> T
        where F: FnOnce() -> Fut, Fut: Future<Output = T>;
    pub fn run_sync<F, Fut, T>(&self, f: F) -> T
        where F: FnOnce() -> Fut, Fut: Future<Output = T>;
}
```

`run_sync` first tries `tokio::runtime::Handle::try_current()`. Inside a running **multi-threaded** runtime (the router's HTTP handler does this) it drives the permit-acquire + closure via `tokio::task::block_in_place(|| handle.block_on(block))` — a bare `Handle::block_on` would panic with "Cannot start a runtime from within a runtime". Inside a current-thread runtime it uses `handle.block_on(block)`. With no active runtime it builds a dedicated current-thread runtime for the duration of the call. This is the right tool for synchronous callers (e.g., a CLI handler invoked outside an async context) that need to bound concurrency. **Caveat:** on the multi-thread scheduler `block_in_place` *parks the calling worker thread* for the duration of the call (the scheduler spins up a replacement thread and the parked thread is unavailable for work stealing until it returns) — `Limiter::run_sync` is therefore the correct sync↔async bridge, but it is **not for throughput-critical paths**; bound its use to control-plane callers that genuinely cannot `await` (`pool.rs:567-595`) and prefer the async `Limiter::run` everywhere else. Burst calling `run_sync` on the runtime would inflate the thread pool, which is exactly why the async path exists.

This is the Rust equivalent of the `credit_flow` sender side: acquire a slot, run the work, release the slot on completion.

### 3.6 `PriorityQueue` — Event Queue

A simple priority queue optimized for the common case where most items have priority 0, exactly like RabbitMQ's `priority_queue.erl` (`queue.rs:7-145`).

**Storage**: a `VecDeque<T>` for the all-zero-priority fast path, plus a `BTreeMap<i32, VecDeque<T>>` for distinct non-zero priorities. A cached `count: usize` makes `len()` and `is_empty()` O(1).

**API**:

```rust
pub struct PriorityQueue<T> { /* simple: VecDeque<T>, prioritized: BTreeMap<i32, VecDeque<T>>, count: usize */ }

impl<T> PriorityQueue<T> {
    pub fn new() -> Self;
    pub fn push(&mut self, item: T, priority: i32);   // priority 0 → VecDeque; non-zero → BTreeMap
    pub fn pop(&mut self) -> Option<T>;               // positive priorities first, then 0, then negative
    pub fn peek(&self) -> Option<(&T, i32)>;          // highest-priority item without removing
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn into_iter(self) -> impl Iterator<Item = (i32, T)>;  // priority order, consumes self
    pub fn drain(&mut self) -> impl Iterator<Item = (i32, T)> + '_;  // same, but borrows
}
```

This is O(log P) for `push` and `pop`, where P is the number of distinct non-zero priorities. It is zero-allocation for the all-zero-priority case.

### 3.7 `CreditFlow` — Chain Backpressure

RabbitMQ's `credit_flow` module throttles publishers end-to-end. Our `CreditFlow` uses explicit message passing between sender and receiver, preserving the exact semantics (`flow.rs:13-98`).

> **Status: production.** The ledger tier feed consumes it in the router: `LedgerTierWorker` owns a `flow::new(CreditSpec)` pair whose `CreditSender` gates the async producer path (`LedgerAgentCoordinator::run_agent` step 5 → `enqueue_with_credit`, `ledger/tiering.rs`, `ledger/orchestrator.rs`) and whose `CreditReceiver::recv()` releases a token after each processed node — so a burst of agent turns cannot grow the feed without bound (the feed channel itself is a bounded `mpsc`; knobs `ledger.tier_credit_limit` / `ledger.tier_credit_more_after`). The review and entity-link workers are the other CreditFlow consumers, but both now compose the shared `CreditGatedPool` (§3.4) rather than constructing a `flow::new` pair by hand — the `CreditSpec` formulas live in one place. The remaining LLM fan-out paths — coral's `LlmRequestQueue` and the router's escalation team-mode parallel votes — are bounded by `ResultPool` (a `cap`-and-`queue_capacity`-limited worker pool whose `submit` provides the backpressure), and team-mode's decomposer "subtasks" are counted for audit but not dispatched to a pool, so there is no separate credit-token producer to throttle there. `flow.rs`'s own tests (`tests/m4.rs`) are the semantics baseline. (The `memory-plugin::MemoryBatch` stub that once consumed `flow::new` was pruned in the 2026-08-15 scaffold review — its body was a log-only stub and no caller constructed it.)

**API**:

```rust
pub struct CreditSpec { pub initial: usize, pub more_after: usize }

pub struct CreditSender  { credit: AtomicIsize, bump_rx: Mutex<mpsc::UnboundedReceiver<usize>>, blocked: AtomicBool }
pub struct CreditReceiver { spec: CreditSpec, counter: AtomicUsize, bump_tx: mpsc::UnboundedSender<usize> }

pub fn new(spec: CreditSpec) -> (CreditSender, CreditReceiver);

impl CreditSender {
    pub async fn send<F, Fut, T>(&self, op: F) -> T
        where F: FnOnce() -> Fut, Fut: Future<Output = T>;
    pub fn is_blocked(&self) -> bool;
    pub fn current_credit(&self) -> isize;
}

impl CreditReceiver {
    pub fn recv(&self);  // increments counter; sends `more_after` as bump when counter ≥ more_after
}
```

The sender's `send` is a CAS loop: load credit, if `> 0` decrement and run `op().await`; if `<= 0` mark `blocked = true` and `await` a bump from the receiver. The receiver's `recv` is non-async: it `fetch_add(1)` and, when the counter crosses `more_after`, resets to 0 and sends a `more_after`-sized bump.

This maps 1:1 to the Erlang `credit_flow` semantics: `send` decrements credit, `ack` (called `recv` here) counts down and sends a `bump_credit` when the counter hits zero.

### 3.8 `PartitionedRouter` — Delegate / Sharding

RabbitMQ's `delegate` module groups PIDs by node and routes them to local delegates to reduce inter-node chatter. In a single-process Rust system, this becomes a key-based router (`router.rs:7-21`).

**API**:

```rust
pub struct PartitionedRouter<K, J: Send + 'static> {
    shards: Vec<WorkerPool<J>>,
    hash: fn(&K) -> usize,
}

impl<K, J: Send + Sync + 'static> PartitionedRouter<K, J> {
    pub fn new(shards: Vec<WorkerPool<J>>, hash: fn(&K) -> usize) -> Self;
    pub async fn submit(&self, key: &K, job: J) -> Result<(), PoolError>;
}
```

The implementation is intentionally thin: it hashes once per submit and forwards to the appropriate shard. This preserves causal ordering: all jobs with the same key always go to the same shard. There is no per-shard statistics, no shard rebalancing, and no fail-over — the primitive is a shim, not a full delegate supervisor. If you need shard observability, build it on top. **Status: compatibility surface (scaffold).** No production caller exists outside its own tests (`tests/m3.rs`) — the 2026-08-15 scaffold review kept it as the keyed-sharding seam (delete `router.rs` + its tests if a genuine second consumer never materializes).

### 3.9 `Runtime` Trait — Pluggable Backend

The `Runtime` trait is defined in `fluent-wvr::runtime` (`fluent-wvr/src/runtime.rs:20-24`) and has three methods:

```rust
pub trait Runtime: Send + Sync + 'static {
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send>>) -> JoinHandle<()>;
    fn sleep(&self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>>;
    fn now(&self) -> Instant;
}
```

**Why `BoxFuture`?** The `Runtime` trait is object-safe so it can be stored as `Arc<dyn Runtime>` in the Control Plane. The cost of one `Box` per `sleep` is negligible because `sleep` is a boundary operation, not a hot-loop inner operation. `spawn` returns `JoinHandle<()>` directly (no Box) because the type is concrete.

**Three backends** ship today:

- `NoopRuntime` (in `fluent-wvr/src/runtime.rs:33-55`) — for `WorkContext::default()`. `spawn` panics with a clear message if called outside a tokio runtime; inside one, it spawns an empty future. `sleep` returns immediately. `now` returns `Instant::now()`.
- `TokioRuntime` (in `fluent-concurrency/src/runtime/tokio.rs:11-26`) — production. Delegates to `tokio::spawn` and `tokio::time::sleep`. Constructed via the convenience function `fluent_concurrency::tokio_runtime() -> Arc<dyn Runtime>` (`lib.rs:19-21`). It also exposes an inherent generic `TokioRuntime::spawn<F>(future)` (`runtime/tokio.rs`) that skips the `Pin<Box<dyn Future>>` boxing — the future is inlined into tokio's task allocation. This is the **hot-path seam** for callers who hold `TokioRuntime` concretely; the trait method (which delegates to the inherent method) stays the `Arc<dyn Runtime>` abstraction.
- `TestRuntime` (in `fluent-concurrency/src/runtime/test.rs:13-56`) — wraps a `tokio::runtime::Handle` plus a seeded `fastrand::Rng` for reproducible non-determinism. Use with `tokio::time::start_paused` for record-replay tests. Note: `TestRuntime::Clone` re-seeds from the stored `seed` field, so cloned runtimes reproduce the **same** deterministic PRNG sequence as the original (they do *not* advance a shared stream). Clones share the underlying `Handle`; the PRNG state is per-clone.

### 3.10 Bonus Primitives

These exist in the crate but are not in the original spec. They earn their place by filling gaps that real consumers hit.

**`ResultPool<T, R, E>`** (`pool.rs:244-334`) — the result-returning variant of `WorkerPool`. The handler is `Fn(T) -> Fut where Fut: Future<Output = Result<R, E>>`, and `submit(job) -> Future<Output = Result<R, ResultPoolError<E>>>`. Internally each `submit` allocates a `tokio::sync::oneshot::channel`; the worker sends its `Result<R, E>` back through the channel. The cost is one `oneshot` per submit; the benefit is that the submitter gets the result as an `await`able future rather than passing through a side channel. This is the canonical pool for "fan out N independent jobs, collect N results" workloads (e.g., `AST_POOL` and `DB_POOL` in `src/guidance/src/runtime.rs`).

```rust
pub enum ResultPoolError<E> { Pool(PoolError), Inner(E), Canceled }

impl<T, R, E> ResultPool<T, R, E> {
    pub fn new<F, Fut>(runtime, cap, queue_capacity, handler: F) -> Self;
    pub fn worker_count(&self) -> usize;
    pub async fn submit(&self, job: T) -> Result<R, ResultPoolError<E>>;
    pub async fn submit_with_abort(&self, job: T, abort: Option<StreamAbort>)
        -> Result<R, ResultPoolError<E>>; // abort cancels the in-flight handler
    pub async fn shutdown(self);
}
```

There is **no** fire-and-forget variant: because the worker protocol requires a `oneshot::Sender`, every submission — even one whose result is ignored — allocates a channel. If you need true fan-out with no result, use `WorkerPool` instead.

**`PriorityResultPool<T, R, E>`** (`pool.rs:433-529`) — priority-ordered variant of `ResultPool`. Jobs are submitted with `submit(job, priority)`; higher priority values are dispatched first, FIFO within the same priority. Internally it uses the `BoundedPriorityQueue` from `queue.rs` — a `PriorityQueue` (ordering) behind a `std::sync::Mutex` (never held across an await, so the async mutex's waker bookkeeping would be pure overhead; poison recovered via `common_core::sync::lock`) with `Queue`-style close-wakes-waiters semantics. **`submit`/`submit_with_abort` apply backpressure**: they block while the queue is at capacity (`PoolError::Full` is the synchronous `BoundedPriorityQueue::push` fast path's error; `Pool(Closed)` surfaces from a submit after shutdown) instead of growing without bound. Workers follow the **drain-then-wait** pattern. `new(runtime, cap, handler)` sizes the queue to `cap * 4`; `with_queue_capacity(runtime, cap, queue_capacity, handler)` overrides it.

**`LlmRequestQueue`** — MOVED (ROADMAP_20260903_LLM M9/M11). The typed
`ResultPool` wrapper for LLM chat completions (`LlmRequestQueue`,
`LlmTask`, `LlmConfig`, `ChatMessage`, `LlmQueueConfig`, `LlmError`) is
owned by `fluent_llm::protocol`; the `Fn(LlmTask) ->
Result<String, LlmError>` handler is supplied at construction time and the
default OpenAI-compatible HTTP handler lives in
`fluent_llm::llm_queue`. This crate stays transport-agnostic: compose the
generic `ResultPool` / `PriorityResultPool` / `StreamAbort` primitives
below for non-LLM work, and import LLM queue types from `fluent_llm`.

```rust
// Canonical paths (M11 deleted fluent_concurrency::llm_queue):
use fluent_llm::protocol::{
    ChatMessage, LlmConfig, LlmError, LlmQueueConfig, LlmRequestQueue, LlmTask,
};
```

`LlmConfig.extra_body_params` is arbitrary JSON merged into every request body (e.g. `num_ctx`, `temperature`, `stop`); `"model"`, `"messages"`, and `"stream"` keys are ignored because those are set explicitly by the chat-completion logic. `LlmConfig::new()` is a `bon` builder (`start_fn = new`).

**`Reserve`** (`reserve.rs:7-94`) — RAII permit on a shared `Arc<AtomicUsize>`. `try_acquire(counter) -> Option<Self>` atomically decrements and returns the permit, or `None` if already at zero. `commit(self)` consumes the permit permanently; otherwise `Drop` returns it to the counter. This is a lower-level primitive than `Limiter`: it does not own the counter (the caller supplies it), and it does not run a closure. Use `Limiter` for "run this with a permit" and `Reserve` for "acquire now, release later" patterns. **Status: compatibility surface (scaffold)** — the crate's own tests are the only in-tree consumer today; the 2026-08-15 scaffold review kept it as the documented low-level permit seam (delete `reserve.rs` + its tests if a genuine second consumer never materializes).

**`StreamAbort`** (`stream.rs:1-...`) — a sticky, `Clone` cancellation signal for long-lived streams. Fires once (`cancel`, idempotent), wakes every currently- and future-waiting task (`cancelled`, sticky), and can be inspected (`is_cancelled`). Carries no I/O — transports drop their connections and management planes issue explicit aborts as a *reaction* to it. `Clone` is an `Arc` bump, so every task that touches a stream (the forwarding task, the downstream body-drop guard, the management-abort watcher) holds its own clone and observes the same state. The canonical wiring is the **body-lifetime rule**: the HTTP body handed to the client is wrapped so its `Drop` fires `cancel()` — the body's lifetime is the single source of truth for "the consumer is gone" — and the forwarding task `select!`s its upstream read against `cancelled()`. **Status: production.** Consumed by the router's streaming dispatch (`OpenAiChatBackend::stream_complete_with_abort`, `dispatch/backend.rs`), which selects its upstream read *and* its body sends against the signal so a blocked `send_data` never wedges the task.

**Abort-aware pool submissions.** `ResultPool::submit_with_abort(job, abort)` and `PriorityResultPool::submit_with_abort(job, priority, abort)` race the handler future against the optional `StreamAbort`: when the signal fires mid-flight the handler future is dropped (an in-flight HTTP call is cancelled, the worker slot is freed) and the submitter observes `ResultPoolError::Canceled` — making that variant *reachable from the submitter side* (previously it fired only on worker panic/shutdown). `fluent_llm::protocol::LlmRequestQueue::submit_with_abort(task, abort)` exposes the same seam for LLM calls; plain `submit` is unchanged (`None` abort). **Status: production** for the pool seam; the protocol queue's abort seam is currently a compatibility surface with no in-tree caller.

**`AffinityScheduler<T, R, E>`** (`affinity.rs:59-191`) — wraps `PriorityResultPool` with session-aware priority boosting and starvation aging. Tracks which session is currently "affine" and gives its tasks a priority bonus (`AgingConfig::affinity_bonus`, default +10). Starved tasks (different session) periodically increase in base priority at `AgingConfig::aging_rate` (default +2 per 5s tick) up to `AgingConfig::max_priority` (default 100). `set_affinity(Option<String>)` switches the active session. `submit(task: ScheduledTask<T>, base_priority)` computes the effective priority and delegates to the pool. Designed for multi-session agent dispatch where context-switching between sessions should be minimized. **Status: production.** Wired at the ledger-agent coordinator's boot path: `ledger.orchestrator.affinity_cap: <cap>` on `RouterConfig` opts the boot into attaching a scheduler via `LedgerAgentCoordinator::build_affinity_scheduler(cap)` (composed through `build_ledger_coordinator`), so the active `run_agent` session is tracked as KV-affine for priority boosting.

**`thread_local_resource!`** (`thread_resource.rs:37-45`) — macro that declares a thread-local `RefCell<Option<T>>` backed by `Default`. The `with_tlr(&STATIC, |res| ...)` accessor takes-or-defaults, calls the closure, and stores the value back after the closure returns. This is the canonical mechanism for per-thread pooled resources (e.g., AST parsers, string builders) that benefit from reuse across calls without allocating each time.

### 3.11 `first_accept_in_order` — the First-Ok Ladder (canonical fallback combinator)

**Status: production.** The single first-Ok fallback combinator, consumed by the router's dispatch chain, backend retry/fallback, and the escalation ladder. Any "try A, then B, then C, take the first success" logic MUST compose this instead of hand-rolling a loop.

**API** (`ladder.rs:38`):

```rust
pub async fn first_accept_in_order<R, E, F, Fut>(
    rungs: impl IntoIterator<Item = F>,
    run: impl Fn(F) -> Fut + Send + Sync,
    stop: impl Fn(&E) -> bool + Send + Sync,
) -> Result<Option<R>, E>
where
    F: Send + Sync + 'static,
    Fut: Future<Output = Result<R, E>> + Send,
    E: Send + 'static,
    R: Send + 'static,
```

**Sync twin** (`ladder.rs`, same module): `first_accept_in_order_sync` mirrors the async semantics exactly (`Ok(Some)` wins, `Ok(None)` skips, `Err(e)` continues unless `stop(&e)`, exhaustion returns the last `Err` or `Ok(None)`) for non-async callers:

```rust
pub fn first_accept_in_order_sync<T, R, E, F, Stop>(
    rungs: impl IntoIterator<Item = T>,
    run: F,
    stop: Stop,
) -> Result<Option<R>, E>
where
    F: FnMut(T) -> Result<Option<R>, E>,
    Stop: FnMut(&E) -> bool,
```

Its one consumer today is the `spacy-rs` synchronous annotation ladder (`run_ladder_sync`), which previously hand-rolled the same first-accept walk with a sequential `if let Some(fetch) … if let Some(encoder) … ArcEager … rule` chain.

- `rungs` — the ordered fallback candidates (owned; no clones or `&mut` counters needed by the caller).
- `run` — one attempt per rung; returned `Ok(r)` short-circuits to `Ok(Some(r))`.
- `stop` — when `true` for an `Err(e)`, the walk **aborts immediately** with `Err(e)` (the terminal error is first-class — no silent swallow). When `false`, the rung is skipped and the next is tried.
- Returns `Ok(None)` when every rung failed non-terminally (clean exhaustion) — the caller decides the fallback (`AllBackendsFailed` etc.).

**Why this shape:** it is monomorphized (no vtable per rung), takes owned rungs (callers like `BackendChain` no longer clone the backend list), and makes the terminal error a first-class `Err` instead of a `last_error` variable captured across the loop. Consumers:

| Consumer | Pattern | Location |
|----------|---------|----------|
| `BackendChain` | `complete`/`stream_complete` fallback across backends | `router/src/dispatch/backend.rs` |
| `dispatch_real` | model fallback walk with attempt counter | `router/src/server/dispatch.rs` |
| `Ladder::try_escalate` | escalation-mode walk (`stop = |_| false`, `.ok().flatten()`) | `router/src/dispatch/escalation/mod.rs` |
| spacy-rs annotation ladder | async `run_ladder_for` over `[LLM, encoder, ArcEager, rule]` | `spacy-rs/src/pipeline.rs` |
| spacy-rs sync annotation ladder | `first_accept_in_order_sync`, `stop = |_| false` | `spacy-rs/src/pipeline.rs` (`run_ladder_sync`) |

**Rule:** any ordered first-success walk must use `first_accept_in_order` (or its sync twin). The router's `match_target` is the one documented exception (start-index skip + last-always-wins + per-rung audit — a different shape; see `target_match.rs`).

## 4. Control Plane / Data Plane Integration (Fluent WVR)

The `fluent-wvr` crate defines `Component`, `WorkUnit`, `FieldAccess`, `Describable`, `SchemaProvider`, `Capability`, `CapabilitySet`, `Runtime`, `WorkContext`, `WorkError`, `WorkOutput`, and `MetadataValue`. `fluent-concurrency` does not redefine these; it **consumes** them.

**WorkContext** (`fluent-wvr/src/work.rs:122-191`) is the per-execution environment carried by every `WorkUnit::execute`. It bundles `rt: Arc<dyn Runtime>`, `caps: CapabilitySet`, `metadata: HashMap<String, MetadataValue>`, `dry_run: bool`, `max_retries: u32`, `timeout_ms: u64`. Three constructors are provided:

- `WorkContext::default()` — uses `NoopRuntime`; safe for dry-run / init paths but panics on `spawn` if not inside a tokio runtime.
- `WorkContext::for_unit(unit, caps)` — uses the unit's `default_timeout_ms()` and a default runtime; the right entry point for a single unit.
- `WorkContext::for_unit_in_batch(batch_rt, batch_caps, |ctx| { ... })` — clones the batch's runtime and caps, then lets the caller mutate; the canonical entry point inside a `SupervisedBatch`.

**Wrappers from `fluent-wvr`**:

- `Instrumented<U>` — wraps any `WorkUnit`/`Component` with `tracing::info!` timing on every `execute`, plus optional `Arc<LatencyHistogram>` recording via `Instrumented::with_metrics(inner, label, histogram)`. The histogram is the canonical latency surface from `common_core::metrics`.
- `ComponentAdapter` — runtime override of `name`, `execute`, and any field on a `Component`. `set_field` forwards via `Arc::get_mut` when possible and otherwise stores an override on the adapter. This is the only wrapper that handles the shared-Arc case correctly. `Instrumented` is a *transparent* wrapper: its `FieldAccess` impls delegate `set_field`/`get_field`/`field_names` straight to the inner type (requiring exclusive access to the inner or interior mutability), matching the delegation semantics of its `WorkUnit` impls.
- `retry_call(max_attempts, base_ms, f)` — the synchronous free-function retry (explicitly blocking, never for `execute` bodies) with jittered-exponential backoff. Returns `Result<RetryResult<T>, E>` where `RetryResult` carries the attempt count. The async canonical path is `common_core::retry::retry_async`.

**Retry**: there is no `WithRetry` wrapper. The single jittered-exponential backoff helper lives in `common_core::retry` (`backoff_ms` / `retry_async`); `SupervisedBatch` drives per-attempt timeout, `WorkError::Timeout` routing, and dependency cancellation on top of it.

**Macros from `fluent-wvr`**:

- `impl_component!(MyType)` and `impl_component!(generic (U: Component + 'static) for Wrapper<U>)` — eliminates the 7-line `as_any`/`as_any_mut` boilerplate that every `Component` implementor would otherwise write.
- `#[derive(FieldAccess, Describable)]` (in `fluent-wvr-macros`) — `#[field(...)]` attributes support `skip`, `desc`, `min`/`max` (numeric), `format`, `max_len`, `sanitize` (trim/lowercase/strip_html/slugify), `pattern` (substring, not regex), `required` (default true), and `empty_is_none` (default true for `Option<String>`). Boundary-string coercion modes (`coerce = "trim,strip_quotes"` and `parse = "number"` for tolerant numerics) are applied by `set_field` before constraints — see `fluent-wvr::coerce` / `fluent-wvr::boundary`.

**Arc blanket impls** (`fluent-wvr/src/lib.rs:50-129`): the type `Arc<dyn Component>` is the universal wire type. Blanket impls provide `WorkUnit`/`FieldAccess`/`Describable`/`Component` for `Arc<dyn Component>` and `WorkUnit` for `Arc<dyn WorkUnit>`. The latter exists for cases where the implementor doesn't need the full `Component` surface. This is the boundary that lets `SupervisedBatch`, `ComponentAdapter`, and any orchestrator dispatch through a uniform `Arc<dyn Component>` without knowing the concrete type.

**Cross-cutting concerns** (timing, rate limiting) are applied via the
`Instrumented` wrapper *before* type erasure, preserving zero-cost inlining.
Retry composes `common_core::retry` (the single jittered-exponential backoff
helper); `SupervisedBatch` has its own per-task retry loop (driven by
`WorkContext::max_retries` / `WorkContext::timeout_ms`, sleeping via the
shared helper) that exists because the SupervisedBatch manages a long-lived set of units
with dependency cancellation, whereas the helper is per-call.

## 5. Performance & Locality Guarantees

| Hot Path | Technique | Why |
|----------|-----------|-----|
| Task scheduling | Tokio's local queue + LIFO slot | We do not add indirection. |
| Worker pool job dispatch | `VecDeque` in `std::sync::Mutex` (scoped, never across `await`) | One short lock per pop; `tokio::sync::Mutex` would add waker overhead for no gain; workers sleep on `Notify`. No `dyn` dispatch per job. |
| Result pool per-submit | `oneshot::channel` per submit | One allocation per job; the worker sends the result through the channel. |
| Priority queue (all same priority) | `VecDeque` fast path | Zero overhead for the common case. |
| SupervisedBatch dependency lookup | `DependencyGraph::dependents_of` (cycle-resilient DFS) | Transitive dependents without scanning the full DAG; cycle is `warn!`-ed, not panicked. |
| SupervisedBatch poll budget | `poll_budget=64`, `wake_by_ref` only when ready-budget exhausted | Prevents one batch starving the executor when many tasks complete in the same wake; no spin when `JoinSet` is `Pending`. |
| Capability gating | `HashMap<TypeId, Arc<dyn Any>>` lookup on `CURRENT_CAPS` via `try_with` | `TypeId` is pointer-sized; no string comparison. `name()` is informational only; absence is `PermissionDenied` (fail-closed). |
| Data transformation | Concrete enums + pattern matching | `WorkUnit::execute` is one vtable call per task; inside it, all work is monomorphized. |
| `Reserve` permit | `AtomicUsize` `fetch_sub`/`fetch_add` | Lock-free, no heap allocation. |
| Poison recovery | `common_core::sync::lock` (`into_inner` on poison) | Queue holds only `VecDeque`/`BTreeMap` with `push`/`pop` that leaves structure valid; wedging on poison would be worse availability loss than recovering the valid deque. |

## 6. Crate Layout (Actual)

The crate is no longer "proposed" — it ships. The current `Cargo.toml`:

**Dependencies:**

- `tokio` (workspace, with the `rt-multi-thread`, `sync`, `time`, `macros` features in production)
- `fluent-wvr` (path = `"../fluent-wvr"`) — for `WorkUnit`, `Component`, `CapabilitySet`, `Runtime`, `ArcIntern`
- `serde`, `serde_json` — for `WorkOutput::data` and `Describable`
- `bon` — derive builder for `LlmConfig` and `ChatMessage`
- `thiserror` — error enums (`PoolError`, `ResultPoolError`, `SupervisedBatchError`, `LlmError`, `CapabilityError`)
- `tracing` — `info!`, `warn!`, `error!` from `Instrumented`, `SupervisedBatch`'s cycle detection, and `Scope`'s panic-during-unwind path
- `reqwest` — backing HTTP client for `NetCapability`
- `common-core` — `LatencyHistogram` (used by `Instrumented::with_metrics` in `fluent-wvr` consumers), `retry` (SupervisedBatch backoff), and `error::IoError`
- `fluent-db` (**optional**, `db` feature, default-on) — `SqlitePool` / `DbCapability` for the database surface. `rusqlite` and `common-core`'s `sqlite` feature are no longer direct deps; they come in transitively only when the `db` feature is on (D11).
- `fastrand` — seeded PRNG for `TestRuntime` and jitter in `common_core::retry` / `retry_call`
- `internment` (with `arc` and `serde` features) — `ArcIntern<str>` for work unit names, dependency asset names, and configuration keys

**Dev-dependencies:**

- `tokio` (with `test-util` feature) — for `start_paused` in tests
- `tempfile` — for filesystem-based tests
- `fluent-wvr-testutil` — `impl_component_for_test!` and `StubComponent` for unit-test scaffolding

No `async-trait` crate (native AFIT is used for the `Pool` traits — no proc-macro, no hidden `Box`), no `bumpalo`, no `crossbeam`. Tokio's channels, `Notify`, and `Semaphore` are sufficient. (One macro: `impl_component!` lives in `fluent-wvr`, not here; derive macros for `FieldAccess` and `Describable` live in `fluent-wvr-macros`.)

## 7. Anti-Patterns Explicitly Rejected

1. **No `async_trait` crate; native AFIT allowed for control-plane traits.** `fluent-concurrency` forbids the `async_trait` proc-macro crate (its `Box<dyn Future>` hides an allocation per call). Native `async fn` in trait (AFIT/RPITIT, stable since Rust 1.75) is allowed for non-object-safe, non-hot-loop control-plane traits — the canonical example is the `Pool` family (`SinkPool`/`RequestPool`/`PriorityRequestPool` in `pool.rs`/`credit_pool.rs`). Hot-loop dispatch (`Limiter::run`, queue `pop`, per-item work) stays concrete generics per `fluent-wvr/SKILL.md:70` ("dyn at request boundary, concrete in tight loop"). The orchestration core (`SupervisedBatch` hand-written `impl Future` `batch.rs:251-365`; `Scope` via `JoinSet`) stays manual `Future` so poll budgets, dependency cancellation, and capability propagation remain compiler-visible.
2. **No `tokio::spawn` without a scope/batch.** The `tokio::spawn` calls in the crate are limited to: `Scope::spawn` (which installs `CURRENT_CAPS`); `SupervisedBatch::spawn_unit` (which installs `CURRENT_CAPS` from `WorkContext.caps`, `batch.rs:206-235`); `Scope::close`/`close_graceful` drains; `WorkerPool`/`ResultPool`/`PriorityResultPool` worker loops; and `DbCapability::query`/`execute` in `fluent-db` (which `spawn_blocking` for sync `rusqlite` work). Every async effect outside these sites flows through `Scope::spawn` or `SupervisedBatch` (both capability-scoped and handle-tracked) or through a pool (which tracks workers). Raw `tokio::spawn` without a scope is an anti-pattern because it loses `CURRENT_CAPS` and is denied at the first gated I/O.
3. **No `dyn Trait` in a per-item loop.** The `PartitionedRouter` hashes once per submit; `PriorityQueue` dispatches via `BTreeMap` keys, not vtables; `CapabilitySet::get` uses `TypeId` (pointer-sized) rather than string comparison; `PartitionedRouter` does no per-job vtable dispatch.
4. **No ambient `tokio::fs::read` or `tokio::time::sleep`.** All I/O is called through a capability method (`FsCapability::read`, `NetCapability::http_get`, `DbCapability::query`, etc.) which calls `check_capability(self)` first. `tokio::time::sleep` is allowed in the framework's own internals (SupervisedBatch retry, SupervisedBatch timeout) and in the `Runtime::sleep` backend, but consumer code is expected to use `Limiter::run`, `common_core::retry`, or `SupervisedBatch` rather than calling it directly.
5. **No automatic restart.** SupervisedBatch instances contain; they do not restart. Restart is a deliberate operator action.

## 8. Dependency Resolution with DependencyGraph

`SupervisedBatch` (`src/fluent-concurrency/src/batch.rs`) composes `fluent_dag::dep_graph::DependencyGraph<K>` for dependency tracking and cancellation. This replaced three hand-rolled `HashMap`s with a single canonical primitive.

### How it works

Each `SupervisedBatch::register(unit)` call registers the unit's `name()` and its return value of `provides()` in the `DependencyGraph<ArcIntern<str>>`. When a unit fails or panics, `SupervisedBatch::cancel_dependents_of(name)` calls `DependencyGraph::dependents_of(name)` (cycle-resilient DFS) to find all transitive dependents and cancels them.

### When to compose DependencyGraph directly

If you need dependency tracking outside a `SupervisedBatch` (e.g., pipeline step DAGs, build-target graphs, session step ordering), compose `DependencyGraph<K>` directly. Examples:

| Consumer | Pattern | Location |
|----------|---------|----------|
| `DependencySession` | Session steps with checkpoint/rewind | `src/router/src/dag_session.rs` |
| `SupervisedBatch` | Task supervision cancellation tree | `src/fluent-concurrency/src/batch.rs` |

**Rule**: Any new dependency-tracking workflow MUST compose `DependencyGraph<K>` rather than re-implementing graph algorithms. See `dag/SKILL.md` for the full API.

---

## 9. Rejected as Scope Creep

Here are examples of what fluent-concurrency does not try to do as a lightweight single-node runtime, compared to RabbitMQ:

- **Actor Hibernation / Idle Backoff**: RabbitMQ's `gen_server2` needs hibernation because it manages hundreds of thousands of idle, long-lived connections. fluent-concurrency is a pipeline execution engine—tasks are spawned to complete work and terminate. Workers parked on `tokio::select!` are sleeping efficiently on native OS epoll/kqueue event loops. Adding an explicit backoff framework here adds unnecessary overhead.
- **Multi-hop Credit Chains**: Our single-hop producer/consumer backpressure is perfectly tailored for a single-node pipeline. We do not need a multi-process AMQP chain.
- **Distributed tracing / OpenTelemetry export**: not in scope; `tracing::info!` is sufficient for the current consumers. If a future consumer needs OTel, the `tracing` crate has compatible subscribers.
- **Loom-style combinatorial scheduler exploration**: the `Runtime` trait is wide enough to add a `LoomRuntime` later, but it is not built today. Q3 from §2 documents this as a future primitive.

## 10. Summary

`fluent-concurrency` is a **thin, safe, opinionated harness** over Tokio. It adds the operational primitives that RabbitMQ proved necessary in production (pools, credit flow, supervision, priority) while keeping the Data Plane as fast and flat as `smol`. It follows the Fluent WVR pattern so the orchestrator sees a uniform interface, and it enforces the five architectural pillars of the manifest without unsafe code, bloat, or overengineering.

The crate consumes the trait crate (`fluent-wvr`) rather than redefining it, applies cross-cutting concerns via wrapper newtypes before type erasure, and exposes three runtime backends (`NoopRuntime`, `TokioRuntime`, `TestRuntime`) so the same code paths run in production, in unit tests, and in record-replay simulations. The result is a single coherent vocabulary for "how do I run this work unit" that holds across the guidance crate, the job-copilot server, the LLM queue, and the benchmark harness.
