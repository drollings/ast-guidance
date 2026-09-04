//! LLM protocol + request queue — the single owner (ROADMAP_20260903_LLM M9).
//!
//! Moved verbatim from `fluent_concurrency::llm_queue`: the protocol data
//! types (`ChatMessage`, `LlmConfig` + `bon` builder, `LlmError` +
//! `is_retryable`, `LlmTask`, `LlmQueueConfig` + `Default`) and the
//! `LlmRequestQueue` executor (a thin wrapper over
//! `fluent_concurrency::pool::ResultPool` with the `submit` /
//! `submit_with_abort` signature and the `ResultPoolError → LlmError`
//! mapping). The only cross-crate dependencies are the generic executor
//! (`fluent_concurrency::pool`, `fluent_concurrency::stream::StreamAbort`),
//! the `fluent_wvr::Runtime` abstraction, and externals
//! (`bon`/`serde`/`thiserror`) — never a domain crate.
//!
//! What stays behind: the generic execution machinery — `ResultPool` /
//! `WorkerPool` / `PriorityResultPool`, `StreamAbort`, `TestRuntime`,
//! `first_accept_in_order` — which this module composes and which is not LLM
//! protocol. `LlmClient` (`client.rs`) stays the sole constructor surface
//! and `llm_queue::build_default_queue` the one-step wiring; both now build
//! on these canonical types.
//!
//! M11 deleted the `fluent-concurrency::llm_queue` byte-identical shim
//! copies (kept through M10 under `#[deprecated]`) with the whole module;
//! the owner goldens in `tests/protocol_parity.rs` are the lasting
//! contract. No `Cargo.toml` edge flipped: the queue composes the generic
//! `fluent_concurrency::pool` executor, so the `fluent-llm →
//! fluent-concurrency` edge stands unchanged (defer-by-default, per M9).
//!
//! Calibration (roadmap §1, M10): the `LlmConfig` builder defaults
//! (`timeout_ms` 2000) and the `extra_body_params` merge rule (`model` /
//! `messages` / `stream` set explicitly, everything else merged) are
//! task-value transport wiring, not producer confidence — a fast timeout is
//! never "the model is sure". They move unchanged; retuning them is M10.
//!
//! # Architecture
//!
//! ```text
//!   LlmClient::chat_complete_async
//!            │
//!            ▼
//!   LlmRequestQueue::submit(LlmTask)
//!            │
//!            ▼
//!   ResultPool::submit  ── enqueue ──▶  bounded bounded worker pool
//!                                                  │
//!                                                  ▼
//!                                  user-supplied handler(LlmTask) → Result<String, LlmError>
//! ```
//!
//! The `Runtime` passed at construction controls how the worker tasks are
//! spawned. In production this is `fluent_concurrency::runtime::tokio::TokioRuntime`
//! (returned by `fluent_concurrency::tokio_runtime()`), which delegates to
//! `tokio::spawn`. In tests with paused time, use `TestRuntime`.

use std::future::Future;
use std::sync::Arc;

use bon::Builder;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use fluent_wvr::Runtime;

use fluent_concurrency::pool::{PoolError, ResultPool, ResultPoolError};

// ─── LLM protocol types ─────────────────────────────────────────────────────

/// Errors returned by LLM chat completions and the request queue.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum LlmError {
    #[error("API error: {0}")]
    Api(String),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("no response from model")]
    NoResponse,
    #[error("rate limited")]
    RateLimited,
}

impl LlmError {
    /// Returns `true` for error variants that indicate a transient condition the
    /// caller *may* wish to retry (transport failures, rate limiting). API
    /// rejections and empty responses are treated as permanent.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Http(_) | Self::RateLimited)
    }
}

/// A single chat message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Per-request LLM configuration. The `LlmClient` carries one of these and
/// clones it into every `LlmTask` it submits.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(start_fn = new)]
pub struct LlmConfig {
    pub api_url: String,
    pub model: String,
    pub think: Option<bool>,
    #[builder(default = 2000)]
    pub timeout_ms: u64,
    /// Arbitrary JSON body parameters merged into every chat completion
    /// request (e.g. `num_ctx`, `temperature`, `stop`).  The keys `"model"`,
    /// `"messages"`, and `"stream"` are ignored if present since those are
    /// set explicitly by the chat completion logic.
    pub extra_body_params: Option<serde_json::Value>,
    #[builder(default)]
    pub debug: bool,
    #[builder(default)]
    pub show_prompts: bool,
}

// ─── Queue payload types ────────────────────────────────────────────────────

/// A single unit of LLM work. Bundles the conversation messages and the
/// per-request config that the worker handler needs to dispatch the HTTP
/// call. The handler is free to use any subset of these fields.
pub struct LlmTask {
    pub messages: Vec<ChatMessage>,
    pub config: LlmConfig,
}

/// Bounded-worker-pool tuning for the queue.
#[derive(Debug, Clone, Copy)]
pub struct LlmQueueConfig {
    pub worker_count: usize,
    pub queue_capacity: usize,
}

impl Default for LlmQueueConfig {
    fn default() -> Self {
        Self {
            worker_count: 1,
            queue_capacity: 100,
        }
    }
}

// ─── The queue primitive ────────────────────────────────────────────────────

/// Async, worker-pool-backed request queue for LLM chat completions.
///
/// Backed by [`ResultPool`]. Each submission is handed to the `handler`
/// closure supplied at construction. The closure returns a future that
/// resolves to the LLM response (or an error).
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use fluent_llm::protocol::{LlmRequestQueue, LlmQueueConfig, LlmTask, LlmConfig, ChatMessage};
/// use fluent_concurrency::tokio_runtime;
///
/// # async fn run() {
/// let runtime = tokio_runtime();
/// let config = LlmQueueConfig { worker_count: 2, queue_capacity: 100 };
/// let queue: Arc<LlmRequestQueue> = Arc::new(LlmRequestQueue::new(
///     runtime,
///     &config,
///     |task: LlmTask| async move {
///         // Call the LLM HTTP endpoint and return its content.
///         Ok::<String, fluent_llm::protocol::LlmError>(String::new())
///     },
/// ));
///
/// let task = LlmTask {
///     messages: vec![ChatMessage { role: "user".into(), content: "hi".into() }],
///     config: LlmConfig::new().api_url("http://localhost:11434/v1".into()).model("code".into()).build(),
/// };
/// let _ = queue.submit(task).await;
/// # }
/// ```
pub struct LlmRequestQueue {
    pool: Arc<ResultPool<LlmTask, String, LlmError>>,
}

impl LlmRequestQueue {
    /// Creates a new queue with `config.worker_count` workers and a queue
    /// capacity of `config.queue_capacity`. The `handler` closure is invoked
    /// for each submitted task; its return value is the LLM response.
    pub fn new<F, Fut>(runtime: Arc<dyn Runtime>, config: &LlmQueueConfig, handler: F) -> Self
    where
        F: Fn(LlmTask) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, LlmError>> + Send,
    {
        let pool = ResultPool::new(runtime, config.worker_count, config.queue_capacity, handler);
        Self {
            pool: Arc::new(pool),
        }
    }

    /// Returns the configured worker count.
    pub fn worker_count(&self) -> usize {
        self.pool.worker_count()
    }

    /// Submits a task and awaits the handler's result. Returns `Err` if the
    /// queue is full or closed, or if the handler returns an error.
    pub async fn submit(&self, task: LlmTask) -> Result<String, LlmError> {
        self.pool.submit(task).await.map_err(map_pool_error)
    }

    /// Submits a task with an abort signal and awaits the handler's result.
    /// When the signal fires mid-flight the handler future is dropped — an
    /// in-flight HTTP call is cancelled and the worker slot is freed — and the
    /// call resolves to `LlmError::Http("queue response canceled")`.
    pub async fn submit_with_abort(
        &self,
        task: LlmTask,
        abort: fluent_concurrency::stream::StreamAbort,
    ) -> Result<String, LlmError> {
        self.pool
            .submit_with_abort(task, Some(abort))
            .await
            .map_err(map_pool_error)
    }
}

/// Maps a `ResultPoolError` to the appropriate `LlmError` variant.
fn map_pool_error(e: ResultPoolError<LlmError>) -> LlmError {
    match e {
        ResultPoolError::Pool(PoolError::Full) => LlmError::Http("queue full".into()),
        ResultPoolError::Pool(PoolError::Closed) => LlmError::Http("queue closed".into()),
        ResultPoolError::Inner(e) => e,
        ResultPoolError::Canceled => LlmError::Http("queue response canceled".into()),
    }
}
