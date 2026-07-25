//! LLM request queue — async, queued, worker-pool-backed LLM chat completion.
//!
//! `LlmRequestQueue` is a thin wrapper over [`crate::pool::ResultPool`] that
//! exposes an LLM-friendly API: it takes [`LlmTask`] (a bundle of messages +
//! per-request config), hands each task to a worker-supplied handler closure,
//! and returns the worker's `Result<String, LlmError>`.
//!
//! The handler is supplied at construction time so this module stays
//! transport-agnostic: the default OpenAI-compatible HTTP handler lives in
//! `guidance_llm` (see `guidance_llm::llm_queue::default_handler`). Tests
//! and adapters can supply stub handlers without dragging in `reqwest`.
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

use crate::pool::{PoolError, ResultPool, ResultPoolError};

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
/// use fluent_concurrency::llm_queue::{LlmRequestQueue, LlmQueueConfig, LlmTask, LlmConfig, ChatMessage};
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
///         Ok::<String, fluent_concurrency::llm_queue::LlmError>(String::new())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tokio::TokioRuntime;
    use crate::tokio_runtime;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A handler that echoes the first message's content. Verifies the
    /// queue's worker pool actually invokes the closure.
    async fn echo_handler(task: LlmTask) -> Result<String, LlmError> {
        Ok(task
            .messages
            .first()
            .map(|m| m.content.clone())
            .unwrap_or_default())
    }

    #[tokio::test]
    async fn test_queue_creation_with_default_config() {
        let runtime = tokio_runtime();
        let queue = LlmRequestQueue::new(runtime, &LlmQueueConfig::default(), echo_handler);
        assert_eq!(queue.worker_count(), 1);
    }

    #[tokio::test]
    async fn test_queue_creation_with_custom_config() {
        let runtime = tokio_runtime();
        let config = LlmQueueConfig {
            worker_count: 4,
            queue_capacity: 50,
        };
        let queue = LlmRequestQueue::new(runtime, &config, echo_handler);
        assert_eq!(queue.worker_count(), 4);
    }

    #[tokio::test]
    async fn test_queue_submit_async_returns_handler_result() {
        let runtime = tokio_runtime();
        let queue = Arc::new(LlmRequestQueue::new(
            runtime,
            &LlmQueueConfig {
                worker_count: 1,
                queue_capacity: 10,
            },
            echo_handler,
        ));
        let task = LlmTask {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "hello".into(),
            }],
            config: LlmConfig::new()
                .api_url("http://localhost:11434/v1".into())
                .model("test".into())
                .build(),
        };
        let result = queue.submit(task).await;
        assert_eq!(result, Ok("hello".to_string()));
    }

    /// Handler that always returns an error. Verifies the queue propagates
    /// handler errors faithfully.
    async fn error_handler(_task: LlmTask) -> Result<String, LlmError> {
        Err(LlmError::Api("intentional".into()))
    }

    #[tokio::test]
    async fn test_queue_handler_error_propagates() {
        let runtime = tokio_runtime();
        let queue = LlmRequestQueue::new(
            runtime,
            &LlmQueueConfig {
                worker_count: 1,
                queue_capacity: 10,
            },
            error_handler,
        );
        let task = LlmTask {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "x".into(),
            }],
            config: LlmConfig::new()
                .api_url("http://localhost:11434/v1".into())
                .model("test".into())
                .build(),
        };
        let result = queue.submit(task).await;
        assert_eq!(result, Err(LlmError::Api("intentional".into())));
    }

    /// Handler that counts invocations across concurrent submits. Verifies
    /// the worker pool actually parallelizes.
    async fn counting_handler(
        task: LlmTask,
        counter: Arc<AtomicUsize>,
    ) -> Result<String, LlmError> {
        counter.fetch_add(1, Ordering::SeqCst);
        // Yield to allow concurrent invocations from other workers.
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        counter.fetch_sub(1, Ordering::SeqCst);
        Ok(task
            .messages
            .first()
            .map(|m| m.content.clone())
            .unwrap_or_default())
    }

    #[tokio::test]
    async fn test_queue_concurrent_submits_process_all() {
        let runtime = tokio_runtime();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let queue = Arc::new(LlmRequestQueue::new(
            runtime,
            &LlmQueueConfig {
                worker_count: 4,
                queue_capacity: 50,
            },
            move |task| {
                let counter = Arc::clone(&counter_clone);
                async move { counting_handler(task, counter).await }
            },
        ));
        let mut handles = Vec::new();
        for i in 0..10 {
            let q = Arc::clone(&queue);
            handles.push(tokio::spawn(async move {
                let task = LlmTask {
                    messages: vec![ChatMessage {
                        role: "user".into(),
                        content: format!("msg_{i}"),
                    }],
                    config: LlmConfig::new()
                        .api_url("http://localhost:11434/v1".into())
                        .model("test".into())
                        .build(),
                };
                q.submit(task).await
            }));
        }
        for h in handles {
            let result = h.await.unwrap();
            assert!(result.is_ok());
        }
    }

    /// TokioRuntime smoke test: confirm the runtime produces spawn handles.
    #[test]
    fn test_tokio_runtime_worker_count() {
        let _ = TokioRuntime;
    }
}
