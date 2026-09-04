//! Default LLM request handler — wires the queue's `ResultPool` to the
//! OpenAI-compatible HTTP transport in `fluent_llm::client`.
//!
//! The `LlmRequestQueue` primitive lives in `fluent_llm::protocol`
//! and is transport-agnostic (it accepts any `Fn(LlmTask) -> Future<...>`
//! handler). This module provides the handler that actually speaks the
//! OpenAI-compatible chat-completions protocol, so callers can build a
//! ready-to-use queue with a single call:
//!
//! ```no_run
//! use std::sync::Arc;
//! use fluent_llm::protocol::{LlmQueueConfig, LlmRequestQueue};
//! use fluent_concurrency::tokio_runtime;
//! use fluent_llm::llm_queue::default_handler;
//!
//! # async fn run() {
//! let queue = Arc::new(LlmRequestQueue::new(
//!     tokio_runtime(),
//!     &LlmQueueConfig { worker_count: 2, queue_capacity: 100 },
//!     default_handler(),
//! ));
//! # }
//! ```
//!
//! Or use [`build_default_queue`] to construct the queue with the recommended
//! defaults in one step.

use std::sync::Arc;

use crate::protocol::{LlmError, LlmQueueConfig, LlmRequestQueue, LlmTask};
use fluent_wvr::Runtime;

use crate::client::chat_complete_http_async;

/// Returns a closure that, for each [`LlmTask`], calls the OpenAI-compatible
/// `chat_complete_http_async` transport and returns its result.
///
/// The closure is `Send + Sync + 'static` so it satisfies the queue's
/// `ResultPool` handler bound.
pub fn default_handler(
) -> impl Fn(LlmTask) -> futures_compat::BoxFuture<'static, Result<String, LlmError>>
       + Send
       + Sync
       + 'static {
    |task: LlmTask| {
        let api_url = task.config.api_url.clone();
        let model = task.config.model.clone();
        let think = task.config.think;
        let timeout_ms = task.config.timeout_ms;
        let extra_body_params = task.config.extra_body_params;
        let debug = task.config.debug;
        let show_prompts = task.config.show_prompts;
        let messages = task.messages;
        Box::pin(async move {
            chat_complete_http_async(
                &api_url,
                &messages,
                &model,
                think,
                timeout_ms,
                extra_body_params.as_ref(),
                debug,
                show_prompts,
            )
            .await
        })
    }
}

/// Returns a `Send + Sync` future wrapper. Hides the `Pin<Box<dyn Future>>`
/// detail from the public surface.
mod futures_compat {
    use std::future::Future;
    use std::pin::Pin;

    pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
}

/// Builds a ready-to-use [`LlmRequestQueue`] wired to the default HTTP
/// transport. The queue's worker pool has `config.worker_count` workers and a
/// bounded queue of `config.queue_capacity` tasks.
pub fn build_default_queue(
    runtime: Arc<dyn Runtime>,
    config: &LlmQueueConfig,
) -> Arc<LlmRequestQueue> {
    Arc::new(LlmRequestQueue::new(runtime, config, default_handler()))
}

#[cfg(test)]
#[path = "../tests/llm_queue.rs"]
mod tests;
