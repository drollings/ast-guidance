use std::sync::Arc;

use fluent_concurrency::pool::{ResultPool, ResultPoolError};
use fluent_wvr::Runtime;

use crate::client::{chat_complete_http_async, ChatMessage, LlmConfig, LlmError};

pub struct LlmTask {
    pub messages: Vec<ChatMessage>,
    pub config: LlmConfig,
}

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

/// Async request queue backed by a `fluent_concurrency::ResultPool`.
///
/// The worker handler calls `chat_complete_http_async` and returns the
/// result directly — no `oneshot::channel` boilerplate needed.
pub struct LlmRequestQueue {
    pool: Arc<ResultPool<LlmTask, String, LlmError>>,
}

impl LlmRequestQueue {
    pub fn new(runtime: Arc<dyn Runtime>, config: &LlmQueueConfig) -> Self {
        let pool = ResultPool::new(
            runtime,
            config.worker_count,
            config.queue_capacity,
            |task: LlmTask| async move {
                chat_complete_http_async(
                    &task.config.api_url,
                    &task.messages,
                    &task.config.model,
                    task.config.think,
                    task.config.timeout_ms,
                    task.config.debug,
                    task.config.show_prompts,
                )
                .await
            },
        );
        Self {
            pool: Arc::new(pool),
        }
    }

    /// Submit a request and await the result. Returns `Err` if the queue is
    /// full or closed, or if the handler fails.
    pub async fn submit_async(
        &self,
        messages: Vec<ChatMessage>,
        config: LlmConfig,
    ) -> Result<String, LlmError> {
        let task = LlmTask { messages, config };
        self.pool.submit(task).await.map_err(|e| match e {
            ResultPoolError::Pool(fluent_concurrency::pool::PoolError::Full) => {
                LlmError::Http("queue full".into())
            }
            ResultPoolError::Pool(fluent_concurrency::pool::PoolError::Closed) => {
                LlmError::Http("queue closed".into())
            }
            ResultPoolError::Inner(e) => e,
            ResultPoolError::Canceled => LlmError::Http("queue response canceled".into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_concurrency::tokio_runtime;

    #[tokio::test]
    async fn test_queue_submit_async_returns_err_when_no_server() {
        let runtime = tokio_runtime();
        let queue = LlmRequestQueue::new(runtime, &LlmQueueConfig::default());
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }];
        let config = LlmConfig::new()
            .api_url("http://localhost:11434/v1".into())
            .model("test".into())
            // Short timeout so the test fails fast (default is 2s, which is
            // already short, but make it explicit).
            .timeout_ms(500)
            .build();

        let result = queue.submit_async(messages, config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_queue_creation_with_default_config() {
        let runtime = tokio_runtime();
        let _queue = LlmRequestQueue::new(runtime, &LlmQueueConfig::default());
        // Just verify construction succeeds without panicking.
    }

    #[tokio::test]
    async fn test_queue_creation_with_custom_config() {
        let runtime = tokio_runtime();
        let config = LlmQueueConfig {
            worker_count: 4,
            queue_capacity: 50,
        };
        let _queue = LlmRequestQueue::new(runtime, &config);
    }
}
