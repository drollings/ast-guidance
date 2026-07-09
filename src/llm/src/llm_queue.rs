use std::sync::Arc;

use fluent_concurrency::pool::{PoolError, WorkerPool};
use fluent_wvr::Runtime;
use tokio::sync::oneshot;

use crate::client::{ChatMessage, LlmConfig, LlmError};

pub struct LlmTask {
    pub messages: Vec<ChatMessage>,
    pub config: LlmConfig,
    pub response_tx: oneshot::Sender<Result<String, LlmError>>,
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

pub struct LlmRequestQueue {
    pool: Arc<WorkerPool<LlmTask>>,
}

impl LlmRequestQueue {
    pub fn new(runtime: Arc<dyn Runtime>, config: &LlmQueueConfig) -> Self {
        let pool = WorkerPool::new(
            runtime,
            config.worker_count,
            config.queue_capacity,
            |task: LlmTask| async move {
                let result = tokio::task::spawn_blocking(move || {
                    make_llm_request(&task.messages, &task.config)
                })
                .await
                .unwrap_or_else(|e| Err(LlmError::Http(e.to_string())));
                let _ = task.response_tx.send(result);
            },
        );
        Self {
            pool: Arc::new(pool),
        }
    }

    pub fn submit(
        &self,
        messages: Vec<ChatMessage>,
        config: LlmConfig,
    ) -> Result<String, LlmError> {
        let (tx, rx) = oneshot::channel();
        let task = LlmTask {
            messages,
            config,
            response_tx: tx,
        };
        let handle = tokio::runtime::Handle::current();
        handle
            .block_on(self.pool.try_submit(task))
            .map_err(|e| match e {
                PoolError::Full => LlmError::Http("queue full".into()),
                PoolError::Closed => LlmError::Http("queue closed".into()),
            })?;
        rx.blocking_recv()
            .map_err(|_| LlmError::Http("queue response canceled".into()))?
    }

    pub async fn submit_async(
        &self,
        messages: Vec<ChatMessage>,
        config: LlmConfig,
    ) -> Result<String, LlmError> {
        let (tx, rx) = oneshot::channel();
        let task = LlmTask {
            messages,
            config,
            response_tx: tx,
        };
        self.pool.try_submit(task).await.map_err(|e| match e {
            PoolError::Full => LlmError::Http("queue full".into()),
            PoolError::Closed => LlmError::Http("queue closed".into()),
        })?;
        rx.await
            .map_err(|_| LlmError::Http("queue response canceled".into()))?
    }
}

fn make_llm_request(messages: &[ChatMessage], config: &LlmConfig) -> Result<String, LlmError> {
    crate::client::chat_complete_http(
        &config.api_url,
        messages,
        &config.model,
        config.think,
        config.debug,
        config.show_prompts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_concurrency::runtime::tokio::TokioRuntime;

    #[test]
    fn test_llm_request_queue_creation() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _guard = rt.enter();
        let queue = LlmRequestQueue::new(Arc::new(TokioRuntime), &LlmQueueConfig::default());
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }];
        let config = LlmConfig::new()
            .api_url("http://localhost:11434/v1".into())
            .model("test".into())
            .build();

        let result = queue.submit(messages, config);
        assert!(result.is_err());
    }
}
