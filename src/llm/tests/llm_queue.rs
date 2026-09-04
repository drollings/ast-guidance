use super::*;
use crate::protocol::{ChatMessage, LlmConfig};
use fluent_concurrency::tokio_runtime;

#[tokio::test]
async fn test_default_handler_routes_to_http() {
    // Smoke test: ensure the handler compiles and returns Err when the
    // LLM endpoint is unreachable. We don't make any actual HTTP call;
    // we just verify the handler's future type matches the queue's bound.
    let runtime = tokio_runtime();
    let queue = build_default_queue(
        runtime,
        &LlmQueueConfig {
            worker_count: 1,
            queue_capacity: 10,
        },
    );
    let task = LlmTask {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }],
        config: LlmConfig::new()
            .api_url("http://127.0.0.1:1/v1".into())
            .model("test".into())
            .timeout_ms(50)
            .build(),
    };
    let result = queue.submit(task).await;
    assert!(
        result.is_err(),
        "unreachable endpoint must surface an error from the queue"
    );
}
