//! ROADMAP_20260903_LLM M9.3 — protocol-ownership parity (tests first).
//!
//! Locks the `fluent_llm::protocol` contract that M9.1 moves out of
//! `fluent_concurrency::llm_queue`: `bon`-builder defaults (`timeout_ms`
//! 2000, `extra_body_params` merge ignoring `model`/`messages`/`stream`),
//! the four `LlmError` variants (+ `is_retryable`), and a queue smoke via
//! `TestRuntime` (mirroring `tests/llm_queue.rs` + the concurrency queue
//! tests) proving the moved executor still dispatches.
//!
//! The dual-path section that pinned the deprecated
//! `fluent_concurrency::llm_queue` copies died with those shims in M11.
//! `LlmClient` stays the sole constructor surface (M9.2) and
//! `llm_queue::build_default_queue` the one-step wiring — both covered by
//! the existing `tests/llm_queue.rs` + `tests/client.rs` suites.

use fluent_llm::protocol::{
    ChatMessage, LlmConfig, LlmError, LlmQueueConfig, LlmRequestQueue, LlmTask,
};

// ── Builder defaults ────────────────────────────────────────────────────────

#[test]
fn llm_config_builder_defaults() {
    let cfg = LlmConfig::new()
        .api_url("http://localhost:11434/v1".into())
        .model("code".into())
        .build();
    assert_eq!(cfg.api_url, "http://localhost:11434/v1");
    assert_eq!(cfg.model, "code");
    assert_eq!(cfg.think, None);
    assert_eq!(cfg.timeout_ms, 2000);
    assert_eq!(cfg.extra_body_params, None);
    assert!(!cfg.debug);
    assert!(!cfg.show_prompts);
}

#[test]
fn llm_config_builder_overrides_stick() {
    let cfg = LlmConfig::new()
        .api_url("http://x/v1".into())
        .model("m".into())
        .think(true)
        .timeout_ms(50)
        .extra_body_params(serde_json::json!({"temperature": 0.5}))
        .debug(true)
        .show_prompts(true)
        .build();
    assert_eq!(cfg.think, Some(true));
    assert_eq!(cfg.timeout_ms, 50);
    assert_eq!(
        cfg.extra_body_params,
        Some(serde_json::json!({"temperature": 0.5}))
    );
    assert!(cfg.debug);
    assert!(cfg.show_prompts);
}

#[test]
fn llm_queue_config_default() {
    let cfg = LlmQueueConfig::default();
    assert_eq!(cfg.worker_count, 1);
    assert_eq!(cfg.queue_capacity, 100);
}

#[test]
fn chat_message_serde_round_trip() {
    let msg = ChatMessage {
        role: "user".into(),
        content: "hi".into(),
    };
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json, serde_json::json!({"role": "user", "content": "hi"}));
    let parsed: ChatMessage = serde_json::from_value(json).unwrap();
    assert_eq!(parsed.role, "user");
    assert_eq!(parsed.content, "hi");
}

// ── extra_body_params merge semantics ───────────────────────────────────────

#[test]
fn extra_body_params_merge_ignores_reserved_keys() {
    // The merge rule (`model`/`messages`/`stream` set explicitly, everything
    // else merged) lives in `openai::build_openai_chat_body`; the protocol
    // test locks it because `LlmConfig::extra_body_params` is the only
    // carrier of those params.
    let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
    let params = serde_json::json!({
        "model": "spoofed",
        "messages": "spoofed",
        "stream": true,
        "temperature": 0.5,
        "num_ctx": 4096,
    });
    let body =
        fluent_llm::openai::build_openai_chat_body("real", &messages, Some(&params), false, None);
    assert_eq!(body["model"], "real");
    assert_eq!(body["messages"], messages);
    assert_eq!(body["stream"], false);
    assert_eq!(body["temperature"], 0.5);
    assert_eq!(body["num_ctx"], 4096);
}

// ── LlmError variants ───────────────────────────────────────────────────────

#[test]
fn llm_error_variants_and_retryability() {
    assert!(LlmError::Http("boom".into()).is_retryable());
    assert!(LlmError::RateLimited.is_retryable());
    assert!(!LlmError::Api("nope".into()).is_retryable());
    assert!(!LlmError::NoResponse.is_retryable());
    assert_eq!(LlmError::NoResponse, LlmError::NoResponse);
    assert_eq!(
        LlmError::Api("intentional".into()),
        LlmError::Api("intentional".into())
    );
    assert_ne!(
        LlmError::Api("a".into()),
        LlmError::Http("a".into()),
        "variants must stay distinct across the move"
    );
}

// ── Queue smoke via TestRuntime ─────────────────────────────────────────────

#[tokio::test]
async fn queue_submit_dispatches_through_test_runtime() {
    use fluent_concurrency::runtime::test::TestRuntime;
    use std::sync::Arc;

    let handle = tokio::runtime::Handle::current();
    let runtime: Arc<dyn fluent_wvr::Runtime> =
        Arc::new(TestRuntime::new(handle, 0x5eed));
    let queue = Arc::new(LlmRequestQueue::new(
        runtime,
        &LlmQueueConfig {
            worker_count: 1,
            queue_capacity: 10,
        },
        |task: LlmTask| async move {
            Ok(task
                .messages
                .first()
                .map(|m| m.content.clone())
                .unwrap_or_default())
        },
    ));
    assert_eq!(queue.worker_count(), 1);
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
    assert_eq!(queue.submit(task).await, Ok("hello".to_string()));
}

#[tokio::test]
async fn queue_propagates_handler_errors() {
    use fluent_concurrency::runtime::test::TestRuntime;
    use std::sync::Arc;

    let handle = tokio::runtime::Handle::current();
    let runtime: Arc<dyn fluent_wvr::Runtime> =
        Arc::new(TestRuntime::new(handle, 0x5eed));
    let queue = LlmRequestQueue::new(
        runtime,
        &LlmQueueConfig::default(),
        |_task: LlmTask| async move { Err(LlmError::Api("intentional".into())) },
    );
    let task = LlmTask {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }],
        config: LlmConfig::new()
            .api_url("http://localhost:11434/v1".into())
            .model("test".into())
            .build(),
    };
    assert_eq!(
        queue.submit(task).await,
        Err(LlmError::Api("intentional".into()))
    );
}

// NOTE (ROADMAP_20260903_LLM M11): the `parity_new_eq_old` dual-path test
// died with the `fluent_concurrency::llm_queue` shims it pinned. The
// builder/merge/variant/smoke goldens above are the lasting contract.
