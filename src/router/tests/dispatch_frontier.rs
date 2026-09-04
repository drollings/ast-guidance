use super::*;
use serde_json::json;

fn default_openai() -> OpenAiBackend {
    OpenAiBackend::new("http://test.local", None)
}

#[test]
fn parse_response_missing_id_defaults_to_unknown() {
    let json = json!({"model": "gpt-4", "choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}});
    let resp = default_openai().parse_response(&json).unwrap();
    assert_eq!(resp.id, "unknown", "missing id should default to 'unknown'");
}

#[test]
fn parse_response_missing_model_defaults_to_unknown() {
    let json = json!({"id": "chatcmpl-123", "choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}});
    let resp = default_openai().parse_response(&json).unwrap();
    assert_eq!(
        resp.model, "unknown",
        "missing model should default to 'unknown'"
    );
}

#[test]
fn parse_response_missing_choices_defaults_to_empty() {
    let json = json!({"id": "chatcmpl-123", "model": "gpt-4"});
    let resp = default_openai().parse_response(&json).unwrap();
    assert!(
        resp.choices.is_empty(),
        "missing choices should default to empty vec"
    );
}

#[test]
fn parse_response_missing_usage_defaults_to_zero() {
    let json = json!({"id": "chatcmpl-123", "model": "gpt-4", "choices": []});
    let resp = default_openai().parse_response(&json).unwrap();
    assert_eq!(resp.usage.prompt_tokens, 0);
    assert_eq!(resp.usage.completion_tokens, 0);
    assert_eq!(resp.usage.total_tokens, 0);
}

#[test]
fn parse_response_null_content_becomes_empty_string() {
    let json = json!({
        "id": "chatcmpl-123",
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": null
            }
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    });
    let resp = default_openai().parse_response(&json).unwrap();
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].message.content.to_string_lossy(), "");
}

#[test]
fn parse_response_full_response_succeeds() {
    let json = json!({
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1680000000,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello!"
            }
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    });
    let resp = default_openai().parse_response(&json).unwrap();
    assert_eq!(resp.id, "chatcmpl-abc123");
    assert_eq!(resp.model, "gpt-4");
    assert_eq!(resp.created, 1680000000);
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].message.content.to_string_lossy(), "Hello!");
    assert_eq!(resp.usage.prompt_tokens, 5);
    assert_eq!(resp.usage.completion_tokens, 10);
    assert_eq!(resp.usage.total_tokens, 15);
}

// ─── D9 characterization: `DispatchError::is_retryable` truth table ─────
// Locks the retryable set: transport/rate-limit (`Http`, `RateLimited`) are
// retryable; every construction/rejection variant is permanent. This mirrors
// `fluent_llm::protocol::LlmError::is_retryable` (`Http` |
// `RateLimited`) — the canonical owner is that type (re-exported by
// `fluent_llm`); this domain predicate stays router-side and must keep the
// same table on the shared variants.
#[test]
fn dispatch_error_retryable_table() {
    assert!(DispatchError::Http("boom".into()).is_retryable());
    assert!(DispatchError::RateLimited.is_retryable());
    assert!(!DispatchError::RequestBuild("bad".into()).is_retryable());
    assert!(!DispatchError::ResponseParse("bad".into()).is_retryable());
    assert!(!DispatchError::StreamParse("bad".into()).is_retryable());
    assert!(!DispatchError::UnsupportedProvider("x".into()).is_retryable());
    assert!(
        !DispatchError::InstanceGroupMiss {
            group: "g".into()
        }
        .is_retryable()
    );
    assert!(!DispatchError::AllBackendsFailed.is_retryable());
}
