//! ROADMAP_20260903_LLM M7.3 — OpenAI envelope goldens (moved, not copied).
//!
//! Canonical home for the OpenAI wire-format goldens: request-body builder
//! defaults, stream-delta parsing, normalization, and the error envelope.
//! Converted from a `#[path]`-included unit-test module to an integration
//! `[[test]]` target (M6 `constants` precedent; owner modules carry no test
//! submodule). The `error_response == legacy error_value` parity lock lived
//! here through M10 (`error_response(msg, "invalid_request_error")`
//! byte-identical to the `common_core::http::error_value(msg)` shim — no
//! behavior change, verified); M11 deleted it with the shim.

use fluent_llm::openai::*;
use serde_json::json;

#[test]
fn build_body_has_thinking_default() {
    let body = build_openai_chat_body(
        "m",
        &json!([{"role": "user", "content": "hi"}]),
        None,
        false,
        None,
    );
    assert_eq!(body["model"], "m");
    assert_eq!(body["stream"], false);
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
}

#[test]
fn build_body_merges_params_and_skips_reserved_keys() {
    let params = json!({
        "temperature": 0.2,
        "max_tokens": 64,
        "model": "override",
        "messages": "override",
        "stream": true,
    });
    let body = build_openai_chat_body(
        "m",
        &json!([{"role": "user", "content": "hi"}]),
        Some(&params),
        false,
        None,
    );
    assert_eq!(body["model"], "m");
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["max_tokens"], 64);
    assert_eq!(body["stream"], false);
    assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
}

#[test]
fn build_body_think_override_wins() {
    let body = build_openai_chat_body(
        "m",
        &json!([]),
        Some(&json!({"chat_template_kwargs": {"enable_thinking": true}})),
        false,
        Some(true),
    );
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": true})
    );
}

#[test]
fn build_body_params_can_override_thinking() {
    let body = build_openai_chat_body(
        "m",
        &json!([]),
        Some(&json!({"chat_template_kwargs": {"enable_thinking": true}})),
        false,
        None,
    );
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": true})
    );
}

#[test]
fn build_body_stream_flag() {
    let body = build_openai_chat_body("m", &json!([]), None, true, None);
    assert_eq!(body["stream"], true);
}

// ── stream-delta parser ──────────────────────────────────────────────

#[test]
fn parse_stream_delta_content() {
    let d = parse_openai_stream_delta(
        r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#,
    )
    .unwrap();
    assert_eq!(d.delta, "Hello");
    assert_eq!(d.finish_reason, None);
}

#[test]
fn parse_stream_delta_finish_reason() {
    let d = parse_openai_stream_delta(
        r#"{"choices":[{"delta":{"content":""},"finish_reason":"stop"}]}"#,
    )
    .unwrap();
    assert_eq!(d.delta, "");
    assert_eq!(d.finish_reason.as_deref(), Some("stop"));
}

#[test]
fn parse_stream_delta_model() {
    let d =
        parse_openai_stream_delta(r#"{"model":"gpt-4","choices":[{"delta":{"content":"x"}}]}"#)
            .unwrap();
    assert_eq!(d.model.as_deref(), Some("gpt-4"));
}

#[test]
fn parse_stream_delta_done_sentinel() {
    assert_eq!(parse_openai_stream_delta("[DONE]"), None);
}

#[test]
fn parse_stream_delta_empty_choices_is_none() {
    // No `choices[0]` — nothing to emit, and not the DONE sentinel.
    assert_eq!(parse_openai_stream_delta(r#"{"choices":[]}"#), None);
}

#[test]
fn parse_stream_delta_non_json_is_none() {
    assert_eq!(parse_openai_stream_delta("garbage"), None);
}

// ── normalization ─────────────────────────────────────────────────────

#[test]
fn normalize_simple_text_request() {
    let body = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let req = normalize_request(body).unwrap();
    assert_eq!(req["model"], "test-model");
    assert_eq!(req["messages"][0]["role"], "user");
    assert_eq!(req["messages"][0]["content"], "hello");
}

#[test]
fn normalize_request_with_session_id() {
    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "session_id": "sess-123"
    });
    let req = normalize_request(body).unwrap();
    assert_eq!(req["session_id"], "sess-123");
}

#[test]
fn normalize_missing_messages_errors() {
    let body = json!({"model": "test"});
    assert!(normalize_request(body).is_err());
}

#[test]
fn normalize_empty_messages_errors() {
    let body = json!({"model": "test", "messages": []});
    assert!(normalize_request(body).is_err());
}

#[test]
fn normalize_null_content_becomes_empty_string() {
    let body = json!({
        "model": "test",
        "messages": [{"role": "assistant", "content": null}]
    });
    let req = normalize_request(body).unwrap();
    assert_eq!(req["messages"][0]["content"], "");
}

#[test]
fn normalize_parts_content_preserved() {
    let body = json!({
        "model": "test",
        "messages": [{"role": "assistant", "content": [
            {"type": "text", "text": "a part"},
            {"type": "image_url", "image_url": {"url": "https://example.test/x.png"}}
        ]}]
    });
    let req = normalize_request(body).unwrap();
    assert_eq!(req["messages"][0]["content"][0]["type"], "text");
    assert_eq!(req["messages"][0]["content"][1]["type"], "image_url");
}

#[test]
fn normalize_tools_preserved() {
    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    });
    let req = normalize_request(body).unwrap();
    assert_eq!(req["tools"][0]["function"]["name"], "lookup");
}

#[test]
fn normalize_response_to_openai_format() {
    let response = json!({
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1000,
        "model": "test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi there"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    });
    let out = normalize_response(&response);
    assert_eq!(out["id"], "resp-1");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(out["choices"][0]["message"]["content"], "hi there");
    assert_eq!(out["usage"]["total_tokens"], 12);
}

#[test]
fn normalize_response_missing_usage_defaults_zero() {
    let response = json!({
        "id": "r", "object": "chat.completion", "created": 0, "model": "m",
        "choices": []
    });
    let out = normalize_response(&response);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn error_response_format() {
    let out = error_response("bad request", "invalid_request_error");
    assert_eq!(out["error"]["message"], "bad request");
    assert_eq!(out["error"]["type"], "invalid_request_error");
}

// NOTE (ROADMAP_20260903_LLM M11): the `parity_error_response_eq_legacy_
// error_value` dual-path test died with the `common_core::http::error_value`
// shim it pinned (M7 verified the shape identical, no behavior change).
// The owner `error_response` goldens above are the lasting contract.

#[test]
fn messages_to_json_passthrough() {
    let msgs = json!([
        {"role": "assistant", "content": "hi", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}
    ]);
    let arr = msgs.as_array().unwrap();
    let out = messages_to_json(arr).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["role"], "assistant");
    assert_eq!(out[0]["tool_calls"][0]["function"]["name"], "lookup");
}
