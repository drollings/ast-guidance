//! OpenAI-compatible chat-completion wire format.
//!
//! Canonical home for the protocol-boundary helpers that used to live (or were
//! hand-rolled) in the router:
//!
//! - the request-body builder that always carries the AGENTS.md-mandated
//!   `chat_template_kwargs: {"enable_thinking": false}` default,
//! - the stream-delta parser shared by every SSE consumer,
//! - the OpenAI-format request/response normalization helpers.
//!
//! Everything here is parameterized on plain `serde_json::Value` (or the LLM
//! crate's own `ChatMessage`) — no router types leak in (R6).

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Request-body builder
// ---------------------------------------------------------------------------

/// Build an OpenAI-compatible chat completion request body.
///
/// Always includes `"chat_template_kwargs": {"enable_thinking": false}` as a
/// default so the model does not emit `<think>...</think>` blocks in its
/// response (AGENTS.md `filter_thinking` contract, request side). `params`
/// (model-level `extra_body_params`) are merged afterwards and may override
/// any key except `model`/`messages`/`stream`, which are set explicitly.
/// `think_override` is the final say: `Some(true)` forces thinking on
/// regardless of the default and `params`.
///
/// `messages` is the pre-serialized `"messages"` array — feed it either
/// `serde_json::to_value(&[ChatMessage])` or the router's normalized message
/// values so the builder never depends on a consumer message type.
pub fn build_openai_chat_body(
    model: &str,
    messages: &Value,
    params: Option<&Value>,
    stream: bool,
    think_override: Option<bool>,
) -> Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": false},
    });

    // Merge model-level params (can override defaults above, e.g. when
    // `params` contains `chat_template_kwargs`).
    if let Some(params) = params {
        if let Some(obj) = params.as_object() {
            for (k, v) in obj {
                if k != "model" && k != "messages" && k != "stream" {
                    body[k] = v.clone();
                }
            }
        }
    }
    // think override from the caller wins over everything.
    if think_override == Some(true) {
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": true});
    }

    body
}

// ---------------------------------------------------------------------------
// Stream-delta parser
// ---------------------------------------------------------------------------

/// One parsed OpenAI chat-completion stream delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiDelta {
    /// Top-level `model` field of the chunk, if present.
    pub model: Option<String>,
    /// `choices[0].delta.content` (empty when the chunk carries only a
    /// `finish_reason`).
    pub delta: String,
    /// `choices[0].finish_reason` (`stop`, `length`, …).
    pub finish_reason: Option<String>,
}

/// Parse one SSE data payload (the text after `data: `) into an
/// [`OpenAiDelta`]. Returns `None` for the `[DONE]` end-of-stream sentinel.
///
/// A chunk with empty or absent `choices` yields `Some(OpenAiDelta)` with an
/// empty `delta` — the caller decides whether to emit anything.
pub fn parse_openai_stream_delta(line: &str) -> Option<OpenAiDelta> {
    if line == "[DONE]" {
        return None;
    }

    let v: Value = serde_json::from_str(line).ok()?;
    let model = v
        .get("model")
        .and_then(Value::as_str)
        .map(ToString::to_string);

    let empty_choices = vec![];
    let choices = v
        .get("choices")
        .and_then(Value::as_array)
        .unwrap_or(&empty_choices);
    let choice = choices.first()?;

    let delta = choice
        .get("delta")
        .and_then(|d| d.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(ToString::to_string);

    Some(OpenAiDelta {
        model,
        delta,
        finish_reason,
    })
}

// ---------------------------------------------------------------------------
// OpenAI-format normalization
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum NormalizeError {
    #[error("missing required field: {0}")]
    MissingField(String),
    #[error("invalid value for field '{field}': {detail}")]
    InvalidValue { field: String, detail: String },
    #[error("JSON parse error: {0}")]
    Parse(String),
}

/// OpenAI-compatible chat completion request shape (inbound normalization).
#[derive(Debug, Deserialize)]
struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub adapter: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// Validate and normalize an OpenAI-compatible chat completion request body
/// into a canonical object with the fields:
/// `model`, `messages`, `temperature`, `max_tokens`, `stream`, `tools`,
/// `tool_choice`, `session_id`, `agent_id`, `adapter`.
///
/// Consumer request types deserialize this object (e.g. the router's
/// `RouterRequest`), so no consumer type leaks into this crate (R6).
pub fn normalize_request(body: serde_json::Value) -> Result<serde_json::Value, NormalizeError> {
    let raw: OpenAiChatRequest =
        serde_json::from_value(body).map_err(|e| NormalizeError::Parse(e.to_string()))?;

    if raw.messages.is_empty() {
        return Err(NormalizeError::MissingField("messages".into()));
    }

    let messages = raw
        .messages
        .into_iter()
        .map(|m| {
            let content = normalize_message_content(m.content)?;
            let mut msg = serde_json::json!({"role": m.role, "content": content});
            if let Some(tc) = m.tool_calls {
                msg["tool_calls"] =
                    serde_json::to_value(tc).map_err(|e| NormalizeError::Parse(e.to_string()))?;
            }
            if let Some(id) = m.tool_call_id {
                msg["tool_call_id"] = Value::String(id);
            }
            Ok(msg)
        })
        .collect::<Result<Vec<_>, NormalizeError>>()?;

    let tools = raw
        .tools
        .map(|t| serde_json::to_value(t).map_err(|e| NormalizeError::Parse(e.to_string())))
        .transpose()?;

    Ok(serde_json::json!({
        "model": raw.model,
        "messages": messages,
        "temperature": raw.temperature,
        "max_tokens": raw.max_tokens,
        "stream": raw.stream,
        "tools": tools,
        "tool_choice": raw.tool_choice,
        "session_id": raw.session_id,
        "agent_id": raw.agent_id,
        "adapter": raw.adapter,
    }))
}

fn normalize_message_content(raw: serde_json::Value) -> Result<serde_json::Value, NormalizeError> {
    match raw {
        serde_json::Value::String(_) | serde_json::Value::Array(_) => Ok(raw),
        serde_json::Value::Null => Ok(Value::String(String::new())),
        _ => Err(NormalizeError::InvalidValue {
            field: "content".into(),
            detail: "content must be a string or array of content parts".into(),
        }),
    }
}

/// Convert pre-serialized consumer messages (in `{role, content, tool_calls?,
/// tool_call_id?}` form) into canonical OpenAI chat-completion message objects.
///
/// This is the shared pass-through for the router's `messages_to_json`; the
/// router serializes its `RouterMessage`s to `Value` first, so the shared
/// helper never sees a router type (R6).
pub fn messages_to_json(
    messages: &[serde_json::Value],
) -> Result<Vec<serde_json::Value>, NormalizeError> {
    Ok(messages.to_vec())
}

/// Normalize a chat-completion response (a `Value` shaped like the consumer's
/// response type: `id`, `object`, `created`, `model`, `choices`, `usage`) into
/// OpenAI-compatible chat completion JSON.
pub fn normalize_response(response: &serde_json::Value) -> serde_json::Value {
    let id = response.get("id").and_then(Value::as_str).unwrap_or("");
    normalize_response_with_id(response, id)
}

pub fn normalize_response_with_id(response: &serde_json::Value, id: &str) -> serde_json::Value {
    let choices: Vec<serde_json::Value> = response
        .get("choices")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .map(|c| {
                    let message = c.get("message").cloned().unwrap_or(Value::Null);
                    serde_json::json!({
                        "index": c.get("index").and_then(Value::as_u64).unwrap_or(0),
                        "message": {
                            "role": message.get("role").and_then(Value::as_str).unwrap_or("assistant"),
                            "content": message.get("content").cloned().unwrap_or(Value::String(String::new())),
                        },
                        "finish_reason": c.get("finish_reason").and_then(Value::as_str).unwrap_or("stop"),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let usage = response.get("usage").cloned().unwrap_or(Value::Null);
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": response.get("created").and_then(Value::as_u64).unwrap_or(0),
        "model": response.get("model").and_then(Value::as_str).unwrap_or(""),
        "choices": choices,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            "completion_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
            "total_tokens": usage.get("total_tokens").and_then(Value::as_u64).unwrap_or(0),
        },
    })
}

/// Build an error response in OpenAI-compatible format.
pub fn error_response(message: &str, error_type: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
