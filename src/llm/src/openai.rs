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

// SSE framing lives in the sibling `sse` module (its single owner) and is
// re-exported here so every SSE consumer imports one protocol path.
pub use crate::sse::drain_sse_lines;

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
///
/// Canonical owner of the `{"error":{"message","type"}}` envelope
/// (ROADMAP_20260903_LLM M7). M11 deleted the `common_core::http`
/// byte-identical `error_value` shim (kept through M10) with its parity
/// test; the `error_response` goldens in `tests/openai.rs` are the lasting
/// contract (shape verified identical, no behavior change).
pub fn error_response(message: &str, error_type: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
        }
    })
}
