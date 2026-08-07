use serde_json::Value;
use thiserror::Error;

use crate::types::{
    RouterChoice, RouterMessage, RouterMessageContent, RouterRequest, RouterResponse, Usage,
};

// Provider request/response builders, reserved for the frontier escalation
// ladder (forward track.  The production dispatch path is `ChatBackend` in
// `dispatch::backend`; this module only owns the wire-format build/parse
// logic for the OpenAI- and Anthropic-style Messages APIs that the ladder
// will compose.

impl DispatchError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, DispatchError::Http(_) | DispatchError::RateLimited)
    }
}

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error("unsupported provider: {0}")]
    UnsupportedProvider(String),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("request build error: {0}")]
    RequestBuild(String),
    #[error("response parse error: {0}")]
    ResponseParse(String),
    #[error("stream parse error: {0}")]
    StreamParse(String),
    #[error("rate limited")]
    RateLimited,
    #[error("no free instance in group: {group}")]
    InstanceGroupMiss { group: String },
    #[error("all backends failed")]
    AllBackendsFailed,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Chunk {
        delta: String,
        finish_reason: Option<String>,
    },
    Done,
    Error(String),
}

pub struct OpenAiBackend {
    pub api_base: String,
    pub api_key: Option<String>,
}

impl OpenAiBackend {
    pub fn new(api_base: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            api_base: api_base.into(),
            api_key,
        }
    }

    pub fn build_request(&self, request: &RouterRequest) -> Result<Value, DispatchError> {
        let messages: Vec<Value> = crate::normalize::messages_to_json(request)
            .map_err(|e| DispatchError::RequestBuild(e.to_string()))?;

        // Canonical body builder (fluent_llm::openai) supplies the
        // AGENTS.md-mandated `chat_template_kwargs: {"enable_thinking": false}`
        // default. The escalation ladder has no `LlmConfig.think` equivalent,
        // so `think_override` is `None`; router request fields are applied
        // after the canonical merge.
        let mut body = fluent_llm::openai::build_openai_chat_body(
            &request.model,
            &Value::Array(messages),
            None,
            false,
            None,
        );

        if let Some(temp) = request.temperature {
            body["temperature"] = Value::Number(serde_json::Number::from_f64(temp).unwrap());
        }
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = Value::Number(serde_json::Number::from(max_tokens));
        }
        if let Some(ref tools) = request.tools {
            body["tools"] = serde_json::to_value(tools).unwrap();
        }
        if let Some(ref tool_choice) = request.tool_choice {
            body["tool_choice"] = Value::String(tool_choice.clone());
        }

        Ok(body)
    }

    pub fn parse_response(&self, body: &Value) -> Result<RouterResponse, DispatchError> {
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let created = body.get("created").and_then(Value::as_u64).unwrap_or(0);

        let choices = body
            .get("choices")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|c| {
                        let index = c.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                        let finish_reason = c
                            .get("finish_reason")
                            .and_then(Value::as_str)
                            .unwrap_or("stop")
                            .to_string();
                        let msg = c.get("message")?;
                        let role = msg
                            .get("role")
                            .and_then(Value::as_str)
                            .unwrap_or("assistant")
                            .to_string();
                        let content = msg
                            .get("content")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        Some(RouterChoice {
                            index,
                            message: RouterMessage {
                                role,
                                content: RouterMessageContent::Text(content),
                                tool_calls: None,
                                tool_call_id: None,
                            },
                            finish_reason,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let usage = body.get("usage").map_or(
            Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            |u| Usage {
                prompt_tokens: u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
                completion_tokens: u
                    .get("completion_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u32,
                total_tokens: u.get("total_tokens").and_then(Value::as_u64).unwrap_or(0) as u32,
            },
        );

        Ok(RouterResponse {
            id,
            object: "chat.completion".into(),
            created,
            model,
            choices,
            usage,
        })
    }

    pub fn parse_stream_event(&self, event: &[u8]) -> Result<StreamEvent, DispatchError> {
        let text = std::str::from_utf8(event)
            .map_err(|e| DispatchError::StreamParse(format!("invalid UTF-8 in stream: {e}")))?;

        if text == "[DONE]" {
            return Ok(StreamEvent::Done);
        }

        // Shared OpenAI stream-delta parser (fluent_llm::openai). `None`
        // here means the payload was unparseable or choice-less — the ladder
        // treats that as a stream-parse error (matching the previous local
        // parser's behavior).
        match crate::normalize::parse_openai_stream_delta(text) {
            Some(delta) => Ok(StreamEvent::Chunk {
                delta: delta.delta,
                finish_reason: delta.finish_reason,
            }),
            None => Err(DispatchError::StreamParse(
                "no choices in stream event".into(),
            )),
        }
    }
}

/// Anthropic Messages-API builders, reserved for the frontier escalation
/// ladder
pub struct Anthropic;

impl Anthropic {
    pub fn build_request(request: &RouterRequest) -> Result<Value, DispatchError> {
        let mut system_parts: Vec<String> = Vec::new();
        let mut messages: Vec<Value> = Vec::new();

        for msg in &request.messages {
            match msg.role.as_str() {
                "system" => {
                    if let RouterMessageContent::Text(s) = &msg.content {
                        system_parts.push(s.clone());
                    }
                }
                role => {
                    let content = match &msg.content {
                        RouterMessageContent::Text(s) => Value::String(s.clone()),
                        RouterMessageContent::Parts(parts) => Value::Array(
                            parts
                                .iter()
                                .map(|p| serde_json::to_value(p).unwrap())
                                .collect(),
                        ),
                    };
                    let mut m = serde_json::json!({ "role": role, "content": content });
                    if let Some(ref id) = msg.tool_call_id {
                        m["tool_call_id"] = Value::String(id.clone());
                    }
                    messages.push(m);
                }
            }
        }

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(4096),
        });

        if !system_parts.is_empty() {
            body["system"] = Value::String(system_parts.join("\n"));
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = Value::Number(serde_json::Number::from_f64(temp).unwrap());
        }

        Ok(body)
    }

    pub fn parse_response(body: &Value) -> Result<RouterResponse, DispatchError> {
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let created = body.get("created").and_then(Value::as_u64).unwrap_or(0);

        let content = body
            .get("content")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(|block| block.get("text"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        let input_tokens = body
            .get("usage")
            .and_then(|u| u.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let output_tokens = body
            .get("usage")
            .and_then(|u| u.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;

        Ok(RouterResponse {
            id,
            object: "chat.completion".into(),
            created,
            model,
            choices: vec![RouterChoice {
                index: 0,
                message: RouterMessage {
                    role: "assistant".into(),
                    content: RouterMessageContent::Text(content),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens: input_tokens,
                completion_tokens: output_tokens,
                total_tokens: input_tokens + output_tokens,
            },
        })
    }

    pub fn parse_stream_event(event: &[u8]) -> Result<StreamEvent, DispatchError> {
        let text = std::str::from_utf8(event)
            .map_err(|e| DispatchError::StreamParse(format!("invalid UTF-8 in stream: {e}")))?;

        if text.starts_with("event: message_stop") || text.contains("[DONE]") {
            return Ok(StreamEvent::Done);
        }

        if let Some(data) = text.strip_prefix("data: ") {
            let v: Value = serde_json::from_str(data)
                .map_err(|e| DispatchError::StreamParse(e.to_string()))?;
            if let Some(delta) = v
                .get("delta")
                .and_then(|d| d.get("text"))
                .and_then(|t| t.as_str())
            {
                Ok(StreamEvent::Chunk {
                    delta: delta.to_string(),
                    finish_reason: None,
                })
            } else {
                Ok(StreamEvent::Chunk {
                    delta: String::new(),
                    finish_reason: Some("stop".into()),
                })
            }
        } else {
            Err(DispatchError::StreamParse("unexpected SSE format".into()))
        }
    }
}

#[cfg(test)]
mod tests {
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
}
