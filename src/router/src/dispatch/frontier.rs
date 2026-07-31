use std::collections::HashMap;
use std::sync::Arc;

use fluent_concurrency::pool::Limiter;
use serde_json::Value;
use thiserror::Error;

use crate::types::{
    RouterChoice, RouterMessage, RouterMessageContent, RouterRequest, RouterResponse, Usage,
};

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

pub trait DispatchBackend: Send + Sync {
    fn provider_name(&self) -> &str;
    fn build_request(&self, request: &RouterRequest) -> Result<Value, DispatchError>;
    fn parse_response(&self, body: &Value) -> Result<RouterResponse, DispatchError>;
    fn parse_stream_event(&self, event: &[u8]) -> Result<StreamEvent, DispatchError>;
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
}

impl DispatchBackend for OpenAiBackend {
    fn provider_name(&self) -> &str {
        "openai"
    }

    fn build_request(&self, request: &RouterRequest) -> Result<Value, DispatchError> {
        let messages: Vec<Value> = crate::normalize::messages_to_json(request)
            .map_err(|e| DispatchError::RequestBuild(e.to_string()))?;

        let mut body = serde_json::json!({
            "model": request.model,
            "messages": messages,
        });

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

    fn parse_response(&self, body: &Value) -> Result<RouterResponse, DispatchError> {
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

    fn parse_stream_event(&self, event: &[u8]) -> Result<StreamEvent, DispatchError> {
        let text = std::str::from_utf8(event)
            .map_err(|e| DispatchError::StreamParse(format!("invalid UTF-8 in stream: {e}")))?;

        if text == "[DONE]" {
            return Ok(StreamEvent::Done);
        }

        let v: Value =
            serde_json::from_str(text).map_err(|e| DispatchError::StreamParse(e.to_string()))?;
        let empty_choices = vec![];
        let choices = v
            .get("choices")
            .and_then(|v| v.as_array())
            .unwrap_or(&empty_choices);

        if let Some(choice) = choices.first() {
            let delta = choice
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            let finish_reason = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .map(ToString::to_string);
            Ok(StreamEvent::Chunk {
                delta,
                finish_reason,
            })
        } else {
            Err(DispatchError::StreamParse(
                "no choices in stream event".into(),
            ))
        }
    }
}

pub struct AnthropicBackend;

impl DispatchBackend for AnthropicBackend {
    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn build_request(&self, request: &RouterRequest) -> Result<Value, DispatchError> {
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

    fn parse_response(&self, body: &Value) -> Result<RouterResponse, DispatchError> {
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

    fn parse_stream_event(&self, event: &[u8]) -> Result<StreamEvent, DispatchError> {
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

pub struct OpenAiCompatBackend {
    pub api_base: String,
    pub api_key: Option<String>,
    pub provider_label: String,
}

impl OpenAiCompatBackend {
    pub fn new(
        api_base: impl Into<String>,
        api_key: Option<String>,
        provider_label: impl Into<String>,
    ) -> Self {
        Self {
            api_base: api_base.into(),
            api_key,
            provider_label: provider_label.into(),
        }
    }
}

impl DispatchBackend for OpenAiCompatBackend {
    fn provider_name(&self) -> &str {
        &self.provider_label
    }

    fn build_request(&self, request: &RouterRequest) -> Result<Value, DispatchError> {
        let openai = OpenAiBackend::new(&self.api_base, self.api_key.clone());
        openai.build_request(request)
    }

    fn parse_response(&self, body: &Value) -> Result<RouterResponse, DispatchError> {
        let openai = OpenAiBackend::new(&self.api_base, self.api_key.clone());
        openai.parse_response(body)
    }

    fn parse_stream_event(&self, event: &[u8]) -> Result<StreamEvent, DispatchError> {
        let openai = OpenAiBackend::new(&self.api_base, self.api_key.clone());
        openai.parse_stream_event(event)
    }
}

struct ProviderConfig {
    backend: Arc<dyn DispatchBackend>,
    api_key: Option<String>,
}

pub struct LlmDispatcher {
    providers: HashMap<String, ProviderConfig>,
    http_client: reqwest::Client,
    limiter: Limiter,
}

impl LlmDispatcher {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            providers: HashMap::new(),
            http_client: reqwest::Client::new(),
            limiter: Limiter::new(max_concurrent),
        }
    }

    pub fn register_backend(
        &mut self,
        name: impl Into<String>,
        backend: Arc<dyn DispatchBackend>,
        api_key: Option<String>,
    ) {
        self.providers
            .insert(name.into(), ProviderConfig { backend, api_key });
    }

    pub fn get_backend(&self, name: &str) -> Option<&Arc<dyn DispatchBackend>> {
        self.providers.get(name).map(|pc| &pc.backend)
    }

    pub async fn dispatch(
        &self,
        provider: &str,
        _model: &str,
        request: &RouterRequest,
    ) -> Result<RouterResponse, DispatchError> {
        let pc = self
            .providers
            .get(provider)
            .ok_or_else(|| DispatchError::UnsupportedProvider(provider.into()))?;

        let body = pc.backend.build_request(request)?;

        let api_url = match provider {
            "openai" => "https://api.openai.com/v1/chat/completions",
            "anthropic" => "https://api.anthropic.com/v1/messages",
            other => {
                if other.starts_with("http://") || other.starts_with("https://") {
                    other
                } else {
                    return Err(DispatchError::UnsupportedProvider(format!(
                        "no known API URL for provider: {other}"
                    )));
                }
            }
        };

        let api_key = pc.api_key.clone();

        let response = self
            .limiter
            .run(|| async {
                let mut req = self.http_client.post(api_url).json(&body);
                if let Some(ref key) = api_key {
                    req = req.header("Authorization", format!("Bearer {key}"));
                }
                req.send()
                    .await
                    .map_err(|e| DispatchError::Http(e.to_string()))?
                    .json::<Value>()
                    .await
                    .map_err(|e| DispatchError::ResponseParse(e.to_string()))
            })
            .await?;

        pc.backend.parse_response(&response)
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
