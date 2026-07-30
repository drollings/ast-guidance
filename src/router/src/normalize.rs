//! Request and response normalization — OpenAI-compatible JSON → RouterRequest
//! and RouterResponse → OpenAI-compatible JSON.

use serde::Deserialize;
use thiserror::Error;

use crate::types::{RouterMessage, RouterMessageContent, RouterRequest, RouterResponse, Usage};

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

/// Normalize an OpenAI-compatible chat completion request JSON into a RouterRequest.
pub fn normalize_request(body: serde_json::Value) -> Result<RouterRequest, NormalizeError> {
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
            let tool_calls = m
                .tool_calls
                .map(|tc| {
                    tc.into_iter()
                        .map(|v| {
                            serde_json::from_value(v)
                                .map_err(|e| NormalizeError::Parse(e.to_string()))
                        })
                        .collect()
                })
                .transpose()?;
            Ok(RouterMessage {
                role: m.role,
                content,
                tool_calls,
                tool_call_id: m.tool_call_id,
            })
        })
        .collect::<Result<Vec<_>, NormalizeError>>()?;

    let tools = raw
        .tools
        .map(|t| {
            t.into_iter()
                .map(|v| {
                    serde_json::from_value(v).map_err(|e| NormalizeError::Parse(e.to_string()))
                })
                .collect()
        })
        .transpose()?;

    Ok(RouterRequest {
        model: raw.model,
        messages,
        temperature: raw.temperature,
        max_tokens: raw.max_tokens,
        stream: raw.stream,
        tools,
        tool_choice: raw.tool_choice,
        session_id: raw.session_id,
        agent_id: raw.agent_id,
        adapter: raw.adapter,
        metadata: Default::default(),
    })
}

fn normalize_message_content(
    raw: serde_json::Value,
) -> Result<RouterMessageContent, NormalizeError> {
    match raw {
        serde_json::Value::String(s) => Ok(RouterMessageContent::Text(s)),
        serde_json::Value::Array(_) => {
            let parts =
                serde_json::from_value(raw).map_err(|e| NormalizeError::Parse(e.to_string()))?;
            Ok(RouterMessageContent::Parts(parts))
        }
        serde_json::Value::Null => Ok(RouterMessageContent::Text(String::new())),
        _ => Err(NormalizeError::InvalidValue {
            field: "content".into(),
            detail: "content must be a string or array of content parts".into(),
        }),
    }
}

/// Normalize a RouterResponse to OpenAI-compatible chat completion JSON.
pub fn normalize_response(response: &RouterResponse) -> serde_json::Value {
    normalize_response_with_id(response, &response.id)
}

pub fn normalize_response_with_id(response: &RouterResponse, id: &str) -> serde_json::Value {
    let choices: Vec<serde_json::Value> = response
        .choices
        .iter()
        .map(|c| {
            serde_json::json!({
                "index": c.index,
                "message": {
                    "role": c.message.role,
                    "content": match &c.message.content {
                        RouterMessageContent::Text(s) => serde_json::Value::String(s.clone()),
                        RouterMessageContent::Parts(parts) => serde_json::to_value(parts).unwrap_or_default(),
                    }
                },
                "finish_reason": c.finish_reason,
            })
        })
        .collect();

    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": response.created,
        "model": response.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
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

/// Convert RouterRequest messages into Vec<serde_json::Value> for dispatch.
/// Used by both server.rs dispatch_to_llm and dispatch::frontier backends.
pub fn messages_to_json(request: &RouterRequest) -> Vec<serde_json::Value> {
    request
        .messages
        .iter()
        .map(|m| {
            let content = match &m.content {
                RouterMessageContent::Text(s) => serde_json::Value::String(s.clone()),
                RouterMessageContent::Parts(parts) => serde_json::Value::Array(
                    parts
                        .iter()
                        .map(|p| serde_json::to_value(p).unwrap())
                        .collect(),
                ),
            };
            let mut msg = serde_json::json!({"role": m.role, "content": content});
            if let Some(ref tc) = m.tool_calls {
                msg["tool_calls"] = serde_json::to_value(tc).unwrap();
            }
            if let Some(ref id) = m.tool_call_id {
                msg["tool_call_id"] = serde_json::Value::String(id.clone());
            }
            msg
        })
        .collect()
}

/// Build a minimal RouterResponse for pipeline rejections.
pub fn rejection_response(_reason: &str, model: &str) -> RouterResponse {
    RouterResponse {
        id: String::new(),
        object: "chat.completion".into(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: model.into(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RouterResponse;

    #[test]
    fn normalize_simple_text_request() {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        });
        let req = normalize_request(body).unwrap();
        assert_eq!(req.model, "test-model");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content.as_text(), "hello");
    }

    #[test]
    fn normalize_request_with_session_id() {
        let body = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "session_id": "sess-123"
        });
        let req = normalize_request(body).unwrap();
        assert_eq!(req.session_id.as_deref(), Some("sess-123"));
    }

    #[test]
    fn normalize_missing_messages_errors() {
        let body = serde_json::json!({"model": "test"});
        assert!(normalize_request(body).is_err());
    }

    #[test]
    fn normalize_empty_messages_errors() {
        let body = serde_json::json!({
            "model": "test",
            "messages": []
        });
        assert!(normalize_request(body).is_err());
    }

    #[test]
    fn normalize_response_to_openai_format() {
        let response = RouterResponse {
            id: "resp-1".into(),
            object: "chat.completion".into(),
            created: 1000,
            model: "test".into(),
            choices: vec![crate::types::RouterChoice {
                index: 0,
                message: RouterMessage {
                    role: "assistant".into(),
                    content: RouterMessageContent::Text("hi there".into()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 2,
                total_tokens: 12,
            },
        };
        let json = normalize_response(&response);
        assert_eq!(json["id"], "resp-1");
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["content"], "hi there");
        assert_eq!(json["usage"]["total_tokens"], 12);
    }

    #[test]
    fn error_response_format() {
        let json = error_response("bad request", "invalid_request_error");
        assert_eq!(json["error"]["message"], "bad request");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }
}
