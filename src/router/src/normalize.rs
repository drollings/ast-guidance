//! Request and response normalization — OpenAI-compatible JSON → RouterRequest
//! and RouterResponse → OpenAI-compatible JSON.
//!
//! Thin adapter over the canonical protocol-boundary helpers in
//! `fluent_llm::openai`. The shared helpers operate on `serde_json::Value`;
//! this module adapts them to the router's typed `RouterRequest`/`RouterResponse`.

use crate::types::{RouterRequest, RouterResponse, Usage};

pub use fluent_llm::openai::NormalizeError;

/// Normalize an OpenAI-compatible chat completion request JSON into a RouterRequest.
///
/// The shared `fluent_llm` normalizer strips non-OpenAI fields; the routing
/// fields the owning llama-server reads (`instance`/`snapshot`/`id_slot`) are
/// re-attached from the original body so they survive into the dispatch target.
pub fn normalize_request(body: serde_json::Value) -> Result<RouterRequest, NormalizeError> {
    let routing: Vec<(String, serde_json::Value)> = ["instance", "snapshot", "id_slot"]
        .iter()
        .filter_map(|key| body.get(key).map(|v| ((*key).to_string(), v.clone())))
        .collect();
    let mut value = fluent_llm::openai::normalize_request(body)?;
    if let serde_json::Value::Object(ref mut obj) = value {
        for (key, value) in routing {
            obj.insert(key, value);
        }
    }
    serde_json::from_value(value).map_err(|e| NormalizeError::Parse(e.to_string()))
}

/// Normalize a RouterResponse to OpenAI-compatible chat completion JSON.
pub fn normalize_response(response: &RouterResponse) -> serde_json::Value {
    let value = serde_json::to_value(response).unwrap_or(serde_json::Value::Null);
    fluent_llm::openai::normalize_response(&value)
}

pub fn normalize_response_with_id(response: &RouterResponse, id: &str) -> serde_json::Value {
    let value = serde_json::to_value(response).unwrap_or(serde_json::Value::Null);
    fluent_llm::openai::normalize_response_with_id(&value, id)
}

/// Build an error response in OpenAI-compatible format.
pub fn error_response(message: &str, error_type: &str) -> serde_json::Value {
    fluent_llm::openai::error_response(message, error_type)
}

/// Convert RouterRequest messages into Vec<serde_json::Value> for dispatch.
/// Used by both server.rs dispatch_to_llm and dispatch::frontier backends.
pub fn messages_to_json(request: &RouterRequest) -> Result<Vec<serde_json::Value>, NormalizeError> {
    let messages = serde_json::to_value(&request.messages)
        .map_err(|e| NormalizeError::Parse(e.to_string()))?;
    let arr = messages.as_array().cloned().unwrap_or_default();
    fluent_llm::openai::messages_to_json(&arr)
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

// Keep the shared error-free helpers importable through this module for the
// small set of callers that reach into `normalize` for the wire helpers.
pub use fluent_llm::openai::{parse_openai_stream_delta, OpenAiDelta};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ContentPart, FunctionCall, ImageUrl, RouterChoice, RouterMessage, RouterMessageContent,
        ToolCall,
    };

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
    fn normalize_request_preserves_routing_fields() {
        // The routing fields the owning llama-server reads survive the
        // normalizer (the shared normalizer strips non-OpenAI keys).
        let body = serde_json::json!({
            "model": "swarm:ledger",
            "messages": [{"role": "user", "content": "hi"}],
            "instance": "scratch",
            "snapshot": "readfiles",
            "id_slot": 3,
        });
        let req = normalize_request(body).unwrap();
        assert_eq!(req.model, "swarm:ledger");
        assert_eq!(req.instance.as_deref(), Some("scratch"));
        assert_eq!(req.snapshot.as_deref(), Some("readfiles"));
        assert_eq!(req.id_slot, Some(3));
    }

    #[test]
    fn normalize_request_absent_routing_fields_stay_none() {
        let body = serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = normalize_request(body).unwrap();
        assert!(req.instance.is_none());
        assert!(req.snapshot.is_none());
        assert!(req.id_slot.is_none());
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
            choices: vec![RouterChoice {
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

    #[test]
    fn messages_to_json_with_parts_and_tool_calls() {
        let request = RouterRequest {
            model: "test".into(),
            messages: vec![RouterMessage {
                role: "assistant".into(),
                content: RouterMessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "a part".into(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.test/x.png".into(),
                        },
                    },
                ]),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "lookup".into(),
                        arguments: "{}".into(),
                    },
                }]),
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            session_id: None,
            agent_id: None,
            adapter: None,
            instance: None,
            snapshot: None,
            id_slot: None,
            metadata: Default::default(),
        };
        let json = messages_to_json(&request).expect("messages serialize");
        assert_eq!(json.len(), 1);
        assert_eq!(json[0]["role"], "assistant");
        assert_eq!(json[0]["content"][0]["type"], "text");
        assert_eq!(json[0]["content"][1]["type"], "image_url");
        assert_eq!(json[0]["tool_calls"][0]["function"]["name"], "lookup");
    }

    #[test]
    fn messages_to_json_text_roundtrip() {
        let request = RouterRequest {
            model: "test".into(),
            messages: vec![RouterMessage {
                role: "user".into(),
                content: RouterMessageContent::Text("hello".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            session_id: None,
            agent_id: None,
            adapter: None,
            instance: None,
            snapshot: None,
            id_slot: None,
            metadata: Default::default(),
        };
        let json = messages_to_json(&request).expect("messages serialize");
        assert_eq!(json[0]["content"], "hello");
    }

    #[test]
    fn normalize_parts_content_round_trips() {
        let body = serde_json::json!({
            "model": "test",
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "a part"},
                {"type": "image_url", "image_url": {"url": "https://example.test/x.png"}}
            ]}]
        });
        let req = normalize_request(body).unwrap();
        let json = messages_to_json(&req).unwrap();
        assert_eq!(json[0]["content"][0]["type"], "text");
        assert_eq!(json[0]["content"][1]["type"], "image_url");
    }
}
