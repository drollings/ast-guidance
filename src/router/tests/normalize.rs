use super::*;
use crate::types::{
    ContentPart, FunctionCall, ImageUrl, RouterChoice, RouterMessage, RouterMessageContent,
    ToolCall, Usage,
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
