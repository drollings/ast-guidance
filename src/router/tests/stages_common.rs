use super::*;
use crate::types::{ContentPart, ImageUrl, RouterMessage, RouterMessageContent};

fn make_ctx_with_messages(messages: Vec<RouterMessage>) -> WorkContext {
    let request = RouterRequest {
        model: "test".into(),
        messages,
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
    let mut ctx = WorkContext::default();
    ctx.set_structured("request", &request);
    ctx
}

#[test]
fn extracts_last_text_message() {
    let ctx = make_ctx_with_messages(vec![
        RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text("earlier".into()),
            tool_calls: None,
            tool_call_id: None,
        },
        RouterMessage {
            role: "assistant".into(),
            content: RouterMessageContent::Text("response".into()),
            tool_calls: None,
            tool_call_id: None,
        },
        RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text("latest".into()),
            tool_calls: None,
            tool_call_id: None,
        },
    ]);
    assert_eq!(extract_user_message(&ctx).unwrap(), "latest");
}

#[test]
fn extracts_text_from_content_parts() {
    let ctx = make_ctx_with_messages(vec![RouterMessage {
        role: "user".into(),
        content: RouterMessageContent::Parts(vec![
            ContentPart::Text {
                text: "About this user:".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.test/x.png".into(),
                },
            },
            ContentPart::Text {
                text: "Daniel".into(),
            },
        ]),
        tool_calls: None,
        tool_call_id: None,
    }]);
    assert_eq!(
        extract_user_message(&ctx).unwrap(),
        "About this user: Daniel"
    );
}

#[test]
fn errors_when_no_user_message() {
    let ctx = make_ctx_with_messages(vec![RouterMessage {
        role: "system".into(),
        content: RouterMessageContent::Text("sys".into()),
        tool_calls: None,
        tool_call_id: None,
    }]);
    let err = extract_user_message(&ctx).unwrap_err();
    assert!(err.to_string().contains("no user message found"));
}

#[test]
fn errors_when_request_missing() {
    let ctx = WorkContext::default();
    let err = extract_user_message(&ctx).unwrap_err();
    assert!(err.to_string().contains("missing request"));
}
