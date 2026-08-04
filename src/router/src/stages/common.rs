//! Shared helpers for pipeline stages.

use fluent_wvr::prelude::*;

use crate::types::{RouterMessageContent, RouterRequest};

/// Extract the last user message from the request carried in
/// `ctx.metadata["request"]`.
///
/// The request is the serialized canonical `RouterRequest`, so content may be
/// either a plain string (`RouterMessageContent::Text`) or an array of content
/// parts (`RouterMessageContent::Parts`, the OpenAI multi-part form used by
/// clients like Brave Leo). Text rendering for both forms lives in the single
/// canonical helper `RouterMessageContent::to_string_lossy` — this function
/// only picks the message, it never re-implements content rendering.
///
/// Selection semantics: the last `role == "user"` message whose content
/// renders to text. A `Text` message is returned verbatim (an empty string is
/// a valid — if degenerate — user message). A `Parts` message with no
/// extractable text (e.g. image-only) is skipped so an earlier text message
/// still resolves, matching the historical skip of non-string content.
pub fn extract_user_message(ctx: &WorkContext) -> Result<String, WorkError> {
    let request_str = get_metadata_string(ctx, "request")
        .ok_or_else(|| WorkError::Execution("missing request".into()))?;

    let request: RouterRequest = serde_json::from_str(&request_str)
        .map_err(|e| WorkError::Execution(e.to_string()))?;

    request
        .messages
        .iter()
        .rev()
        .find_map(|m| {
            if m.role != "user" {
                return None;
            }
            match &m.content {
                RouterMessageContent::Text(s) => Some(s.clone()),
                RouterMessageContent::Parts(_) => {
                    let text = m.content.to_string_lossy();
                    (!text.trim().is_empty()).then_some(text)
                }
            }
        })
        .ok_or_else(|| WorkError::Execution("no user message found".into()))
}

pub fn get_metadata_string(ctx: &WorkContext, key: &str) -> Option<String> {
    ctx.metadata.get(key).and_then(|v| match v {
        MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
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
            metadata: Default::default(),
        };
        let mut ctx = WorkContext::default();
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(serde_json::to_string(&request).unwrap()),
        );
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
}
