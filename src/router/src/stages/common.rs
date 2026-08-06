//! Shared helpers for pipeline stages.

use serde_json::{Map, Value};

use fluent_wvr::prelude::*;

use crate::types::{RouterMessageContent, RouterRequest};

/// Ensure a floating-point field exists on a classifier/tree JSON object,
/// coercing a string-valued number back to numeric. The shared "surviving
/// normalization" used by both the flat `ClassifierStage` sanitizer and the
/// M4 classification-tree parser.
pub(crate) fn coerce_float(obj: &mut Map<String, Value>, key: &str, default: f64) {
    match obj.get(key) {
        None => {
            if let Some(n) = serde_json::Number::from_f64(default) {
                obj.insert(key.into(), Value::Number(n));
            }
        }
        Some(Value::String(s)) => {
            if let Ok(n) = s.parse::<f64>() {
                if let Some(num) = serde_json::Number::from_f64(n) {
                    obj[key] = Value::Number(num);
                }
            }
        }
        _ => {}
    }
}

/// Ensure an unsigned-integer field exists on a classifier/tree JSON object,
/// coercing from float or string. The shared "surviving normalization" for the
/// `complexity` axis.
pub(crate) fn coerce_u8(obj: &mut Map<String, Value>, key: &str, default: u8) {
    match obj.get(key) {
        None => {
            obj.insert(key.into(), Value::Number(serde_json::Number::from(default)));
        }
        Some(Value::Number(n)) => {
            let as_u8 = n
                .as_u64()
                .map(|i| i.min(u64::from(u8::MAX)) as u8)
                .or_else(|| n.as_f64().map(|f| f.round().min(f64::from(u8::MAX)) as u8));
            if let Some(v) = as_u8 {
                obj[key] = Value::Number(serde_json::Number::from(v));
            }
        }
        Some(Value::String(s)) => {
            let as_u8 = s
                .parse::<u8>()
                .ok()
                .or_else(|| s.parse::<f64>().ok().map(|f| f.round() as u8));
            if let Some(v) = as_u8 {
                obj[key] = Value::Number(serde_json::Number::from(v));
            }
        }
        _ => {}
    }
}

/// Ensure a string field exists on a classifier/tree JSON object with a
/// default when absent.
pub(crate) fn coerce_string(obj: &mut Map<String, Value>, key: &str, default: &str) {
    if !obj.contains_key(key) {
        obj.insert(key.into(), Value::String(default.into()));
    }
}

/// Extract the last user message from the request carried in
/// `ctx.structured["request"]`.
///
/// The request is the structured canonical `RouterRequest` (a typed
/// `serde_json::Value` in the structured channel, not a JSON string), so
/// content may be either a plain string (`RouterMessageContent::Text`) or an
/// array of content parts (`RouterMessageContent::Parts`, the OpenAI
/// multi-part form used by clients like Brave Leo). Text rendering for both
/// forms lives in the single canonical helper
/// `RouterMessageContent::to_string_lossy` — this function only picks the
/// message, it never re-implements content rendering.
///
/// Selection semantics: the last `role == "user"` message whose content
/// renders to text. A `Text` message is returned verbatim (an empty string is
/// a valid — if degenerate — user message). A `Parts` message with no
/// extractable text (e.g. image-only) is skipped so an earlier text message
/// still resolves, matching the historical skip of non-string content.
pub fn extract_user_message(ctx: &WorkContext) -> Result<String, WorkError> {
    let request: RouterRequest = ctx
        .structured("request")
        .map_err(|e| WorkError::Execution(format!("missing request: {e}")))?;

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
}
