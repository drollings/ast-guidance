//! Unified request/response types — OpenAI-compatible protocol boundary for
//! the pipeline. The canonical shapes that every stage and provider backend reads from.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterRequest {
    pub model: String,
    pub messages: Vec<RouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<RouterTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// Instance or group to route to (forwarded to the owning llama-server).
    /// Overrides any instance component of `model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// KV snapshot to switch into the target slot before serving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
    /// Slot to target for snapshot switching (defaults to 0 on the server).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_slot: Option<i32>,
    /// Internal pipeline metadata (anonymize map, routing annotations, etc.)
    /// Not sent to frontier providers. Serialized as empty object when absent.
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterMessage {
    pub role: String,
    pub content: RouterMessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RouterMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl RouterMessageContent {
    pub fn as_text(&self) -> &str {
        match self {
            Self::Text(s) => s,
            Self::Parts(_) => "",
        }
    }

    pub fn to_string_lossy(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<RouterChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterChoice {
    pub index: u32,
    pub message: RouterMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> RouterRequest {
        serde_json::from_value(serde_json::json!({
            "model": "fast",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 128,
            "stream": false,
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "session_id": "s1",
        }))
        .expect("deserialize sample request")
    }

    #[test]
    fn router_request_serde_round_trip() {
        let req = sample_request();
        assert_eq!(req.model, "fast");
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.session_id.as_deref(), Some("s1"));
        let back: RouterRequest =
            serde_json::from_str(&serde_json::to_string(&req).expect("serialize")).expect("round trip");
        assert_eq!(back.model, "fast");
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.tools.as_ref().expect("tools").len(), 1);
    }

    #[test]
    fn router_request_optional_fields_default_none() {
        let req: RouterRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [],
        }))
        .expect("deserialize");
        assert_eq!(req.temperature, None);
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.stream, None);
        assert!(req.tools.is_none());
        assert!(req.session_id.is_none());
        assert!(req.instance.is_none());
        assert!(req.snapshot.is_none());
        assert!(req.metadata.is_empty());
    }

    #[test]
    fn router_request_preserves_routing_fields() {
        let req: RouterRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [],
            "instance": "g1",
            "snapshot": "snap",
            "id_slot": 3,
            "agent_id": "a",
            "adapter": "std",
            "metadata": {"k": "v"},
        }))
        .expect("deserialize");
        assert_eq!(req.instance.as_deref(), Some("g1"));
        assert_eq!(req.snapshot.as_deref(), Some("snap"));
        assert_eq!(req.id_slot, Some(3));
        assert_eq!(req.metadata["k"], "v");
        // Routing fields are preserved through a serde round trip.
        let back: RouterRequest =
            serde_json::from_str(&serde_json::to_string(&req).expect("serialize")).expect("round trip");
        assert_eq!(back.instance.as_deref(), Some("g1"));
        assert_eq!(back.snapshot.as_deref(), Some("snap"));
    }

    #[test]
    fn router_message_content_text() {
        let c = RouterMessageContent::Text("hi".into());
        assert_eq!(c.as_text(), "hi");
        assert_eq!(c.to_string_lossy(), "hi");
    }

    #[test]
    fn router_message_content_parts() {
        let c = RouterMessageContent::Parts(vec![
            ContentPart::Text { text: "one".into() },
            ContentPart::ImageUrl { image_url: ImageUrl { url: "http://img".into() } },
            ContentPart::Text { text: "two".into() },
        ]);
        assert_eq!(c.as_text(), "");
        assert_eq!(c.to_string_lossy(), "one two");
    }

    #[test]
    fn router_message_content_untagged_serde() {
        // Untagged: a plain string parses to Text; an array of parts to Parts.
        let text: RouterMessageContent = serde_json::from_str("\"plain\"").expect("text");
        assert!(matches!(text, RouterMessageContent::Text(_)));
        let parts: RouterMessageContent = serde_json::from_str(
            r#"[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"http://i"}}]"#,
        )
        .expect("parts");
        assert!(matches!(parts, RouterMessageContent::Parts(_)));
        // Round trips preserve the variant.
        let back: RouterMessageContent =
            serde_json::from_str(&serde_json::to_string(&parts).expect("serialize")).expect("round trip");
        assert!(matches!(back, RouterMessageContent::Parts(_)));
    }

    #[test]
    fn content_part_tagged_serde() {
        let t: ContentPart = serde_json::from_str(r#"{"type":"text","text":"x"}"#).expect("text part");
        assert!(matches!(t, ContentPart::Text { .. }));
        let i: ContentPart =
            serde_json::from_str(r#"{"type":"image_url","image_url":{"url":"http://u"}}"#).expect("image");
        assert!(matches!(i, ContentPart::ImageUrl { .. }));
    }

    #[test]
    fn router_message_tool_fields_round_trip() {
        let msg: RouterMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [{
                "id": "t1", "type": "function",
                "function": {"name": "fn", "arguments": "{}"}
            }],
            "tool_call_id": "t0",
        }))
        .expect("deserialize");
        let calls = msg.tool_calls.as_ref().expect("tool calls");
        assert_eq!(calls[0].call_type, "function");
        assert_eq!(calls[0].function.name, "fn");
        assert_eq!(msg.tool_call_id.as_deref(), Some("t0"));
        let back: RouterMessage =
            serde_json::from_str(&serde_json::to_string(&msg).expect("serialize")).expect("round trip");
        assert_eq!(back.role, "assistant");
    }

    #[test]
    fn router_response_serde_round_trip() {
        let resp: RouterResponse = serde_json::from_value(serde_json::json!({
            "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"}
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }))
        .expect("deserialize");
        assert_eq!(resp.choices[0].message.content.as_text(), "hi");
        assert_eq!(resp.usage.total_tokens, 3);
        let back: RouterResponse =
            serde_json::from_str(&serde_json::to_string(&resp).expect("serialize")).expect("round trip");
        assert_eq!(back.id, "cmpl-1");
    }

    #[test]
    fn usage_defaults_to_zero() {
        assert_eq!(Usage::default().total_tokens, 0);
    }
}
