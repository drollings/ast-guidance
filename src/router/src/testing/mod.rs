pub mod mock;

pub use mock::{
    default_transcript, load_transcript_file, transcript_provider_from_entries,
    MockDispatchContext, MockTranscriptEntry, TranscriptProvider,
};

use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};

/// Build a `RouterRequest` with default fields for testing.
/// All fields except `model` and `messages` use sensible defaults
/// (all `None`, empty metadata).
pub fn test_request(text: &str) -> RouterRequest {
    RouterRequest {
        model: "test-model".into(),
        messages: vec![RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text(text.into()),
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
        metadata: Default::default(),
    }
}

/// Extract the text content of the first message in a `RouterRequest`.
pub fn text_of(result: &RouterRequest) -> String {
    result.messages[0].content.to_string_lossy()
}
