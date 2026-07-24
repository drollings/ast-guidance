//! SSE streaming handler — translates RouterResponse chunks into
//! OpenAI-compatible streaming delta chunks.

use crate::types::RouterChoice;

/// SSE streaming handler. Translates RouterChoice chunks into
/// OpenAI-compatible streaming delta chunks.
pub struct StreamingHandler {
    buffer: String,
    chunk_index: u32,
    request_id: String,
    model: String,
}

impl StreamingHandler {
    pub fn new(request_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            buffer: String::new(),
            chunk_index: 0,
            request_id: request_id.into(),
            model: model.into(),
        }
    }

    /// Format a single delta chunk as an SSE `data:` line.
    /// Returns the SSE-formatted string including `\n\n` terminator.
    pub fn format_chunk(&mut self, delta: &str, finish_reason: Option<&str>) -> String {
        let chunk = serde_json::json!({
            "id": self.request_id,
            "object": "chat.completion.chunk",
            "created": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": delta
                },
                "finish_reason": finish_reason,
            }],
        });

        self.buffer.push_str(delta);
        self.chunk_index += 1;

        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap_or_default())
    }

    /// Format a single choice as an SSE delta chunk.
    pub fn format_choice_chunk(&mut self, choice: &RouterChoice) -> String {
        let content = match &choice.message.content {
            crate::types::RouterMessageContent::Text(s) => s.as_str(),
            crate::types::RouterMessageContent::Parts(_) => "",
        };

        self.format_chunk(content, Some(&choice.finish_reason))
    }

    /// Format the stream termination marker.
    pub fn format_done(&self) -> String {
        "data: [DONE]\n\n".to_string()
    }

    pub fn chunk_count(&self) -> u32 {
        self.chunk_index
    }

    pub fn accumulated_content(&self) -> &str {
        &self.buffer
    }

    /// Return the accumulated content with `<thinking>...</thinking>` blocks
    /// stripped. Intended for session context / KV cache accumulation where
    /// reasoning tokens should not pollute the working context.
    pub fn filtered_content(&self) -> String {
        let mut result = String::with_capacity(self.buffer.len());
        let mut in_thinking = false;
        let mut pos = 0;
        let bytes = self.buffer.as_bytes();

        while pos < bytes.len() {
            if in_thinking {
                if let Some(end) = find_subseq(bytes, pos, b"</thinking>") {
                    in_thinking = false;
                    pos = end + b"</thinking>".len();
                } else {
                    break;
                }
            } else if let Some(start) = find_subseq(bytes, pos, b"<thinking>") {
                result.push_str(&self.buffer[pos..start]);
                if let Some(end) = find_subseq(bytes, start + b"<thinking>".len(), b"</thinking>") {
                    pos = end + b"</thinking>".len();
                } else {
                    return result;
                }
            } else {
                result.push_str(&self.buffer[pos..]);
                return result;
            }
        }

        result
    }
}

/// Find the first occurrence of a subsequence in bytes starting from `start`.
fn find_subseq(haystack: &[u8], start: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    haystack[start..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| start + i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RouterMessage, RouterMessageContent, RouterChoice};

    #[test]
    fn format_single_chunk() {
        let mut h = StreamingHandler::new("req-1", "test-model");
        let line = h.format_chunk("hello", None);
        assert!(line.starts_with("data: "));
        assert!(line.ends_with("\n\n"));
        assert!(line.contains("\"delta\""));
        assert!(line.contains("\"content\":\"hello\""));
        assert_eq!(h.chunk_count(), 1);
    }

    #[test]
    fn format_chunk_with_finish_reason() {
        let mut h = StreamingHandler::new("req-2", "gpt-4");
        let line = h.format_chunk("world", Some("stop"));
        assert!(line.contains("\"finish_reason\":\"stop\""));
        assert!(line.contains("\"content\":\"world\""));
    }

    #[test]
    fn format_choice_chunk() {
        let mut h = StreamingHandler::new("req-3", "gpt-4");
        let choice = RouterChoice {
            index: 0,
            message: RouterMessage {
                role: "assistant".into(),
                content: RouterMessageContent::Text("done".into()),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".into(),
        };
        let line = h.format_choice_chunk(&choice);
        assert!(line.contains("\"content\":\"done\""));
        assert!(line.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn format_done_marker() {
        let h = StreamingHandler::new("req-1", "test");
        assert_eq!(h.format_done(), "data: [DONE]\n\n");
    }

    #[test]
    fn accumulated_content() {
        let mut h = StreamingHandler::new("req-1", "test");
        h.format_chunk("hello", None);
        h.format_chunk(" world", None);
        assert_eq!(h.accumulated_content(), "hello world");
    }

    #[test]
    fn filtered_content_strips_thinking_blocks() {
        let mut h = StreamingHandler::new("req-1", "test");
        h.format_chunk("Hello ", None);
        h.format_chunk("<thinking>let me think", None);
        h.format_chunk(" carefully</thinking>", None);
        h.format_chunk(" world", None);
        assert_eq!(h.filtered_content(), "Hello  world");
    }

    #[test]
    fn filtered_content_handles_unclosed_thinking() {
        let mut h = StreamingHandler::new("req-1", "test");
        h.format_chunk("A ", None);
        h.format_chunk("<thinking>unclosed", None);
        assert_eq!(h.filtered_content(), "A ");
    }

    #[test]
    fn filtered_content_no_thinking_noop() {
        let mut h = StreamingHandler::new("req-1", "test");
        h.format_chunk("hello", None);
        h.format_chunk(" world", None);
        assert_eq!(h.filtered_content(), "hello world");
    }

    #[test]
    fn filtered_content_multiple_thinking_blocks() {
        let mut h = StreamingHandler::new("req-1", "test");
        h.format_chunk("A", None);
        h.format_chunk("<thinking>skip</thinking>", None);
        h.format_chunk("B", None);
        h.format_chunk("<thinking>skip2</thinking>", None);
        h.format_chunk("C", None);
        assert_eq!(h.filtered_content(), "ABC");
    }

    #[test]
    fn filtered_content_thinking_at_start() {
        let mut h = StreamingHandler::new("req-1", "test");
        h.format_chunk("<thinking>reasoning</thinking>", None);
        h.format_chunk("result", None);
        assert_eq!(h.filtered_content(), "result");
    }

    #[test]
    fn filtered_content_thinking_at_end() {
        let mut h = StreamingHandler::new("req-1", "test");
        h.format_chunk("result", None);
        h.format_chunk("<thinking>reasoning</thinking>", None);
        assert_eq!(h.filtered_content(), "result");
    }
}