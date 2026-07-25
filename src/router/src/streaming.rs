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

    pub fn filtered_content(&self) -> String {
        strip_thinking_blocks(&self.buffer)
    }
}

use common_core::string::find_subseq;

/// Tag pair definitions for thinking blocks. Tried in order.
const THINKING_PAIRS: &[(&[u8], &[u8])] = &[
    (b"\x3cthink\x3e", b"\x3c/think\x3e"), // Ollama: <think>...</think>
    (b"\x3cthinking\x3e", b"\x3c/thinking\x3e"), // Claude/Gemini: <thinking>...</thinking>
];

/// Strip content between start and end markers. Returns the text with content
/// between each matching pair removed. If a start marker is found without a
/// matching end marker, everything from the start marker onward is stripped.
fn strip_tag_pairs(
    text: &str,
    pairs: &[(&[u8], &[u8])],
) -> String {
    let mut result = String::with_capacity(text.len());
    let mut pos = 0;
    let bytes = text.as_bytes();

    while pos < bytes.len() {
        let mut earliest: Option<usize> = None;
        let mut matched_pair: Option<(&[u8], &[u8])> = None;

        for &(start_mark, end_mark) in pairs {
            if let Some(start) = find_subseq(bytes, pos, start_mark) {
                if earliest.map_or(true, |e| start < e) {
                    earliest = Some(start);
                    matched_pair = Some((start_mark, end_mark));
                }
            }
        }

        match matched_pair {
            Some((start_mark, end_mark)) => {
                let start = earliest.unwrap();
                result.push_str(&text[pos..start]);
                let after_start = start + start_mark.len();
                if let Some(end) = find_subseq(bytes, after_start, end_mark) {
                    pos = end + end_mark.len();
                } else {
                    return result;
                }
            }
            None => {
                result.push_str(&text[pos..]);
                return result;
            }
        }
    }

    result
}

/// Strip ` thinking ...  response\n` plain-text delimiters
/// (DeepSeek R1, unsloth thinking). The end delimiter must be followed by a
/// newline or end-of-string.
fn strip_plain_thinking(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut pos = 0;
    let bytes = text.as_bytes();

    while pos < bytes.len() {
        match find_subseq(bytes, pos, b" thinking") {
            Some(start) => {
                let after_start = start + 9;
                match find_subseq(bytes, after_start, b" response") {
                    Some(end) if end + 9 >= bytes.len()
                        || bytes[end + 9] == b'\n' =>
                    {
                        result.push_str(&text[pos..start]);
                        let after_end = end + 9;
                        pos = if after_end < bytes.len() { after_end + 1 } else { after_end };
                    }
                    _ => return result,
                }
            }
            None => {
                result.push_str(&text[pos..]);
                return result;
            }
        }
    }

    result
}

/// Strip thinking blocks from the given text. Handles multiple formats:
/// - `<think>...</think>` (Ollama-style XML tags)
/// - `<thinking>...</thinking>` (Claude, Gemini, some local models)
/// - ` thinking ...  response\n` (DeepSeek R1, unsloth thinking)
/// Tags can appear anywhere in the content and blocks may be unclosed.
pub fn strip_thinking_blocks(text: &str) -> String {
    let tagged = strip_tag_pairs(text, THINKING_PAIRS);
    if tagged != text {
        return tagged;
    }
    strip_plain_thinking(text)
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

    #[test]
    fn strip_thinking_blocks_free_function() {
        assert_eq!(strip_thinking_blocks("Hello  world"), "Hello  world");
        assert_eq!(
            strip_thinking_blocks("Hello <thinking>reason</thinking> world"),
            "Hello  world"
        );
        assert_eq!(
            strip_thinking_blocks("<thinking>a</thinking>B<thinking>c</thinking>D"),
            "BD"
        );
        assert_eq!(strip_thinking_blocks("start<thinking>unclosed"), "start");
        assert_eq!(
            strip_thinking_blocks("<thinking>only thinking</thinking>"),
            ""
        );
    }

    #[test]
    fn strip_ollama_thinking_blocks() {
        assert_eq!(
            strip_thinking_blocks("\x3cthink\x3ereason\x3c/think\x3eresult"),
            "result"
        );
        assert_eq!(
            strip_thinking_blocks("before \x3cthink\x3ereason\x3c/think\x3e after"),
            "before  after"
        );
        assert_eq!(
            strip_thinking_blocks("\x3cthink\x3eunclosed"),
            ""
        );
        assert_eq!(
            strip_thinking_blocks("A\x3cthink\x3eB\x3c/think\x3eC\x3cthink\x3eD\x3c/think\x3eE"),
            "ACE"
        );
        assert_eq!(strip_thinking_blocks("no tags here"), "no tags here");
    }

    #[test]
    fn strip_ollama_thinking_at_any_position() {
        assert_eq!(
            strip_thinking_blocks("prefix \x3cthink\x3e middle stuff \x3c/think\x3e suffix"),
            "prefix  suffix"
        );
    }

    #[test]
    fn strip_thinking_blocks_multiple_formats() {
        assert_eq!(
            strip_thinking_blocks("\x3cthink\x3eollama\x3c/think\x3eplain\x3cthinking\x3exml\x3c/thinking\x3eend"),
            "plainend"
        );
    }

    #[test]
    fn strip_plain_thinking_blocks() {
        assert_eq!(
            strip_thinking_blocks(" thinking let me check response\nThe answer is 4"),
            "The answer is 4"
        );
        assert_eq!(
            strip_thinking_blocks("Hello  thinking let me think response\n result"),
            "Hello  result"
        );
        assert_eq!(
            strip_thinking_blocks(" thinking unclosed"),
            ""
        );
        assert_eq!(strip_thinking_blocks("normal text response here"), "normal text response here");
        assert_eq!(
            strip_thinking_blocks(" thinking a response\nB thinking c response\nD"),
            "BD"
        );
        assert_eq!(
            strip_thinking_blocks(" thinking only thinking response\n"),
            ""
        );
    }

    #[test]
    fn strip_plain_thinking_multiple_blocks() {
        assert_eq!(
            strip_thinking_blocks(" thinking a\n response\nB thinking c\n response\nD"),
            "BD"
        );
    }

    #[test]
    fn strip_plain_thinking_only_thinking_response() {
        assert_eq!(
            strip_thinking_blocks(" thinking only thinking response\n"),
            ""
        );
    }

    #[test]
    fn strip_plain_thinking_respects_word_boundary() {
        assert_eq!(
            strip_thinking_blocks("rethinking a plan"),
            "rethinking a plan"
        );
    }
}