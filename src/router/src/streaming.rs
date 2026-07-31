//! SSE streaming handler — translates RouterResponse chunks into
//! OpenAI-compatible streaming delta chunks.

use crate::types::RouterChoice;

/// SSE streaming handler. Translates RouterChoice chunks into
/// OpenAI-compatible streaming delta chunks.
pub struct StreamingHandler {
    buffer: String,
    filtered_buffer: String,
    chunk_index: u32,
    request_id: String,
    model: String,
    filter_thinking: bool,
    /// When filter_thinking is true, tracks content inside an unclosed
    /// thinking block across chunks so partial think tags are never
    /// leaked to the client.
    in_think_block: bool,
    think_pending: String,
    /// Held-back trailing text that is a proper prefix of an open/close tag
    /// (e.g. `<thi`), waiting for the next chunk to complete it.
    pending: String,
}

impl StreamingHandler {
    pub fn new(request_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            buffer: String::new(),
            filtered_buffer: String::new(),
            chunk_index: 0,
            request_id: request_id.into(),
            model: model.into(),
            filter_thinking: false,
            in_think_block: false,
            think_pending: String::new(),
            pending: String::new(),
        }
    }

    #[must_use]
    pub fn with_filter_thinking(mut self, enabled: bool) -> Self {
        self.filter_thinking = enabled;
        self
    }

    /// Format a single delta chunk as an SSE `data:` line.
    /// Returns the SSE-formatted string including `\n\n` terminator.
    /// When `filter_thinking` is true, thinking blocks are stripped from
    /// the delta before emission, correctly handling blocks that span
    /// multiple chunks.
    pub fn format_chunk(&mut self, delta: &str, finish_reason: Option<&str>) -> String {
        self.buffer.push_str(delta);

        let content_to_send = if self.filter_thinking {
            self.filter_think_block(delta)
        } else {
            self.chunk_index += 1;
            delta.to_string()
        };

        if content_to_send.is_empty() && finish_reason.is_none() {
            // Entire chunk was inside a think block — emit nothing
            // (unless there's a finish_reason that needs to be sent).
            return String::new();
        }

        self.filtered_buffer.push_str(&content_to_send);

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
                    "content": content_to_send
                },
                "finish_reason": finish_reason,
            }],
        });

        format!(
            "data: {}\n\n",
            serde_json::to_string(&chunk).unwrap_or_default()
        )
    }

    /// Filter thinking blocks from a delta, handling cross-chunk blocks.
    /// Returns the text to emit (empty if entirely inside a think block).
    ///
    /// Tag splits are handled at the *suffix* level: a trailing run of the
    /// input that is a proper prefix of an open/close tag is held in `pending`
    /// until the next chunk can complete it, so `<thi` + `nk>` or
    /// `</thi` + `nk>` never leaks a partial tag to the client.
    fn filter_think_block(&mut self, delta: &str) -> String {
        const OPEN_TAGS: &[&str] = &["<think>", "<thinking>"];
        const CLOSE_TAGS: &[&str] = &["</think>", "</thinking>"];

        let mut combined = String::with_capacity(self.pending.len() + delta.len());
        combined.push_str(&self.pending);
        combined.push_str(delta);
        self.pending.clear();

        let mut remaining: &str = &combined;
        let mut output = String::new();

        loop {
            if self.in_think_block {
                // Check for any closing tag in remaining
                let mut earliest_close: Option<(usize, &str)> = None;
                for ct in CLOSE_TAGS {
                    if let Some(pos) = remaining.find(ct) {
                        if earliest_close.is_none_or(|(e, _)| pos < e) {
                            earliest_close = Some((pos, ct));
                        }
                    }
                }

                if let Some((pos, ct)) = earliest_close {
                    // Close the think block — emit nothing for its content
                    self.in_think_block = false;
                    self.think_pending.clear();
                    remaining = &remaining[pos + ct.len()..];
                    continue;
                }
                // Still inside the block — discard the definite content but
                // hold back a trailing close-tag prefix that may complete in
                // the next chunk (otherwise a split `</thi` + `nk>` close
                // would be missed and the tail would leak).
                let hold = Self::tag_prefix_len(remaining, CLOSE_TAGS);
                self.think_pending.push_str(&remaining[..remaining.len() - hold]);
                self.pending.push_str(&remaining[remaining.len() - hold..]);
                self.chunk_index += 1;
                return output;
            }

            // Not inside a think block — scan for opening tags
            let mut earliest_open: Option<(usize, &str)> = None;
            for ot in OPEN_TAGS {
                if let Some(pos) = remaining.find(ot) {
                    if earliest_open.is_none_or(|(e, _)| pos < e) {
                        earliest_open = Some((pos, ot));
                    }
                }
            }

            if let Some((pos, ot)) = earliest_open {
                output.push_str(&remaining[..pos]);
                self.in_think_block = true;
                self.think_pending.clear();
                remaining = &remaining[pos + ot.len()..];
                continue;
            }

            // No complete open tag — emit everything except a trailing run
            // that could be the start of a tag split across chunks.
            let hold = Self::tag_prefix_len(remaining, OPEN_TAGS);
            output.push_str(&remaining[..remaining.len() - hold]);
            self.pending.push_str(&remaining[remaining.len() - hold..]);
            self.chunk_index += 1;
            return output;
        }
    }

    /// Length of the longest suffix of `s` that is a proper prefix of any tag
    /// in `tags` — i.e. a run that could grow into a full tag once the next
    /// chunk arrives.
    fn tag_prefix_len(s: &str, tags: &[&str]) -> usize {
        let mut best = 0;
        for tag in tags {
            for len in 1..tag.len() {
                if s.ends_with(&tag[..len]) {
                    best = best.max(len);
                }
            }
        }
        best
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

    /// Raw accumulated content as received from the LLM (including any
    /// thinking blocks).
    pub fn accumulated_content(&self) -> &str {
        &self.buffer
    }

    /// Accumulated content with thinking blocks stripped.
    pub fn filtered_content(&self) -> String {
        if self.filter_thinking {
            self.filtered_buffer.clone()
        } else {
            common_core::string::strip_thinking_blocks(&self.buffer)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RouterChoice, RouterMessage, RouterMessageContent};
    use common_core::string::strip_thinking_blocks;

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
    fn filtered_content_open_tag_split_across_chunks() {
        let mut h = StreamingHandler::new("req-1", "test").with_filter_thinking(true);
        h.format_chunk("Hello <thi", None);
        h.format_chunk("nk>secret reasoning</think>the answer", None);
        assert_eq!(
            h.filtered_content(),
            "Hello the answer",
            "a split `<thi`+`nk>` open tag must never leak a partial tag"
        );
    }

    #[test]
    fn filtered_content_close_tag_split_across_chunks() {
        let mut h = StreamingHandler::new("req-1", "test").with_filter_thinking(true);
        h.format_chunk("A <think>secret</thi", None);
        h.format_chunk("nk>B", None);
        assert_eq!(
            h.filtered_content(),
            "A B",
            "a split `</thi`+`nk>` close tag must not leak its tail"
        );
    }

    #[test]
    fn filtered_content_incomplete_tag_prefix_not_emitted_partial() {
        let mut h = StreamingHandler::new("req-1", "test").with_filter_thinking(true);
        // A lone `<` at a chunk boundary is held back, not emitted as a
        // partial tag.
        h.format_chunk("value <", None);
        h.format_chunk("", None);
        assert_eq!(
            h.filtered_content(),
            "value ",
            "an incomplete tag prefix must be held back, not leaked"
        );
        // It only resolves to real text when a non-tag continuation arrives.
        h.format_chunk("input", None);
        assert_eq!(h.filtered_content(), "value <input");
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
        assert_eq!(strip_thinking_blocks("\x3cthink\x3eunclosed"), "");
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
            strip_thinking_blocks(
                "\x3cthink\x3eollama\x3c/think\x3eplain\x3cthinking\x3exml\x3c/thinking\x3eend"
            ),
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
        assert_eq!(strip_thinking_blocks(" thinking unclosed"), "");
        assert_eq!(
            strip_thinking_blocks("normal text response here"),
            "normal text response here"
        );
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
