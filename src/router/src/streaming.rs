//! SSE streaming handler — translates RouterResponse chunks into
//! OpenAI-compatible streaming delta chunks.

use std::sync::{Arc, Mutex};

use common_core::string::StreamingThinkFilter;
use common_core::sync::lock;

use crate::types::RouterChoice;

/// Best-effort stream finalization sink (M5). The streaming backend's task
/// writes the assembled answer here once the stream ends; the HTTP handler
/// waits on it (bounded) and records the content into the ledger + session
/// step. `finalize` is idempotent in effect — the last writer wins.
#[derive(Clone)]
pub struct StreamAnswer {
    content: Arc<Mutex<Option<String>>>,
    notify: Arc<tokio::sync::Notify>,
}

impl StreamAnswer {
    pub fn new() -> Self {
        Self {
            content: Arc::new(Mutex::new(None)),
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Finalize with the assembled content, waking any waiter.
    pub fn finalize(&self, content: String) {
        *lock(&self.content) = Some(content);
        self.notify.notify_waiters();
    }

    /// The finalized content, if the stream has completed.
    pub fn get(&self) -> Option<String> {
        lock(&self.content).clone()
    }

    /// Wait up to `timeout` for the stream to finalize, then return the
    /// assembled content (or `None` on timeout).
    pub async fn wait(&self, timeout: std::time::Duration) -> Option<String> {
        if let Some(content) = self.get() {
            return Some(content);
        }
        tokio::time::timeout(timeout, self.notify.notified())
            .await
            .ok()?;
        self.get()
    }
}

impl Default for StreamAnswer {
    fn default() -> Self {
        Self::new()
    }
}

/// SSE streaming handler. Translates RouterChoice chunks into
/// OpenAI-compatible streaming delta chunks.
pub struct StreamingHandler {
    buffer: String,
    filtered_buffer: String,
    chunk_index: u32,
    request_id: String,
    model: String,
    filter_thinking: bool,
    /// When filter_thinking is true, delegates cross-chunk think-block
    /// filtering to the shared `StreamingThinkFilter` value type.
    filter: StreamingThinkFilter,
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
            filter: StreamingThinkFilter::new(),
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

        self.chunk_index += 1;
        let content_to_send = if self.filter_thinking {
            self.filter_think_block(delta)
        } else {
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
    /// Delegates to the canonical `StreamingThinkFilter` in `common-core`,
    /// which holds back trailing tag prefixes at the *suffix* level so
    /// `<thi` + `nk>` or `</thi` + `nk>` never leaks a partial tag.
    fn filter_think_block(&mut self, delta: &str) -> String {
        self.filter.push(delta)
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
    fn stream_answer_finalizes_content() {
        let answer = StreamAnswer::new();
        assert_eq!(answer.get(), None);
        answer.finalize("assembled".into());
        assert_eq!(answer.get().as_deref(), Some("assembled"));
    }

    #[tokio::test]
    async fn stream_answer_wait_returns_after_finalize() {
        let answer = StreamAnswer::new();
        let waiter = answer.clone();
        let task = tokio::spawn(async move {
            waiter
                .wait(std::time::Duration::from_millis(2000))
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        answer.finalize("done".into());
        assert_eq!(task.await.expect("waiter completes").as_deref(), Some("done"));
    }

    #[tokio::test]
    async fn stream_answer_wait_times_out_without_finalize() {
        let answer = StreamAnswer::new();
        let content = answer
            .wait(std::time::Duration::from_millis(50))
            .await;
        assert_eq!(content, None);
    }

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
