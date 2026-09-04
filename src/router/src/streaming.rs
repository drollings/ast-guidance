//! SSE streaming handler — translates RouterResponse chunks into
//! OpenAI-compatible streaming delta chunks.

use std::sync::{Arc, Mutex};

use common_core::sync::lock;
use fluent_llm::thinking::{strip_thinking_blocks, StreamingThinkFilter};

use crate::types::RouterChoice;

/// Best-effort stream finalization sink. The streaming backend's task
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
    /// Delegates to the canonical `StreamingThinkFilter` in `fluent-llm`,
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
            strip_thinking_blocks(&self.buffer)
        }
    }
}

#[cfg(test)]
#[path = "../tests/streaming.rs"]
mod tests;
