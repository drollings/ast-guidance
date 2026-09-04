//! Summarization and result acceptance.
//!
//! Provides `ResultScorer` (scores agent/frontier responses against rubric)
//! and `Summarizer` (condenses full responses into compact form).
//!
//! Both accept `Arc<dyn ChatBackend>` — inject `StubChatBackend` in tests,
//! `LlmClient` in production.

use std::sync::Arc;

use fluent_llm::client::ChatBackend;
use fluent_llm::{ChatMessage, LlmError};
use fluent_wvr::prelude::*;
use serde::{Deserialize, Serialize};

use crate::stages::common::{extract_user_message, get_metadata_string};

/// A scored agent/frontier response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    #[serde(default)]
    pub content: String,
    pub score: f64,
    pub accepted: bool,
    pub reason: String,
    pub summary: String,
}

/// Scores agent/frontier responses against a rubric using a local model.
///
/// Inputs from `WorkContext`:
/// - `structured["request"]` — original user query (structured `RouterRequest`)
/// - `"response"` — agent/frontier response text
///
/// Output: `ScoredResult` in `WorkOutput.data`
pub struct ResultScorer {
    name: ArcIntern<str>,
    client: Arc<dyn ChatBackend>,
    acceptance_threshold: f64,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl ResultScorer {
    pub fn new(client: Arc<dyn ChatBackend>, acceptance_threshold: f64) -> Self {
        Self {
            name: ArcIntern::from("pipeline.scorer"),
            client,
            acceptance_threshold,
            depends: vec![ArcIntern::from("pipeline.stage5.output")],
            provides: vec![ArcIntern::from("pipeline.scorer.output")],
        }
    }
}

#[allow(clippy::needless_raw_string_hashes)]
const SCORING_SYSTEM_PROMPT: &str = r#"You are a response quality scorer. Evaluate the assistant's response to the user's query.

Output JSON only:
{
  "score": 0.0-1.0,
  "accepted": true/false,
  "reason": "brief explanation of the score",
  "summary": "one-line summary of the response"
}

Rubric:
- Correctness: accurately addresses the query
- Completeness: covers all aspects of the query
- Clarity: well-written and easy to understand
- Safety: avoids harmful content

Score >= 0.7: response is acceptable
Score between 0.3 and 0.7: borderline, may need review
Score < 0.3: response is garbage or incoherent

Only output valid JSON, no other text."#;

impl WorkUnit for ResultScorer {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let query = extract_user_message(ctx)?;
        let response = get_metadata_string(ctx, "response")
            .ok_or_else(|| WorkError::Execution("missing response in context metadata".into()))?;

        let prompt = format!("User query: {query}\n\nAssistant response:\n{response}");

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: SCORING_SYSTEM_PROMPT.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: prompt,
            },
        ];

        let response_text = self
            .client
            .chat_complete(&messages)
            .map_err(|e| WorkError::Execution(format!("scorer error: {e}")))?;

        // M2: LLM-produced text goes through the shared tolerant codec
        // (`parse_typed`: pristine fast path → fence-strip → extract →
        // repair). Pristine JSON takes the fast path with identical values;
        // fence/prose-wrapped replies are recovered instead of erroring.
        let mut scored: ScoredResult =
            fluent_llm::parse_typed(&response_text, &serde_json::Value::Null, |_| {})
                .map_err(|e| WorkError::Execution(format!("scorer parse error: {e}")))?;

        scored.content = response;
        scored.accepted = scored.score >= self.acceptance_threshold;

        if !scored.accepted {
            scored.summary = common_core::string::first_sentence(&scored.summary);
        }

        WorkOutput::typed("scored", &scored)
    }
}

impl_fieldless!(ResultScorer);

impl Describable for ResultScorer {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(ResultScorer);

/// Condenses a full agent/frontier response into a compact summary.
///
/// Input from `WorkContext` metadata:
/// - `"content"` — the text to summarize
///
/// Output: `{"summary": "..."}` in `WorkOutput.data`
pub struct Summarizer {
    name: ArcIntern<str>,
    client: Arc<dyn ChatBackend>,
    max_summary_tokens: u32,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl Summarizer {
    pub fn new(client: Arc<dyn ChatBackend>, max_summary_tokens: u32) -> Self {
        Self {
            name: ArcIntern::from("pipeline.summarizer"),
            client,
            max_summary_tokens,
            depends: vec![ArcIntern::from("pipeline.scorer.output")],
            provides: vec![ArcIntern::from("pipeline.summarizer.output")],
        }
    }

    pub fn summarize_text(&self, text: &str) -> Result<String, LlmError> {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: format!(
                    "Summarize the following text in {} tokens or fewer. \
                     Be concise but preserve key facts and decisions. \
                     Output only the summary, no preamble.",
                    self.max_summary_tokens
                ),
            },
            ChatMessage {
                role: "user".into(),
                content: text.into(),
            },
        ];

        self.client.chat_complete(&messages)
    }
}

const SUMMARIZE_SYSTEM_PROMPT: &str = r"You are a text summarizer. Condense the following content into a compact form.
Preserve key facts, decisions, and action items. Be concise.
Output only the summary, no preamble.";

impl WorkUnit for Summarizer {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let content = get_metadata_string(ctx, "content")
            .ok_or_else(|| WorkError::Execution("missing content in context metadata".into()))?;

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: SUMMARIZE_SYSTEM_PROMPT.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: format!(
                    "Summarize in {} tokens or fewer:\n\n{}",
                    self.max_summary_tokens, content
                ),
            },
        ];

        let summary = self
            .client
            .chat_complete(&messages)
            .map_err(|e| WorkError::Execution(format!("summarizer error: {e}")))?;

        WorkOutput::typed("summarized", &serde_json::json!({"summary": summary}))
    }
}

impl_fieldless!(Summarizer);

impl Describable for Summarizer {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(Summarizer);

#[cfg(test)]
#[path = "../tests/summarization.rs"]
mod tests;
