//! Summarization and result acceptance.
//!
//! Provides `ResultScorer` (scores agent/frontier responses against rubric)
//! and `Summarizer` (condenses full responses into compact form).
//!
//! Both use `ChatBackend` for testability — inject `StubChatBackend` in tests,
//! `LlmClient` in production.

use fluent_wvr::prelude::*;
use guidance_llm::client::ChatBackend;
use guidance_llm::{ChatMessage, LlmClient, LlmConfig, LlmError};
use serde::{Deserialize, Serialize};

use crate::stages::common::{extract_user_message, get_metadata_string};

/// A scored agent/frontier response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    /// The full response content that was scored.
    /// Has `#[serde(default)]` because the LLM judge JSON doesn't include it;
    /// it is populated from context by `ResultScorer::execute`.
    #[serde(default)]
    pub content: String,
    /// Quality score 0.0-1.0 from the LLM judge.
    pub score: f64,
    /// Whether the response passed the acceptance threshold.
    pub accepted: bool,
    /// Human-readable explanation of the score.
    pub reason: String,
    /// Compact summary (single line for rejected, fuller for accepted).
    pub summary: String,
}

/// Scores agent/frontier responses against a rubric using a local model.
///
/// Inputs from `WorkContext` metadata:
/// - `"request"` — original user query (JSON array of messages)
/// - `"response"` — agent/frontier response text
///
/// Output: `ScoredResult` in `WorkOutput.data`
pub struct ResultScorer {
    name: ArcIntern<str>,
    backend: Option<Box<dyn ChatBackend>>,
    config: LlmConfig,
    acceptance_threshold: f64,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl ResultScorer {
    pub fn new(config: LlmConfig, acceptance_threshold: f64) -> Self {
        Self {
            name: ArcIntern::from("pipeline.scorer"),
            backend: None,
            config,
            acceptance_threshold,
            depends: vec![ArcIntern::from("pipeline.stage5.output")],
            provides: vec![ArcIntern::from("pipeline.scorer.output")],
        }
    }

    pub fn with_chat_backend(
        config: LlmConfig,
        backend: Box<dyn ChatBackend>,
        acceptance_threshold: f64,
    ) -> Self {
        Self {
            name: ArcIntern::from("pipeline.scorer"),
            backend: Some(backend),
            config,
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

        let prompt = format!(
            "User query: {query}\n\nAssistant response:\n{response}",
        );

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

        let response_text = if let Some(ref backend) = self.backend {
            backend
                .chat_complete(&messages)
                .map_err(|e| WorkError::Execution(format!("scorer error: {e}")))?
        } else {
            let client = LlmClient::with_config(self.config.clone());
            client
                .chat_complete(&messages)
                .map_err(|e| WorkError::Execution(format!("scorer error: {e}")))?
        };

        let mut scored: ScoredResult = serde_json::from_str(&response_text)
            .map_err(|e| WorkError::Execution(format!("scorer parse error: {e}")))?;

        scored.content = response;
        scored.accepted = scored.score >= self.acceptance_threshold;

        if !scored.accepted {
            scored.summary = truncate_at_sentence(&scored.summary);
        }

        WorkOutput::typed("scored", &scored)
    }
}

impl FieldAccess for ResultScorer {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "ResultScorer has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "ResultScorer has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

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
    backend: Option<Box<dyn ChatBackend>>,
    config: LlmConfig,
    max_summary_tokens: u32,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl Summarizer {
    pub fn new(config: LlmConfig, max_summary_tokens: u32) -> Self {
        Self {
            name: ArcIntern::from("pipeline.summarizer"),
            backend: None,
            config,
            max_summary_tokens,
            depends: vec![ArcIntern::from("pipeline.scorer.output")],
            provides: vec![ArcIntern::from("pipeline.summarizer.output")],
        }
    }

    pub fn with_chat_backend(
        config: LlmConfig,
        backend: Box<dyn ChatBackend>,
        max_summary_tokens: u32,
    ) -> Self {
        Self {
            name: ArcIntern::from("pipeline.summarizer"),
            backend: Some(backend),
            config,
            max_summary_tokens,
            depends: vec![ArcIntern::from("pipeline.scorer.output")],
            provides: vec![ArcIntern::from("pipeline.summarizer.output")],
        }
    }

    /// Summarize text directly — bypasses WorkUnit dispatch.
    /// Useful when the text is already in-hand.
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

        if let Some(ref backend) = self.backend {
            backend.chat_complete(&messages)
        } else {
            let client = LlmClient::with_config(self.config.clone());
            client.chat_complete(&messages)
        }
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

        let summary = if let Some(ref backend) = self.backend {
            backend
                .chat_complete(&messages)
                .map_err(|e| WorkError::Execution(format!("summarizer error: {e}")))?
        } else {
            let client = LlmClient::with_config(self.config.clone());
            client
                .chat_complete(&messages)
                .map_err(|e| WorkError::Execution(format!("summarizer error: {e}")))?
        };

        WorkOutput::typed("summarized", &serde_json::json!({"summary": summary}))
    }
}

impl FieldAccess for Summarizer {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "Summarizer has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "Summarizer has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

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

/// Truncate text at the first sentence-ending punctuation (. ! ?).
fn truncate_at_sentence(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(idx) = trimmed.find(['.', '!', '?']) {
        trimmed[..=idx].trim().to_string()
    } else {
        trimmed.chars().take(120).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_at_sentence_with_period() {
        assert_eq!(truncate_at_sentence("Hello world. More text"), "Hello world.");
    }

    #[test]
    fn test_truncate_at_sentence_with_exclamation() {
        assert_eq!(truncate_at_sentence("Great answer! Follow up"), "Great answer!");
    }

    #[test]
    fn test_truncate_at_sentence_no_punctuation() {
        let result = truncate_at_sentence("Single sentence no punctuation");
        assert!(result.len() <= 120);
        assert_eq!(result, "Single sentence no punctuation");
    }

    #[test]
    fn test_truncate_at_sentence_empty() {
        assert_eq!(truncate_at_sentence(""), "");
    }

    #[test]
    fn test_truncate_at_sentence_whitespace() {
        assert_eq!(truncate_at_sentence("  "), "");
    }

    #[test]
    fn test_result_scorer_name() {
        let config = LlmConfig::new()
            .api_url("http://test".into())
            .model("test-model".into())
            .build();
        let scorer = ResultScorer::new(config, 0.7);
        assert_eq!(scorer.name(), "pipeline.scorer");
    }

    #[test]
    fn test_result_scorer_describable() {
        let config = LlmConfig::new()
            .api_url("http://test".into())
            .model("test-model".into())
            .build();
        let scorer = ResultScorer::new(config, 0.7);
        let desc = scorer.describe();
        assert_eq!(desc["type"], "object");
    }

    #[test]
    fn test_result_scorer_missing_response() {
        let config = LlmConfig::new()
            .api_url("http://test".into())
            .model("test-model".into())
            .build();
        let scorer = ResultScorer::new(config, 0.7);
        let ctx = WorkContext::default();
        let result = scorer.execute(&ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_summarizer_name() {
        let config = LlmConfig::new()
            .api_url("http://test".into())
            .model("test-model".into())
            .build();
        let summarizer = Summarizer::new(config, 50);
        assert_eq!(summarizer.name(), "pipeline.summarizer");
    }

    #[test]
    fn test_summarizer_describable() {
        let config = LlmConfig::new()
            .api_url("http://test".into())
            .model("test-model".into())
            .build();
        let summarizer = Summarizer::new(config, 50);
        let desc = summarizer.describe();
        assert_eq!(desc["type"], "object");
    }

    #[test]
    fn test_summarizer_missing_content() {
        let config = LlmConfig::new()
            .api_url("http://test".into())
            .model("test-model".into())
            .build();
        let summarizer = Summarizer::new(config, 50);
        let ctx = WorkContext::default();
        let result = summarizer.execute(&ctx);
        assert!(result.is_err());
    }
}
