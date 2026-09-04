use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use common_core::sync::lock;
use fluent_llm::client::ChatBackend;
use fluent_llm::{ChatMessage, LlmError};
use serde::{Deserialize, Serialize};

use crate::pipeline::RoutingTarget;
use crate::types::{RouterMessage, RouterMessageContent, RouterResponse, Usage};

pub struct TranscriptProvider {
    entries: HashMap<String, String>,
    default_response: String,
}

impl TranscriptProvider {
    pub fn new(entries: HashMap<String, String>) -> Self {
        Self {
            entries,
            default_response: Self::default_pass_response(),
        }
    }

    #[must_use]
    pub fn with_default(mut self, response: String) -> Self {
        self.default_response = response;
        self
    }

    /// Dual-shape pass response: the flat classifier reads `target`, the
    /// tree engine reads `route` — both resolve identically so transcript
    /// tests work whichever engine the config selects.
    fn default_pass_response() -> String {
        serde_json::json!({
            "action": "route",
            "target": "fast",
            "route": "fast",
            "coherence_score": 0.95,
            "coherence": 0.95,
            "safety_score": 0.9,
            "safety": 0.9,
            "complexity": 5,
            "intent": "question",
            "reason": "well-formed factual query",
        })
        .to_string()
    }
}

impl ChatBackend for TranscriptProvider {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        let user_msg = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| {
                let content = m.content.trim();
                if content.is_empty() {
                    None
                } else {
                    Some(content)
                }
            })
            .unwrap_or("");

        if user_msg.is_empty() {
            return Ok(self.default_response.clone());
        }

        Ok(self
            .entries
            .get(user_msg)
            .cloned()
            .unwrap_or_else(|| self.default_response.clone()))
    }
}

pub fn default_transcript() -> TranscriptProvider {
    TranscriptProvider::new(HashMap::new())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockTranscriptEntry {
    pub user_message: String,
    pub classifier_response: String,
    #[serde(default)]
    pub expected_route: Option<String>,
    /// When set, the resolved routing target must have dispatched through this
    /// `model_groups` name (i.e. `RoutingTarget.group` equals it). This is the
    /// intent→model_group assertion hook for the config-synced integration
    /// tests: `expected_route` proves the *route* resolved, this proves the
    /// dispatch went through the *group* the route maps to in the config.
    #[serde(default)]
    pub expect_model_group: Option<String>,
    #[serde(default)]
    pub dispatch_response: Option<String>,
    #[serde(default)]
    pub rejected: bool,
    #[serde(default)]
    pub reject_reason_contains: Option<String>,
}

pub fn load_transcript_file(
    path: impl AsRef<Path>,
) -> Result<Vec<MockTranscriptEntry>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path.as_ref())?;
    Ok(serde_json::from_str(&content)?)
}

pub fn transcript_provider_from_entries(entries: &[MockTranscriptEntry]) -> TranscriptProvider {
    let map: HashMap<String, String> = entries
        .iter()
        .map(|e| (e.user_message.clone(), e.classifier_response.clone()))
        .collect();
    TranscriptProvider::new(map)
}

pub struct MockDispatchContext {
    pub transcripts: Arc<Vec<MockTranscriptEntry>>,
    pub failures: std::sync::Mutex<Vec<String>>,
    /// Model names that should NOT be mocked — these make real LLM calls.
    pub except_models: Vec<String>,
}

impl MockDispatchContext {
    pub fn new(transcripts: Vec<MockTranscriptEntry>, except_models: Vec<String>) -> Self {
        Self {
            transcripts: Arc::new(transcripts),
            failures: std::sync::Mutex::new(Vec::new()),
            except_models,
        }
    }

    /// Reference to the underlying transcript entries.
    pub fn transcripts(&self) -> &[MockTranscriptEntry] {
        &self.transcripts
    }

    /// Returns `true` if the given model name is in the except list
    /// (i.e., should make real LLM calls instead of returning canned responses).
    pub fn is_model_excepted(&self, model: &str) -> bool {
        self.except_models.iter().any(|m| m == model)
    }

    pub fn lookup(&self, user_message: &str) -> Option<&MockTranscriptEntry> {
        let result = self
            .transcripts
            .iter()
            .find(|t| t.user_message == user_message);
        tracing::debug!(target: "router.mock",
            user_message_len = user_message.len(),
            found = result.is_some(),
            "transcript lookup"
        );
        result
    }

    pub fn validate_route(
        &self,
        entry: &MockTranscriptEntry,
        routing_target: Option<&RoutingTarget>,
    ) {
        let result = match (&entry.expected_route, routing_target) {
            (None, None) => Ok(()),
            (Some(expected), Some(actual)) => {
                let actual_route = actual.target_name.as_deref().unwrap_or(&actual.model);
                if actual_route == expected {
                    Ok(())
                } else {
                    Err(format!(
                        "route mismatch for '{}': expected '{}', got '{}' (model={}, url={})",
                        entry.user_message, expected, actual_route, actual.model, actual.url
                    ))
                }
            }
            (Some(expected), None) => Err(format!(
                "route mismatch for '{}': expected '{}', but no routing target was set",
                entry.user_message, expected
            )),
            (None, Some(actual)) => {
                let actual_route = actual.target_name.as_deref().unwrap_or(&actual.model);
                Err(format!(
                    "route mismatch for '{}': expected no route, but got '{}' (model={})",
                    entry.user_message, actual_route, actual.model
                ))
            }
        };

        if let Err(msg) = result {
            lock(&self.failures).push(msg);
        }

        // Independent model-group check: when the entry declares the expected
        // `model_groups` name, verify the resolved target was dispatched
        // through exactly that group (intent → model_group, config-derived).
        if let (Some(expected_group), Some(actual)) = (&entry.expect_model_group, routing_target) {
            let actual_group = actual.group.as_deref().unwrap_or("");
            if actual_group != expected_group {
                lock(&self.failures).push(format!(
                    "group mismatch for '{}': expected model_group '{}', got '{}' (route={}, model={})",
                    entry.user_message,
                    expected_group,
                    actual_group,
                    actual.target_name.as_deref().unwrap_or(""),
                    actual.model,
                ));
            }
        }
    }

    pub fn validate_rejection(&self, entry: &MockTranscriptEntry, reject_reason: &str) {
        if let Some(expected_substr) = &entry.reject_reason_contains {
            if !reject_reason.contains(expected_substr.as_str()) {
                lock(&self.failures).push(format!(
                    "rejection reason mismatch for '{}': expected reason to contain '{}', got '{}'",
                    entry.user_message, expected_substr, reject_reason
                ));
            }
        }
    }

    pub fn dispatch_response(
        &self,
        entry: &MockTranscriptEntry,
        model_name: &str,
    ) -> RouterResponse {
        RouterResponse {
            id: "mock-resp".into(),
            object: "chat.completion".into(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: model_name.to_string(),
            choices: vec![crate::types::RouterChoice {
                index: 0,
                message: RouterMessage {
                    role: "assistant".into(),
                    content: RouterMessageContent::Text(
                        entry.dispatch_response.clone().unwrap_or_default(),
                    ),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage::default(),
        }
    }

    pub fn take_failures(&self) -> Vec<String> {
        std::mem::take(&mut lock(&self.failures))
    }
}
