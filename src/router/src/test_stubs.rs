use std::collections::VecDeque;
use std::sync::Mutex;

use fluent_wvr::prelude::*;
use guidance_llm::client::ChatBackend;
use guidance_llm::{ChatMessage, LlmError};

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};

pub struct StubChatBackend {
    responses: Mutex<VecDeque<String>>,
}

impl StubChatBackend {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }

    pub fn always(response: impl Into<String>) -> Self {
        Self::new(vec![response.into()])
    }
}

impl ChatBackend for StubChatBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        let mut queue = self.responses.lock().unwrap_or_else(|e| e.into_inner());
        queue.pop_front().ok_or(LlmError::NoResponse)
    }
}

// ── Test stage stubs for DAG pipeline tests ──────────────────────────────

/// A minimal `WorkUnit` stage that always passes, returning a
/// `StageDecision` with `Passed` verdict and a configurable reason.
pub struct SimplePassStage {
    name: ArcIntern<str>,
    reason: String,
}

impl SimplePassStage {
    pub fn new(stage_name: &str, reason: &str) -> Self {
        Self {
            name: ArcIntern::from(stage_name.to_string()),
            reason: reason.to_string(),
        }
    }
}

impl WorkUnit for SimplePassStage {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        WorkOutput::typed(
            "passed",
            &StageDecision {
                stage: PipelineStage::Router,
                verdict: StageVerdict::Passed,
                score: None,
                reason: self.reason.clone(),
                latency_ms: 0,
                metadata: serde_json::Value::Object(Default::default()),
            },
        )
    }
}

impl FieldAccess for SimplePassStage {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "SimplePassStage has no configurable fields".into(),
        ))
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "SimplePassStage has no configurable fields".into(),
        ))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for SimplePassStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}
impl_component!(SimplePassStage);

/// A `WorkUnit` stage that fails N times then succeeds.  Used by
/// `RetryClassifier` tests.
pub struct FailingStage {
    name: ArcIntern<str>,
    failures_remaining: Mutex<usize>,
}

impl FailingStage {
    pub fn new(stage_name: &str, failure_count: usize) -> Self {
        Self {
            name: ArcIntern::from(stage_name.to_string()),
            failures_remaining: Mutex::new(failure_count),
        }
    }
}

impl WorkUnit for FailingStage {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let mut remaining = self.failures_remaining.lock().unwrap();
        if *remaining > 0 {
            *remaining -= 1;
            WorkOutput::typed(
                "fallback",
                &StageDecision {
                    stage: PipelineStage::Classifier,
                    verdict: StageVerdict::Passed,
                    score: Some(1.0),
                    reason: format!("parse error: attempt would be {} more failures", *remaining + 1),
                    latency_ms: 0,
                    metadata: serde_json::json!({"fallback": true}),
                },
            )
        } else {
            WorkOutput::typed(
                "classified",
                &StageDecision {
                    stage: PipelineStage::Classifier,
                    verdict: StageVerdict::Passed,
                    score: Some(0.95),
                    reason: "intent=code, action=route".into(),
                    latency_ms: 0,
                    metadata: serde_json::json!({"intent": "code", "fallback": false}),
                },
            )
        }
    }
}

impl FieldAccess for FailingStage {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound("FailingStage has no configurable fields".into()))
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound("FailingStage has no configurable fields".into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for FailingStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}
impl_component!(FailingStage);
