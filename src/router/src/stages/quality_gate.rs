//! Stage 2: QualityGate — classifies intent + prompt quality using a fast local model.
//! Returns `Passed` if coherence >= threshold; `Rejected` otherwise.

use fluent_wvr::prelude::*;
use guidance_llm::{ChatMessage, LlmClient, LlmConfig};
use serde::{Deserialize, Serialize};

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityClassification {
    pub intent: String,
    pub coherence_score: f64,
    pub is_code: bool,
    pub language: String,
}

pub struct QualityGate {
    name: ArcIntern<str>,
    client: LlmClient,
    quality_threshold: f64,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl QualityGate {
    pub fn new(config: LlmConfig, quality_threshold: f64) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage2.quality_gate"),
            client: LlmClient::with_config(config),
            quality_threshold,
            depends: vec![ArcIntern::from("pipeline.stage1.output")],
            provides: vec![ArcIntern::from("pipeline.stage2.output")],
        }
    }
}

const QUALITY_GATE_SYSTEM_PROMPT: &str = r#"You are a prompt quality classifier. Analyze the user's input and output JSON:
{
  "intent": "question" | "command" | "creative" | "code" | "garbage" | "chitchat",
  "coherence_score": 0.0-1.0,
  "is_code": true/false,
  "language": "english" | "other" | "code"
}
Coherence < 0.3 is garbage/incoherent input. Only output JSON."#;

impl WorkUnit for QualityGate {
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
        let input = extract_user_message(ctx)?;

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: QUALITY_GATE_SYSTEM_PROMPT.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: input,
            },
        ];

        let response = self
            .client
            .chat_complete(&messages)
            .map_err(|e| WorkError::Execution(format!("quality gate error: {e}")))?;

        let classification: QualityClassification = serde_json::from_str(&response)
            .map_err(|e| WorkError::Execution(format!("quality gate parse error: {e}")))?;

        let passed = classification.intent != "garbage"
            && classification.coherence_score >= self.quality_threshold;

        WorkOutput::typed(
            "classified",
            &StageDecision {
                stage: PipelineStage::QualityGate,
                verdict: if passed {
                    StageVerdict::Passed
                } else {
                    StageVerdict::Rejected
                },
                score: Some(classification.coherence_score),
                reason: if passed {
                    format!(
                        "intent={}, coherence={:.2}",
                        classification.intent, classification.coherence_score
                    )
                } else {
                    format!(
                        "rejected: intent={}, coherence={:.2}",
                        classification.intent, classification.coherence_score
                    )
                },
                latency_ms: 0,
                metadata: serde_json::json!({
                    "intent": classification.intent,
                    "coherence": classification.coherence_score,
                    "is_code": classification.is_code,
                    "language": classification.language,
                }),
            },
        )
    }
}

impl FieldAccess for QualityGate {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "QualityGate has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "QualityGate has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for QualityGate {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(QualityGate);