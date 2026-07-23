//! Stage 4: GuardrailCheck — policy/safety check, distinct from quality gate.
//! Always runs before frontier dispatch; optionally before local agents.

use fluent_wvr::prelude::*;
use guidance_llm::{ChatMessage, LlmClient, LlmConfig};
use serde::{Deserialize, Serialize};

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailResult {
    pub safety_score: f64,
    pub categories: Vec<String>,
    pub blocked: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

pub struct GuardrailCheck {
    name: ArcIntern<str>,
    client: LlmClient,
    #[allow(dead_code)]
    blocked_topics: Vec<String>,
    #[allow(dead_code)]
    check_local_agents: bool,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl GuardrailCheck {
    pub fn new(
        config: LlmConfig,
        blocked_topics: Vec<String>,
        check_local_agents: bool,
    ) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage4.guardrail"),
            client: LlmClient::with_config(config),
            blocked_topics,
            check_local_agents,
            depends: vec![ArcIntern::from("pipeline.stage3.output")],
            provides: vec![ArcIntern::from("pipeline.stage4.output")],
        }
    }
}

const GUARDRAIL_SYSTEM_PROMPT: &str = r#"You are a content safety classifier. Analyze the user's input for policy violations and output JSON:
{
  "safety_score": 0.0-1.0 (1.0 = completely safe, 0.0 = dangerous),
  "categories": ["violence", "hate", "self_harm", "illegal", "pii_leak", "none"],
  "blocked": true/false,
  "reason": "optional explanation if blocked"
}
Block if safety_score < 0.5. Only output JSON."#;

impl WorkUnit for GuardrailCheck {
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
                content: GUARDRAIL_SYSTEM_PROMPT.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: input,
            },
        ];

        let response = self
            .client
            .chat_complete(&messages)
            .map_err(|e| WorkError::Execution(format!("guardrail error: {e}")))?;

        let guard_result: GuardrailResult = serde_json::from_str(&response)
            .map_err(|e| WorkError::Execution(format!("guardrail parse error: {e}")))?;

        let passed = !guard_result.blocked && guard_result.safety_score >= 0.5;

        WorkOutput::typed(
            "checked",
            &StageDecision {
                stage: PipelineStage::GuardrailCheck,
                verdict: if passed {
                    StageVerdict::Passed
                } else {
                    StageVerdict::Rejected
                },
                score: Some(guard_result.safety_score),
                reason: if passed {
                    "guardrail passed".into()
                } else {
                    format!(
                        "blocked: {}",
                        guard_result
                            .reason
                            .as_deref()
                            .unwrap_or("policy violation")
                    )
                },
                latency_ms: 0,
                metadata: serde_json::json!({
                    "safety_score": guard_result.safety_score,
                    "categories": guard_result.categories,
                    "blocked": guard_result.blocked,
                }),
            },
        )
    }
}

impl FieldAccess for GuardrailCheck {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "GuardrailCheck has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "GuardrailCheck has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for GuardrailCheck {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(GuardrailCheck);