//! Stage 3: PlanningRefinementAgent — restructures/clarifies prompts before
//! expensive work begins. Optional (config-driven). Uses a local model.

use fluent_wvr::prelude::*;
use guidance_llm::{ChatMessage, LlmClient, LlmConfig};
use serde::{Deserialize, Serialize};

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningResult {
    pub needs_restructuring: bool,
    pub complexity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedPrompt {
    pub original_intent: String,
    pub refined_intent: String,
    pub refined_prompt: String,
    pub subtasks: Vec<String>,
    pub clarity_score: f64,
}

pub struct PlanningRefinementAgent {
    name: ArcIntern<str>,
    client: LlmClient,
    enabled: bool,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl PlanningRefinementAgent {
    pub fn new(config: LlmConfig, enabled: bool) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage3.planning"),
            client: LlmClient::with_config(config),
            enabled,
            depends: vec![ArcIntern::from("pipeline.stage2.output")],
            provides: vec![ArcIntern::from("pipeline.stage3.output")],
        }
    }
}

const PLANNING_SYSTEM_PROMPT: &str = r#"You are a prompt restructuring assistant. Given a user's input, determine if it needs restructuring and output JSON:
{
  "original_intent": "brief description of user's intent",
  "refined_intent": "clarified description after restructuring",
  "refined_prompt": "the restructured prompt text",
  "subtasks": ["subtask 1", "subtask 2"],
  "clarity_score": 0.0-1.0
}
If the prompt is already clear and well-structured, return the original unchanged with clarity_score >= 0.8. Only output JSON."#;

impl WorkUnit for PlanningRefinementAgent {
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
        if !self.enabled {
            return WorkOutput::typed(
                "skipped",
                &StageDecision {
                    stage: PipelineStage::PlanningRefinement,
                    verdict: StageVerdict::Skipped,
                    score: None,
                    reason: "planning disabled in config".into(),
                    latency_ms: 0,
                    metadata: serde_json::json!({}),
                },
            );
        }

        let input = extract_user_message(ctx)?;

        let screening = screen_prompt(&input);
        if !screening.needs_restructuring {
            return WorkOutput::typed(
                "passed",
                &StageDecision {
                    stage: PipelineStage::PlanningRefinement,
                    verdict: StageVerdict::Passed,
                    score: Some(screening.complexity_score),
                    reason: "prompt does not need restructuring".into(),
                    latency_ms: 0,
                    metadata: serde_json::json!({
                        "complexity": screening.complexity_score
                    }),
                },
            );
        }

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: PLANNING_SYSTEM_PROMPT.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: input,
            },
        ];

        let response = self
            .client
            .chat_complete(&messages)
            .map_err(|e| WorkError::Execution(format!("planning error: {e}")))?;

        let refined: RefinedPrompt = serde_json::from_str(&response)
            .map_err(|e| WorkError::Execution(format!("planning parse error: {e}")))?;

        WorkOutput::typed(
            "rerouted",
            &StageDecision {
                stage: PipelineStage::PlanningRefinement,
                verdict: StageVerdict::Rerouted,
                score: Some(refined.clarity_score),
                reason: format!(
                    "restructured: {} -> {}",
                    refined.original_intent, refined.refined_intent
                ),
                latency_ms: 0,
                metadata: serde_json::json!({
                    "rewritten_request": refined.refined_prompt,
                    "subtasks": refined.subtasks,
                    "original_intent": refined.original_intent,
                    "refined_intent": refined.refined_intent,
                }),
            },
        )
    }
}

fn screen_prompt(input: &str) -> ScreeningResult {
    let word_count = input.split_whitespace().count();
    let needs_restructuring = word_count > 100 || (input.contains('\n') && word_count > 50);
    let complexity_score = if needs_restructuring { 0.4 } else { 0.85 };

    ScreeningResult {
        needs_restructuring,
        complexity_score,
    }
}

impl FieldAccess for PlanningRefinementAgent {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "PlanningRefinementAgent has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "PlanningRefinementAgent has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for PlanningRefinementAgent {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(PlanningRefinementAgent);