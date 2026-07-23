//! Stage 1: DeterministicPreFilter — command dispatch and PII flagging.
//! No model calls. Two sub-stages: command regex check, PII pattern detection.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use fluent_wvr::prelude::*;
use regex::Regex;

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;

static COMMAND_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[/.](\w[\w-]*)(?:\s+(.*))?$").unwrap());

type CommandHandler = Arc<dyn Fn(&[String]) -> Result<String, String> + Send + Sync>;

pub struct DeterministicPreFilter {
    name: ArcIntern<str>,
    command_registry: HashMap<String, CommandHandler>,
    pii_patterns: Vec<(String, Regex)>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl DeterministicPreFilter {
    pub fn new() -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage1.deterministic"),
            command_registry: default_command_registry(),
            pii_patterns: default_pii_patterns(),
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.stage1.output")],
        }
    }

    #[must_use]
    pub fn with_command(
        mut self,
        name: impl Into<String>,
        handler: CommandHandler,
    ) -> Self {
        self.command_registry.insert(name.into(), handler);
        self
    }

    #[must_use]
    pub fn with_pii_patterns(mut self, patterns: Vec<(String, Regex)>) -> Self {
        self.pii_patterns = patterns;
        self
    }
}

impl Default for DeterministicPreFilter {
    fn default() -> Self {
        Self::new()
    }
}

fn default_command_registry() -> HashMap<String, CommandHandler> {
    let mut registry: HashMap<String, CommandHandler> = HashMap::new();

    registry.insert(
        "help".into(),
        Arc::new(|_args| Ok("Available commands: /help, /stats, /checkpoint <name>".into())),
    );

    registry.insert(
        "stats".into(),
        Arc::new(|_args| Ok("Router statistics not yet available.".into())),
    );

    registry.insert(
        "checkpoint".into(),
        Arc::new(|args| {
            if args.is_empty() {
                Err("usage: /checkpoint <name>".into())
            } else {
                Ok(format!("checkpoint '{}' created.", args[0]))
            }
        }),
    );

    registry
}

fn default_pii_patterns() -> Vec<(String, Regex)> {
    vec![
        (
            "ssn".into(),
            Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
        ),
        (
            "card_number".into(),
            Regex::new(r"\b(?:\d[ -]*?){13,19}\b").unwrap(),
        ),
        (
            "email".into(),
            Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap(),
        ),
        (
            "phone".into(),
            Regex::new(r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b").unwrap(),
        ),
    ]
}

impl WorkUnit for DeterministicPreFilter {
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
        let trimmed = input.trim();

        if let Some(captures) = COMMAND_RE.captures(trimmed) {
            let cmd = captures.get(1).map_or("", |m| m.as_str());
            let args: Vec<String> = captures
                .get(2)
                .map(|m| m.as_str().split_whitespace().map(String::from).collect())
                .unwrap_or_default();

            if let Some(handler) = self.command_registry.get(cmd) {
                match handler(&args) {
                    Ok(result) => {
                        return WorkOutput::typed(
                            "command_dispatched",
                            &StageDecision {
                                stage: PipelineStage::DeterministicPreFilter,
                                verdict: StageVerdict::Rejected,
                                score: Some(1.0),
                                reason: format!("command '{cmd}' executed deterministically"),
                                latency_ms: 0,
                                metadata: serde_json::json!({
                                    "command_result": result,
                                    "command": cmd
                                }),
                            },
                        );
                    }
                    Err(e) => {
                        return WorkOutput::typed(
                            "command_error",
                            &StageDecision {
                                stage: PipelineStage::DeterministicPreFilter,
                                verdict: StageVerdict::Rejected,
                                score: Some(1.0),
                                reason: format!("command '{cmd}' parse error: {e}"),
                                latency_ms: 0,
                                metadata: serde_json::json!({}),
                            },
                        );
                    }
                }
            }

            return WorkOutput::typed(
                "unknown_command",
                &StageDecision {
                    stage: PipelineStage::DeterministicPreFilter,
                    verdict: StageVerdict::Rejected,
                    score: Some(1.0),
                    reason: format!("unknown command: '{cmd}'"),
                    latency_ms: 0,
                    metadata: serde_json::json!({}),
                },
            );
        }

        let mut pii_found: Vec<String> = Vec::new();
        for (class, pattern) in &self.pii_patterns {
            if pattern.is_match(&input) {
                pii_found.push(class.clone());
            }
        }

        WorkOutput::typed(
            "passed",
            &StageDecision {
                stage: PipelineStage::DeterministicPreFilter,
                verdict: StageVerdict::Passed,
                score: Some(1.0),
                reason: if pii_found.is_empty() {
                    "no command, no PII flags".into()
                } else {
                    format!("flagged PII classes: {}", pii_found.join(", "))
                },
                latency_ms: 0,
                metadata: serde_json::json!({ "pii_classes": pii_found }),
            },
        )
    }
}

impl FieldAccess for DeterministicPreFilter {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "DeterministicPreFilter has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "DeterministicPreFilter has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for DeterministicPreFilter {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }
}

impl_component!(DeterministicPreFilter);