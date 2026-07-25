use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use fluent_wvr::prelude::*;
use regex::Regex;

use crate::filters::{DeterministicFilterEngine, FilterContext, FilterDecision};
use crate::filters::regex_filter::RegexFilter;
use crate::config::RejectPatterns;
use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;

static COMMAND_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[/.,](\w[\w-]*)(?:\s+(.*))?$").unwrap());

type CommandHandler = Arc<dyn Fn(&[String]) -> Result<String, String> + Send + Sync>;

pub struct DeterministicPreFilter {
    name: ArcIntern<str>,
    command_registry: HashMap<String, CommandHandler>,
    filter_engine: DeterministicFilterEngine,
    commands_enabled: bool,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl DeterministicPreFilter {
    fn builtin_commands() -> HashMap<String, CommandHandler> {
        let mut cmds: HashMap<String, CommandHandler> = HashMap::new();
        cmds.insert(
            "help".into(),
            Arc::new(|_args| Ok("Available commands: /help, /stats, /checkpoint <name>".into())),
        );
        cmds.insert(
            "stats".into(),
            Arc::new(|_args| Ok("Router statistics not yet available.".into())),
        );
        cmds.insert(
            "checkpoint".into(),
            Arc::new(|args| {
                if args.is_empty() {
                    Err("usage: /checkpoint <name>".into())
                } else {
                    Ok(format!("checkpoint '{}' created.", args[0]))
                }
            }),
        );
        cmds
    }

    pub fn new() -> Self {
        let command_registry = Self::builtin_commands();

        Self {
            name: ArcIntern::from("pipeline.stage1.deterministic"),
            command_registry,
            filter_engine: DeterministicFilterEngine::new(),
            commands_enabled: true,
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.stage1.output")],
        }
    }

    pub fn from_config(config: &RejectPatterns) -> Self {
        let mut engine = DeterministicFilterEngine::new();
        for entry in &config.patterns {
            if let Some(filter) = RegexFilter::from_entry(entry) {
                engine.add_filter(Box::new(filter));
            }
        }

        let mut command_registry = Self::builtin_commands();
        if let Some(ref cmd) = config.commands {
            for (name, handler) in build_command_registry(&cmd.handlers) {
                command_registry.insert(name, handler);
            }
        }

        Self {
            name: ArcIntern::from("pipeline.stage1.deterministic"),
            command_registry,
            filter_engine: engine,
            commands_enabled: true,
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
        self.commands_enabled = true;
        self
    }
}

impl Default for DeterministicPreFilter {
    fn default() -> Self {
        Self::new()
    }
}

fn build_command_registry(handlers: &HashMap<String, String>) -> HashMap<String, CommandHandler> {
    let mut registry: HashMap<String, CommandHandler> = HashMap::new();
    for (cmd, template) in handlers {
        let template = template.clone();
        registry.insert(
            cmd.clone(),
            Arc::new(move |args| {
                let mut result = template.clone();
                for (i, arg) in args.iter().enumerate() {
                    result = result.replace(&format!("${}", i + 1), arg);
                }
                Ok(result)
            }),
        );
    }
    registry
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

        tracing::debug!(target: "router.pipeline.stage1", input_len = input.len(), "deterministic pre-filter");

        if self.commands_enabled {
            if let Some(captures) = COMMAND_RE.captures(trimmed) {
                let cmd = captures.get(1).map_or("", |m| m.as_str());
                let args: Vec<String> = captures
                    .get(2)
                    .map(|m| m.as_str().split_whitespace().map(String::from).collect())
                    .unwrap_or_default();

                tracing::info!(target: "router.pipeline.stage1", command = %cmd, args = ?args, "command detected");

                if let Some(handler) = self.command_registry.get(cmd) {
                    match handler(&args) {
                        Ok(result) => {
                            tracing::info!(target: "router.pipeline.stage1", command = %cmd, "command dispatched");
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
                            tracing::warn!(target: "router.pipeline.stage1", command = %cmd, error = %e, "command handler error");
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

                // Check filter engine for commands that matched pattern
                let filter_ctx = FilterContext {
                    user_message: trimmed.to_string(),
                    is_frontier_bound: false,
                };
                if let Some(decision) = self.filter_engine.evaluate(&filter_ctx) {
                    if let FilterDecision::HardReject { pattern, message } = decision {
                        tracing::info!(target: "router.pipeline.stage1", pattern = %pattern, "hard reject on command");
                        return WorkOutput::typed(
                            "rejected",
                            &StageDecision {
                                stage: PipelineStage::DeterministicPreFilter,
                                verdict: StageVerdict::Rejected,
                                score: Some(1.0),
                                reason: format!("pattern match '{pattern}': {message}"),
                                latency_ms: 0,
                                metadata: serde_json::json!({ "blacklist": pattern, "http_code": 422 }),
                            },
                        );
                    }
                }

                tracing::info!(target: "router.pipeline.stage1", command = %cmd, "unknown command rejected");
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
        }

        // Run filter engine on the input
        let filter_ctx = FilterContext {
            user_message: input.clone(),
            is_frontier_bound: false,
        };
        if let Some(decision) = self.filter_engine.evaluate(&filter_ctx) {
            match decision {
                FilterDecision::HardReject { pattern, message } => {
                    tracing::info!(target: "router.pipeline.stage1", pattern = %pattern, "hard reject");
                    return WorkOutput::typed(
                        "rejected",
                        &StageDecision {
                            stage: PipelineStage::DeterministicPreFilter,
                            verdict: StageVerdict::Rejected,
                            score: Some(1.0),
                            reason: format!("pattern match '{pattern}': {message}"),
                            latency_ms: 0,
                            metadata: serde_json::json!({ "blacklist": pattern, "http_code": 422 }),
                        },
                    );
                }
                FilterDecision::OutputFilter { action, matched_pattern, codewords, matches } => {
                    tracing::info!(target: "router.pipeline.stage1", pattern = %matched_pattern, action = ?action, match_count = matches.len(), "output_filter flagged");
                    return WorkOutput::typed(
                        "output_filter_flagged",
                        &StageDecision {
                            stage: PipelineStage::DeterministicPreFilter,
                            verdict: StageVerdict::Passed,
                            score: Some(1.0),
                            reason: format!("PII flagged for output filtering: {matched_pattern}"),
                            latency_ms: 0,
                            metadata: serde_json::json!({
                                "pii_filter": {
                                    "pattern": matched_pattern,
                                    "action": action,
                                    "codewords": codewords,
                                    "matches": matches,
                                }
                            }),
                        },
                    );
                }
                FilterDecision::SoftRedirect { .. } => {}
            }
        }

        tracing::debug!(target: "router.pipeline.stage1", "passed — no command, no PII");
        WorkOutput::typed(
            "passed",
            &StageDecision {
                stage: PipelineStage::DeterministicPreFilter,
                verdict: StageVerdict::Passed,
                score: Some(1.0),
                reason: "no command, no PII flags".into(),
                latency_ms: 0,
                metadata: serde_json::json!({ "pii_classes": <Vec<String>>::new() }),
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
