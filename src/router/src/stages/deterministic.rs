//! Stage 1: DeterministicPreFilter — regex-based blacklist and command dispatch.
//! No model calls. Configurable via `RejectPatterns`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use fluent_wvr::prelude::*;
use regex::Regex;

use crate::config::{PatternEntry, RejectPatterns};
use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
use crate::stages::common::extract_user_message;

static COMMAND_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[/.](\w[\w-]*)(?:\s+(.*))?$").unwrap());

type CommandHandler = Arc<dyn Fn(&[String]) -> Result<String, String> + Send + Sync>;

pub struct DeterministicPreFilter {
    name: ArcIntern<str>,
    command_registry: HashMap<String, CommandHandler>,
    patterns: Vec<CachedPatternEntry>,
    commands_enabled: bool,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

struct CachedPatternEntry {
    name: String,
    http_code: u16,
    error_message: Option<String>,
    patterns: Vec<Regex>,
}

impl DeterministicPreFilter {
    pub fn new() -> Self {
        let patterns: Vec<CachedPatternEntry> = vec![
            ("ssn", vec![r"\b\d{3}-\d{2}-\d{4}\b"]),
            ("card_number", vec![r"\b(?:\d[ -]*?){13,19}\b"]),
            ("email", vec![r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"]),
            ("phone", vec![r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"]),
        ]
        .into_iter()
        .map(|(name, regexes)| {
            let re: Vec<Regex> = regexes
                .iter()
                .filter_map(|r| Regex::new(r).ok())
                .collect();
            CachedPatternEntry {
                name: name.into(),
                http_code: 400,
                error_message: Some("Request contains sensitive personal information".into()),
                patterns: re,
            }
        })
        .collect();

        let mut command_registry: HashMap<String, CommandHandler> = HashMap::new();
        command_registry.insert(
            "help".into(),
            Arc::new(|_args| Ok("Available commands: /help, /stats, /checkpoint <name>".into())),
        );
        command_registry.insert(
            "stats".into(),
            Arc::new(|_args| Ok("Router statistics not yet available.".into())),
        );
        command_registry.insert(
            "checkpoint".into(),
            Arc::new(|args| {
                if args.is_empty() {
                    Err("usage: /checkpoint <name>".into())
                } else {
                    Ok(format!("checkpoint '{}' created.", args[0]))
                }
            }),
        );

        Self {
            name: ArcIntern::from("pipeline.stage1.deterministic"),
            command_registry,
            patterns,
            commands_enabled: true,
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.stage1.output")],
        }
    }

    pub fn from_config(config: &RejectPatterns) -> Self {
        let mut patterns: Vec<CachedPatternEntry> = Vec::new();
        for entry in &config.patterns {
            let re: Vec<Regex> = entry
                .regexes
                .iter()
                .filter_map(|r| Regex::new(r).ok())
                .collect();
            if !re.is_empty() {
                patterns.push(CachedPatternEntry {
                    name: entry.name.clone(),
                    http_code: entry.http_code,
                    error_message: entry.error_message.clone(),
                    patterns: re,
                });
            }
        }

        let command_registry: HashMap<String, CommandHandler> = config
            .commands
            .as_ref()
            .map(|cmd| {
                build_command_registry(&cmd.handlers)
            })
            .unwrap_or_default();
        let commands_enabled = config.commands.is_some();

        Self {
            name: ArcIntern::from("pipeline.stage1.deterministic"),
            command_registry,
            patterns,
            commands_enabled,
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

    #[must_use]
    pub fn with_blacklist(mut self, entries: Vec<PatternEntry>) -> Self {
        for entry in entries {
            let re: Vec<Regex> = entry
                .regexes
                .iter()
                .filter_map(|r| Regex::new(r).ok())
                .collect();
            if !re.is_empty() {
                self.patterns.push(CachedPatternEntry {
                    name: entry.name,
                    http_code: entry.http_code,
                    error_message: entry.error_message,
                    patterns: re,
                });
            }
        }
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
    if !registry.contains_key("checkpoint") {
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

        if self.commands_enabled {
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

                let matched = self.patterns.iter().find(|entry| {
                    entry.patterns.iter().any(|re| re.is_match(trimmed))
                });

                if let Some(entry) = matched {
                    let msg = entry
                        .error_message
                        .clone()
                        .unwrap_or_else(|| format!("blocked by '{}'", entry.name));
                    return WorkOutput::typed(
                        "rejected",
                        &StageDecision {
                            stage: PipelineStage::DeterministicPreFilter,
                            verdict: StageVerdict::Rejected,
                            score: Some(1.0),
                            reason: format!("pattern match '{}': {msg}", entry.name),
                            latency_ms: 0,
                            metadata: serde_json::json!({
                                "blacklist": entry.name,
                                "http_code": entry.http_code
                            }),
                        },
                    );
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
        }

        let mut pii_found: Vec<String> = Vec::new();
        for entry in &self.patterns {
            if entry.patterns.iter().any(|re| re.is_match(&input)) {
                pii_found.push(entry.name.clone());
            }
        }

        if !pii_found.is_empty() {
            return WorkOutput::typed(
                "rejected",
                &StageDecision {
                    stage: PipelineStage::DeterministicPreFilter,
                    verdict: StageVerdict::Rejected,
                    score: Some(1.0),
                    reason: format!("blocked: patterns: {}", pii_found.join(", ")),
                    latency_ms: 0,
                    metadata: serde_json::json!({ "pii_classes": pii_found }),
                },
            );
        }

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
