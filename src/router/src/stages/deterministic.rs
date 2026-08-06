use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use fluent_wvr::prelude::*;
use regex::Regex;

use crate::config::RejectPatterns;
use crate::filters::injection_detect::InjectionDetectFilter;
use crate::filters::regex_filter::RegexFilter;
use crate::filters::{DeterministicFilterEngine, FilterContext, FilterDecision};
use crate::pipeline_types::{
    PiiVerdict, PipelineStage, StageDecision, StageDecisionProducer, StageMetadata, StageVerdict,
};
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
        let filter_engine = builtin_filter_engine();

        Self {
            name: ArcIntern::from("pipeline.stage1.deterministic"),
            command_registry,
            filter_engine,
            commands_enabled: true,
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.stage1.output")],
        }
    }

    /// Run the stage-1 filter engine over output text — the response
    /// re-scan used by the escalation ladder's filter mode
    /// (`dispatch::escalation`.  `None` means the output is clean
    /// (accepted).
    pub(crate) fn scan_output(&self, text: &str) -> Option<FilterDecision> {
        let ctx = FilterContext::frontier(text.to_string());
        self.filter_engine.evaluate(&ctx)
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
    pub fn with_command(mut self, name: impl Into<String>, handler: CommandHandler) -> Self {
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

/// The stage-1 filter engine over the builtin PII/injection patterns — the
/// canonical engine shared by `DeterministicPreFilter::new` and the
/// escalation ladder's filter-mode response re-scan.
pub(crate) fn builtin_filter_engine() -> DeterministicFilterEngine {
    use crate::config::{PatternEntry, RejectPatterns};

    let pii_patterns = fluent_llm::pii_patterns::pii_patterns();
    let pii_map: std::collections::HashMap<&str, &str> = pii_patterns
        .iter()
        .map(|p| (p.name, p.regex.as_str()))
        .collect();
    fn pii_regex<'a>(map: &'a HashMap<&str, &str>, key: &str, fallback: &'a str) -> String {
        map.get(key).copied().unwrap_or(fallback).to_string()
    }
    let map = &pii_map;

    let patterns = vec![
        PatternEntry {
            name: "ssn".into(),
            outcome: crate::config::FilterOutcome::OutputFilter,
            filter_action: Some(crate::config::FilterAction::Redact),
            confidence_gate: crate::config::ConfidenceGate::None,
            scope: vec![crate::config::FilterScope::Any],
            http_code: 422,
            error_message: Some("SSN detected".into()),
            regexes: vec![pii_regex(map, "ssn", r"\b\d{3}-\d{2}-\d{4}\b")],
        },
        PatternEntry {
            name: "card_number".into(),
            outcome: crate::config::FilterOutcome::OutputFilter,
            filter_action: Some(crate::config::FilterAction::Redact),
            confidence_gate: crate::config::ConfidenceGate::LuhnValid,
            scope: vec![crate::config::FilterScope::Any],
            http_code: 422,
            error_message: Some("Credit card number detected".into()),
            regexes: vec![r"\b(?:\d[ -]*?){13,19}\b".into()],
        },
        PatternEntry {
            name: "email".into(),
            outcome: crate::config::FilterOutcome::OutputFilter,
            filter_action: Some(crate::config::FilterAction::Anonymize),
            confidence_gate: crate::config::ConfidenceGate::None,
            scope: vec![crate::config::FilterScope::Any],
            http_code: 422,
            error_message: Some("Email detected".into()),
            regexes: vec![pii_regex(
                map,
                "email",
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            )],
        },
        PatternEntry {
            name: "phone".into(),
            outcome: crate::config::FilterOutcome::OutputFilter,
            filter_action: Some(crate::config::FilterAction::Anonymize),
            confidence_gate: crate::config::ConfidenceGate::None,
            scope: vec![crate::config::FilterScope::Any],
            http_code: 422,
            error_message: Some("Phone number detected".into()),
            regexes: vec![pii_regex(
                map,
                "phone",
                r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            )],
        },
        PatternEntry {
            name: "api_key".into(),
            outcome: crate::config::FilterOutcome::HardReject,
            filter_action: None,
            confidence_gate: crate::config::ConfidenceGate::None,
            scope: vec![crate::config::FilterScope::Any],
            http_code: 422,
            error_message: Some("API key detected".into()),
            regexes: vec![pii_regex(
                map,
                "api_key",
                r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*[^\s]{8,}",
            )],
        },
    ];

    let builtin = RejectPatterns {
        patterns,
        commands: None,
    };
    let mut engine = DeterministicFilterEngine::new();
    for entry in &builtin.patterns {
        if let Some(filter) = RegexFilter::from_entry(entry) {
            engine.add_filter(Box::new(filter));
        }
    }
    engine.add_filter(Box::new(InjectionDetectFilter::new(0.30)));
    engine
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

impl DeterministicPreFilter {
    /// Produce the stage decision and its `WorkOutput` message (M5.4). The
    /// typed `StageDecision` is built here once and consumed directly by the
    /// orchestrator's typed handoff; `WorkUnit::execute` wraps the same result
    /// into `WorkOutput` for the composition path.
    fn decide(&self, ctx: &WorkContext) -> Result<(String, StageDecision), WorkError> {
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
                            crate::audit::emit(
                                "filter",
                                serde_json::json!({
                                    "stage": "deterministic",
                                    "verdict": "command_dispatched",
                                    "command": cmd,
                                }),
                            );
                            let mut metadata = StageMetadata::default();
                            metadata.set_command_result(&result);
                            metadata.insert("command", serde_json::Value::String(cmd.into()));
                            return Ok((
                                "command_dispatched".into(),
                                StageDecision {
                                    stage: PipelineStage::DeterministicPreFilter,
                                    verdict: StageVerdict::Rejected,
                                    score: Some(1.0),
                                    reason: format!("command '{cmd}' executed deterministically"),
                                    latency_ms: 0,
                                    metadata: metadata.into_value(),
                                },
                            ));
                        }
                        Err(e) => {
                            tracing::warn!(target: "router.pipeline.stage1", command = %cmd, error = %e, "command handler error");
                            crate::audit::emit(
                                "filter",
                                serde_json::json!({
                                    "stage": "deterministic",
                                    "verdict": "command_error",
                                    "command": cmd,
                                    "error": e,
                                }),
                            );
                            return Ok((
                                "command_error".into(),
                                StageDecision {
                                    stage: PipelineStage::DeterministicPreFilter,
                                    verdict: StageVerdict::Rejected,
                                    score: Some(1.0),
                                    reason: format!("command '{cmd}' parse error: {e}"),
                                    latency_ms: 0,
                                    metadata: serde_json::json!({}),
                                },
                            ));
                        }
                    }
                }

                // Check filter engine for commands that matched pattern
                let filter_ctx = FilterContext::pipeline(trimmed.to_string());
                if let Some(FilterDecision::HardReject { pattern, message }) =
                    self.filter_engine.evaluate(&filter_ctx)
                {
                    tracing::info!(target: "router.pipeline.stage1", pattern = %pattern, "hard reject on command");
                    crate::audit::emit(
                        "filter",
                        serde_json::json!({
                            "stage": "deterministic",
                            "verdict": "hard_reject",
                            "pattern": pattern,
                            "message": message,
                        }),
                    );
                    return Ok((
                        "rejected".into(),
                        StageDecision {
                            stage: PipelineStage::DeterministicPreFilter,
                            verdict: StageVerdict::Rejected,
                            score: Some(1.0),
                            reason: format!("pattern match '{pattern}': {message}"),
                            latency_ms: 0,
                            metadata: serde_json::json!({ "blacklist": pattern, "http_code": 422 }),
                        },
                    ));
                }

                tracing::info!(target: "router.pipeline.stage1", command = %cmd, "unknown command rejected");
                crate::audit::emit(
                    "filter",
                    serde_json::json!({
                        "stage": "deterministic",
                        "verdict": "unknown_command",
                        "command": cmd,
                    }),
                );
                return Ok((
                    "unknown_command".into(),
                    StageDecision {
                        stage: PipelineStage::DeterministicPreFilter,
                        verdict: StageVerdict::Rejected,
                        score: Some(1.0),
                        reason: format!("unknown command: '{cmd}'"),
                        latency_ms: 0,
                        metadata: serde_json::json!({}),
                    },
                ));
            }
        }

        // Run filter engine on the input
        let filter_ctx = FilterContext::pipeline(input.clone());
        if let Some(decision) = self.filter_engine.evaluate(&filter_ctx) {
            match decision {
                FilterDecision::HardReject { pattern, message } => {
                    tracing::info!(target: "router.pipeline.stage1", pattern = %pattern, "hard reject");
                    crate::audit::emit(
                        "filter",
                        serde_json::json!({
                            "stage": "deterministic",
                            "verdict": "hard_reject",
                            "pattern": pattern,
                            "message": message,
                        }),
                    );
                    return Ok((
                        "rejected".into(),
                        StageDecision {
                            stage: PipelineStage::DeterministicPreFilter,
                            verdict: StageVerdict::Rejected,
                            score: Some(1.0),
                            reason: format!("pattern match '{pattern}': {message}"),
                            latency_ms: 0,
                            metadata: serde_json::json!({ "blacklist": pattern, "http_code": 422 }),
                        },
                    ));
                }
                FilterDecision::OutputFilter {
                    action,
                    matched_pattern,
                    codewords,
                    matches,
                } => {
                    tracing::info!(target: "router.pipeline.stage1", pattern = %matched_pattern, action = ?action, match_count = matches.len(), "output_filter flagged");
                    crate::audit::emit(
                        "filter",
                        serde_json::json!({
                            "stage": "deterministic",
                            "verdict": "output_filter",
                            "pattern": matched_pattern,
                            "action": action,
                            "match_count": matches.len(),
                        }),
                    );
                    let reason = format!("PII flagged for output filtering: {matched_pattern}");
                    let mut metadata = StageMetadata::default();
                    metadata.set_pii_filter(&PiiVerdict {
                        pattern: matched_pattern,
                        action,
                        codewords,
                        matches,
                    });
                    return Ok((
                        "output_filter_flagged".into(),
                        StageDecision {
                            stage: PipelineStage::DeterministicPreFilter,
                            verdict: StageVerdict::Passed,
                            score: Some(1.0),
                            reason,
                            latency_ms: 0,
                            metadata: metadata.into_value(),
                        },
                    ));
                }
                FilterDecision::SoftRedirect { .. } => {}
            }
        }

        tracing::debug!(target: "router.pipeline.stage1", "passed — no command, no PII");
        crate::audit::emit(
            "filter",
            serde_json::json!({
                "stage": "deterministic",
                "verdict": "passed",
            }),
        );
        Ok((
            "passed".into(),
            StageDecision {
                stage: PipelineStage::DeterministicPreFilter,
                verdict: StageVerdict::Passed,
                score: Some(1.0),
                reason: "no command, no PII flags".into(),
                latency_ms: 0,
                metadata: serde_json::json!({ "pii_classes": <Vec<String>>::new() }),
            },
        ))
    }
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
        let (message, decision) = self.decide(ctx)?;
        WorkOutput::typed(message, &decision)
    }
}

impl StageDecisionProducer for DeterministicPreFilter {
    fn stage_kind(&self) -> PipelineStage {
        PipelineStage::DeterministicPreFilter
    }

    fn evaluate(
        &self,
        ctx: &WorkContext,
        _prior: &[StageDecision],
    ) -> Result<StageDecision, WorkError> {
        Ok(self.decide(ctx)?.1)
    }
}

impl_fieldless!(DeterministicPreFilter);

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
