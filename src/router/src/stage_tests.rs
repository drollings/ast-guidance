#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fluent_wvr::prelude::*;

    use crate::config::{
        ConfidenceGate, FilterAction, FilterOutcome, FilterScope, PatternEntry, RejectPatterns,
    };
    use crate::pipeline::PipelineOrchestrator;
    use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
    use crate::stages::deterministic::DeterministicPreFilter;

    fn make_pii_filter() -> DeterministicPreFilter {
        let patterns = RejectPatterns {
            patterns: vec![
                PatternEntry {
                    name: "ssn".into(),
                    outcome: FilterOutcome::OutputFilter,
                    filter_action: Some(FilterAction::Redact),
                    confidence_gate: ConfidenceGate::None,
                    scope: vec![FilterScope::Any],
                    http_code: 422,
                    error_message: Some("SSN detected".into()),
                    regexes: vec![r"\b\d{3}-\d{2}-\d{4}\b".into()],
                },
                PatternEntry {
                    name: "card_number".into(),
                    outcome: FilterOutcome::OutputFilter,
                    filter_action: Some(FilterAction::Redact),
                    confidence_gate: ConfidenceGate::LuhnValid,
                    scope: vec![FilterScope::Any],
                    http_code: 422,
                    error_message: Some("Credit card detected".into()),
                    regexes: vec![r"\b(?:\d[ -]*?){13,19}\b".into()],
                },
                PatternEntry {
                    name: "email".into(),
                    outcome: FilterOutcome::OutputFilter,
                    filter_action: Some(FilterAction::Anonymize),
                    confidence_gate: ConfidenceGate::None,
                    scope: vec![FilterScope::Any],
                    http_code: 422,
                    error_message: Some("Email detected".into()),
                    regexes: vec![r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b".into()],
                },
                PatternEntry {
                    name: "phone".into(),
                    outcome: FilterOutcome::OutputFilter,
                    filter_action: Some(FilterAction::Anonymize),
                    confidence_gate: ConfidenceGate::None,
                    scope: vec![FilterScope::Any],
                    http_code: 422,
                    error_message: Some("Phone detected".into()),
                    regexes: vec![
                        r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b".into(),
                    ],
                },
                PatternEntry {
                    name: "api_key".into(),
                    outcome: FilterOutcome::HardReject,
                    filter_action: None,
                    confidence_gate: ConfidenceGate::None,
                    scope: vec![FilterScope::Any],
                    http_code: 422,
                    error_message: Some("API key detected".into()),
                    regexes: vec![
                        r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*[^\s]{8,}".into(),
                    ],
                },
            ],
            commands: None,
        };
        DeterministicPreFilter::from_config(&patterns)
    }

    fn make_ctx(user_text: &str) -> WorkContext {
        let request_json = serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": user_text}
            ]
        });
        let mut ctx = WorkContext::default();
        ctx.set_structured("request", &request_json);
        ctx
    }

    // ── Stage 1: DeterministicPreFilter ──────────────────────────────────────

    #[test]
    fn test_deterministic_command_help() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("/help");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.stage, PipelineStage::DeterministicPreFilter);
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("help"));
    }

    #[test]
    fn test_deterministic_command_stats() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("/stats");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("stats"));
    }

    #[test]
    fn test_deterministic_command_checkpoint_with_arg() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("/checkpoint my-snapshot");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("checkpoint"));
        assert!(decision
            .metadata
            .get("command_result")
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.contains("my-snapshot")));
    }

    #[test]
    fn test_deterministic_command_checkpoint_no_arg() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("/checkpoint");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("usage"));
    }

    #[test]
    fn test_deterministic_unknown_command() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("/nonexistent arg1 arg2");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("unknown command"));
    }

    #[test]
    fn test_deterministic_dot_command() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx(".help");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("help"));
    }

    #[test]
    fn test_deterministic_prose_passes() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("What is the capital of France?");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert!(decision.reason.contains("no command, no PII flags"));
    }

    #[test]
    fn test_deterministic_pii_email_detected() {
        let filter = make_pii_filter();
        let ctx = make_ctx("My email is user@example.com");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert_eq!(output.message, "output_filter_flagged");
        let pii_filter = decision
            .metadata
            .get("pii_filter")
            .expect("pii_filter metadata");
        assert_eq!(pii_filter["pattern"], "email");
    }

    #[test]
    fn test_deterministic_pii_ssn_detected() {
        let filter = make_pii_filter();
        let ctx = make_ctx("My SSN is 123-45-6789");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let pii_filter = decision
            .metadata
            .get("pii_filter")
            .expect("pii_filter metadata");
        assert_eq!(pii_filter["pattern"], "ssn");
    }

    #[test]
    fn test_deterministic_pii_card_number_detected() {
        let filter = make_pii_filter();
        let ctx = make_ctx("card: 4111-1111-1111-1111");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let pii_filter = decision
            .metadata
            .get("pii_filter")
            .expect("pii_filter metadata");
        assert_eq!(pii_filter["pattern"], "card_number");
    }

    #[test]
    fn test_deterministic_pii_phone_detected() {
        let filter = make_pii_filter();
        let ctx = make_ctx("Call me at (555) 123-4567");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let pii_filter = decision
            .metadata
            .get("pii_filter")
            .expect("pii_filter metadata");
        assert_eq!(pii_filter["pattern"], "phone");
    }

    #[test]
    fn test_deterministic_multiple_pii_first_match_wins() {
        let filter = make_pii_filter();
        let ctx = make_ctx("My email is user@example.com and my SSN is 123-45-6789");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let pii_filter = decision
            .metadata
            .get("pii_filter")
            .expect("pii_filter metadata");
        // First filter in insertion order that matches is "ssn" (position 0)
        assert_eq!(
            pii_filter["pattern"], "ssn",
            "ssn filter is first in insertion order"
        );
    }

    #[test]
    fn test_deterministic_prose_with_api_key_rejected() {
        let filter = make_pii_filter();
        let ctx = make_ctx("My token=sk-abc123def456ghi789");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("api_key"));
    }

    // ── PipelineOrchestrator ─────────────────────────────────────────────────

    #[test]
    fn test_pipeline_empty_stages_returns_complete() {
        let orchestrator = PipelineOrchestrator::new(vec![]);
        let ctx = WorkContext::default();
        let output = orchestrator.execute(&ctx).expect("execute");
        let result: crate::pipeline::PipelineResult = output.data_as().expect("data_as");
        assert!(!result.rejected);
        assert_eq!(result.decisions.len(), 0);
    }

    #[test]
    fn test_pipeline_single_deterministic_stage_prose() {
        let stage = Arc::new(DeterministicPreFilter::new());
        let orchestrator = PipelineOrchestrator::new(vec![stage]);
        let ctx = make_ctx("What is Rust?");
        let output = orchestrator.execute(&ctx).expect("execute");
        let result: crate::pipeline::PipelineResult = output.data_as().expect("data_as");
        assert!(!result.rejected);
        assert_eq!(result.decisions.len(), 1);
        assert_eq!(result.decisions[0].verdict, StageVerdict::Passed);
    }

    #[test]
    fn test_pipeline_single_deterministic_stage_command() {
        let stage = Arc::new(DeterministicPreFilter::new());
        let orchestrator = PipelineOrchestrator::new(vec![stage]);
        let ctx = make_ctx("/help");
        let output = orchestrator.execute(&ctx).expect("execute");
        let result: crate::pipeline::PipelineResult = output.data_as().expect("data_as");
        assert!(result.rejected);
        assert_eq!(result.decisions.len(), 1);
        assert_eq!(result.decisions[0].verdict, StageVerdict::Rejected);
    }

    #[test]
    fn test_pipeline_orchestrator_name() {
        let orchestrator = PipelineOrchestrator::new(vec![]);
        assert_eq!(orchestrator.name(), "pipeline.orchestrator");
    }

    #[test]
    fn test_pipeline_orchestrator_provides() {
        let orchestrator = PipelineOrchestrator::new(vec![]);
        assert_eq!(orchestrator.provides().len(), 1);
        assert_eq!(&*orchestrator.provides()[0], "pipeline.result");
    }

    #[test]
    fn test_pipeline_orchestrator_builder() {
        let stage = Arc::new(DeterministicPreFilter::new());
        let orchestrator = PipelineOrchestrator::builder().push(stage).build();
        assert_eq!(orchestrator.name(), "pipeline.orchestrator");
    }

    #[test]
    fn test_deterministic_prefilter_describable() {
        let filter = DeterministicPreFilter::new();
        let desc = filter.describe();
        assert_eq!(desc["type"], "object");
    }

    // ── Stage 2: ClassifierStage concurrency limiter ────────────────────────

    /// Backend that tracks the maximum number of concurrently executing
    /// `chat_complete` calls, so a `Limiter`'s cap is observable.
    struct TrackingBackend {
        active: std::sync::atomic::AtomicUsize,
        max_active: std::sync::atomic::AtomicUsize,
    }

    impl fluent_llm::client::ChatBackend for TrackingBackend {
        fn chat_complete(
            &self,
            _messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            use std::sync::atomic::Ordering;
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(serde_json::json!({
                "action": "respond",
                "coherence_score": 0.9,
                "safety_score": 0.9,
                "reason": "ok",
                "intent": "question",
                "response": "hello",
            })
            .to_string())
        }
    }

    #[test]
    fn classifier_limiter_serializes_concurrent_calls() {
        use std::collections::HashMap;
        use std::sync::atomic::Ordering;

        use crate::config::RoutingConfig;
        use crate::stages::classifier::ClassifierStage;

        let backend = Arc::new(TrackingBackend {
            active: std::sync::atomic::AtomicUsize::new(0),
            max_active: std::sync::atomic::AtomicUsize::new(0),
        });
        let tracking = Arc::clone(&backend);
        let routing_config = RoutingConfig {
            routes: HashMap::new(),
            models: HashMap::new(),
            model_groups: HashMap::new(),
            system_prompt: String::new(),
            safety_threshold: 0.5,
            default_route: "fast".into(),
            score_matrix: None,
        };
        let limiter = Arc::new(fluent_concurrency::pool::Limiter::new(1));
        let stage = ClassifierStage::new(
            backend as Arc<dyn fluent_llm::client::ChatBackend>,
            routing_config,
            0.7,
            None,
            false,
            1,
            "fast",
            limiter,
        );

        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let mut ctx = WorkContext::default();
                    ctx.set_structured(
                        "request",
                        &serde_json::json!({
                            "model": "test",
                            "messages": [{"role": "user", "content": "hello"}],
                        }),
                    );
                    let output = stage.execute(&ctx).expect("execute");
                    let _decision: StageDecision = output.data_as().expect("data_as");
                });
            }
        });

        assert_eq!(
            tracking.max_active.load(Ordering::SeqCst),
            1,
            "a Limiter::new(1) must serialize classifier calls"
        );
    }

    // ── Stage 2: ClassifierStage in M4 classification-tree mode ─────────────

    /// A backend that records every system prompt and always returns the
    /// supplied classifier verdict.
    struct TreeRecordingBackend {
        prompts: Arc<std::sync::Mutex<Vec<String>>>,
        response: String,
    }

    impl fluent_llm::client::ChatBackend for TreeRecordingBackend {
        fn chat_complete(
            &self,
            messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            self.prompts.lock().unwrap().extend(
                messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .map(|m| m.content.clone()),
            );
            Ok(self.response.clone())
        }
    }

    fn tree_test_config() -> crate::config::RouterConfig {
        serde_json::from_str(
            r#"{
                "pipelines": {"default": {"deterministic_prefilter": false, "classifier": true}},
                "models": {
                    "fast": {"endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 1e-6, "cost_output": 6e-6, "cost_cached_read": 4e-7, "speed": 8},
                    "code-model": {"endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "code-model", "intelligence": 5, "cost_input": 5e-6, "cost_output": 3e-5, "cost_cached_read": 2e-6, "speed": 5}
                },
                "model_groups": {
                    "fast": ["fast"],
                    "code": ["code-model"]
                },
                "routes": {
                    "code": {"group": "code", "pipelines": ["default"], "description": "code"},
                    "local": {"group": "fast", "pipelines": ["default"], "description": "local"}
                },
                "default_route": "local",
                "classification": {
                    "root": {
                        "type": "classifier",
                        "description": "request router",
                        "model": "fast",
                        "coherence_threshold": 0.4,
                        "safety_threshold": 0.3,
                        "children": [
                            {
                                "key": "code",
                                "description": "programming and implementation",
                                "node": { "type": "terminal", "route": "code", "group": "code" }
                            },
                            {
                                "key": "general",
                                "description": "everything else",
                                "node": {
                                    "type": "fallback",
                                    "node": { "type": "terminal", "route": "local", "group": "fast" }
                                }
                            }
                        ]
                    }
                }
            }"#,
        )
        .expect("valid tree config")
    }

    #[test]
    fn classifier_stage_tree_mode_produces_routing_target() {
        use crate::pipeline::RoutingTarget;

        let config = tree_test_config();
        let prompts = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let backend: Arc<dyn fluent_llm::client::ChatBackend> = Arc::new(TreeRecordingBackend {
            prompts: prompts.clone(),
            response: serde_json::json!({
                "route": "code",
                "coherence": 0.9,
                "safety": 0.9,
                "complexity": 6,
                "reason": "code query",
            })
            .to_string(),
        });

        // The stage is built through the pipeline builder (exercises the M4
        // tree-engine construction path, not a hand-built engine).
        let pipeline = config
            .build_named_pipeline_with_backend("default", Some(backend))
            .expect("tree pipeline should build");

        let mut ctx = WorkContext::default();
        ctx.set_structured(
            "request",
            &serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "help me write a sort"}],
            }),
        );

        let output = pipeline.execute(&ctx).expect("pipeline executes");
        let result: crate::pipeline::PipelineResult = output.data_as().expect("pipeline result");
        assert!(!result.rejected);
        let rt: RoutingTarget = result
            .routing_target
            .expect("tree should produce a routing target");
        assert_eq!(rt.target_name.as_deref(), Some("code"));
        assert_eq!(rt.model, "code-model");
        assert_eq!(rt.group.as_deref(), Some("code"));

        // The auto-generated prompt was sent to the backend.
        let captured = prompts.lock().unwrap().clone();
        assert_eq!(captured.len(), 1, "exactly one tree classifier call");
        assert!(
            captured[0].contains("- code: programming and implementation"),
            "auto-constructed prompt lists child routes, got: {}",
            captured[0]
        );
        assert!(
            captured[0].contains("\"route\": \"<exactly one of: code>\""),
            "three-axis route enum, got: {}",
            captured[0]
        );
    }

    #[test]
    fn classifier_stage_tree_mode_rejects_below_threshold() {
        let config = tree_test_config();
        let backend: Arc<dyn fluent_llm::client::ChatBackend> = Arc::new(TreeRecordingBackend {
            prompts: Arc::new(std::sync::Mutex::new(Vec::new())),
            response: serde_json::json!({
                "route": "code",
                "coherence": 0.1,
                "safety": 0.9,
                "complexity": 1,
                "reason": "garbage",
            })
            .to_string(),
        });

        let pipeline = config
            .build_named_pipeline_with_backend("default", Some(backend))
            .expect("tree pipeline should build");

        let mut ctx = WorkContext::default();
        ctx.set_structured(
            "request",
            &serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "asdf qwerty"}],
            }),
        );

        let output = pipeline.execute(&ctx).expect("pipeline executes");
        let result: crate::pipeline::PipelineResult = output.data_as().expect("pipeline result");
        assert!(result.rejected);
        assert!(
            result
                .reject_reason
                .as_deref()
                .is_some_and(|r| r.contains("coherence")),
            "rejection should mention coherence, got: {:?}",
            result.reject_reason
        );
    }

    // ── M5: ScoreMatrix as the routing decision engine ─────────────────────

    const DEFAULT_MATRIX_ROUTES: &str = r#"{
        "plan":  {"bands": {"completeness": [0.0, 0.5]}},
        "local": {"bands": {"completeness": [0.7, 1.0], "risk": [0.0, 0.4]}},
        "rigor": {"bands": {"completeness": [0.7, 1.0], "risk": [0.4, 1.0]}}
    }"#;

    fn matrix_config(authoritative: bool, matrix_routes: &str) -> crate::config::RouterConfig {
        serde_json::from_str(&format!(
            r#"{{
                "pipelines": {{
                    "default": {{
                        "classifier": true,
                        "classifier_model": "fast",
                        "score_matrix": {{
                            "dimensions": ["coherence", "complexity", "completeness", "risk"],
                            "weights": [0.3, 0.2, 0.3, 0.2],
                            "routes": {matrix_routes}
                        }},
                        "score_matrix_authoritative": {authoritative}
                    }}
                }},
                "classifier_model": "fast",
                "models": {{
                    "fast": {{ "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 10 }},
                    "code-model": {{ "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "code-model", "intelligence": 5, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 5 }},
                    "local-model": {{ "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "local-model", "intelligence": 3, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 8 }}
                }},
                "model_groups": {{
                    "code": ["code-model"],
                    "local": ["local-model"]
                }},
                "routes": {{
                    "code": {{ "group": "code", "pipelines": ["default"] }},
                    "local": {{ "group": "local", "pipelines": ["default"] }}
                }},
                "default_route": "fast"
            }}"#
        ))
        .expect("valid matrix config")
    }

    fn run_matrix_pipeline(
        config: &crate::config::RouterConfig,
        response: &str,
    ) -> crate::pipeline::PipelineResult {
        let backend: Arc<dyn fluent_llm::client::ChatBackend> =
            Arc::new(crate::test_stubs::StubChatBackend::always(response));
        let pipeline = config
            .build_named_pipeline_with_backend("default", Some(backend))
            .expect("matrix pipeline should build");
        let mut ctx = WorkContext::default();
        ctx.set_structured(
            "request",
            &serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "help me write a sort"}],
            }),
        );
        let output = pipeline.execute(&ctx).expect("pipeline executes");
        output.data_as().expect("pipeline result")
    }

    #[test]
    fn score_matrix_authoritative_matrix_decides_over_llm_target() {
        // The LLM says route to "code", but with authoritative scoring the
        // matrix's top route is "local" (completeness 0.9 + risk 0.1) — the
        // matrix wins and dispatch resolves through the shared path.
        let config = matrix_config(true, DEFAULT_MATRIX_ROUTES);
        let result = run_matrix_pipeline(
            &config,
            &serde_json::json!({
                "action": "route",
                "target": "code",
                "coherence_score": 0.9,
                "safety_score": 0.9,
                "complexity": 3,
                "completeness": 0.9,
                "risk": 0.1,
                "reason": "code request",
            })
            .to_string(),
        );
        assert!(!result.rejected);
        let rt = result.routing_target.expect("matrix route must dispatch");
        assert_eq!(
            rt.target_name.as_deref(),
            Some("local"),
            "matrix top route decides over the LLM's target"
        );
        assert_eq!(rt.model, "local-model");
    }

    #[test]
    fn score_matrix_authoritative_falls_back_when_no_band_matches() {
        // Completeness 0.6 matches no band (plan needs ≤0.5, local/rigor need
        // ≥0.7) — no matrix route → the LLM path resolves unchanged.
        let config = matrix_config(true, DEFAULT_MATRIX_ROUTES);
        let result = run_matrix_pipeline(
            &config,
            &serde_json::json!({
                "action": "route",
                "target": "code",
                "coherence_score": 0.9,
                "safety_score": 0.9,
                "complexity": 3,
                "completeness": 0.6,
                "risk": 0.1,
                "reason": "code request",
            })
            .to_string(),
        );
        assert!(!result.rejected);
        let rt = result.routing_target.expect("LLM fallback must dispatch");
        assert_eq!(
            rt.target_name.as_deref(),
            Some("code"),
            "no matrix band match -> LLM path fallback"
        );
        assert_eq!(rt.model, "code-model");
    }

    #[test]
    fn score_matrix_authoritative_thresholds_reject_first() {
        // Coherence below threshold gates before the matrix is consulted.
        let config = matrix_config(true, DEFAULT_MATRIX_ROUTES);
        let result = run_matrix_pipeline(
            &config,
            &serde_json::json!({
                "action": "route",
                "target": "code",
                "coherence_score": 0.1,
                "safety_score": 0.9,
                "complexity": 3,
                "completeness": 0.9,
                "risk": 0.1,
                "reason": "garbage",
            })
            .to_string(),
        );
        assert!(result.rejected);
        assert!(
            result
                .reject_reason
                .as_deref()
                .is_some_and(|r| r.contains("coherence")),
            "gating rejection must precede the matrix, got: {:?}",
            result.reject_reason
        );
    }

    #[test]
    fn score_matrix_authoritative_emits_scored_route_audit_metadata() {
        let config = matrix_config(true, DEFAULT_MATRIX_ROUTES);
        let result = run_matrix_pipeline(
            &config,
            &serde_json::json!({
                "action": "route",
                "target": "code",
                "coherence_score": 0.9,
                "safety_score": 0.9,
                "complexity": 3,
                "completeness": 0.9,
                "risk": 0.1,
                "reason": "code request",
            })
            .to_string(),
        );
        let decision = result
            .decisions
            .iter()
            .find(|d| d.stage == crate::pipeline_types::PipelineStage::Classifier)
            .expect("classifier decision");
        let metadata = &decision.metadata;
        assert_eq!(
            metadata["scored_route"]["route"],
            serde_json::json!("local"),
            "audit metadata must name the decided route"
        );
        assert!(
            metadata["scored_routes"].is_array(),
            "full ranking stays legible for the audit trail"
        );
    }

    #[test]
    fn score_matrix_authoritative_respond_route_preserves_direct_response() {
        // A matrix whose only matching route is "respond" must yield a direct
        // response (no dispatch target), reusing the output.response handling.
        let config = matrix_config(
            true,
            r#"{
                "respond": {"bands": {"completeness": [0.0, 0.5]}}
            }"#,
        );
        let result = run_matrix_pipeline(
            &config,
            &serde_json::json!({
                "action": "route",
                "target": "code",
                "coherence_score": 0.9,
                "safety_score": 0.9,
                "complexity": 2,
                "completeness": 0.3,
                "risk": 0.1,
                "response": "the direct answer",
                "reason": "trivial",
            })
            .to_string(),
        );
        assert!(!result.rejected);
        assert!(
            result.routing_target.is_none(),
            "matrix 'respond' must not dispatch"
        );
        assert_eq!(
            result.classifier_response.as_deref(),
            Some("the direct answer"),
            "direct response preserved for the matrix 'respond' route"
        );
    }

    #[test]
    fn score_matrix_default_off_uses_llm_path() {
        // `score_matrix_authoritative` defaults to false: existing behavior
        // (LLM `action`/`target`) is untouched.
        let config = matrix_config(false, DEFAULT_MATRIX_ROUTES);
        let result = run_matrix_pipeline(
            &config,
            &serde_json::json!({
                "action": "route",
                "target": "code",
                "coherence_score": 0.9,
                "safety_score": 0.9,
                "complexity": 3,
                "completeness": 0.9,
                "risk": 0.1,
                "reason": "code request",
            })
            .to_string(),
        );
        let rt = result.routing_target.expect("LLM path must dispatch");
        assert_eq!(
            rt.target_name.as_deref(),
            Some("code"),
            "default-off keeps the LLM path"
        );
        assert_eq!(rt.model, "code-model");
    }

    // ── M6: RetryClassifier wired into the production builder ──────────────

    /// A `MakeWriter` that captures formatted log lines for assertions.
    #[derive(Clone, Default)]
    struct LogCapture(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(buf).into_owned());
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl tracing_subscriber::fmt::MakeWriter<'_> for LogCapture {
        type Writer = Self;
        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
        use tracing_subscriber::layer::SubscriberExt;
        let capture = LogCapture::default();
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_writer(capture.clone())
                .with_ansi(false)
                .with_target(true),
        );
        let result = tracing::subscriber::with_default(subscriber, f);
        let logs = capture.0.lock().unwrap().clone();
        (result, logs)
    }

    /// A `ChatBackend` that fails JSON parsing the first two calls (garbage
    /// output) then returns the supplied valid classifier response, recording
    /// every system prompt it receives.
    struct RetryFailBackend {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        success_response: String,
    }

    impl fluent_llm::client::ChatBackend for RetryFailBackend {
        fn chat_complete(
            &self,
            messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.prompts.lock().unwrap().extend(
                messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .map(|m| m.content.clone()),
            );
            if self.calls.load(std::sync::atomic::Ordering::SeqCst) < 3 {
                Ok("this is definitely not json".into())
            } else {
                Ok(self.success_response.clone())
            }
        }
    }

    fn retry_config(retry_max: u32) -> crate::config::RouterConfig {
        serde_json::from_str(&format!(
            r#"{{
                "pipelines": {{
                    "default": {{
                        "classifier": true,
                        "classifier_model": "fast",
                        "classifier_retry_max": {retry_max},
                        "classifier_retry_prompts": ["corrective prompt 1", "corrective prompt 2"]
                    }}
                }},
                "classifier_model": "fast",
                "models": {{
                    "fast": {{ "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 10 }},
                    "code-model": {{ "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "code-model", "intelligence": 5, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 5 }}
                }},
                "model_groups": {{
                    "fast": ["fast"],
                    "code": ["code-model"]
                }},
                "routes": {{
                    "code": {{ "group": "code", "pipelines": ["default"] }}
                }},
                "default_route": "fast"
            }}"#
        ))
        .expect("valid retry config")
    }

    #[test]
    fn retry_classifier_recovers_through_real_pipeline() {
        use std::sync::atomic::Ordering;

        let config = retry_config(2);
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prompts = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let backend = RetryFailBackend {
            calls: calls.clone(),
            prompts: prompts.clone(),
            success_response: serde_json::json!({
                "action": "route",
                "target": "code",
                "coherence_score": 0.9,
                "safety_score": 0.9,
                "complexity": 3,
                "intent": "code",
                "reason": "recovered",
            })
            .to_string(),
        };

        let (result, logs) = capture_logs(|| {
            let pipeline = config
                .build_named_pipeline_with_backend("default", Some(std::sync::Arc::new(backend)))
                .expect("retry pipeline should build");
            let mut ctx = WorkContext::default();
            ctx.set_structured(
                "request",
                &serde_json::json!({
                    "model": "test",
                    "messages": [{"role": "user", "content": "write a sort"}],
                }),
            );
            let output = pipeline.execute(&ctx).expect("pipeline executes");
            output.data_as::<crate::pipeline::PipelineResult>().expect("pipeline result")
        });

        // Final decision is non-fallback and dispatched through the LLM target.
        assert!(!result.rejected);
        let rt = result.routing_target.expect("routing target");
        assert_eq!(rt.model, "code-model");
        let decision = result
            .decisions
            .iter()
            .find(|d| d.stage == crate::pipeline_types::PipelineStage::Classifier)
            .expect("classifier decision");
        assert_eq!(
            decision.metadata["fallback"],
            serde_json::json!(false),
            "final decision must be non-fallback"
        );

        // Exactly one initial call + two retries reached the backend.
        assert_eq!(calls.load(Ordering::SeqCst), 3, "initial + 2 retries");

        // The escalating corrective prompts were injected per retry attempt.
        let recorded = prompts.lock().unwrap().clone();
        assert_eq!(recorded.len(), 3);
        assert_eq!(recorded[1], "corrective prompt 1");
        assert_eq!(recorded[2], "corrective prompt 2");

        // The retry attempts are observable: RetryClassifier logs each retry,
        // and ClassifierStage logs the injected `classifier_retry_attempt`.
        let joined = logs.join("\n");
        assert!(
            joined.contains("retry=1") && joined.contains("retry=2"),
            "retry attempts must be logged, got:\n{joined}"
        );
        assert!(
            joined.contains("retry_attempt=0") && joined.contains("retry_attempt=1"),
            "classifier_retry_attempt must be observable in classifier logs, got:\n{joined}"
        );
    }

    #[test]
    fn retry_disabled_by_default_is_byte_for_byte_unchanged() {
        // Defaults: retry disabled (0) with the two stock prompts.
        let defaults = crate::config::builder::PipelineParams::default();
        assert_eq!(defaults.classifier_retry_max, 0);
        assert_eq!(defaults.classifier_retry_prompts.len(), 2);

        // A config that omits the retry fields must deserialize to the same
        // defaults.
        let config: crate::config::RouterConfig = serde_json::from_str(
            r#"{
                "pipelines": {"default": {"classifier": true, "classifier_model": "fast"}},
                "classifier_model": "fast",
                "models": {
                    "fast": {"endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 10}
                },
                "model_groups": {"fast": ["fast"]},
                "routes": {},
                "default_route": "fast"
            }"#,
        )
        .expect("valid config");
        let params = &config.pipelines["default"];
        assert_eq!(params.classifier_retry_max, 0);
        assert_eq!(params.classifier_retry_prompts.len(), 2);

        // Behaviorally: a garbage classifier response makes exactly ONE backend
        // call (the classifier is NOT wrapped in a retry decorator).
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let prompts = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let backend = RetryFailBackend {
            calls: calls.clone(),
            prompts,
            success_response: "{}".into(),
        };
        let pipeline = config
            .build_named_pipeline_with_backend("default", Some(std::sync::Arc::new(backend)))
            .expect("pipeline builds");
        let mut ctx = WorkContext::default();
        ctx.set_structured(
            "request",
            &serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "write a sort"}],
            }),
        );
        let _ = pipeline.execute(&ctx).expect("pipeline executes");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "default retry_max=0 must not retry"
        );
    }

    #[test]
    fn retry_classifier_tree_mode_round_trips() {
        // Retry wrapping must not disturb the classification-tree path: the
        // wrapper delegates to the inner tree-driven stage, whose engine
        // produces the final decision.
        let config = tree_test_config();
        let mut with_retry = config;
        with_retry.pipelines.get_mut("default").unwrap().classifier_retry_max = 2;
        let backend: Arc<dyn fluent_llm::client::ChatBackend> = Arc::new(TreeRecordingBackend {
            prompts: Arc::new(std::sync::Mutex::new(Vec::new())),
            response: serde_json::json!({
                "route": "code",
                "coherence": 0.9,
                "safety": 0.9,
                "complexity": 6,
                "reason": "code query",
            })
            .to_string(),
        });
        let pipeline = with_retry
            .build_named_pipeline_with_backend("default", Some(backend))
            .expect("tree pipeline with retry should build");
        let mut ctx = WorkContext::default();
        ctx.set_structured(
            "request",
            &serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "help me write a sort"}],
            }),
        );
        let output = pipeline.execute(&ctx).expect("pipeline executes");
        let result: crate::pipeline::PipelineResult = output.data_as().expect("pipeline result");
        assert!(!result.rejected);
        let rt = result.routing_target.expect("tree engine must produce a target");
        assert_eq!(rt.model, "code-model");
    }
}
