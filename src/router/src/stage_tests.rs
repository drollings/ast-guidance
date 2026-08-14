#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fluent_wvr::prelude::*;

    use crate::config::{
        ConfidenceGate, FilterAction, FilterOutcome, FilterScope, PatternEntry, RejectPatterns,
    };
    use crate::pipeline::PipelineOrchestrator;
    use crate::pipeline_types::{PipelineStage, StageDecision, StageMetadata, StageVerdict};
    use crate::stages::deterministic::DeterministicPreFilter;
    use crate::test_support::capture_logs;

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
            None,
            crate::config::ClassifierFailurePolicy::Reject,
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

    // ── Stage 2: ClassifierStage failure policy ─────────────────────────

    /// A backend that always fails its LLM call.
    struct AlwaysFailBackend;

    impl fluent_llm::client::ChatBackend for AlwaysFailBackend {
        fn chat_complete(
            &self,
            _messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            Err(fluent_llm::LlmError::Api("simulated outage".into()))
        }
    }

    /// A backend that always returns the given raw text (to exercise the
    /// parse-fallback path).
    struct FixedResponseBackend {
        response: String,
    }

    impl fluent_llm::client::ChatBackend for FixedResponseBackend {
        fn chat_complete(
            &self,
            _messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            Ok(self.response.clone())
        }
    }

    fn classifier_stage_with_policy(
        backend: Arc<dyn fluent_llm::client::ChatBackend>,
        policy: crate::config::ClassifierFailurePolicy,
    ) -> crate::stages::classifier::ClassifierStage {
        let routing_config = crate::config::RoutingConfig {
            routes: std::collections::HashMap::new(),
            models: std::collections::HashMap::new(),
            model_groups: std::collections::HashMap::new(),
            system_prompt: String::new(),
            safety_threshold: 0.5,
            default_route: "fast".into(),
            score_matrix: None,
        };
        crate::stages::classifier::ClassifierStage::new(
            backend,
            routing_config,
            0.7,
            None,
            false,
            1,
            "fast",
            Arc::new(fluent_concurrency::pool::Limiter::new(4)),
            None,
            policy,
        )
    }

    fn classifier_ctx() -> WorkContext {
        let mut ctx = WorkContext::default();
        ctx.set_structured(
            "request",
            &serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            }),
        );
        ctx
    }

    #[test]
    fn classifier_reject_policy_rejects_on_llm_error() {
        let stage = classifier_stage_with_policy(
            Arc::new(AlwaysFailBackend),
            crate::config::ClassifierFailurePolicy::Reject,
        );
        let decision: StageDecision = stage
            .execute(&classifier_ctx())
            .expect("execute")
            .data_as()
            .expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("LLM error"));
        // No fabricated 1.0 scores may appear on the reject path.
        assert!(
            !decision.reason.contains("coherence")
                && !decision.reason.contains("1.00"),
            "reject path must not fabricate scores: {}",
            decision.reason
        );
    }

    #[test]
    fn classifier_reject_policy_rejects_on_parse_failure() {
        let stage = classifier_stage_with_policy(
            Arc::new(FixedResponseBackend {
                response: "this is not json".into(),
            }),
            crate::config::ClassifierFailurePolicy::Reject,
        );
        let decision: StageDecision = stage
            .execute(&classifier_ctx())
            .expect("execute")
            .data_as()
            .expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("classifier failure"));
    }

    #[test]
    fn classifier_route_to_default_truthful_uses_zero_scores() {
        let stage = classifier_stage_with_policy(
            Arc::new(AlwaysFailBackend),
            crate::config::ClassifierFailurePolicy::RouteToDefaultTruthful,
        );
        let decision: StageDecision = stage
            .execute(&classifier_ctx())
            .expect("execute")
            .data_as()
            .expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let meta = decision.metadata.as_object().expect("metadata object");
        assert_eq!(meta.get("coherence_score").and_then(|v| v.as_f64()), Some(0.0));
        assert_eq!(meta.get("safety_score").and_then(|v| v.as_f64()), Some(0.0));
    }

    #[test]
    fn classifier_legacy_fail_open_keeps_high_scores() {
        let stage = classifier_stage_with_policy(
            Arc::new(AlwaysFailBackend),
            crate::config::ClassifierFailurePolicy::LegacyFailOpen,
        );
        let decision: StageDecision = stage
            .execute(&classifier_ctx())
            .expect("execute")
            .data_as()
            .expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        let meta = decision.metadata.as_object().expect("metadata object");
        assert_eq!(meta.get("coherence_score").and_then(|v| v.as_f64()), Some(1.0));
        assert_eq!(meta.get("safety_score").and_then(|v| v.as_f64()), Some(1.0));
    }

    // ── Stage 2: ClassifierStage in classification-tree mode ─────────────

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

        // The stage is built through the pipeline builder (exercises the
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

    // ── ScoreMatrix as the routing decision engine ─────────────────────

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

    // ── RetryClassifier wired into the production builder ──────────────

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
            output
                .data_as::<crate::pipeline::PipelineResult>()
                .expect("pipeline result")
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

    /// A backend that always returns garbage (never recovers), to exercise the
    /// retries-exhausted path.
    struct AlwaysGarbageBackend;

    impl fluent_llm::client::ChatBackend for AlwaysGarbageBackend {
        fn chat_complete(
            &self,
            _messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            Ok("not json at all".into())
        }
    }

    #[test]
    fn retry_exhausts_then_default_policy_rejects() {
        // After max_retries, the default (Reject) classifier failure policy
        // yields a `Rejected` verdict — never a fabricated 1.0-score dispatch.
        let config = retry_config(2);
        let pipeline = config
            .build_named_pipeline_with_backend(
                "default",
                Some(std::sync::Arc::new(AlwaysGarbageBackend)),
            )
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
        let result: crate::pipeline::PipelineResult = output.data_as().expect("pipeline result");

        assert!(result.rejected, "default policy must fail closed");
        assert!(result.routing_target.is_none(), "no dispatch on reject");

        let decision = result
            .decisions
            .iter()
            .find(|d| d.stage == crate::pipeline_types::PipelineStage::Classifier)
            .expect("classifier decision");
        assert_eq!(
            decision.verdict,
            crate::pipeline_types::StageVerdict::Rejected,
            "after retries exhaust the default policy rejects"
        );
        // The reject path must not carry fabricated 1.0 scores.
        let meta = decision.metadata.as_object().expect("metadata object");
        assert_eq!(meta.get("coherence_score"), None);
        assert_eq!(meta.get("safety_score"), None);
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
        with_retry
            .pipelines
            .get_mut("default")
            .unwrap()
            .classifier_retry_max = 2;
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
        let rt = result
            .routing_target
            .expect("tree engine must produce a target");
        assert_eq!(rt.model, "code-model");
    }

    // ── Target-matching ladder wired into the flat classifier path ───────

    /// A flat config whose `code` route resolves to the 2-member group
    /// `[swarm, qwen3.6-27b]` — the shipped ladder shape. `target_match` is
    /// configurable so the static-preserves-today test shares the same shape.
    fn ladder_config(target_match: &str) -> crate::config::RouterConfig {
        serde_json::from_str(&format!(
            r#"{{
                "pipelines": {{
                    "default": {{
                        "classifier": true,
                        "classifier_model": "fast",
                        "target_match": "{target_match}"
                    }}
                }},
                "classifier_model": "fast",
                "models": {{
                    "fast": {{ "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 10 }},
                    "swarm": {{ "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "swarm", "intelligence": 2, "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.4, "speed": 9 }},
                    "qwen3.6-27b": {{ "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "qwen3.6-27b", "intelligence": 6, "cost_input": 5.0, "cost_output": 5.0, "cost_cached_read": 2.0, "speed": 4 }}
                }},
                "model_groups": {{
                    "code": ["swarm", "qwen3.6-27b"]
                }},
                "routes": {{
                    "code": {{ "group": "code", "pipelines": ["default"] }}
                }},
                "default_route": "fast"
            }}"#
        ))
        .expect("valid ladder config")
    }

    /// Run the flat pipeline with a queued `StubChatBackend`. The first
    /// response is the classifier verdict; the remaining ones are the
    /// per-candidate self-assessments, consumed in order by the ladder.
    fn run_ladder_pipeline(
        config: &crate::config::RouterConfig,
        responses: Vec<String>,
    ) -> crate::pipeline::PipelineResult {
        let backend: Arc<dyn fluent_llm::client::ChatBackend> =
            Arc::new(crate::test_stubs::StubChatBackend::new(responses));
        let pipeline = config
            .build_named_pipeline_with_backend("default", Some(backend))
            .expect("ladder pipeline should build");
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

    fn classifier_verdict(complexity: u8, target: &str) -> String {
        serde_json::json!({
            "action": "route",
            "target": target,
            "coherence_score": 0.9,
            "safety_score": 0.9,
            "complexity": complexity,
            "completeness": 0.9,
            "risk": 0.1,
            "reason": "code request",
        })
        .to_string()
    }

    fn assessment(complexity: u8, reason: &str) -> String {
        serde_json::json!({
            "complexity": complexity,
            "reason": reason,
        })
        .to_string()
    }

    #[test]
    fn target_match_ladder_climbs_to_more_intelligent_member() {
        // Classifier routes to "code" with complexity 1 (start index 0 →
        // swarm self-assesses first). Swarm reports 7 > its intelligence 2 →
        // escalate; qwen3.6-27b reports 5 <= 6 → match. The matched target is
        // the more-intelligent member, exactly 2 self-assessment calls made.
        let config = ladder_config("self_assess");
        let result = run_ladder_pipeline(
            &config,
            vec![
                classifier_verdict(1, "code"),
                assessment(7, "hard"),
                assessment(5, "ok"),
            ],
        );
        assert!(!result.rejected);
        let rt = result.routing_target.expect("must dispatch");
        assert_eq!(rt.target_name.as_deref(), Some("code"));
        assert_eq!(rt.group.as_deref(), Some("code"));
        assert_eq!(rt.model, "qwen3.6-27b");
        // Mechanical-failure fallbacks = the group tail (empty after the last
        // member) plus any `all_dispatch_targets` entries not already included
        // — here the primary group's other member, preserving today's
        // cross-group resilience list.
        let fb: Vec<&str> = rt.fallbacks.iter().map(|f| f.model.as_str()).collect();
        assert_eq!(fb, vec!["swarm"]);
    }

    #[test]
    fn target_match_ladder_matches_first_qualifying_member() {
        // Swarm self-assesses 1 <= 2 → matches immediately. The single
        // assessment is enough to prove exactly one self-assessment call: an
        // unexpected second call would pop `None` from the queue and escalate
        // conservatively to qwen (last member), changing the result.
        let config = ladder_config("self_assess");
        let result = run_ladder_pipeline(
            &config,
            vec![classifier_verdict(1, "code"), assessment(1, "easy")],
        );
        assert!(!result.rejected);
        let rt = result.routing_target.expect("must dispatch");
        assert_eq!(rt.model, "swarm");
        assert_eq!(rt.target_name.as_deref(), Some("code"));
        // The group tail becomes the mechanical-failure fallback list.
        let fb: Vec<&str> = rt.fallbacks.iter().map(|f| f.model.as_str()).collect();
        assert_eq!(fb, vec!["qwen3.6-27b"]);
    }

    #[test]
    fn target_match_static_reproduces_todays_cheapest_qualifying_pick() {
        // `target_match: "static"` disables the ladder entirely: the cheapest
        // qualifying model (swarm) is picked at resolution time, and only the
        // single classifier response is consumed — no self-assessment calls.
        let config = ladder_config("static");
        let result = run_ladder_pipeline(&config, vec![classifier_verdict(1, "code")]);
        assert!(!result.rejected);
        let rt = result.routing_target.expect("must dispatch");
        assert_eq!(rt.model, "swarm");
        assert_eq!(rt.target_name.as_deref(), Some("code"));
        assert_eq!(rt.group.as_deref(), Some("code"));
    }

    #[test]
    fn target_match_single_member_group_skips_ladder() {
        // Default `self_assess` with a single-member group: nothing to climb,
        // so it resolves statically (no extra LLM call). An extra call would
        // pop `None` and — if the ladder erroneously ran — land on the sole
        // member anyway; the audit/assessment path must stay silent. We assert
        // via the queued response count: exactly the one classifier call.
        let mut config = ladder_config("self_assess");
        config.model_groups.insert(
            "code".into(),
            crate::config::ModelGroup::Array(vec!["qwen3.6-27b".into()]),
        );
        let result = run_ladder_pipeline(&config, vec![classifier_verdict(1, "code")]);
        assert!(!result.rejected);
        let rt = result.routing_target.expect("must dispatch");
        assert_eq!(rt.model, "qwen3.6-27b");
    }

    #[test]
    fn target_match_ladder_applies_to_matrix_authoritative_branch() {
        // The matrix-decides route is resolved through the same ladder (DRY):
        // the matrix's top route "code" climbs its 2-member group, escalating
        // swarm (reports 7 > 2) and matching qwen (reports 5 <= 6).
        let config = serde_json::from_str(
            r#"{
                "pipelines": {
                    "default": {
                        "classifier": true,
                        "classifier_model": "fast",
                        "target_match": "self_assess",
                        "score_matrix": {
                            "dimensions": ["coherence", "complexity", "completeness", "risk"],
                            "weights": [0.3, 0.2, 0.3, 0.2],
                            "routes": {
                                "code": { "bands": { "completeness": [0.0, 1.0] } }
                            }
                        },
                        "score_matrix_authoritative": true
                    }
                },
                "classifier_model": "fast",
                "models": {
                    "fast": { "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 10 },
                    "swarm": { "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "swarm", "intelligence": 2, "cost_input": 1.0, "cost_output": 1.0, "cost_cached_read": 0.4, "speed": 9 },
                    "qwen3.6-27b": { "endpoint": "http://upstream.test:8080/v1/chat/completions", "name": "qwen3.6-27b", "intelligence": 6, "cost_input": 5.0, "cost_output": 5.0, "cost_cached_read": 2.0, "speed": 4 }
                },
                "model_groups": {
                    "code": ["swarm", "qwen3.6-27b"]
                },
                "routes": {
                    "code": { "group": "code", "pipelines": ["default"] }
                },
                "default_route": "fast"
            }"#,
        )
        .expect("valid matrix ladder config");

        let result = run_ladder_pipeline(
            &config,
            vec![
                classifier_verdict(1, "code"),
                assessment(7, "hard"),
                assessment(5, "ok"),
            ],
        );
        assert!(!result.rejected);
        let rt = result.routing_target.expect("must dispatch");
        assert_eq!(rt.target_name.as_deref(), Some("code"));
        assert_eq!(rt.model, "qwen3.6-27b");
    }

    // ── Route-level `always_route`: never let the classifier answer directly ─

    #[test]
    fn always_route_forces_dispatch_over_classifier_respond() {
        use crate::config::RoutingConfig;
        use crate::stages::classifier::ClassifierStage;

        // The classifier model is overconfident and answers a prose prompt
        // directly; the route is configured `always_route: true`, so the stage
        // must override action=respond into a dispatch to the route's group.
        let prompt_log = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let backend: Arc<dyn fluent_llm::client::ChatBackend> = Arc::new(TreeRecordingBackend {
            prompts: Arc::clone(&prompt_log),
            response: serde_json::json!({
                "action": "respond",
                "coherence_score": 0.9,
                "safety_score": 0.9,
                "reason": "i can write prose",
                "intent": "prose",
                "response": "once upon a time...",
            })
            .to_string(),
        });
        let routing_config: RoutingConfig = serde_json::from_value(serde_json::json!({
            "routes": {
                "prose": { "group": "prose", "pipelines": ["default"], "description": "creative", "always_route": true },
                "local": { "group": "default", "pipelines": ["default"], "description": "qa" }
            },
            "models": {
                "gemma": { "endpoint": "http://x/v1/chat/completions", "name": "gemma", "intelligence": 6, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 5 },
                "swarm": { "endpoint": "http://y/v1/chat/completions", "name": "swarm", "intelligence": 2, "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 8 }
            },
            "model_groups": {
                "prose": ["gemma"],
                "default": ["swarm"]
            },
            "system_prompt": "",
            "safety_threshold": 0.5,
            "default_route": "local"
        }))
        .expect("valid routing config");

        let limiter = Arc::new(fluent_concurrency::pool::Limiter::new(2));
        let stage = ClassifierStage::new(
            backend,
            routing_config.clone(),
            0.7,
            None,
            false,
            2,
            "swarm",
            limiter,
            None,
            crate::config::ClassifierFailurePolicy::Reject,
        );

        let mut ctx = WorkContext::default();
        ctx.set_structured(
            "request",
            &serde_json::json!({
                "model": "prose",
                "messages": [{"role": "user", "content": "Write a 400-word gothic story..."}],
            }),
        );
        let output = stage.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        let metadata = StageMetadata::from(decision.metadata.clone());
        // The direct response must have been overridden into a routing target.
        assert!(
            metadata.response().is_none(),
            "always_route must not produce a direct response"
        );
        let rt = metadata.routing_target().expect("must dispatch");
        assert_eq!(rt.target_name.as_deref(), Some("prose"));
        assert_eq!(rt.model, "gemma");

        // The system prompt advertises the dispatch rule so the LLM routes even
        // without the hard enforcement.
        let recorded = prompt_log.lock().unwrap().clone();
        assert!(
            recorded.iter().any(|p| p.contains("ALWAYS dispatch") && p.contains("prose")),
            "prompt must teach the always-route rule: {recorded:?}"
        );

        // A `local`-style route (always_route off) keeps the direct response.
        let mut ctx_local = WorkContext::default();
        ctx_local.set_structured(
            "request",
            &serde_json::json!({
                "model": "local",
                "messages": [{"role": "user", "content": "what is 2+2?"}],
            }),
        );
        let output_local = stage.execute(&ctx_local).expect("execute");
        let decision_local: StageDecision = output_local.data_as().expect("data_as");
        let metadata_local = StageMetadata::from(decision_local.metadata.clone());
        assert_eq!(metadata_local.response().as_deref(), Some("once upon a time..."));
        assert!(
            metadata_local.routing_target().is_none(),
            "non-always-route keeps the direct classifier answer"
        );
    }
}
