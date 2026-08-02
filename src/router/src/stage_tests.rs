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
        ctx.metadata.insert(
            "request".into(),
            MetadataValue::String(request_json.to_string()),
        );
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
            .map_or(false, |s| s.contains("my-snapshot")));
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

    impl guidance_llm::client::ChatBackend for TrackingBackend {
        fn chat_complete(
            &self,
            _messages: &[guidance_llm::ChatMessage],
        ) -> Result<String, guidance_llm::LlmError> {
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
            backend as Arc<dyn guidance_llm::client::ChatBackend>,
            routing_config,
            0.7,
            None,
            1,
            "fast",
            limiter,
        );

        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let mut ctx = WorkContext::default();
                    ctx.metadata.insert(
                        "request".into(),
                        MetadataValue::String(
                            serde_json::json!({
                                "model": "test",
                                "messages": [{"role": "user", "content": "hello"}],
                            })
                            .to_string(),
                        ),
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
}
