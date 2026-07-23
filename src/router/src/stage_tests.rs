#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fluent_wvr::prelude::*;

    use crate::pipeline::PipelineOrchestrator;
    use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};
    use crate::stages::deterministic::DeterministicPreFilter;
    use crate::stages::router::{RouterStage, RoutingPolicy};

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
        assert!(decision.reason.contains("no PII flags"));
    }

    #[test]
    fn test_deterministic_pii_email_detected() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("My email is user@example.com");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("email"));
        let pii_classes = decision
            .metadata
            .get("pii_classes")
            .and_then(|v| v.as_array())
            .expect("pii_classes array");
        assert!(pii_classes.iter().any(|c| c.as_str() == Some("email")));
    }

    #[test]
    fn test_deterministic_pii_ssn_detected() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("My SSN is 123-45-6789");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Rejected);
        assert!(decision.reason.contains("ssn"));
    }

    #[test]
    fn test_deterministic_pii_card_number_detected() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("card: 4111-1111-1111-1111");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        let pii = decision
            .metadata
            .get("pii_classes")
            .and_then(|v| v.as_array())
            .expect("pii_classes array");
        assert!(pii.iter().any(|c| c.as_str() == Some("card_number")));
    }

    #[test]
    fn test_deterministic_pii_phone_detected() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("Call me at (555) 123-4567");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        let pii = decision
            .metadata
            .get("pii_classes")
            .and_then(|v| v.as_array())
            .expect("pii_classes array");
        assert!(pii.iter().any(|c| c.as_str() == Some("phone")));
    }

    #[test]
    fn test_deterministic_multiple_pii_detected() {
        let filter = DeterministicPreFilter::new();
        let ctx = make_ctx("My email is user@example.com and my SSN is 123-45-6789");
        let output = filter.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        let pii = decision
            .metadata
            .get("pii_classes")
            .and_then(|v| v.as_array())
            .expect("pii_classes array");
        assert!(pii.len() >= 2);
    }

    // ── RouterStage ──────────────────────────────────────────────────────────

    #[test]
    fn test_router_local_first_policy() {
        let stage = RouterStage::new(RoutingPolicy::LocalFirst);
        let ctx = WorkContext::default();
        let output = stage.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert!(decision.reason.contains("LocalFirst"));
    }

    #[test]
    fn test_router_frontier_only_policy() {
        let stage = RouterStage::new(RoutingPolicy::FrontierOnly);
        let ctx = WorkContext::default();
        let output = stage.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert!(decision.reason.contains("FrontierOnly"));
    }

    #[test]
    fn test_router_auto_routing_policy() {
        let stage = RouterStage::new(RoutingPolicy::AutoRouting {
            classifier_model: "test-model".into(),
        });
        let ctx = WorkContext::default();
        let output = stage.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert!(decision.reason.contains("auto-routing"));
    }

    #[test]
    fn test_router_cost_minimizing_policy() {
        let stage = RouterStage::new(RoutingPolicy::CostMinimizing);
        let ctx = WorkContext::default();
        let output = stage.execute(&ctx).expect("execute");
        let decision: StageDecision = output.data_as().expect("data_as");
        assert_eq!(decision.verdict, StageVerdict::Passed);
        assert!(decision.reason.contains("LocalFirst"));
    }

    // ── PipelineOrchestrator ─────────────────────────────────────────────────

    #[test]
    fn test_pipeline_empty_stages_returns_complete() {
        let orchestrator = PipelineOrchestrator::new(vec![]);
        let ctx = WorkContext::default();
        let output = orchestrator.execute(&ctx).expect("execute");
        let result: crate::pipeline::PipelineResult =
            output.data_as().expect("data_as");
        assert!(!result.rejected);
        assert_eq!(result.decisions.len(), 0);
    }

    #[test]
    fn test_pipeline_single_deterministic_stage_prose() {
        let stage = Arc::new(DeterministicPreFilter::new());
        let orchestrator = PipelineOrchestrator::new(vec![stage]);
        let ctx = make_ctx("What is Rust?");
        let output = orchestrator.execute(&ctx).expect("execute");
        let result: crate::pipeline::PipelineResult =
            output.data_as().expect("data_as");
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
        let result: crate::pipeline::PipelineResult =
            output.data_as().expect("data_as");
        assert!(result.rejected);
        assert_eq!(result.decisions.len(), 1);
        assert_eq!(result.decisions[0].verdict, StageVerdict::Rejected);
    }

    #[test]
    fn test_pipeline_multiple_stages_sequential() {
        let stage1 = Arc::new(DeterministicPreFilter::new());
        let stage2 = Arc::new(RouterStage::new(RoutingPolicy::LocalFirst));
        let orchestrator = PipelineOrchestrator::new(vec![stage1, stage2]);
        let ctx = make_ctx("What is Rust?");
        let output = orchestrator.execute(&ctx).expect("execute");
        let result: crate::pipeline::PipelineResult =
            output.data_as().expect("data_as");
        assert!(!result.rejected);
        assert_eq!(result.decisions.len(), 2);
        assert_eq!(result.decisions[0].verdict, StageVerdict::Passed);
        assert_eq!(result.decisions[1].verdict, StageVerdict::Passed);
    }

    #[test]
    fn test_pipeline_early_termination_on_rejected() {
        let stage1 = Arc::new(DeterministicPreFilter::new());
        let stage2 = Arc::new(RouterStage::new(RoutingPolicy::LocalFirst));
        let orchestrator = PipelineOrchestrator::new(vec![stage1, stage2]);
        // Command dispatch should reject immediately, RouterStage never runs
        let ctx = make_ctx("/help");
        let output = orchestrator.execute(&ctx).expect("execute");
        let result: crate::pipeline::PipelineResult =
            output.data_as().expect("data_as");
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
        assert_eq!(
            &*orchestrator.provides()[0],
            "pipeline.result"
        );
    }

    #[test]
    fn test_pipeline_orchestrator_builder() {
        let stage = Arc::new(DeterministicPreFilter::new());
        let orchestrator = PipelineOrchestrator::builder()
            .push(stage)
            .build();
        assert_eq!(orchestrator.name(), "pipeline.orchestrator");
    }

    #[test]
    fn test_deterministic_prefilter_describable() {
        let filter = DeterministicPreFilter::new();
        let desc = filter.describe();
        assert_eq!(desc["type"], "object");
    }

    #[test]
    fn test_router_stage_describable() {
        let stage = RouterStage::new(RoutingPolicy::LocalFirst);
        let desc = stage.describe();
        assert_eq!(desc["type"], "object");
    }
}