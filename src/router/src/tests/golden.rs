//! Golden test set for the router pipeline.
//!
//! A checked-in labeled corpus covering intent categories, quality levels,
//! PII presence, and adversarial edge cases. Tests validate that pipeline
//! stages produce the expected decisions for each case.
//!
//! All tests use the real `PipelineOrchestrator` with `TranscriptProvider`
//! injected — no LLM, no network.

use crate::pipeline_types::PipelineStage;
use crate::testing::mock::TranscriptProvider;
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};
use std::sync::Arc;

use crate::config::RouterConfig;
use crate::pipeline::{PipelineOrchestrator, PipelineResult};
use fluent_wvr::prelude::*;
use guidance_llm::client::ChatBackend;

struct GoldenCase {
    name: &'static str,
    input: &'static str,
    expected_reject_stage: Option<PipelineStage>,
    expected_pii: &'static [&'static str],
}

fn default_provider() -> TranscriptProvider {
    TranscriptProvider::new(std::collections::HashMap::new())
}


fn make_test_config() -> RouterConfig {
    match serde_json::from_str::<RouterConfig>(r#"{
        "pipelines": {"default": {"deterministic_prefilter": true, "classifier": true}},
        "models": {"fast": {"endpoint": "http://localhost:8080/v1/chat/completions", "name": "fast", "intelligence": 1, "cost_input": 0.000001, "cost_output": 0.000006, "cost_cached_read": 0.0000004, "speed": 10, "total_timeout_ms": 5000, "idle_timeout_ms": 2000, "stream": false, "filter_thinking": false, "retry_count": 0, "retry_base_interval_s": 1}},
        "model_groups": {"fast": ["fast"]},
        "routes": {"fast": {"group": "fast", "pipelines": ["default"]}},
        "default_route": "fast"
    }"#) {
        Ok(c) => c,
        Err(e) => panic!("invalid test config: {e}"),
    }
}

fn make_pipeline(provider: TranscriptProvider) -> PipelineOrchestrator {
    let config = make_test_config();
    let backend = Arc::new(provider) as Arc<dyn ChatBackend>;
    config
        .build_named_pipeline_with_backend("default", Some(backend))
        .expect("default pipeline should build")
}

fn make_request(text: &str) -> RouterRequest {
    RouterRequest {
        model: "test-model".into(),
        messages: vec![RouterMessage {
            role: "user".into(),
            content: RouterMessageContent::Text(text.into()),
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        max_tokens: None,
        stream: None,
        tools: None,
        tool_choice: None,
        session_id: Some("golden-test-session".into()),
        agent_id: None,
        adapter: None,
        metadata: Default::default(),
    }
}

fn route(pipeline: &PipelineOrchestrator, request: &RouterRequest) -> Result<PipelineResult, WorkError> {
    let request_json = serde_json::to_string(request)
        .map_err(|e| WorkError::Execution(format!("serialization error: {e}")))?;
    let mut ctx = WorkContext::default();
    ctx.metadata.insert("request".into(), MetadataValue::String(request_json));
    let output = pipeline.execute(&ctx)?;
    output.data_take().map_err(|e| WorkError::Execution(e.to_string()))
}

const INTENT_CASES: &[GoldenCase] = &[
    GoldenCase {
        name: "question_what_is_rust",
        input: "What is Rust?",
        expected_reject_stage: None,
        expected_pii: &[],
    },
    GoldenCase {
        name: "command_with_slash",
        input: "/help",
        expected_reject_stage: Some(PipelineStage::DeterministicPreFilter),
        expected_pii: &[],
    },
    GoldenCase {
        name: "command_with_dot",
        input: ".stats",
        expected_reject_stage: Some(PipelineStage::DeterministicPreFilter),
        expected_pii: &[],
    },
    GoldenCase {
        name: "creative_prompt",
        input: "Write a poem about a digital ocean.",
        expected_reject_stage: None,
        expected_pii: &[],
    },
    GoldenCase {
        name: "code_request",
        input: "Write a Rust function to compute Fibonacci numbers.",
        expected_reject_stage: None,
        expected_pii: &[],
    },
    GoldenCase {
        name: "chitchat",
        input: "Hello! How are you today?",
        expected_reject_stage: None,
        expected_pii: &[],
    },
];

const PII_CASES: &[GoldenCase] = &[
    GoldenCase {
        name: "ssn_present",
        input: "My SSN is 123-45-6789.",
        expected_reject_stage: Some(PipelineStage::DeterministicPreFilter),
        expected_pii: &["ssn"],
    },
    GoldenCase {
        name: "email_present",
        input: "Contact me at user@example.com.",
        expected_reject_stage: Some(PipelineStage::DeterministicPreFilter),
        expected_pii: &["email"],
    },
    GoldenCase {
        name: "card_number_present",
        input: "My card is 4111-1111-1111-1111.",
        expected_reject_stage: Some(PipelineStage::DeterministicPreFilter),
        expected_pii: &["card_number"],
    },
    GoldenCase {
        name: "phone_present",
        input: "Call me at (555) 123-4567.",
        expected_reject_stage: Some(PipelineStage::DeterministicPreFilter),
        expected_pii: &["phone"],
    },
    GoldenCase {
        name: "multiple_pii",
        input: "Email: user@example.com, SSN: 123-45-6789, phone: 555-123-4567.",
        expected_reject_stage: Some(PipelineStage::DeterministicPreFilter),
        expected_pii: &["email", "ssn", "phone"],
    },
    GoldenCase {
        name: "no_pii",
        input: "What is the capital of France?",
        expected_reject_stage: None,
        expected_pii: &[],
    },
];

const ADVERSARIAL_CASES: &[GoldenCase] = &[
    GoldenCase {
        name: "prose_resembling_command",
        input: ".help I need information about Rust",
        expected_reject_stage: Some(PipelineStage::DeterministicPreFilter),
        expected_pii: &[],
    },
    GoldenCase {
        name: "empty_message",
        input: "",
        expected_reject_stage: None,
        expected_pii: &[],
    },
    GoldenCase {
        name: "special_characters",
        input: "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
        expected_reject_stage: None,
        expected_pii: &[],
    },
];

fn run_golden_cases(cases: &[GoldenCase], provider: TranscriptProvider) {
    let pipeline = make_pipeline(provider);

    for case in cases {
        let request = make_request(case.input);
        let result = route(&pipeline, &request);

        match case.expected_reject_stage {
            None => {
                assert!(
                    result.is_ok(),
                    "case '{}' expected pipeline to complete, got error: {:?}",
                    case.name,
                    result.err()
                );
                if let Ok(pipeline_result) = &result {
                    assert!(
                        !pipeline_result.rejected,
                        "case '{}' should not be rejected, but was rejected with: {:?}",
                        case.name,
                        pipeline_result.reject_reason
                    );
                }
            }
            Some(expected_stage) => {
                assert!(
                    result.is_ok(),
                    "case '{}' pipeline should handle rejection gracefully, got error: {:?}",
                    case.name,
                    result.err()
                );
                if let Ok(pipeline_result) = &result {
                    assert!(
                        pipeline_result.rejected,
                        "case '{}' should be rejected",
                        case.name
                    );
                    let reject_stage = &pipeline_result.decisions[0].stage;
                    assert_eq!(
                        reject_stage, &expected_stage,
                        "case '{}': expected rejection at {:?}, got {:?} with decision: {:?}",
                        case.name, expected_stage, reject_stage, pipeline_result.decisions[0]
                    );
                }
            }
        }

        if !case.expected_pii.is_empty() {
            if let Ok(pipeline_result) = &result {
                for decision in &pipeline_result.decisions {
                    if decision.stage == PipelineStage::DeterministicPreFilter {
                        let pii = decision
                            .metadata
                            .get("pii_classes")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter().filter_map(|c| c.as_str()).collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        for expected in case.expected_pii {
                            assert!(
                                pii.contains(expected),
                                "case '{}': expected PII class '{}' to be detected, got: {:?}",
                                case.name,
                                expected,
                                pii
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn golden_intent_categories() {
    run_golden_cases(INTENT_CASES, default_provider());
}

#[test]
fn golden_pii_detection() {
    run_golden_cases(PII_CASES, default_provider());
}

#[test]
fn golden_adversarial_cases() {
    run_golden_cases(ADVERSARIAL_CASES, default_provider());
}

#[test]
fn golden_pipeline_has_all_stages() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request("What is the capital of France?");
    let result = route(&pipeline, &request).expect("pipeline should complete");
    assert!(!result.rejected, "normal request should not be rejected");
    assert!(
        result.decisions.len() >= 2,
        "pipeline should have at least 2 decisions, got {}",
        result.decisions.len()
    );
}

#[test]
fn golden_command_is_rejected() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request("/help");
    let result = route(&pipeline, &request).expect("pipeline should handle command");
    assert!(result.rejected, "command should be rejected");
}

#[test]
fn golden_unknown_command_is_rejected() {
    let pipeline = make_pipeline(default_provider());
    let request = make_request("/nonexistent_command_xyz");
    let result = route(&pipeline, &request).expect("pipeline should handle unknown cmd");
    assert!(result.rejected, "unknown command should be rejected");
    assert!(
        result.reject_reason.unwrap_or_default().contains("unknown command"),
        "reject reason should mention unknown command"
    );
}
