//! Golden test set for the router pipeline.
//!
//! A checked-in labeled corpus covering intent categories, quality levels,
//! PII presence, and adversarial edge cases. Tests validate that mock
//! pipeline stages produce the expected decisions for each case.
//!
//! All tests use `MockRouter` — no LLM, no network.

use crate::pipeline_types::PipelineStage;
use crate::testing::mock::{MockFixtures, MockRouter};
use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};

/// A single golden test case.
struct GoldenCase {
    /// Display name for the test case.
    name: &'static str,
    /// Input user message.
    input: &'static str,
    /// Expected stage that rejects (or None if pipeline should complete).
    expected_reject_stage: Option<PipelineStage>,
    /// Expected PII classes (may be empty).
    expected_pii: &'static [&'static str],
}

/// Build a `MockFixtures` instance that returns "passed" for all
/// LLM-dependent stages, so the golden test only exercises the
/// deterministic pre-filter and router stages.
fn default_pass_fixtures() -> MockFixtures {
    MockFixtures::new()
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

// ── Intent Categories ───────────────────────────────────────────────────

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

// ── PII Cases ───────────────────────────────────────────────────────────

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

// ── Adversarial / Edge Cases ────────────────────────────────────────────

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

// ── Helpers ─────────────────────────────────────────────────────────────

fn run_golden_cases(cases: &[GoldenCase], fixtures: MockFixtures) {
    let router = MockRouter::new(fixtures);

    for case in cases {
        let request = make_request(case.input);
        let result = router.route(&request);

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

        // Validate PII classes were detected
        if !case.expected_pii.is_empty() {
            if let Ok(pipeline_result) = &result {
                for decision in &pipeline_result.decisions {
                    if decision.stage == PipelineStage::DeterministicPreFilter {
                        let pii = decision
                            .metadata
                            .get("pii_classes")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|c| c.as_str())
                                    .collect::<Vec<_>>()
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

// ── Tests ───────────────────────────────────────────────────────────────

#[test]
fn golden_intent_categories() {
    run_golden_cases(INTENT_CASES, default_pass_fixtures());
}

#[test]
fn golden_pii_detection() {
    run_golden_cases(PII_CASES, default_pass_fixtures());
}

#[test]
fn golden_adversarial_cases() {
    run_golden_cases(ADVERSARIAL_CASES, default_pass_fixtures());
}

#[test]
fn golden_pipeline_has_all_stages() {
    let fixtures = default_pass_fixtures();
    let router = MockRouter::new(fixtures);

    let request = make_request("What is the capital of France?");
    let result = router.route(&request).expect("pipeline should complete");
    assert!(!result.rejected, "normal request should not be rejected");
    assert!(
        result.decisions.len() >= 2,
        "pipeline should have at least 2 decisions, got {}",
        result.decisions.len()
    );
}

#[test]
fn golden_command_is_rejected() {
    let fixtures = default_pass_fixtures();
    let router = MockRouter::new(fixtures);

    let request = make_request("/help");
    let result = router.route(&request).expect("pipeline should handle command");
    assert!(result.rejected, "command should be rejected");
}

#[test]
fn golden_unknown_command_is_rejected() {
    let fixtures = default_pass_fixtures();
    let router = MockRouter::new(fixtures);

    let request = make_request("/nonexistent_command_xyz");
    let result = router
        .route(&request)
        .expect("pipeline should handle unknown cmd");
    assert!(result.rejected, "unknown command should be rejected");
    assert!(
        result
            .reject_reason
            .unwrap_or_default()
            .contains("unknown command"),
        "reject reason should mention unknown command"
    );
}
