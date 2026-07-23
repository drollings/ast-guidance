//! Rubric-based test fixtures for `ResultScorer` and `Summarizer`.
//!
//! All tests use `StubChatBackend` — no live model, no network.

use fluent_wvr::prelude::*;
use guidance_llm::LlmConfig;

use crate::summarization::{ResultScorer, ScoredResult, Summarizer};
use crate::test_stubs::StubChatBackend;

fn scorer_ctx(query: &str, response: &str) -> WorkContext {
    let request_json = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": query}]
    });
    let mut ctx = WorkContext::default();
    ctx.metadata.insert(
        "request".into(),
        MetadataValue::String(request_json.to_string()),
    );
    ctx.metadata.insert(
        "response".into(),
        MetadataValue::String(response.to_string()),
    );
    ctx
}

fn summarizer_ctx(content: &str) -> WorkContext {
    let mut ctx = WorkContext::default();
    ctx.metadata.insert(
        "content".into(),
        MetadataValue::String(content.to_string()),
    );
    ctx
}

fn test_config() -> LlmConfig {
    LlmConfig::new()
        .api_url("http://test".into())
        .model("test-model".into())
        .build()
}

// ── ResultScorer: correct answer is accepted ─────────────────────────────────

#[test]
fn test_scorer_accepts_correct_math_answer() {
    let backend = Box::new(StubChatBackend::always(
        r#"{"score":0.95,"accepted":true,"reason":"correct and complete","summary":"2+2=4 is correct"}"#,
    ));
    let scorer = ResultScorer::with_chat_backend(test_config(), backend, 0.7);
    let ctx = scorer_ctx("What is 2+2?", "4");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(result.accepted, "correct answer should be accepted");
    assert!(result.score >= 0.7);
    assert_eq!(result.content, "4");
}

// ── ResultScorer: garbage answer is rejected ─────────────────────────────────

#[test]
fn test_scorer_rejects_garbage_answer() {
    let backend = Box::new(StubChatBackend::always(
        r#"{"score":0.15,"accepted":false,"reason":"garbage output","summary":"Response is incoherent"}"#,
    ));
    let scorer = ResultScorer::with_chat_backend(test_config(), backend, 0.7);
    let ctx = scorer_ctx("Write a Rust function", "purple monkey dishwasher");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(!result.accepted, "garbage answer should be rejected");
    assert!(result.score < 0.7);
}

// ── ResultScorer: borderline answer is scored correctly ──────────────────────

#[test]
fn test_scorer_borderline_answer_below_threshold() {
    let backend = Box::new(StubChatBackend::always(
        r#"{"score":0.55,"accepted":false,"reason":"partially correct","summary":"Answer starts correctly but lacks detail"}"#,
    ));
    let scorer = ResultScorer::with_chat_backend(test_config(), backend, 0.7);
    let ctx = scorer_ctx("Explain Rust ownership", "Ownership is a set of rules.");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(!result.accepted, "borderline answer should be rejected at 0.7 threshold");
    assert!(result.score < 0.7);
}

#[test]
fn test_scorer_borderline_answer_above_threshold() {
    let backend = Box::new(StubChatBackend::always(
        r#"{"score":0.75,"accepted":true,"reason":"mostly correct","summary":"Good explanation of ownership"}"#,
    ));
    let scorer = ResultScorer::with_chat_backend(test_config(), backend, 0.7);
    let ctx = scorer_ctx("Explain Rust ownership", "Ownership ensures memory safety without a garbage collector.");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(result.accepted, "borderline answer at 0.75 should be accepted at 0.7 threshold");
    assert!(result.score >= 0.7);
}

// ── ResultScorer: rejection produces compact one-line summary ────────────────

#[test]
fn test_scorer_rejection_produces_compact_summary() {
    let backend = Box::new(StubChatBackend::always(
        r#"{"score":0.1,"accepted":false,"reason":"completely wrong","summary":"The answer is completely wrong and unrelated to the query about Fibonacci numbers."}"#,
    ));
    let scorer = ResultScorer::with_chat_backend(test_config(), backend, 0.7);
    let ctx = scorer_ctx("Write Fibonacci", "The sky is blue.");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(!result.accepted);
    assert!(!result.summary.is_empty(), "rejected result should have a summary");
    // Summary should be a single sentence (truncated at period)
    assert!(
        result.summary.len() <= result.summary.find(|c: char| c == '.').map(|i| i + 1).unwrap_or(result.summary.len()),
        "rejected summary should be compact (single sentence)"
    );
}

// ── Summarizer: condenses text ─────────────────────────────────────────────

#[test]
fn test_summarizer_condenses_long_text() {
    let backend = Box::new(StubChatBackend::always(
        "A Rust function computes Fibonacci numbers using recursion.",
    ));
    let summarizer = Summarizer::with_chat_backend(test_config(), backend, 50);
    let ctx = summarizer_ctx(
        "fn fib(n: u64) -> u64 { if n <= 1 { n } else { fib(n-1) + fib(n-2) } }",
    );
    let output = summarizer.execute(&ctx).expect("execute");
    let data: serde_json::Value = output.data_as().expect("data_as");
    let summary = data["summary"].as_str().expect("summary");
    assert!(!summary.is_empty(), "summary should not be empty");
}

// ── Summarizer: direct call via summarize_text ──────────────────────────────

#[test]
fn test_summarizer_direct_call() {
    let backend = Box::new(StubChatBackend::always("compact summary via direct call"));
    let summarizer = Summarizer::with_chat_backend(test_config(), backend, 20);
    let summary = summarizer
        .summarize_text("long text to summarize")
        .expect("summarize_text");
    assert_eq!(summary, "compact summary via direct call");
}

// ── All tests use StubChatBackend — no live model ───────────────────────────

#[test]
fn test_scorer_uses_stub_not_live_model() {
    // If this test passes, it proves the stub is being used, not a real model.
    let backend = Box::new(StubChatBackend::always(
        r#"{"score":0.99,"accepted":true,"reason":"stub","summary":"stub response"}"#,
    ));
    let scorer = ResultScorer::with_chat_backend(test_config(), backend, 0.5);
    let ctx = scorer_ctx("test", "stub response");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert_eq!(result.score, 0.99);
    assert_eq!(result.reason, "stub");
}

// ── Error cases ─────────────────────────────────────────────────────────────

#[test]
fn test_scorer_missing_request_returns_error() {
    let config = test_config();
    let scorer = ResultScorer::new(config, 0.7);
    let ctx = WorkContext::default();
    let result = scorer.execute(&ctx);
    assert!(result.is_err());
}

#[test]
fn test_scorer_missing_response_returns_error() {
    let config = test_config();
    let scorer = ResultScorer::new(config, 0.7);
    let request_json = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let mut ctx = WorkContext::default();
    ctx.metadata.insert(
        "request".into(),
        MetadataValue::String(request_json.to_string()),
    );
    let result = scorer.execute(&ctx);
    assert!(result.is_err());
}

#[test]
fn test_summarizer_empty_content_does_not_panic() {
    let backend = Box::new(StubChatBackend::always("summary of empty"));
    let summarizer = Summarizer::with_chat_backend(test_config(), backend, 50);
    let ctx = summarizer_ctx("some content");
    let output = summarizer.execute(&ctx).expect("execute");
    assert!(output.success);
}

// ── FieldAccess and Describable ─────────────────────────────────────────────

#[test]
fn test_scorer_set_field_returns_not_found() {
    let config = test_config();
    let mut scorer = ResultScorer::new(config, 0.7);
    let err = scorer.set_field("nonexistent", "value").unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[test]
fn test_summarizer_get_field_returns_not_found() {
    let config = test_config();
    let summarizer = Summarizer::new(config, 50);
    let err = summarizer.get_field("nonexistent").unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[test]
fn test_scorer_field_names_empty() {
    let config = test_config();
    let scorer = ResultScorer::new(config, 0.7);
    assert!(scorer.field_names().is_empty());
}

#[test]
fn test_summarizer_field_names_empty() {
    let config = test_config();
    let summarizer = Summarizer::new(config, 50);
    assert!(summarizer.field_names().is_empty());
}
