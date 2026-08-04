//! Rubric-based test fixtures for `ResultScorer` and `Summarizer`.
//!
//! All tests use `StubChatBackend` — no live model, no network.
// Stub scores are compared against literal thresholds — deliberate.
#![allow(clippy::float_cmp)]

use std::sync::Arc;

use fluent_wvr::prelude::*;

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
    ctx.metadata
        .insert("content".into(), MetadataValue::String(content.to_string()));
    ctx
}

fn stub_client(response: &str) -> Arc<dyn guidance_llm::client::ChatBackend> {
    Arc::new(StubChatBackend::always(response))
}

// ── ResultScorer: correct answer is accepted ─────────────────────────────────

#[test]
fn test_scorer_accepts_correct_math_answer() {
    let scorer = ResultScorer::new(
        stub_client(
            r#"{"score":0.95,"accepted":true,"reason":"correct and complete","summary":"2+2=4 is correct"}"#,
        ),
        0.7,
    );
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
    let scorer = ResultScorer::new(
        stub_client(
            r#"{"score":0.15,"accepted":false,"reason":"garbage output","summary":"Response is incoherent"}"#,
        ),
        0.7,
    );
    let ctx = scorer_ctx("Write a Rust function", "purple monkey dishwasher");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(!result.accepted, "garbage answer should be rejected");
    assert!(result.score < 0.7);
}

// ── ResultScorer: borderline answer is scored correctly ──────────────────────

#[test]
fn test_scorer_borderline_answer_below_threshold() {
    let scorer = ResultScorer::new(
        stub_client(
            r#"{"score":0.55,"accepted":false,"reason":"partially correct","summary":"Answer starts correctly but lacks detail"}"#,
        ),
        0.7,
    );
    let ctx = scorer_ctx("Explain Rust ownership", "Ownership is a set of rules.");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(
        !result.accepted,
        "borderline answer should be rejected at 0.7 threshold"
    );
    assert!(result.score < 0.7);
}

#[test]
fn test_scorer_borderline_answer_above_threshold() {
    let scorer = ResultScorer::new(
        stub_client(
            r#"{"score":0.75,"accepted":true,"reason":"mostly correct","summary":"Good explanation of ownership"}"#,
        ),
        0.7,
    );
    let ctx = scorer_ctx(
        "Explain Rust ownership",
        "Ownership ensures memory safety without a garbage collector.",
    );
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(
        result.accepted,
        "borderline answer at 0.75 should be accepted at 0.7 threshold"
    );
    assert!(result.score >= 0.7);
}

// ── ResultScorer: rejection produces compact one-line summary ────────────────

#[test]
fn test_scorer_rejection_produces_compact_summary() {
    let scorer = ResultScorer::new(
        stub_client(
            r#"{"score":0.1,"accepted":false,"reason":"completely wrong","summary":"The answer is completely wrong and unrelated to the query about Fibonacci numbers."}"#,
        ),
        0.7,
    );
    let ctx = scorer_ctx("Write Fibonacci", "The sky is blue.");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert!(!result.accepted);
    assert!(
        !result.summary.is_empty(),
        "rejected result should have a summary"
    );
    assert!(
        result.summary.len()
            <= result
                .summary
                .find('.')
                .map_or(result.summary.len(), |i| i + 1),
        "rejected summary should be compact (single sentence)"
    );
}

// ── Summarizer: condenses text ─────────────────────────────────────────────

#[test]
fn test_summarizer_condenses_long_text() {
    let summarizer = Summarizer::new(
        stub_client("A Rust function computes Fibonacci numbers using recursion."),
        50,
    );
    let ctx =
        summarizer_ctx("fn fib(n: u64) -> u64 { if n <= 1 { n } else { fib(n-1) + fib(n-2) } }");
    let output = summarizer.execute(&ctx).expect("execute");
    let data: serde_json::Value = output.data_as().expect("data_as");
    let summary = data["summary"].as_str().expect("summary");
    assert!(!summary.is_empty(), "summary should not be empty");
}

// ── Summarizer: direct call via summarize_text ──────────────────────────────

#[test]
fn test_summarizer_direct_call() {
    let summarizer = Summarizer::new(stub_client("compact summary via direct call"), 20);
    let summary = summarizer
        .summarize_text("long text to summarize")
        .expect("summarize_text");
    assert_eq!(summary, "compact summary via direct call");
}

// ── All tests use StubChatBackend — no live model ───────────────────────────

#[test]
fn test_scorer_uses_stub_not_live_model() {
    let scorer = ResultScorer::new(
        stub_client(r#"{"score":0.99,"accepted":true,"reason":"stub","summary":"stub response"}"#),
        0.5,
    );
    let ctx = scorer_ctx("test", "stub response");
    let output = scorer.execute(&ctx).expect("execute");
    let result: ScoredResult = output.data_as().expect("data_as");
    assert_eq!(result.score, 0.99);
    assert_eq!(result.reason, "stub");
}

// ── Error cases ─────────────────────────────────────────────────────────────

#[test]
fn test_scorer_missing_request_returns_error() {
    let scorer = ResultScorer::new(
        stub_client(r#"{"score":0.5,"accepted":false,"reason":"test","summary":"test summary"}"#),
        0.7,
    );
    let ctx = WorkContext::default();
    let result = scorer.execute(&ctx);
    assert!(result.is_err());
}

#[test]
fn test_scorer_missing_response_returns_error() {
    let scorer = ResultScorer::new(
        stub_client(r#"{"score":0.5,"accepted":false,"reason":"test","summary":"test summary"}"#),
        0.7,
    );
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
    let summarizer = Summarizer::new(stub_client("summary of empty"), 50);
    let ctx = summarizer_ctx("some content");
    let output = summarizer.execute(&ctx).expect("execute");
    assert!(output.success);
}

// ── FieldAccess and Describable ─────────────────────────────────────────────

#[test]
fn test_scorer_set_field_returns_not_found() {
    let mut scorer = ResultScorer::new(
        stub_client(r#"{"score":0.5,"accepted":false,"reason":"test","summary":"test"}"#),
        0.7,
    );
    let err = scorer.set_field("nonexistent", "value").unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[test]
fn test_summarizer_get_field_returns_not_found() {
    let summarizer = Summarizer::new(stub_client("test"), 50);
    let err = summarizer.get_field("nonexistent").unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[test]
fn test_scorer_field_names_empty() {
    let scorer = ResultScorer::new(
        stub_client(r#"{"score":0.5,"accepted":false,"reason":"test","summary":"test"}"#),
        0.7,
    );
    assert!(scorer.field_names().is_empty());
}

#[test]
fn test_summarizer_field_names_empty() {
    let summarizer = Summarizer::new(stub_client("test"), 50);
    assert!(summarizer.field_names().is_empty());
}
