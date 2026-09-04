use super::*;
use crate::test_stubs::StubChatBackend;

fn test_client() -> Arc<dyn ChatBackend> {
    Arc::new(StubChatBackend::always(
        r#"{"score": 0.5, "accepted": false, "reason": "test", "summary": "test summary"}"#,
    ))
}

#[test]
fn test_result_scorer_name() {
    let scorer = ResultScorer::new(test_client(), 0.7);
    assert_eq!(scorer.name(), "pipeline.scorer");
}

#[test]
fn test_result_scorer_describable() {
    let scorer = ResultScorer::new(test_client(), 0.7);
    let desc = scorer.describe();
    assert_eq!(desc["type"], "object");
}

#[test]
fn test_result_scorer_missing_response() {
    let scorer = ResultScorer::new(test_client(), 0.7);
    let ctx = WorkContext::default();
    let result = scorer.execute(&ctx);
    assert!(result.is_err());
}

#[test]
fn test_result_scorer_accepts_pristine_json() {
    // Characterization (M2.1): pristine LLM JSON parses to identical values.
    // Must pass unchanged after the tolerant-codec migration.
    let scorer = ResultScorer::new(test_client(), 0.7);
    let mut ctx = WorkContext::default();
    ctx.set_structured(
        "request",
        &serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
        }),
    );
    ctx.metadata.insert(
        "response".into(),
        MetadataValue::String("the full response text".into()),
    );
    let out = scorer.execute(&ctx).expect("pristine JSON must score");
    let scored: ScoredResult = out.data_as().expect("typed ScoredResult");
    assert_eq!(scored.score, 0.5);
    assert!(!scored.accepted, "0.5 < 0.7 threshold");
    assert_eq!(scored.content, "the full response text");
    assert_eq!(scored.reason, "test");
}

#[test]
fn test_result_scorer_recovers_fenced_json() {
    // M2.6: the intended widening — fence/prose-wrapped LLM JSON is
    // recovered by the tolerant codec instead of erroring.
    let scorer = ResultScorer::new(
        Arc::new(StubChatBackend::always(
            "Here is my score:\n```json\n{\"score\": 0.9, \"accepted\": true, \"reason\": \"good\", \"summary\": \"a good answer\"}\n```",
        )),
        0.7,
    );
    let mut ctx = WorkContext::default();
    ctx.set_structured(
        "request",
        &serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
        }),
    );
    ctx.metadata.insert(
        "response".into(),
        MetadataValue::String("the full response text".into()),
    );
    let out = scorer.execute(&ctx).expect("fenced JSON must recover");
    let scored: ScoredResult = out.data_as().expect("typed ScoredResult");
    assert_eq!(scored.score, 0.9);
    assert!(scored.accepted, "0.9 >= 0.7 threshold");
}

#[test]
fn test_result_scorer_rejects_garbage() {
    // Characterization (M2.1): non-JSON LLM text is a hard scorer error today.
    // The tolerant-codec migration intentionally widens this for
    // fence/prose-wrapped JSON; pure garbage must stay an error.
    let scorer = ResultScorer::new(
        Arc::new(StubChatBackend::always("definitely not json {{{")),
        0.7,
    );
    let mut ctx = WorkContext::default();
    ctx.set_structured(
        "request",
        &serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "hello"}],
        }),
    );
    ctx.metadata.insert(
        "response".into(),
        MetadataValue::String("the full response text".into()),
    );
    let err = scorer.execute(&ctx).expect_err("garbage must fail");
    assert!(
        err.to_string().contains("scorer parse error"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_summarizer_name() {
    let summarizer = Summarizer::new(test_client(), 50);
    assert_eq!(summarizer.name(), "pipeline.summarizer");
}

#[test]
fn test_summarizer_describable() {
    let summarizer = Summarizer::new(test_client(), 50);
    let desc = summarizer.describe();
    assert_eq!(desc["type"], "object");
}

#[test]
fn test_summarizer_missing_content() {
    let summarizer = Summarizer::new(test_client(), 50);
    let ctx = WorkContext::default();
    let result = summarizer.execute(&ctx);
    assert!(result.is_err());
}
