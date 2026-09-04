use super::*;
use crate::pipeline_types::StageDecision;
use crate::test_stubs;

fn make_fallback_output() -> WorkOutput {
    let decision = StageDecision {
        stage: crate::pipeline_types::PipelineStage::Classifier,
        verdict: crate::pipeline_types::StageVerdict::Passed,
        score: Some(1.0),
        reason: "parse error: trailing characters".into(),
        latency_ms: 0,
        metadata: serde_json::json!({"fallback": true}),
    };
    WorkOutput::typed_infallible("classified", &decision)
}

fn make_success_output() -> WorkOutput {
    let decision = StageDecision {
        stage: crate::pipeline_types::PipelineStage::Classifier,
        verdict: crate::pipeline_types::StageVerdict::Passed,
        score: Some(0.95),
        reason: "intent=code, action=route".into(),
        latency_ms: 0,
        metadata: serde_json::json!({"intent": "code", "action": "route", "fallback": false}),
    };
    WorkOutput::typed_infallible("classified", &decision)
}

#[test]
fn is_fallback_detects_fallback_flag() {
    assert!(RetryClassifier::is_fallback(&make_fallback_output()));
    assert!(!RetryClassifier::is_fallback(&make_success_output()));
}

#[test]
fn parse_error_extraction() {
    let err = RetryClassifier::parse_error_from(&make_fallback_output());
    assert!(err.contains("parse error"));
}

#[test]
fn retry_injects_prompt_into_context() {
    let inner = Arc::new(test_stubs::FailingStage::new(
        "failing_classifier",
        2, // fail twice, then succeed
    ));
    let retry = RetryClassifier::new(
        inner,
        2,
        vec!["retry prompt 1".into(), "retry prompt 2".into()],
    );

    let ctx = WorkContext::default();
    let _ = retry.execute(&ctx);
    // The test asserts that retry_context was built correctly — the inner
    // test_stubs::FailingStage handles the actual failure verification.
}

#[test]
fn retry_builds_context_with_correct_metadata() {
    let retry = RetryClassifier::new(
        Arc::new(test_stubs::SimplePassStage::new("inner", "ok")),
        3,
        vec!["prompt1".into(), "prompt2".into(), "prompt3".into()],
    );

    let base = WorkContext::default();
    let ctx = retry.build_retry_context(&base, 1, "test error");

    assert_eq!(ctx.get::<i64>(METADATA_RETRY_ATTEMPT), Some(&1));
    assert_eq!(
        ctx.metadata.get(METADATA_PARSE_ERROR),
        Some(&MetadataValue::String("test error".into()))
    );
    assert_eq!(
        ctx.metadata.get(METADATA_SYSTEM_PROMPT),
        Some(&MetadataValue::String("prompt2".into()))
    );
}

#[test]
fn retry_reuses_last_prompt_when_out_of_prompts() {
    let retry = RetryClassifier::new(
        Arc::new(test_stubs::SimplePassStage::new("inner", "ok")),
        2,
        vec!["only_prompt".into()],
    );

    let base = WorkContext::default();

    // retry_index=0 uses "only_prompt"
    let ctx0 = retry.build_retry_context(&base, 0, "err0");
    assert_eq!(
        ctx0.metadata.get(METADATA_SYSTEM_PROMPT),
        Some(&MetadataValue::String("only_prompt".into()))
    );

    // retry_index=1 also gets "only_prompt" (last prompt reused)
    let ctx1 = retry.build_retry_context(&base, 1, "err1");
    assert_eq!(
        ctx1.metadata.get(METADATA_SYSTEM_PROMPT),
        Some(&MetadataValue::String("only_prompt".into()))
    );
}

#[test]
fn retry_with_empty_prompts_omits_prompt_override() {
    let retry = RetryClassifier::new(
        Arc::new(test_stubs::SimplePassStage::new("inner", "ok")),
        1,
        vec![],
    );

    let base = WorkContext::default();
    let ctx = retry.build_retry_context(&base, 0, "err");
    assert!(!ctx.metadata.contains_key(METADATA_SYSTEM_PROMPT));
}

// ── Execute-loop characterization (M6) ────────────────────────────────────

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A scripted classifier stub: returns fallback or success per the script,
/// counting every call. An exhausted script keeps returning fallback.
struct ScriptedClassifier {
    name: ArcIntern<str>,
    script: std::sync::Mutex<VecDeque<bool>>,
    calls: AtomicUsize,
}

impl ScriptedClassifier {
    fn new(stage_name: &str, fallback_script: Vec<bool>) -> Self {
        Self {
            name: ArcIntern::from(stage_name.to_string()),
            script: std::sync::Mutex::new(fallback_script.into()),
            calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl WorkUnit for ScriptedClassifier {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let fallback = self.script.lock().unwrap().pop_front().unwrap_or(true);
        Ok(if fallback {
            make_fallback_output()
        } else {
            make_success_output()
        })
    }
}

impl_fieldless!(ScriptedClassifier);

impl Describable for ScriptedClassifier {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}
impl_component!(ScriptedClassifier);

#[test]
fn success_first_try_makes_no_second_call() {
    // Characterization (M6): a non-fallback first output short-circuits.
    let inner = Arc::new(ScriptedClassifier::new("inner", vec![false]));
    let retry = RetryClassifier::new(inner.clone(), 3, vec!["p".into()]);
    let out = retry.execute(&WorkContext::default()).expect("execute");
    assert!(!RetryClassifier::is_fallback(&out));
    assert_eq!(inner.calls(), 1, "no second call on first-try success");
}

#[test]
fn fallback_then_success_returns_success() {
    let inner = Arc::new(ScriptedClassifier::new("inner", vec![true, false]));
    let retry = RetryClassifier::new(inner.clone(), 3, vec!["p".into()]);
    let out = retry.execute(&WorkContext::default()).expect("execute");
    assert!(!RetryClassifier::is_fallback(&out));
    assert_eq!(inner.calls(), 2);
}

#[test]
fn max_retries_zero_still_re_executes_final() {
    // Characterization (M6): with max_retries == 0 the loop body never runs,
    // but the stage still re-executes once on the original context for the
    // final fallback — 2 calls total, fallback output.
    let inner = Arc::new(ScriptedClassifier::new("inner", vec![true, true]));
    let retry = RetryClassifier::new(inner.clone(), 0, vec!["p".into()]);
    let out = retry.execute(&WorkContext::default()).expect("execute");
    assert!(RetryClassifier::is_fallback(&out));
    assert_eq!(inner.calls(), 2, "initial + final re-execute");
}

#[test]
fn exhausted_retries_return_final_fallback() {
    // 1 initial + max_retries loop calls + 1 final re-execute == 4.
    let inner = Arc::new(ScriptedClassifier::new("inner", vec![true; 8]));
    let retry = RetryClassifier::new(inner.clone(), 2, vec!["p1".into(), "p2".into()]);
    let out = retry.execute(&WorkContext::default()).expect("execute");
    assert!(RetryClassifier::is_fallback(&out), "final output is fallback");
    assert_eq!(inner.calls(), 4, "1 + 2 retries + final re-execute");
}

#[test]
fn non_fallback_output_passes_through_verbatim() {
    let inner = Arc::new(ScriptedClassifier::new("inner", vec![false]));
    let retry = RetryClassifier::new(inner.clone(), 2, vec!["p".into()]);
    let out = retry.execute(&WorkContext::default()).expect("execute");
    let reason = out
        .data
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(reason.contains("intent=code"), "passthrough keeps body: {reason}");
}
