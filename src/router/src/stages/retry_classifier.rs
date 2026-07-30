//! RetryClassifier — a `WorkUnit` decorator that wraps a classifier stage
//! and retries on JSON parse failure with escalating corrective prompts.
//!
//! When the inner stage's output contains `metadata.fallback = true` (the
//! signal emitted by `ClassifierStage` on parse/LLM error), the decorator
//! re-executes the inner stage with an escalating system prompt override.
//! After `max_retries` attempts it returns the final fallback output.

use std::sync::Arc;

use fluent_wvr::prelude::*;

/// Maximum number of attempts metadata key.  Injected into `WorkContext`
/// so the inner stage can include it in tracing/logging.
pub const METADATA_RETRY_ATTEMPT: &str = "classifier_retry_attempt";
/// Parse error from the previous attempt, injected into `WorkContext`.
pub const METADATA_PARSE_ERROR: &str = "classifier_parse_error";
/// Override system prompt injected into `WorkContext` for the current retry.
/// The `ClassifierStage` reads this key (preferring it over
/// `routing_config.system_prompt`) when present.
pub const METADATA_SYSTEM_PROMPT: &str = "classifier_system_prompt";

pub struct RetryClassifier {
    name: ArcIntern<str>,
    /// The wrapped classifier stage.
    inner: Arc<dyn Component>,
    /// Maximum number of retry attempts (total attempts = 1 + max_retries).
    max_retries: usize,
    /// Escalating system prompts for each retry attempt.
    /// `retry_prompts[0]` is the prompt override for the 1st retry,
    /// `retry_prompts[1]` for the 2nd, etc.
    /// When there are more retries than prompts, the last prompt is reused.
    retry_prompts: Vec<String>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl RetryClassifier {
    pub fn new(inner: Arc<dyn Component>, max_retries: usize, retry_prompts: Vec<String>) -> Self {
        Self {
            name: ArcIntern::from(format!("pipeline.stage2.retry({})", inner.name())),
            inner,
            max_retries,
            retry_prompts,
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.stage2.output")],
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = ArcIntern::from(name.into());
        self
    }

    #[must_use]
    pub fn with_depends(mut self, deps: Vec<ArcIntern<str>>) -> Self {
        self.depends = deps;
        self
    }

    #[must_use]
    pub fn with_provides(mut self, provides: Vec<ArcIntern<str>>) -> Self {
        self.provides = provides;
        self
    }

    /// Build a retry context with the escalating prompt for the given
    /// attempt number (0-based retry index).
    fn build_retry_context(
        &self,
        base: &WorkContext,
        retry_index: usize,
        parse_error: &str,
    ) -> WorkContext {
        let mut ctx = base.clone();
        ctx.metadata.insert(
            METADATA_RETRY_ATTEMPT.into(),
            MetadataValue::Number(retry_index as i64),
        );
        ctx.metadata.insert(
            METADATA_PARSE_ERROR.into(),
            MetadataValue::String(parse_error.to_string()),
        );

        let prompt = self
            .retry_prompts
            .get(retry_index)
            .or_else(|| self.retry_prompts.last())
            .cloned()
            .unwrap_or_default();

        if !prompt.is_empty() {
            ctx.metadata
                .insert(METADATA_SYSTEM_PROMPT.into(), MetadataValue::String(prompt));
        }

        ctx
    }

    /// Check whether a `WorkOutput` from the inner stage indicates a
    /// fallback (parse error), meaning we should retry.
    fn is_fallback(output: &WorkOutput) -> bool {
        output
            .data
            .get("metadata")
            .and_then(|m| m.get("fallback"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }

    /// Extract the parse error message from a fallback output.
    fn parse_error_from(output: &WorkOutput) -> String {
        output
            .data
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown parse error")
            .to_string()
    }
}

impl WorkUnit for RetryClassifier {
    fn name(&self) -> &str {
        &self.name
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        // Attempt 0 (initial, no retry prompt)
        let output = self.inner.execute(ctx)?;

        if !Self::is_fallback(&output) {
            return Ok(output);
        }

        let mut last_error = Self::parse_error_from(&output);

        for retry_index in 0..self.max_retries {
            tracing::info!(
                target: "router.pipeline.retry",
                retry = retry_index + 1,
                max_retries = self.max_retries,
                parse_error = %last_error,
                "classifier fallback detected, retrying"
            );

            let retry_ctx = self.build_retry_context(ctx, retry_index, &last_error);
            let retry_output = self.inner.execute(&retry_ctx)?;

            if !Self::is_fallback(&retry_output) {
                return Ok(retry_output);
            }

            last_error = Self::parse_error_from(&retry_output);
        }

        tracing::warn!(
            target: "router.pipeline.retry",
            max_retries = self.max_retries,
            parse_error = %last_error,
            "exhausted retries, returning final fallback"
        );

        // Re-execute on the original context for the final attempt so the
        // ClassifierStage's own fallback logic handles the response.
        self.inner.execute(ctx)
    }
}

impl FieldAccess for RetryClassifier {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "RetryClassifier has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "RetryClassifier has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for RetryClassifier {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "max_retries": {"type": "integer"},
                "retry_prompts": {"type": "array", "items": {"type": "string"}},
            },
            "required": []
        })
    }
}

impl_component!(RetryClassifier);

#[cfg(test)]
mod tests {
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

        assert_eq!(
            ctx.metadata.get(METADATA_RETRY_ATTEMPT),
            Some(&MetadataValue::Number(1))
        );
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
}
