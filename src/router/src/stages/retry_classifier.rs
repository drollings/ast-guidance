//! RetryClassifier — a `WorkUnit` decorator that wraps a classifier stage
//! and retries on JSON parse failure with escalating corrective prompts.
//!
//! When the inner stage's output contains `metadata.fallback = true` (the
//! signal emitted by `ClassifierStage` on parse/LLM error), the decorator
//! re-executes the inner stage with an escalating system prompt override.
//! After `max_retries` attempts it returns the final fallback output.
//!
//! NOTE (P6, evaluated-and-declined): this loop deliberately does NOT compose
//! `common_core::retry::retry_async` (or `SupervisedBatch`). Those retry on a
//! typed `Err`; this retries on an *output marker* (`metadata.fallback`) while
//! threading a *different* `WorkContext` per attempt (escalating prompt,
//! attempt number, parse error) and re-executing on the *original* context
//! when exhausted. Adapting marker→`Err`→marker plus per-attempt context
//! plumbing outweighs the ~15-line loop — structure-destroying, not
//! structure-sharing. Future *error-typed* retries should use
//! `common_core::retry`; see `stages_retry_classifier.rs` loop tests.

use std::sync::Arc;

use fluent_wvr::prelude::*;

/// Maximum number of attempts key.  Injected into the `WorkContext` typed
/// store (`ctx.set::<i64>(METADATA_RETRY_ATTEMPT, attempt)`) so the inner
/// stage can include it in tracing/logging. The attempt number is a concrete
/// `i64`, so it lives in the typed store rather than the stringly-typed
/// `metadata` channel (see the `WorkContext` decision rule).
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
        ctx.set(METADATA_RETRY_ATTEMPT, retry_index as i64);
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

    /// Whether a classifier decision is a parse-failure fallback (same marker
    /// `is_fallback` reads off the `WorkOutput` boundary, without the
    /// serialize round-trip).
    fn is_fallback_decision(decision: &crate::pipeline_types::StageDecision) -> bool {
        crate::pipeline_types::StageMetadata::from(decision.metadata.clone())
            .fallback()
            .unwrap_or(false)
    }
}

impl RetryClassifier {
    /// Typed evaluation: mirrors the `execute` retry loop but threads the
    /// inner classifier's dispatch target by value. The orchestrator's
    /// producer path calls this so a retry-wrapped classifier still publishes
    /// to the typed store.
    pub(crate) fn evaluate_with_target(
        &self,
        ctx: &WorkContext,
        prior: &[crate::pipeline_types::StageDecision],
    ) -> Result<
        (
            crate::pipeline_types::StageDecision,
            Option<crate::pipeline::RoutingTarget>,
        ),
        WorkError,
    > {
        let Some(classifier) = self.inner.as_any().downcast_ref::<crate::stages::classifier::ClassifierStage>()
        else {
            // Unknown inner: the serialization boundary carries no target.
            let decision: crate::pipeline_types::StageDecision =
                self.inner.execute(ctx).and_then(|output| {
                    output
                        .data_take()
                        .map_err(|e| WorkError::Execution(e.to_string()))
                })?;
            return Ok((decision, None));
        };

        // Attempt 0 (initial, no retry prompt).
        let (decision, target) = classifier.evaluate_with_target(ctx, prior)?;
        if !Self::is_fallback_decision(&decision) {
            return Ok((decision, target));
        }
        let mut last_error = decision.reason.clone();

        for retry_index in 0..self.max_retries {
            tracing::info!(
                target: "router.pipeline.retry",
                retry = retry_index + 1,
                max_retries = self.max_retries,
                parse_error = %last_error,
                "classifier fallback detected, retrying",
            );
            let retry_ctx = self.build_retry_context(ctx, retry_index, &last_error);
            let (retry_decision, retry_target) =
                classifier.evaluate_with_target(&retry_ctx, prior)?;
            if !Self::is_fallback_decision(&retry_decision) {
                return Ok((retry_decision, retry_target));
            }
            last_error.clone_from(&retry_decision.reason);
        }

        tracing::warn!(
            target: "router.pipeline.retry",
            max_retries = self.max_retries,
            parse_error = %last_error,
            "exhausted retries, returning final fallback",
        );

        // Re-evaluate on the original context for the final attempt so the
        // classifier's own fallback logic handles the response.
        classifier.evaluate_with_target(ctx, prior)
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

impl_fieldless!(RetryClassifier);

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
#[path = "../../tests/stages_retry_classifier.rs"]
mod tests;
