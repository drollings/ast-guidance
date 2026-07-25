//! SwitchStage — a `WorkUnit` that branches to one of several sub-pipelines
//! based on a metadata field value.  Plugs into any `PipelineOrchestrator` or
//! `PipelineGraph` stage list.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_wvr::prelude::*;

use crate::pipeline::PipelineOrchestrator;

/// Evaluates a metadata field at runtime and delegates to one of several
/// pre-built sub-pipelines.  The matching value is the string representation
/// of the metadata entry (for `String` / `Number` / `Float` / `Bool`).
///
/// # Branch resolution order
/// 1. Exact match against the first segment of `field_path` in `ctx.metadata`.
/// 2. Fallback to the wildcard key `"*"` if present.
/// 3. Fallback to `default_branch` if set.
/// 4. Returns `WorkError::Execution` when no branch matches.
///
/// # Typical use
///
/// ```text
///          ┌──────────────┐
///          │  classifier   │  StageDecision.metadata["intent"] = "code"
///          └──────┬───────┘
///                 │
///          ┌──────▼───────┐
///          │   switch on   │
///          │   "intent"    │
///          └─┬───┬───┬────┘
///    "code" ▼   │   ▼ "question"
///   ┌────────┐ │  ┌──────────┐
///   │code    │ │  │ fast     │
///   │pipeline│ │  │ dispatch │
///   └────────┘ │  └──────────┘
///          "*" ▼
///         ┌──────────┐
///         │ default  │
///         │ fallback │
///         └──────────┘
/// ```
pub struct SwitchStage {
    name: ArcIntern<str>,
    /// Metadata key to switch on.  Read directly from `ctx.metadata` after
    /// the preceding stages have been executed and their decision fields
    /// promoted into the context by `PipelineGraph`.
    field_key: String,
    /// Exact-match branches.  The special key `"*"` is a wildcard that
    /// catches any value not matched by any other key.
    branches: HashMap<String, Arc<PipelineOrchestrator>>,
    /// Default branch used when no exact or wildcard match was found.
    /// When `None` and no match, the stage returns an error.
    default_branch: Option<Arc<PipelineOrchestrator>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl SwitchStage {
    pub fn new(field_key: impl Into<String>) -> Self {
        Self {
            name: ArcIntern::from("pipeline.stage.switch"),
            field_key: field_key.into(),
            branches: HashMap::new(),
            default_branch: None,
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.stage.switch.output")],
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

    /// Register a branch keyed by the exact metadata string value.
    /// Use `"*"` as the key for a wildcard catch-all.
    #[must_use]
    pub fn branch(mut self, value: impl Into<String>, pipeline: PipelineOrchestrator) -> Self {
        self.branches
            .insert(value.into(), Arc::new(pipeline));
        self
    }

    /// Set the default branch used when no exact or wildcard match is found.
    #[must_use]
    pub fn default_branch(mut self, pipeline: PipelineOrchestrator) -> Self {
        self.default_branch = Some(Arc::new(pipeline));
        self
    }

    /// Read the switch field value from WorkContext metadata.
    /// Returns the string representation (Number → decimal, Bool → "true"/"false").
    fn read_field(&self, ctx: &WorkContext) -> Option<String> {
        ctx.metadata.get(&self.field_key).and_then(metadata_value_to_string)
    }

    /// Select the matching sub-pipeline.  Resolution order:
    /// exact → wildcard → default → error.
    fn select_pipeline(
        &self,
        value: &str,
    ) -> Result<&Arc<PipelineOrchestrator>, WorkError> {
        if let Some(p) = self.branches.get(value) {
            return Ok(p);
        }
        if let Some(p) = self.branches.get("*") {
            return Ok(p);
        }
        if let Some(p) = &self.default_branch {
            return Ok(p);
        }
        Err(WorkError::Execution(format!(
            "switch '{}': no branch for value '{}'",
            self.field_key, value,
        )))
    }
}

impl WorkUnit for SwitchStage {
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
        let value = self.read_field(ctx).unwrap_or_default();
        tracing::info!(
            target: "router.pipeline.switch",
            field = %self.field_key,
            value = %value,
            branches = self.branches.len(),
            "switch evaluating"
        );

        let pipeline = self.select_pipeline(&value)?;
        tracing::info!(
            target: "router.pipeline.switch",
            field = %self.field_key,
            value = %value,
            branch_pipeline = %pipeline.name(),
            "switch selected branch, delegating"
        );

        pipeline.execute(ctx)
    }
}

impl FieldAccess for SwitchStage {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(
            "SwitchStage has no configurable fields".into(),
        ))
    }

    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(
            "SwitchStage has no configurable fields".into(),
        ))
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for SwitchStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "field_key": {"type": "string", "description": "metadata key to switch on"},
                "branches": {"type": "object", "description": "value -> sub-pipeline mapping"},
            },
            "required": ["field_key"]
        })
    }
}

impl_component!(SwitchStage);

// ── helpers ──────────────────────────────────────────────────────────────

/// Convert a `MetadataValue` to its string representation.
pub(crate) fn metadata_value_to_string(v: &MetadataValue) -> Option<String> {
    match v {
        MetadataValue::String(s) => Some(s.clone()),
        MetadataValue::Number(n) => Some(n.to_string()),
        MetadataValue::Float(f) => Some(f.to_string()),
        MetadataValue::Bool(b) => Some(if *b { "true".into() } else { "false".into() }),
        MetadataValue::Null => None,
    }
}

/// Promote a `StageDecision`'s metadata fields into a `WorkContext`'s
/// metadata map so that downstream stages (notably `SwitchStage`) can
/// read them directly via `ctx.metadata.get()`.
///
/// Only scalar values (string, number, bool) are promoted; arrays and
/// objects are skipped.
pub(crate) fn promote_decision_metadata(
    target: &mut HashMap<String, MetadataValue>,
    prefix: &str,
    metadata: &serde_json::Value,
) {
    if let Some(obj) = metadata.as_object() {
        for (k, v) in obj {
            let key = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{prefix}.{k}")
            };
            if let Some(mv) = json_to_metadata_value(v) {
                // Also insert the bare key (without prefix) for simple
                // switch-on-single-metadata-field use cases.
                if !prefix.is_empty() {
                    target.insert(k.clone(), mv.clone());
                }
                target.insert(key, mv);
            }
        }
    }
}

fn json_to_metadata_value(v: &serde_json::Value) -> Option<MetadataValue> {
    match v {
        serde_json::Value::String(s) => Some(MetadataValue::String(s.clone())),
        serde_json::Value::Number(n) => {
            n.as_i64()
                .map(MetadataValue::Number)
                .or_else(|| n.as_f64().map(MetadataValue::Float))
        }
        serde_json::Value::Bool(b) => Some(MetadataValue::Bool(*b)),
        serde_json::Value::Null => Some(MetadataValue::Null),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineResult;
    use crate::test_stubs;

    fn make_pass_stage(name: &str, message: &str) -> PipelineOrchestrator {
        let stage = test_stubs::SimplePassStage::new(name, message);
        PipelineOrchestrator::new(vec![Arc::new(stage)])
    }

    #[test]
    fn switch_selects_correct_branch_from_metadata() {
        let code_pipeline = make_pass_stage("code_branch", "code executed");
        let question_pipeline = make_pass_stage("question_branch", "question executed");

        let switch = SwitchStage::new("intent")
            .with_name("switch.test")
            .branch("code", code_pipeline)
            .branch("question", question_pipeline);

        let mut ctx = WorkContext::default();
        ctx.metadata
            .insert("intent".into(), MetadataValue::String("code".into()));

        let result = switch.execute(&ctx).unwrap();
        let pr: PipelineResult = result.data_as().unwrap();
        assert!(!pr.rejected);
    }

    #[test]
    fn switch_uses_wildcard_when_no_exact_match() {
        let fallback = make_pass_stage("fallback_branch", "fallback executed");
        let switch = SwitchStage::new("intent")
            .with_name("switch.wildcard")
            .branch("*", fallback);

        let mut ctx = WorkContext::default();
        ctx.metadata
            .insert("intent".into(), MetadataValue::String("unknown".into()));

        let result = switch.execute(&ctx).unwrap();
        let pr: PipelineResult = result.data_as().unwrap();
        assert!(!pr.rejected);
    }

    #[test]
    fn switch_uses_default_when_no_match_and_no_wildcard() {
        let default = make_pass_stage("default_branch", "default executed");
        let switch = SwitchStage::new("intent")
            .with_name("switch.default")
            .default_branch(default);

        let mut ctx = WorkContext::default();
        ctx.metadata
            .insert("intent".into(), MetadataValue::String("unknown".into()));

        let result = switch.execute(&ctx).unwrap();
        let pr: PipelineResult = result.data_as().unwrap();
        assert!(!pr.rejected);
    }

    #[test]
    fn switch_errors_when_no_branch_matches_at_all() {
        let switch = SwitchStage::new("intent").with_name("switch.no_match");
        let mut ctx = WorkContext::default();
        ctx.metadata
            .insert("intent".into(), MetadataValue::String("missing".into()));

        let err = switch.execute(&ctx).unwrap_err();
        assert!(
            err.to_string().contains("no branch"),
            "expected 'no branch' error, got: {err}"
        );
    }

    #[test]
    fn switch_handles_missing_metadata_gracefully() {
        let default = make_pass_stage("default_branch", "default executed");
        let switch = SwitchStage::new("intent")
            .with_name("switch.missing_key")
            .default_branch(default);

        let ctx = WorkContext::default(); // no "intent" key
        let result = switch.execute(&ctx).unwrap();
        // Should fall through to default
        assert!(result.success);
    }

    #[test]
    fn promote_decision_metadata_copies_scalars() {
        let mut target: HashMap<String, MetadataValue> = HashMap::new();
        let metadata = serde_json::json!({
            "intent": "code",
            "complexity": 5,
            "coherence_score": 0.95,
            "fallback": false,
        });
        promote_decision_metadata(&mut target, "classifier", &metadata);

        assert_eq!(target.get("intent"), Some(&MetadataValue::String("code".into())));
        assert_eq!(target.get("complexity"), Some(&MetadataValue::Number(5)));
        assert_eq!(
            target.get("classifier.intent"),
            Some(&MetadataValue::String("code".into()))
        );
        assert_eq!(target.get("fallback"), Some(&MetadataValue::Bool(false)));
    }

    #[test]
    fn promote_decision_metadata_skips_arrays_and_objects() {
        let mut target: HashMap<String, MetadataValue> = HashMap::new();
        let metadata = serde_json::json!({
            "tags": ["a", "b"],
            "nested": {"x": 1},
        });
        promote_decision_metadata(&mut target, "classifier", &metadata);
        assert!(target.is_empty(), "arrays and objects must not be promoted");
    }

    #[test]
    fn metadata_value_to_string_conversions() {
        assert_eq!(
            metadata_value_to_string(&MetadataValue::String("hello".into())),
            Some("hello".into())
        );
        assert_eq!(
            metadata_value_to_string(&MetadataValue::Number(42)),
            Some("42".into())
        );
        assert_eq!(
            metadata_value_to_string(&MetadataValue::Float(3.14)),
            Some("3.14".into())
        );
        assert_eq!(
            metadata_value_to_string(&MetadataValue::Bool(true)),
            Some("true".into())
        );
        assert_eq!(metadata_value_to_string(&MetadataValue::Null), None);
    }
}
