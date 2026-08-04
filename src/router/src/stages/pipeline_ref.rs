//! PipelineRefStage — a `WorkUnit` that delegates to a named pipeline
//! resolved at runtime from a `PipelineRegistry`.

use std::collections::HashMap;
use std::sync::Arc;

use fluent_wvr::prelude::*;

/// A stage that looks up a named pipeline in a shared registry and delegates
/// execution to it.  This is what the JSON `{"type": "pipeline_ref", "name": "..."}`
/// config node maps to.
///
/// The registry is shared across all `PipelineRefStage` instances so that
/// pipelines can reference each other by name without circular `Arc` issues
/// (the registry is built in a single pass before any stage executes).
pub struct PipelineRefStage {
    name: ArcIntern<str>,
    /// Name used to look up the target pipeline in the registry.
    pipeline_name: String,
    /// Shared registry mapping pipeline names to their constructed components.
    /// Built once at config time and frozen thereafter.
    registry: Arc<HashMap<String, Arc<dyn Component>>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl PipelineRefStage {
    pub fn new(
        pipeline_name: impl Into<String>,
        registry: Arc<HashMap<String, Arc<dyn Component>>>,
    ) -> Self {
        let pn = pipeline_name.into();
        Self {
            name: ArcIntern::from(format!("pipeline.stage.ref.{pn}")),
            pipeline_name: pn,
            registry,
            depends: vec![],
            provides: vec![ArcIntern::from("pipeline.stage.ref.output")],
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

    fn resolve(&self) -> Result<&Arc<dyn Component>, WorkError> {
        self.registry.get(&self.pipeline_name).ok_or_else(|| {
            WorkError::Dependency(format!(
                "pipeline '{}' not found in registry ({} entries registered)",
                self.pipeline_name,
                self.registry.len(),
            ))
        })
    }
}

impl WorkUnit for PipelineRefStage {
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
        tracing::info!(
            target: "router.pipeline.ref",
            pipeline_name = %self.pipeline_name,
            "delegating to referenced pipeline"
        );
        let pipeline = self.resolve()?;
        pipeline.execute(ctx)
    }
}

impl_fieldless!(PipelineRefStage);

impl Describable for PipelineRefStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pipeline_name": {"type": "string"},
            },
            "required": ["pipeline_name"]
        })
    }
}

impl_component!(PipelineRefStage);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineOrchestrator;
    use crate::test_stubs;

    #[test]
    fn pipeline_ref_resolves_and_delegates() {
        let target = PipelineOrchestrator::new(vec![Arc::new(test_stubs::SimplePassStage::new(
            "inner",
            "produced by inner",
        ))]);
        let target_arc: Arc<dyn Component> = Arc::new(target);

        let mut registry: HashMap<String, Arc<dyn Component>> = HashMap::new();
        registry.insert("code_router".into(), target_arc);
        let registry = Arc::new(registry);

        let ref_stage = PipelineRefStage::new("code_router", registry);
        let ctx = WorkContext::default();

        let result = ref_stage.execute(&ctx).unwrap();
        assert!(result.success);
    }

    #[test]
    fn pipeline_ref_errors_on_unknown_name() {
        let registry = Arc::new(HashMap::<String, Arc<dyn Component>>::new());
        let ref_stage = PipelineRefStage::new("missing", registry);
        let ctx = WorkContext::default();

        let err = ref_stage.execute(&ctx).unwrap_err();
        assert!(
            err.to_string().contains("not found"),
            "expected 'not found' error, got: {err}"
        );
    }
}
