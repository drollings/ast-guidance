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
