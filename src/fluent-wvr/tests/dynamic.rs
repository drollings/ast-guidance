#![allow(unused_imports)]
#[allow(unused_imports)]
use fluent_wvr::dynamic::*;
use std::collections::HashMap;
use std::sync::Arc;
use fluent_wvr::prelude::*;
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};


use fluent_wvr::component_downcast_ref;
use fluent_wvr::prelude::*;

fn echo_executor(
    _ctx: &WorkContext,
    config: &HashMap<String, String>,
) -> Result<WorkOutput, WorkError> {
    let retries = config.get("retries").cloned().unwrap_or_default();
    Ok(WorkOutput::ok_with_data(
        format!("configured retries={retries}"),
        serde_json::json!({ "retries": retries }),
    ))
}

#[test]
fn assembles_and_executes_from_config_map() {
    let mut comp = DynamicComponent::new(
        "db.tool",
        Arc::new(echo_executor) as DynamicExecutor,
    )
    .with_field_keys(&["retries", "timeout_ms"]);

    comp.set_field("retries", "3").unwrap();
    assert_eq!(comp.get_field("retries").unwrap(), "3");
    assert_eq!(comp.field_names(), &["retries", "timeout_ms"]);

    let ctx = WorkContext::default();
    let out = comp.execute(&ctx).unwrap();
    assert!(out.success);
    assert_eq!(out.data["retries"], "3");
}

#[test]
fn unknown_field_errors_on_get() {
    let comp = DynamicComponent::new(
        "db.tool",
        Arc::new(echo_executor) as DynamicExecutor,
    );
    assert!(matches!(
        comp.get_field("missing"),
        Err(FieldError::NotFound(_))
    ));
}

#[test]
fn erases_to_uniform_handle() {
    let comp = DynamicComponent::new(
        "db.tool",
        Arc::new(echo_executor) as DynamicExecutor,
    );
    let handle: Arc<dyn Component> = Arc::new(comp);
    let ctx = WorkContext::default();
    assert!(handle.execute(&ctx).is_ok());
    assert!(handle.describe().get("type").is_some());
    assert!(component_downcast_ref::<DynamicComponent>(&*handle).is_some());
}
