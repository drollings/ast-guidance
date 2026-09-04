#![allow(unused_imports)]
#[allow(unused_imports)]
use fluent_wvr::traits::*;
use std::sync::Arc;
use fluent_wvr::prelude::*;
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};


#[test]
fn field_error_display_variants() {
    assert_eq!(
        format!("{}", FieldError::NotFound("x".into())),
        "field not found: x"
    );
    assert_eq!(
        format!("{}", FieldError::Parse("bad".into())),
        "field parse error: bad"
    );
    assert_eq!(
        format!("{}", FieldError::Constraint("too big".into())),
        "constraint violation: too big"
    );
    assert_eq!(
        format!("{}", FieldError::ReadOnly("port".into(), "shared".into())),
        "field \"port\" is read-only on a shared Arc: shared"
    );
}

#[test]
fn field_error_equality_and_partial_eq() {
    assert_eq!(FieldError::NotFound("a".into()), FieldError::NotFound("a".into()));
    assert_ne!(FieldError::NotFound("a".into()), FieldError::NotFound("b".into()));
    assert_ne!(FieldError::Parse("p".into()), FieldError::Constraint("p".into()));
}

#[test]
fn field_schema_serde_round_trip() {
    let schema = FieldSchema {
        name: "port".into(),
        type_name: "u16".into(),
        description: Some("listen port".into()),
        min: Some(1.0),
        max: Some(65535.0),
        required: true,
        format: Some("duration".into()),
        max_len: Some(10),
        sanitize: Some("trim,lowercase".into()),
        pattern: Some("http".into()),
        coerce: Some("trim,strip_quotes".into()),
        parse: Some("number".into()),
    };
    let back: FieldSchema =
        serde_json::from_str(&serde_json::to_string(&schema).expect("serialize"))
            .expect("round trip");
    assert_eq!(back.name, "port");
    assert_eq!(back.coerce.as_deref(), Some("trim,strip_quotes"));
    assert_eq!(back.parse.as_deref(), Some("number"));
}

#[test]
fn field_schema_defaults_coerce_and_parse() {
    // `coerce`/`parse` default to None so older schema JSON round-trips.
    let json = r#"{"name":"p","type_name":"u16","required":true}"#;
    let schema: FieldSchema = serde_json::from_str(json).expect("deserialize");
    assert_eq!(schema.coerce, None);
    assert_eq!(schema.parse, None);
    assert!(schema.required);
}

#[test]
fn component_downcast_round_trip() {
    let unit = fluent_wvr::test_support::MockUnit::ok("mock");
    let mut comp: Arc<dyn Component> = Arc::new(unit);
    assert!(component_downcast_ref::<fluent_wvr::test_support::MockUnit>(&*comp).is_some());
    assert!(component_downcast_ref::<String>(&*comp).is_none());
    // Mutable downcast works when the Arc is exclusively owned.
    assert!(component_downcast_mut::<fluent_wvr::test_support::MockUnit>(&mut comp).is_some());
}

#[test]
fn arc_component_ext_mutability() {
    let unit = fluent_wvr::test_support::MockUnit::ok("mock");
    let mut arc: Arc<dyn Component> = Arc::new(unit);
    // Exclusive ownership: mutable access is available.
    assert!(arc.try_as_any_mut().is_some());
    // Shared: no mutable access.
    let mut shared = Arc::clone(&arc);
    assert!(shared.try_as_any_mut().is_none());
}
