//! Expansion tests for the fluent-wvr derive/macro surface.
//!
//! `fluent-wvr-macros` is a proc-macro crate, so its own `#[cfg(test)]` unit
//! tests cannot invoke the proc macros (a proc macro is only usable from a
//! separate crate). These integration tests are that separate crate: they
//! depend on `fluent-wvr` and expand `#[derive(FieldAccess, Describable)]`,
//! `fluent_wvr::impl_fieldless!`, and `fluent_wvr::impl_component!`
//! (concrete + generic) on real fixture types, then assert the generated
//! behavior. Covers ROADMAP_20260816_TESTS M3.7.

use fluent_wvr::prelude::*;
use fluent_wvr::FieldError;
use internment::ArcIntern;
use std::sync::Arc;

// ── FieldAccess + Describable derive on a struct with the full attribute
//    surface ───────────────────────────────────────────────────────────────

#[derive(FieldAccess, Describable)]
struct ToolConfig {
    #[field(desc = "TCP listen port", min = 1, max = 65535)]
    pub port: u16,
    #[field(desc = "Host address", max_len = 255)]
    pub host: String,
    #[field(desc = "Enable verbose logging")]
    pub verbose: bool,
    #[field(desc = "API endpoint URL", pattern = "https://")]
    pub endpoint: String,
    #[field(desc = "Display name", sanitize = "lowercase")]
    pub name: String,
    #[field(skip)]
    pub internal: u64,
}

#[test]
fn derive_field_access_set_and_get() {
    let mut cfg = ToolConfig {
        port: 8080,
        host: "localhost".into(),
        verbose: false,
        endpoint: "https://api.example.com".into(),
        name: "Server".into(),
        internal: 7,
    };
    cfg.set_field("port", "9000").unwrap();
    assert_eq!(cfg.get_field("port").unwrap(), "9000");
    cfg.set_field("verbose", "true").unwrap();
    assert_eq!(cfg.get_field("verbose").unwrap(), "true");
    cfg.set_field("host", "router.local").unwrap();
    assert_eq!(cfg.get_field("host").unwrap(), "router.local");
}

#[test]
fn derive_field_access_field_names() {
    let cfg = ToolConfig {
        port: 1,
        host: "h".into(),
        verbose: false,
        endpoint: "https://e".into(),
        name: "n".into(),
        internal: 0,
    };
    // The `skip` field is excluded from field_names.
    let names = cfg.field_names();
    assert!(names.contains(&"port"));
    assert!(names.contains(&"host"));
    assert!(names.contains(&"verbose"));
    assert!(names.contains(&"endpoint"));
    assert!(names.contains(&"name"));
    assert!(!names.contains(&"internal"));
}

#[test]
fn derive_field_access_enforces_constraints() {
    let mut cfg = ToolConfig {
        port: 1,
        host: "h".into(),
        verbose: false,
        endpoint: "https://e".into(),
        name: "n".into(),
        internal: 0,
    };
    // min/max on port.
    assert!(matches!(cfg.set_field("port", "0"), Err(FieldError::Constraint(_))));
    assert!(matches!(cfg.set_field("port", "65536"), Err(FieldError::Constraint(_))));
    // non-numeric parse error.
    assert!(matches!(cfg.set_field("port", "abc"), Err(FieldError::Parse(_))));
    // max_len on host.
    let long = "x".repeat(300);
    assert!(matches!(cfg.set_field("host", &long), Err(FieldError::Constraint(_))));
    // pattern (substring) on endpoint.
    assert!(matches!(
        cfg.set_field("endpoint", "http://plain"),
        Err(FieldError::Constraint(_))
    ));
    cfg.set_field("endpoint", "https://ok").unwrap();
    // unknown field.
    assert!(matches!(cfg.set_field("nope", "x"), Err(FieldError::NotFound(_))));
}

#[test]
fn derive_field_access_sanitizes() {
    let mut cfg = ToolConfig {
        port: 1,
        host: "h".into(),
        verbose: false,
        endpoint: "https://e".into(),
        name: "n".into(),
        internal: 0,
    };
    // sanitize = "lowercase".
    cfg.set_field("name", "MixedCase").unwrap();
    assert_eq!(cfg.get_field("name").unwrap(), "mixedcase");
}

#[test]
fn derive_describable_emits_schema() {
    let cfg = ToolConfig {
        port: 1,
        host: "h".into(),
        verbose: false,
        endpoint: "https://e".into(),
        name: "n".into(),
        internal: 0,
    };
    let schema = cfg.describe();
    let props = &schema["properties"];
    assert_eq!(props["port"]["type"], "integer");
    assert_eq!(props["host"]["type"], "string");
    assert_eq!(props["verbose"]["type"], "boolean");
    // The `skip` field does not appear in the schema.
    assert!(props.get("internal").is_none());
}

// ── impl_fieldless! on a fieldless unit ──────────────────────────────────

struct FieldlessUnit;
impl WorkUnit for FieldlessUnit {
    fn name(&self) -> &str {
        "fieldless"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("fieldless"))
    }
}
impl Describable for FieldlessUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}
fluent_wvr::impl_fieldless!(FieldlessUnit);
fluent_wvr::impl_component!(FieldlessUnit);

#[test]
fn impl_fieldless_rejects_all_fields() {
    let mut unit = FieldlessUnit;
    assert!(matches!(unit.set_field("x", "y"), Err(FieldError::NotFound(_))));
    assert!(matches!(unit.get_field("x"), Err(FieldError::NotFound(_))));
    assert!(unit.field_names().is_empty());
    // The type is still a full Component: it can be erased and downcast.
    let comp: Arc<dyn Component> = Arc::new(FieldlessUnit);
    assert!(fluent_wvr::component_downcast_ref::<FieldlessUnit>(&*comp).is_some());
}

// ── impl_component!(generic ...) on a wrapper ─────────────────────────────

struct Wrapper<U>(U);
impl<U: Component> WorkUnit for Wrapper<U> {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        self.0.depends()
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        self.0.provides()
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        self.0.execute(ctx)
    }
}
impl<U: Component> FieldAccess for Wrapper<U> {
    fn set_field(&mut self, n: &str, v: &str) -> Result<(), FieldError> {
        self.0.set_field(n, v)
    }
    fn get_field(&self, n: &str) -> Result<String, FieldError> {
        self.0.get_field(n)
    }
    fn field_names(&self) -> &'static [&'static str] {
        self.0.field_names()
    }
}
impl<U: Component> Describable for Wrapper<U> {
    fn describe(&self) -> serde_json::Value {
        self.0.describe()
    }
}
fluent_wvr::impl_component!(generic (U: Component + 'static) for Wrapper<U>);

#[test]
fn impl_component_generic_expands() {
    let inner: Arc<dyn Component> = Arc::new(FieldlessUnit);
    let comp: Arc<dyn Component> = Arc::new(Wrapper(Arc::clone(&inner)));
    assert!(
        fluent_wvr::component_downcast_ref::<Wrapper<Arc<dyn Component>>>(&*comp).is_some()
    );
    assert_eq!(comp.name(), "fieldless");
}
