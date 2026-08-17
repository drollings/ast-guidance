//! Crate-local test support for fluent-wvr.
//!
//! **Why this cannot live in `fluent-wvr-testutil`:** testutil depends on
//! `fluent-wvr`; moving these stubs there would create a dependency cycle.
//! This module is the sanctioned exception (ROADMAP_20260816_TESTS.md §2.4 /
//! M1.5) — the stubs deliberately match the testutil API (`StubComponent`
//! semantics: `ok_unit` / `failing_unit` shorthands) so the seam stays thin
//! and documented, and so reviewers don't "fix" it into a cycle.
//!
//! The `TestComponent` (formerly in `tests.rs`) and the `CloneableUnit` /
//! `MockUnit` / `AdapterHost` / `DepProvider` / `FieldedHost` /
//! `ConstrainedHost` wrapper stubs (formerly inline in `wrapper.rs`) were
//! consolidated here so every suite shares one set.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use internment::ArcIntern;

use crate::prelude::*;

/// A `Component` with a mutable `value` field and a `name`.
///
/// `execute` returns `computed: {value * 2}` so tests can observe field
/// mutation through the return message.
pub struct TestComponent {
    pub name: ArcIntern<str>,
    pub value: i32,
}

impl FieldAccess for TestComponent {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        match name {
            "value" => {
                self.value = value.parse().map_err(|_| FieldError::Parse(value.into()))?;
                Ok(())
            }
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "value" => Ok(self.value.to_string()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["value"]
    }
}

impl Describable for TestComponent {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"name": &*self.name, "value": self.value})
    }
}

impl WorkUnit for TestComponent {
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
        Ok(WorkOutput::ok(format!("computed: {}", self.value * 2)))
    }
}

impl_component!(TestComponent);

/// A `Clone`-able fieldless unit, used by `Instrumented::clone` tests.
#[derive(Clone)]
pub struct CloneableUnit;

impl WorkUnit for CloneableUnit {
    fn name(&self) -> &str {
        "cloneable"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("ok"))
    }
}

impl FieldAccess for CloneableUnit {
    fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
        Ok(())
    }
    fn get_field(&self, _: &str) -> Result<String, FieldError> {
        Ok(String::new())
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for CloneableUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::Value::Object(serde_json::Map::new())
    }
}

impl_component!(CloneableUnit);

/// A configurable stub `Component`: succeeds or fails on `execute` and tracks
/// how many times it ran (via `call_count`).
pub struct MockUnit {
    pub name: ArcIntern<str>,
    pub should_fail: bool,
    pub call_count: AtomicUsize,
}

impl MockUnit {
    pub fn ok(name: &str) -> Self {
        Self {
            name: ArcIntern::from(name),
            should_fail: false,
            call_count: AtomicUsize::new(0),
        }
    }

    pub fn fail(name: &str) -> Self {
        Self {
            name: ArcIntern::from(name),
            should_fail: true,
            call_count: AtomicUsize::new(0),
        }
    }
}

impl WorkUnit for MockUnit {
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
        self.call_count.fetch_add(1, Ordering::SeqCst);
        if self.should_fail {
            Err(WorkError::Execution("failed".into()))
        } else {
            Ok(WorkOutput::ok("done"))
        }
    }
}

impl FieldAccess for MockUnit {
    fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
        Ok(())
    }
    fn get_field(&self, _: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound("test type: no fields".into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for MockUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

impl_component!(MockUnit);

/// A succeeding unit erased to `Arc<dyn Component>` (testutil-style shorthand).
pub fn ok_unit(name: &str) -> Arc<dyn Component> {
    Arc::new(MockUnit::ok(name))
}

/// A failing unit erased to `Arc<dyn Component>` (testutil-style shorthand).
pub fn failing_unit(name: &str) -> Arc<dyn Component> {
    Arc::new(MockUnit::fail(name))
}

/// A `Component` that is deliberately NOT a `MockUnit`, used to confirm the
/// `ComponentAdapter` works when wrapping an arbitrary `Arc<dyn Component>`.
pub struct AdapterHost {
    pub name: ArcIntern<str>,
    pub last_message: Arc<std::sync::Mutex<String>>,
}

impl AdapterHost {
    pub fn new(name: &str) -> Self {
        Self {
            name: ArcIntern::from(name),
            last_message: Arc::new(std::sync::Mutex::new(String::new())),
        }
    }
}

impl FieldAccess for AdapterHost {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Ok(())
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound("no fields".into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for AdapterHost {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"name": &*self.name})
    }
}

impl_component!(AdapterHost);

impl WorkUnit for AdapterHost {
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
        let mut g = self.last_message.lock().unwrap();
        *g = "from_inner".to_string();
        Ok(WorkOutput::ok("from_inner"))
    }
}

/// A `Component` with non-empty `depends`/`provides`, for the adapter's
/// dependency-forwarding tests.
pub struct DepProvider {
    pub name: ArcIntern<str>,
    pub deps: Vec<ArcIntern<str>>,
    pub provs: Vec<ArcIntern<str>>,
}

impl WorkUnit for DepProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.deps
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provs
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("ok"))
    }
}

impl FieldAccess for DepProvider {
    fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
        Ok(())
    }
    fn get_field(&self, _: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound("none".into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for DepProvider {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

impl_component!(DepProvider);

/// A `Component` with three `field_names`, for the adapter's
/// `field_names`-forwarding tests.
pub struct FieldedHost {
    pub name: ArcIntern<str>,
}

impl WorkUnit for FieldedHost {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("ok"))
    }
}

impl FieldAccess for FieldedHost {
    fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> {
        Ok(())
    }
    fn get_field(&self, _: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound("none".into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["host_a", "host_b", "host_c"]
    }
}

impl Describable for FieldedHost {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

impl_component!(FieldedHost);

/// A `Component` whose `port` field is constrained to `<= 1024`, for the
/// adapter's constraint/parse error-propagation tests.
pub struct ConstrainedHost {
    pub port: u16,
}

impl FieldAccess for ConstrainedHost {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        if name == "port" {
            let v: u16 = value
                .parse()
                .map_err(|_| FieldError::Parse(format!("invalid u16 for 'port': {value}")))?;
            if v > 1024 {
                return Err(FieldError::Constraint(
                    "port: value above maximum 1024".into(),
                ));
            }
            self.port = v;
            Ok(())
        } else {
            Err(FieldError::NotFound(name.into()))
        }
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "port" => Ok(self.port.to_string()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["port"]
    }
}

impl Describable for ConstrainedHost {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

impl_component!(ConstrainedHost);

impl WorkUnit for ConstrainedHost {
    fn name(&self) -> &str {
        "constrained_host"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("done"))
    }
}