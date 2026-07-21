use std::any::Any;
use std::sync::Arc;

use internment::ArcIntern;
use serde::{Deserialize, Serialize};

use crate::work::{WorkError, WorkOutput};

#[derive(Error, Debug, PartialEq, Eq)]
pub enum FieldError {
    #[error("field not found: {0}")]
    NotFound(String),
    #[error("field parse error: {0}")]
    Parse(String),
    #[error("constraint violation: {0}")]
    Constraint(String),
}
use thiserror::Error;

pub trait FieldAccess {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError>;
    fn get_field(&self, name: &str) -> Result<String, FieldError>;
    fn field_names(&self) -> &'static [&'static str];
}

pub trait Describable {
    fn describe(&self) -> serde_json::Value;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub type_name: String,
    pub description: Option<String>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub required: bool,
    /// Optional JSON Schema `format` hint (e.g. "url", "duration", "email").
    /// Informational only; not enforced by `set_field`.
    pub format: Option<String>,
    pub max_len: Option<usize>,
    pub sanitize: Option<String>,
    /// Substring pattern. The value must contain this string. Not a regex.
    pub pattern: Option<String>,
}

pub trait SchemaProvider {
    fn schema(&self) -> Vec<FieldSchema>;
}

/// The core execution unit in the component system. Every `Component` implements
/// this trait. The `Zone` supervisor and `MiddlewareChain` operate on `WorkUnit`s.
///
/// # Examples
///
/// ```
/// use fluent_wvr::{WorkUnit, WorkContext, WorkOutput, WorkError};
/// use internment::ArcIntern;
///
/// struct PingUnit;
///
/// impl WorkUnit for PingUnit {
///     fn name(&self) -> &str { "ping" }
///     fn depends(&self) -> &[ArcIntern<str>] { &[] }
///     fn provides(&self) -> &[ArcIntern<str>] { &[] }
///     fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
///         Ok(WorkOutput::ok("pong"))
///     }
/// }
///
/// let unit = PingUnit;
/// assert_eq!(unit.name(), "ping");
/// ```
pub trait WorkUnit: Send + Sync {
    fn name(&self) -> &str;
    fn depends(&self) -> &[ArcIntern<str>];
    fn provides(&self) -> &[ArcIntern<str>];
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError>;

    /// Returns the default timeout in milliseconds for this unit.
    /// Override this to set a unit-specific timeout. Default: 30,000ms.
    fn default_timeout_ms(&self) -> u64 {
        30_000
    }

    /// Returns the Rust type name of this unit (e.g. "L3GraphUnit").
    /// Useful for logging, metrics aggregation, and debugging.
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}
use crate::work::WorkContext;

/// A fully-featured component: `FieldAccess` + `Describable` + `WorkUnit` + `Send + Sync`.
///
/// Any type that implements all four traits can implement `Component`. The derive
/// macros (`#[derive(FieldAccess, Describable)]`) plus a manual `WorkUnit` impl
/// is the 80% path. `Component` requires `as_any()`/`as_any_mut()` for runtime
/// type identification.
///
/// # Examples
///
/// ```no_run
/// use fluent_wvr::{Component, WorkUnit, WorkContext, WorkOutput, WorkError,
///     FieldAccess, Describable, FieldError, impl_component};
/// use internment::ArcIntern;
///
/// struct MyUnit { port: u16 }
///
/// impl WorkUnit for MyUnit {
///     fn name(&self) -> &str { "my_unit" }
///     fn depends(&self) -> &[ArcIntern<str>] { &[] }
///     fn provides(&self) -> &[ArcIntern<str>] { &[] }
///     fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
///         Ok(WorkOutput::ok("done"))
///     }
/// }
/// impl FieldAccess for MyUnit {
///     fn set_field(&mut self, _: &str, _: &str) -> Result<(), FieldError> { Ok(()) }
///     fn get_field(&self, _: &str) -> Result<String, FieldError> { Err(FieldError::NotFound("none".into())) }
///     fn field_names(&self) -> &'static [&'static str] { &[] }
/// }
/// impl Describable for MyUnit {
///     fn describe(&self) -> serde_json::Value { serde_json::json!({}) }
/// }
/// impl_component!(MyUnit);
///
/// // MyUnit is now a Component — can be wrapped in Arc<dyn Component>.
/// let _comp: std::sync::Arc<dyn Component> = std::sync::Arc::new(MyUnit { port: 8080 });
/// ```
pub trait Component: FieldAccess + Describable + WorkUnit + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Optional trait for components that can be persisted to storage.
///
/// **Deferred:** This trait is a design placeholder — no blanket impl exists,
/// and no in-tree component implements it yet. A second consumer that needs
/// to serialize component state (e.g., to a database or over the wire) should
/// implement this trait on the specific types that need persistence. The base
/// `Component` trait intentionally does NOT require `Serialize` because most
/// components hold non-serializable state (`Arc<dyn Provider>`, `Mutex<Plugin>`).
pub trait PersistableComponent: Component {
    fn serialize_state(&self) -> Result<serde_json::Value, WorkError>;
}

/// Downcast a `dyn Component` to a concrete type. Returns `None` if the type doesn't match.
pub fn component_downcast_ref<T: 'static>(comp: &dyn Component) -> Option<&T> {
    comp.as_any().downcast_ref::<T>()
}

/// Mutable downcast a `dyn Component` to a concrete type. Returns `None` if the type doesn't match.
pub fn component_downcast_mut<T: 'static>(comp: &mut dyn Component) -> Option<&mut T> {
    comp.as_any_mut().downcast_mut::<T>()
}

/// Extension trait for safe mutable access through `Arc<dyn Component>`.
///
/// Use this when you can't guarantee exclusive ownership of the `Arc`.
/// If the `Arc` is shared, `try_as_any_mut` returns `None` and you can
/// decide whether to clone-and-mutate, defer, or error.
pub trait ComponentArcExt {
    fn try_as_any_mut(&mut self) -> Option<&mut dyn std::any::Any>;
}

impl ComponentArcExt for Arc<dyn Component> {
    fn try_as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Arc::get_mut(self).map(Component::as_any_mut)
    }
}
