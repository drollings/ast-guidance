#![forbid(unsafe_code)]

//! ## Fluent WVR — Framework Trait Crate
//!
//! This is a **framework trait** crate — the Rust equivalent of a header-only
//! interface.  It defines the core `Component`, `WorkUnit`, `FieldAccess`, and
//! `Describable` traits that the DAG executor, Coral, and ContentNode crates
//! implement and consume.
//!
//! **Design contract:**
//! - No implementation logic beyond blanket impls and helper types
//! - No domain-specific dependencies (no rusqlite, no LLM, no guidance-types)
//! - The thinness is intentional — value is in the trait boundaries
//! - If a derive macro (`#[derive(FieldAccess)]`) is added later, it goes here
//!
//! Consumers: `fluent-dag`, `coral-context`, `guidance-content-node`

extern crate self as fluent_wvr;

pub mod wrapper;

pub use fluent_wvr_macros::{Describable, FieldAccess};
pub use internment::ArcIntern;
use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::task::JoinHandle;

/// Typed metadata value for `WorkContext`.
///
/// Replaces the old `Vec<(String, String)>` with a type-safe, structured
/// representation. Supports string, integer, float, boolean, and null values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataValue {
    String(String),
    Number(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        MetadataValue::String(s.to_string())
    }
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        MetadataValue::String(s)
    }
}

impl From<i64> for MetadataValue {
    fn from(n: i64) -> Self {
        MetadataValue::Number(n)
    }
}

impl From<f64> for MetadataValue {
    fn from(f: f64) -> Self {
        MetadataValue::Float(f)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        MetadataValue::Bool(b)
    }
}

/// Legacy type alias for backward compatibility. Prefer `MetadataValue`.
pub type MetadataEntry = (String, String);

/// A capability token that can be placed in a `CapabilitySet` to gate access
/// to resources (network, filesystem, database).
///
/// # Examples
///
/// ```
/// use fluent_wvr::Capability;
///
/// struct NetCapability;
/// impl Capability for NetCapability {
///     fn name(&self) -> &'static str { "net" }
/// }
/// ```
pub trait Capability: Send + Sync + 'static {
    fn name(&self) -> &'static str;
}

/// A type-map of capability tokens, used to gate access to resources.
///
/// # Examples
///
/// ```
/// use fluent_wvr::{Capability, CapabilitySet};
///
/// struct FsCapability;
/// impl Capability for FsCapability {
///     fn name(&self) -> &'static str { "fs" }
/// }
///
/// let caps = CapabilitySet::new().with(FsCapability);
/// assert!(caps.get::<FsCapability>().is_some());
/// ```
#[derive(Default, Debug)]
pub struct CapabilitySet {
    caps: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Clone for CapabilitySet {
    fn clone(&self) -> Self {
        Self {
            caps: self.caps.clone(),
        }
    }
}

impl CapabilitySet {
    pub fn new() -> Self {
        Self {
            caps: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with<C: Capability>(mut self, cap: C) -> Self {
        self.caps.insert(TypeId::of::<C>(), Arc::new(cap));
        self
    }

    pub fn get<C: Capability>(&self) -> Option<&C> {
        self.caps
            .get(&TypeId::of::<C>())
            .and_then(|arc| (&**arc as &dyn Any).downcast_ref::<C>())
    }
}

pub struct Reserve {
    counter: Arc<std::sync::atomic::AtomicUsize>,
    committed: bool,
}

impl Reserve {
    /// Attempt to acquire a permit from the counter.
    ///
    /// Returns `None` if the counter is already at zero (no permits available).
    /// Does NOT underflow — this is the safe alternative to `new()`.
    pub fn try_acquire(counter: Arc<std::sync::atomic::AtomicUsize>) -> Option<Self> {
        let prev = counter.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        if prev == 0 {
            // Underflow would occur — restore counter and return None
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            None
        } else {
            Some(Self {
                counter,
                committed: false,
            })
        }
    }

    pub fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for Reserve {
    fn drop(&mut self) {
        if !self.committed {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// Async runtime abstraction. All async primitives in the workspace accept
/// this trait so that production code uses `tokio` and tests can substitute
/// a deterministic runtime.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use fluent_wvr::Runtime;
/// use fluent_concurrency::runtime::tokio::TokioRuntime;
///
/// let rt: Arc<dyn Runtime> = Arc::new(TokioRuntime);
/// rt.spawn(Box::pin(async { /* background work */ }));
/// ```
pub trait Runtime: Send + Sync + 'static {
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send>>) -> JoinHandle<()>;
    fn sleep(&self, duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>>;
    fn now(&self) -> Instant;
}

/// A no-op runtime for contexts where `spawn` and `sleep` are never called.
///
/// `spawn` logs a warning and returns a dummy `JoinHandle`. If called outside
/// a tokio runtime, it panics with a clear message directing the caller to
/// provide a real `Runtime`. `sleep` returns immediately. This runtime is
/// intended for testing or initialization code that doesn't actually need
/// async execution.
pub struct NoopRuntime;

impl Runtime for NoopRuntime {
    fn spawn(&self, _future: Pin<Box<dyn Future<Output = ()> + Send>>) -> JoinHandle<()> {
        tracing::warn!("NoopRuntime::spawn called — no runtime configured; task will not execute");
        let has_runtime = tokio::runtime::Handle::try_current().is_ok();
        assert!(
            has_runtime,
            "NoopRuntime::spawn called outside a tokio runtime. \
             Either supply a real Runtime (e.g. via WorkContext with rt: tokio_runtime()) \
             or use NoopRuntime only for dry-run / init code paths."
        );
        tokio::spawn(async {})
    }

    fn sleep(&self, _duration: Duration) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async {})
    }

    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[derive(Error, Debug)]
pub enum FieldError {
    #[error("field not found: {0}")]
    NotFound(String),
    #[error("field parse error: {0}")]
    Parse(String),
    #[error("constraint violation: {0}")]
    Constraint(String),
}

#[derive(Error, Debug)]
pub enum WorkError {
    #[error("execution failed: {0}")]
    Execution(String),
    #[error("dependency not satisfied: {0}")]
    Dependency(String),
    #[error("timeout after {duration_ms}ms ({unit})")]
    Timeout { duration_ms: u64, unit: String },
}

/// Execution context passed to `WorkUnit::execute`. Carries configuration
/// (dry-run, retries, timeout), typed metadata, a runtime, and a capability set.
///
/// # Examples
///
/// ```no_run
/// use fluent_wvr::{WorkContext, CapabilitySet};
/// use std::sync::Arc;
///
/// let ctx = WorkContext {
///     dry_run: true,
///     max_retries: 3,
///     timeout_ms: 10_000,
///     metadata: Default::default(),
///     rt: Arc::new(fluent_concurrency::runtime::tokio::TokioRuntime),
///     caps: CapabilitySet::new(),
/// };
/// assert!(ctx.dry_run);
/// ```
#[derive(Clone)]
pub struct WorkContext {
    pub dry_run: bool,
    pub max_retries: u32,
    pub timeout_ms: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub rt: Arc<dyn Runtime>,
    pub caps: CapabilitySet,
}

impl std::fmt::Debug for WorkContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkContext")
            .field("dry_run", &self.dry_run)
            .field("max_retries", &self.max_retries)
            .field("timeout_ms", &self.timeout_ms)
            .field("metadata", &self.metadata)
            .field("rt", &"<dyn Runtime>")
            .field("caps", &self.caps)
            .finish()
    }
}

impl Default for WorkContext {
    fn default() -> Self {
        Self {
            dry_run: false,
            max_retries: 0,
            timeout_ms: 30_000,
            metadata: HashMap::new(),
            rt: Arc::new(NoopRuntime),
            caps: CapabilitySet::new(),
        }
    }
}

impl WorkContext {
    /// Construct a `WorkContext` for a specific unit with a given capability set.
    /// Uses the unit's `default_timeout_ms()` and a default runtime.
    pub fn for_unit(unit: &dyn WorkUnit, caps: CapabilitySet) -> Self {
        Self {
            dry_run: false,
            max_retries: 0,
            timeout_ms: unit.default_timeout_ms(),
            metadata: HashMap::new(),
            rt: Arc::new(NoopRuntime),
            caps,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkOutput {
    pub success: bool,
    pub message: String,
    pub data: serde_json::Value,
}

impl WorkOutput {
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: serde_json::Value::Null,
        }
    }
    pub fn ok_with_data(message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success: true,
            message: message.into(),
            data,
        }
    }
    pub fn fail(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            data: serde_json::Value::Null,
        }
    }
}

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
    pub format: Option<String>,
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
///         WorkOutput::ok("pong")
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
}

/// A fully-featured component: `FieldAccess` + `Describable` + `WorkUnit` + `Send + Sync`.
///
/// Any type that implements all four traits automatically implements `Component`
/// via a blanket impl. The derive macros (`#[derive(FieldAccess, Describable)]`)
/// plus a manual `WorkUnit` impl is the 80% path.
///
/// # Examples
///
/// ```no_run
/// use fluent_wvr::{Component, WorkUnit, WorkContext, WorkOutput, WorkError,
///     FieldAccess, Describable, FieldError};
/// use internment::ArcIntern;
///
/// struct MyUnit { port: u16 }
///
/// impl WorkUnit for MyUnit {
///     fn name(&self) -> &str { "my_unit" }
///     fn depends(&self) -> &[ArcIntern<str>] { &[] }
///     fn provides(&self) -> &[ArcIntern<str>] { &[] }
///     fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
///         WorkOutput::ok("done")
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
///
/// // MyUnit is now a Component — can be wrapped in Arc<dyn Component>.
/// let _comp: std::sync::Arc<dyn Component> = std::sync::Arc::new(MyUnit { port: 8080 });
/// ```
pub trait Component: FieldAccess + Describable + WorkUnit + Send + Sync {}
impl<T: FieldAccess + Describable + WorkUnit + Send + Sync> Component for T {}

impl WorkUnit for Arc<dyn WorkUnit> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        (**self).depends()
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        (**self).provides()
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        (**self).execute(ctx)
    }
    fn default_timeout_ms(&self) -> u64 {
        (**self).default_timeout_ms()
    }
}

impl WorkUnit for Arc<dyn Component> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        (**self).depends()
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        (**self).provides()
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        (**self).execute(ctx)
    }
    fn default_timeout_ms(&self) -> u64 {
        (**self).default_timeout_ms()
    }
}

impl FieldAccess for Arc<dyn Component> {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        Arc::get_mut(self)
            .ok_or_else(|| {
                FieldError::NotFound("Arc has multiple owners; configure before wrapping".into())
            })?
            .set_field(name, value)
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        (**self).get_field(name)
    }
    fn field_names(&self) -> &'static [&'static str] {
        (**self).field_names()
    }
}

impl Describable for Arc<dyn Component> {
    fn describe(&self) -> serde_json::Value {
        (**self).describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestComponent {
        name: ArcIntern<str>,
        value: i32,
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

    #[test]
    fn test_field_access() {
        let mut comp = TestComponent {
            name: ArcIntern::from("test"),
            value: 42,
        };
        assert_eq!(comp.get_field("value").unwrap(), "42");
        comp.set_field("value", "99").unwrap();
        assert_eq!(comp.get_field("value").unwrap(), "99");
        assert!(comp.set_field("nonexistent", "x").is_err());
    }
    #[test]
    fn test_work_context_default() {
        let ctx = WorkContext::default();
        assert!(!ctx.dry_run);
        assert_eq!(ctx.timeout_ms, 30000);
    }
    #[test]
    fn test_work_output_helpers() {
        assert!(WorkOutput::ok("done").success);
        assert!(!WorkOutput::fail("error").success);
    }
    #[test]
    fn test_component_trait_object() {
        let comp = TestComponent {
            name: ArcIntern::from("test"),
            value: 10,
        };
        let boxed: Box<dyn Component> = Box::new(comp);
        assert_eq!(boxed.name(), "test");
    }

    // --- Derive macro tests ---

    #[derive(FieldAccess, Describable)]
    struct BasicConfig {
        name: String,
        count: u32,
        enabled: bool,
    }

    impl WorkUnit for BasicConfig {
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
            Ok(WorkOutput::ok("done"))
        }
    }

    #[test]
    fn test_derive_field_access_basic() {
        let mut cfg = BasicConfig {
            name: "test".into(),
            count: 5,
            enabled: true,
        };
        assert_eq!(cfg.get_field("name").unwrap(), "test");
        assert_eq!(cfg.get_field("count").unwrap(), "5");
        assert_eq!(cfg.get_field("enabled").unwrap(), "true");
        cfg.set_field("count", "10").unwrap();
        assert_eq!(cfg.get_field("count").unwrap(), "10");
        assert!(cfg.set_field("nonexistent", "x").is_err());
    }

    #[test]
    fn test_derive_field_names() {
        let cfg = BasicConfig {
            name: "".into(),
            count: 0,
            enabled: false,
        };
        let names = cfg.field_names();
        assert!(names.contains(&"name"));
        assert!(names.contains(&"count"));
        assert!(names.contains(&"enabled"));
    }

    #[test]
    fn test_derive_describable_basic() {
        let cfg = BasicConfig {
            name: "test".into(),
            count: 5,
            enabled: true,
        };
        let schema = cfg.describe();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(schema["properties"]["count"]["type"], "integer");
        assert_eq!(schema["properties"]["enabled"]["type"], "boolean");
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("name")));
        assert!(required.contains(&serde_json::json!("count")));
    }

    #[derive(FieldAccess, Describable)]
    struct ConstrainedConfig {
        #[field(desc = "TCP port", min = 1, max = 65535)]
        port: u16,
        #[field(desc = "Retry count", min = 0, max = 10)]
        retries: u32,
        #[field(desc = "Host name")]
        host: String,
    }

    impl WorkUnit for ConstrainedConfig {
        fn name(&self) -> &str {
            &self.host
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

    #[test]
    fn test_derive_field_access_constraint_valid() {
        let mut cfg = ConstrainedConfig {
            port: 8080,
            retries: 3,
            host: "localhost".into(),
        };
        cfg.set_field("port", "9000").unwrap();
        assert_eq!(cfg.port, 9000);
        cfg.set_field("retries", "5").unwrap();
        assert_eq!(cfg.retries, 5);
    }

    #[test]
    fn test_derive_field_access_constraint_below_min() {
        let mut cfg = ConstrainedConfig {
            port: 8080,
            retries: 3,
            host: "localhost".into(),
        };
        let err = cfg.set_field("port", "0").unwrap_err();
        match err {
            FieldError::Constraint(msg) => {
                assert!(msg.contains("below minimum"), "unexpected: {}", msg);
            }
            other => panic!("expected Constraint, got {:?}", other),
        }
    }

    #[test]
    fn test_derive_field_access_constraint_above_max() {
        let mut cfg = ConstrainedConfig {
            port: 8080,
            retries: 3,
            host: "localhost".into(),
        };
        let err = cfg.set_field("port", "70000").unwrap_err();
        match err {
            FieldError::Constraint(msg) => {
                assert!(msg.contains("above maximum"), "unexpected: {}", msg);
            }
            other => panic!("expected Constraint, got {:?}", other),
        }
    }

    #[test]
    fn test_derive_field_access_constraint_zero_min() {
        let mut cfg = ConstrainedConfig {
            port: 8080,
            retries: 3,
            host: "localhost".into(),
        };
        cfg.set_field("retries", "0").unwrap();
        assert_eq!(cfg.retries, 0);
    }

    #[test]
    fn test_derive_describable_with_constraints() {
        let cfg = ConstrainedConfig {
            port: 8080,
            retries: 3,
            host: "localhost".into(),
        };
        let schema = cfg.describe();
        let port_schema = &schema["properties"]["port"];
        assert_eq!(port_schema["type"], "integer");
        assert_eq!(port_schema["description"], "TCP port");
        assert_eq!(port_schema["minimum"], "1");
        assert_eq!(port_schema["maximum"], "65535");
    }

    #[test]
    fn test_schema_provider() {
        use super::SchemaProvider;
        let cfg = ConstrainedConfig {
            port: 8080,
            retries: 3,
            host: "localhost".into(),
        };
        let fields = cfg.schema();
        assert_eq!(fields.len(), 3);
        let port = &fields[0];
        assert_eq!(port.name, "port");
        assert_eq!(port.type_name, "u16");
        assert_eq!(port.description.as_deref(), Some("TCP port"));
        assert_eq!(port.min, Some(1.0));
        assert_eq!(port.max, Some(65535.0));
        assert!(port.required);
        let host = &fields[2];
        assert_eq!(host.name, "host");
        assert_eq!(host.type_name, "String");
        assert!(host.min.is_none());
    }

    #[test]
    fn test_derive_component_blanket_impl() {
        let cfg = ConstrainedConfig {
            port: 8080,
            retries: 3,
            host: "localhost".into(),
        };
        let boxed: Box<dyn Component> = Box::new(cfg);
        assert_eq!(boxed.field_names().len(), 3);
    }

    #[derive(FieldAccess)]
    struct FloatMinConfig {
        #[field(min = 1.5)]
        scale: f64,
    }

    #[test]
    fn field_min_float_sets_min_not_max() {
        let mut c = FloatMinConfig { scale: 2.0 };
        // 0.5 is below min=1.5, should fail
        assert!(c.set_field("scale", "0.5").is_err());
        // 2.0 is above min=1.5, should succeed
        assert!(c.set_field("scale", "2.0").is_ok());
        // 1.5 is exactly min, should succeed
        assert!(c.set_field("scale", "1.5").is_ok());
    }

    #[derive(Describable)]
    #[allow(dead_code)]
    struct OptionalConfig {
        name: String,
        #[field(required = false)]
        nickname: Option<String>,
    }

    impl WorkUnit for OptionalConfig {
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
            Ok(WorkOutput::ok("done"))
        }
    }

    #[test]
    fn describable_required_false_excludes_from_required_array() {
        let c = OptionalConfig {
            name: "x".into(),
            nickname: None,
        };
        let desc = c.describe();
        let required = desc["required"].as_array().unwrap();
        assert!(
            required.iter().any(|v| v == "name"),
            "name should be required"
        );
        assert!(
            !required.iter().any(|v| v == "nickname"),
            "nickname should not be required"
        );
    }

    #[test]
    fn schema_provider_required_false() {
        let c = OptionalConfig {
            name: "x".into(),
            nickname: None,
        };
        let fields = c.schema();
        let name_field = fields.iter().find(|f| f.name == "name").unwrap();
        assert!(name_field.required);
        let nick_field = fields.iter().find(|f| f.name == "nickname").unwrap();
        assert!(!nick_field.required);
    }

    #[derive(Describable)]
    #[allow(dead_code)]
    struct FormatConfig {
        #[field(desc = "Endpoint URL", format = "url")]
        endpoint: String,
        #[field(desc = "Timeout", format = "duration")]
        timeout_ms: u64,
        name: String,
    }

    impl WorkUnit for FormatConfig {
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
            Ok(WorkOutput::ok("done"))
        }
    }

    #[test]
    fn format_attribute_reaches_describable_json() {
        let c = FormatConfig {
            endpoint: "https://example.com".into(),
            timeout_ms: 5000,
            name: "test".into(),
        };
        let desc = c.describe();
        let endpoint_schema = &desc["properties"]["endpoint"];
        assert_eq!(endpoint_schema["description"], "Endpoint URL");
        assert_eq!(endpoint_schema["format"], "url");
        let timeout_schema = &desc["properties"]["timeout_ms"];
        assert_eq!(timeout_schema["format"], "duration");
    }

    #[test]
    fn format_attribute_reaches_field_schema() {
        use super::SchemaProvider;
        let c = FormatConfig {
            endpoint: "https://example.com".into(),
            timeout_ms: 5000,
            name: "test".into(),
        };
        let fields = c.schema();
        let endpoint_field = fields.iter().find(|f| f.name == "endpoint").unwrap();
        assert_eq!(endpoint_field.format.as_deref(), Some("url"));
        let timeout_field = fields.iter().find(|f| f.name == "timeout_ms").unwrap();
        assert_eq!(timeout_field.format.as_deref(), Some("duration"));
        let name_field = fields.iter().find(|f| f.name == "name").unwrap();
        assert!(name_field.format.is_none());
    }

    #[test]
    fn noop_runtime_spawn_panics_outside_tokio_with_clear_message() {
        let rt = NoopRuntime;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = rt.spawn(Box::pin(async {}));
        }));
        assert!(result.is_err(), "should panic outside tokio runtime");
        let payload = result.unwrap_err();
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            panic!("unexpected panic payload type");
        };
        assert!(
            msg.contains("NoopRuntime::spawn called outside a tokio runtime"),
            "panic message should be clear, got: {msg}"
        );
        assert!(
            msg.contains("supply a real Runtime"),
            "panic message should suggest fix, got: {msg}"
        );
    }
}
