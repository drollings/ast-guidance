use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::capability::CapabilitySet;
use crate::metadata::MetadataValue;
use crate::runtime::{NoopRuntime, Runtime};
use crate::store::OutputStore;
use crate::traits::WorkUnit;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum WorkError {
    #[error("execution failed: {0}")]
    Execution(String),
    #[error("dependency not satisfied: {0}")]
    Dependency(String),
    #[error("timeout after {duration_ms}ms ({unit})")]
    Timeout { duration_ms: u64, unit: String },
}

impl WorkError {
    /// Whether the error represents a transient condition worth retrying.
    ///
    /// `Execution` is a *permanent* failure — the unit ran and failed, so a
    /// retry would burn backoff budget without a recovery signal. `Dependency`
    /// (a prerequisite not yet satisfied) and `Timeout` (attempt budget
    /// exhausted, e.g. by transient overload) are retryable.
    ///
    /// This is the canonical predicate for retry loops that drive
    /// `WorkUnit::execute` (the `Zone` supervisor and any future caller);
    /// `common_core::retry::retry_async` short-circuits on `false`.
    pub fn is_retryable(&self) -> bool {
        matches!(self, WorkError::Dependency(_) | WorkError::Timeout { .. })
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

    /// Create a successful output with typed data serialized to JSON.
    /// Returns `Err(WorkError::Execution(...))` on serialization failure.
    /// This is the canonical way to pass structured data between components.
    pub fn typed<T: serde::Serialize>(
        message: impl Into<String>,
        data: &T,
    ) -> Result<Self, WorkError> {
        let data = serde_json::to_value(data)
            .map_err(|e| WorkError::Execution(format!("serialize typed data: {e}")))?;
        Ok(Self {
            success: true,
            message: message.into(),
            data,
        })
    }

    /// Same as `typed` but panics on serialization failure.
    /// Use this when the input is trusted and you want a one-liner.
    /// For untrusted input, prefer `typed` to handle the error gracefully.
    pub fn typed_infallible<T: serde::Serialize>(message: impl Into<String>, data: &T) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: serde_json::to_value(data).expect("serialize typed data"),
        }
    }

    /// Deserialize the `data` field back to a typed value (borrowed).
    pub fn data_as<T: for<'de> Deserialize<'de>>(&self) -> Result<T, WorkError> {
        serde_json::from_value(self.data.clone())
            .map_err(|e| WorkError::Execution(format!("deserialize data: {e}")))
    }

    /// Deserialize the `data` field back to a typed value, consuming `self`.
    /// Avoids the clone that `data_as` requires.
    pub fn data_take<T: for<'de> Deserialize<'de>>(self) -> Result<T, WorkError> {
        serde_json::from_value(self.data)
            .map_err(|e| WorkError::Execution(format!("deserialize data: {e}")))
    }
}

impl std::fmt::Display for WorkOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.success {
            write!(f, "OK: {}", self.message)
        } else {
            write!(f, "FAIL: {}", self.message)
        }
    }
}

/// Execution context passed to `WorkUnit::execute`. Carries configuration
/// (dry-run, retries, timeout), typed metadata, a runtime, and a capability set.
///
/// # Data channels — the decision rule
///
/// A `WorkContext` exposes four data channels with different typing fidelity.
/// The rule for choosing between them is:
///
/// 1. **`outputs` (the typed store, [`OutputStore`])** — the *primary* channel
///    for in-process handoff between stages / work units. Values are written
///    with [`WorkContext::set::<T>`] and read with [`WorkContext::get::<T>`],
///    so the type is pinned at the call site and no serialize/deserialize
///    round-trip happens between producers and consumers in the same process.
/// 2. **`WorkOutput.data` (`serde_json::Value`)** — reserved exclusively for
///    payloads that genuinely cross a serialization boundary: WASM units,
///    network dispatch, and a `WorkUnit`'s output consumed by a generic
///    orchestrator. Do not use it for same-process handoffs.
/// 3. **`metadata` (`HashMap<String, MetadataValue>`)** — only for genuinely
///    dynamic / stringly-typed fields such as debug annotations, where the
///    value's type is not known at the call site.
/// 4. **`structured` (`HashMap<String, serde_json::Value>`)** — the JSON
///    handoff channel for payloads that are natively `serde_json::Value`-shaped
///    (e.g. a serialized `RouterRequest`). Prefer `outputs` whenever the type
///    is known at compile time.
///
/// # Examples
///
/// ```no_run
/// use fluent_wvr::{WorkContext, CapabilitySet, NoopRuntime};
/// use std::sync::Arc;
///
/// let mut ctx = WorkContext {
///     dry_run: true,
///     max_retries: 3,
///     timeout_ms: 10_000,
///     metadata: Default::default(),
///     structured: Default::default(),
///     outputs: Default::default(),
///     rt: Arc::new(NoopRuntime),
///     caps: CapabilitySet::new(),
/// };
/// ctx.set("attempt", 2_i32);
/// assert_eq!(ctx.get::<i32>("attempt"), Some(&2));
/// ```
#[derive(Clone)]
pub struct WorkContext {
    pub dry_run: bool,
    pub max_retries: u32,
    pub timeout_ms: u64,
    pub metadata: HashMap<String, MetadataValue>,
    /// Typed/structured metadata channel. Unlike `metadata` (scalar
    /// `MetadataValue`), this holds arbitrary `serde_json::Value` payloads —
    /// the canonical home for structured handoffs such as the serialized
    /// `RouterRequest` (`"request"`), bound chart entities (`"entities"`),
    /// and per-stage outputs (`"stage.{id}"`). Crosses crate boundaries as
    /// `serde_json::Value` only (never a domain type).
    ///
    /// Prefer the typed store (`outputs`) whenever the payload type is known
    /// at compile time — see the decision rule above.
    pub structured: HashMap<String, serde_json::Value>,
    /// Typed in-process handoff accumulator (the primary inter-unit channel).
    ///
    /// Write with [`WorkContext::set::<T>`] / [`OutputStore::set`], read with
    /// [`WorkContext::get::<T>`] / [`OutputStore::get`]. See the struct-level
    /// decision rule for when this is preferred over `data`/`metadata`.
    pub outputs: OutputStore,
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
            .field("structured", &self.structured)
            .field("outputs", &self.outputs)
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
            structured: HashMap::new(),
            outputs: OutputStore::default(),
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
            structured: HashMap::new(),
            outputs: OutputStore::default(),
            rt: Arc::new(NoopRuntime),
            caps,
        }
    }

    /// Store a typed value for in-process handoff under `key`.
    ///
    /// The canonical writer for the typed store — equivalent to
    /// `self.outputs.set::<T>(key, value)`. No serialization is performed.
    pub fn set<T: Send + Sync + 'static>(&mut self, key: impl Into<String>, value: T) {
        self.outputs.set(key, value);
    }

    /// Read a typed value previously written with [`WorkContext::set`].
    ///
    /// Returns `None` if `key` is absent or holds a different type. This is
    /// the canonical reader for the typed store; no deserialization is
    /// performed.
    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        self.outputs.get::<T>(key)
    }

    /// Read a structured value by key, deserializing it to a typed value.
    /// The single place that does `serde_json::from_value` for the channel.
    pub fn structured<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<T, WorkError> {
        let value = self.structured.get(key).ok_or_else(|| {
            WorkError::Execution(format!("structured metadata key not found: {key}"))
        })?;
        serde_json::from_value(value.clone())
            .map_err(|e| WorkError::Execution(format!("structured metadata key {key:?}: {e}")))
    }

    /// Write a structured value by key, serializing a typed value.
    /// The single place that does `serde_json::to_value` for the channel.
    /// A serialization failure is silently skipped (matches the typed-setter
    /// convention of `StageMetadata`).
    pub fn set_structured<T: serde::Serialize>(&mut self, key: &str, value: &T) {
        if let Ok(v) = serde_json::to_value(value) {
            self.structured.insert(key.to_string(), v);
        }
    }

    /// Returns a new `WorkContext` with `rt` and `caps` cloned from the zone's
    /// defaults, with `max_retries`/`timeout_ms`/etc overridden by the supplied
    /// closure.
    ///
    /// For batched registration of identical contexts, construct one `WorkContext`
    /// and reuse it with `register_with_context` directly.
    pub fn for_unit_in_zone(
        zone_rt: &Arc<dyn Runtime>,
        zone_caps: &CapabilitySet,
        mutate: impl FnOnce(&mut WorkContext),
    ) -> Self {
        let mut ctx = WorkContext {
            rt: Arc::clone(zone_rt),
            caps: zone_caps.clone(),
            ..WorkContext::default()
        };
        mutate(&mut ctx);
        ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::FieldError;

    #[test]
    fn test_work_context_default() {
        let ctx = WorkContext::default();
        assert!(!ctx.dry_run);
        assert_eq!(ctx.timeout_ms, 30000);
        assert!(ctx.outputs.is_empty());
    }

    #[test]
    fn work_context_typed_channel_handoff() {
        let mut ctx = WorkContext::default();
        ctx.set("stage", "deterministic".to_string());
        ctx.set("attempt", 1_i32);

        assert_eq!(
            ctx.get::<String>("stage"),
            Some(&"deterministic".to_string())
        );
        assert_eq!(ctx.get::<i32>("attempt"), Some(&1));
        assert_eq!(ctx.get::<u32>("attempt"), None, "wrong type reads None");
        assert_eq!(ctx.get::<i32>("absent"), None);

        let clone = ctx.clone();
        assert_eq!(
            clone.get::<String>("stage"),
            Some(&"deterministic".to_string()),
            "clone shares the typed allocation"
        );
    }

    #[test]
    fn work_context_typed_channel_overwrite() {
        let mut ctx = WorkContext::default();
        ctx.set("verdict", "passed".to_string());
        ctx.set("verdict", "rerouted".to_string());
        assert_eq!(ctx.get::<String>("verdict"), Some(&"rerouted".to_string()));
    }

    #[test]
    fn test_work_output_helpers() {
        assert!(WorkOutput::ok("done").success);
        assert!(!WorkOutput::fail("error").success);
    }

    #[test]
    fn work_output_typed_and_data_as_roundtrip() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct MyData {
            x: i32,
            label: String,
        }
        let data = MyData {
            x: 42,
            label: "hello".into(),
        };
        let output = WorkOutput::typed("ok", &data).unwrap();
        assert!(output.success);
        assert_eq!(output.message, "ok");

        let recovered: MyData = output.data_as().unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn work_output_typed_returns_err_on_unserializable() {
        /// A type whose `Serialize` impl always fails.
        struct AlwaysFails;
        impl serde::Serialize for AlwaysFails {
            fn serialize<S: serde::Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("always fails"))
            }
        }
        let result = WorkOutput::typed("fail", &AlwaysFails);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("execution failed") || msg.contains("serialize"),
            "error message should describe serialization failure, got: {msg}"
        );
    }

    #[test]
    fn work_output_data_take_consumes_self() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct MyData {
            x: i32,
        }
        let data = MyData { x: 42 };
        let output = WorkOutput::typed("ok", &data).unwrap();

        let recovered: MyData = output.data_take().unwrap();
        assert_eq!(recovered, data);
        // output is consumed — cannot call data_as anymore
    }

    #[test]
    fn work_output_typed_infallible_roundtrip() {
        #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
        struct MyData {
            x: i32,
        }
        let data = MyData { x: 42 };
        let output = WorkOutput::typed_infallible("ok", &data);
        assert!(output.success);
        let recovered: MyData = output.data_as().unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn work_output_display_ok() {
        let out = WorkOutput::ok("done");
        assert_eq!(format!("{}", out), "OK: done");
    }

    #[test]
    fn work_output_display_fail() {
        let out = WorkOutput::fail("oops");
        assert_eq!(format!("{}", out), "FAIL: oops");
    }

    #[test]
    fn work_error_partial_eq() {
        let a = WorkError::Execution("boom".into());
        let b = WorkError::Execution("boom".into());
        assert_eq!(a, b);
    }

    #[test]
    fn work_error_is_retryable_classification() {
        // Execution is a permanent failure — never retry.
        assert!(!WorkError::Execution("boom".into()).is_retryable());
        // Dependency (prerequisite not yet satisfied) is transient — retry.
        assert!(WorkError::Dependency("awaiting asset".into()).is_retryable());
        // Timeout (budget exhausted under load) is transient — retry.
        assert!(WorkError::Timeout {
            duration_ms: 100,
            unit: "u".into()
        }
        .is_retryable());
    }

    #[test]
    fn field_error_partial_eq() {
        let a = FieldError::NotFound("x".into());
        let b = FieldError::NotFound("x".into());
        assert_eq!(a, b);
    }
}
