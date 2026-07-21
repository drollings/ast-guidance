use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::capability::CapabilitySet;
use crate::metadata::MetadataValue;
use crate::runtime::{NoopRuntime, Runtime};
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
/// # Examples
///
/// ```no_run
/// use fluent_wvr::{WorkContext, CapabilitySet, NoopRuntime};
/// use std::sync::Arc;
///
/// let ctx = WorkContext {
///     dry_run: true,
///     max_retries: 3,
///     timeout_ms: 10_000,
///     metadata: Default::default(),
///     rt: Arc::new(NoopRuntime),
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
    fn field_error_partial_eq() {
        let a = FieldError::NotFound("x".into());
        let b = FieldError::NotFound("x".into());
        assert_eq!(a, b);
    }
}
