//! `DynamicComponent` — a `Component` assembled at runtime from a config map.
//!
//! **Status: compatibility surface (scaffold).** This is the database-driven
//! config path from SKILL §13: a component whose fields come from key/value
//! config (e.g. `SELECT key, value FROM tool_config`) and whose executable
//! body is injected as a closure. It exists so the `Component` interface is
//! uniformly available to *any* runtime-assembled unit — Rust struct, WASM
//! plugin, or DB config — without branching on origin. It has no production
//! consumer in-tree today; keep it as the forward-compat seam for future
//! runtime reconfiguration (do not prune as dead code).

use std::collections::HashMap;
use std::sync::Arc;

use internment::ArcIntern;

use crate::traits::{Describable, FieldAccess, FieldError};
use crate::work::{WorkContext, WorkError, WorkOutput};
use crate::{impl_component, WorkUnit};

/// Executable body of a dynamically assembled component.
///
/// The closure receives the live config map (as it stands at execute time)
/// and the `WorkContext`, and returns a `WorkOutput`. Signature is chosen to
/// be implementable from a DB-row fetch, an HTTP callback, or a WASM call
/// without capturing anything beyond what those paths already own.
pub type DynamicExecutor =
    Arc<dyn Fn(&WorkContext, &HashMap<String, String>) -> Result<WorkOutput, WorkError> + Send + Sync>;

/// Compatibility surface (scaffold) — see ROADMAP_20260901_FIXES_4.md M0
/// A runtime-assembled `Component`.
///
/// Fields are stored in an interior `Mutex<HashMap<String, String>>` so
/// `set_field` works on `&mut self` after construction (matching `WasmComponent`).
/// `field_names` returns the declared schema keys; unknown keys are still
/// storable (the map is open-ended) but are reported as config entries in
/// `describe`.
#[doc(hidden)]
#[allow(dead_code)]
pub struct DynamicComponent {
    name: ArcIntern<str>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
    fields: std::sync::Mutex<HashMap<String, String>>,
    /// Declared configurable keys, in canonical order — `field_names()`.
    field_keys: &'static [&'static str],
    executor: DynamicExecutor,
}

impl DynamicComponent {
    /// Build a dynamic component with a fixed name and an executor body.
    pub fn new(name: impl Into<String>, executor: DynamicExecutor) -> Self {
        Self {
            name: ArcIntern::from(name.into()),
            depends: vec![],
            provides: vec![],
            fields: std::sync::Mutex::new(HashMap::new()),
            field_keys: &[],
            executor,
        }
    }

    /// Declare the configurable field keys (`field_names()` contract).
    #[must_use]
    pub fn with_field_keys(mut self, keys: &'static [&'static str]) -> Self {
        self.field_keys = keys;
        self
    }

    /// Pre-seed the config map (e.g. rows from a `tool_config` query).
    #[must_use]
    pub fn with_config(self, config: HashMap<String, String>) -> Self {
        *self.fields.lock().expect("dynamic component config poisoned") = config;
        self
    }

    #[must_use]
    pub fn with_depends(mut self, deps: &[ArcIntern<str>]) -> Self {
        self.depends = deps.to_vec();
        self
    }

    #[must_use]
    pub fn with_provides(mut self, prov: &[ArcIntern<str>]) -> Self {
        self.provides = prov.to_vec();
        self
    }

    /// Read the current config map (for the executor and tests).
    pub fn config(&self) -> HashMap<String, String> {
        self.fields.lock().expect("dynamic component config poisoned").clone()
    }
}

impl FieldAccess for DynamicComponent {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        self.fields
            .lock()
            .expect("dynamic component config poisoned")
            .insert(name.to_string(), value.to_string());
        Ok(())
    }

    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        self.fields
            .lock()
            .expect("dynamic component config poisoned")
            .get(name)
            .cloned()
            .ok_or_else(|| FieldError::NotFound(name.into()))
    }

    fn field_names(&self) -> &'static [&'static str] {
        self.field_keys
    }
}

impl Describable for DynamicComponent {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "type": "dynamic",
            "config": self.config(),
        })
    }
}

impl WorkUnit for DynamicComponent {
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
        (self.executor)(ctx, &self.config())
    }
}

impl_component!(DynamicComponent);
