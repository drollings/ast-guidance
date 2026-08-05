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
//! - No domain-specific dependencies (no rusqlite, no LLM, no fluent-types)
//! - The thinness is intentional — value is in the trait boundaries
//! - If a derive macro (`#[derive(FieldAccess)]`) is added later, it goes here
//!
//! Consumers: `fluent-dag`, `coral-context`, `content-node`

extern crate self as fluent_wvr;

pub mod capability;
pub mod macros;
pub mod metadata;
pub mod runtime;
pub mod store;
pub mod traits;
pub mod work;

pub mod prelude;
pub mod wrapper;

pub use capability::{check_capability, Capability, CapabilityError, CapabilitySet, CURRENT_CAPS};
pub use fluent_wvr_macros::{Describable, FieldAccess};
pub use internment::ArcIntern;
pub use metadata::MetadataValue;
pub use runtime::{NoopRuntime, Runtime};
pub use store::OutputStore;
pub use traits::{
    component_downcast_mut, component_downcast_ref, Component, ComponentArcExt, Describable,
    FieldAccess, FieldError, FieldSchema, PersistableComponent, SchemaProvider, WorkUnit,
};
pub use work::{WorkContext, WorkError, WorkOutput};

// Re-export string utilities from common-core for use in derive macros and consumers.
pub use common_core::string::{
    contains_ident_word, contains_ignore_case, contains_word, looks_like_identifier, slugify,
    strip_html, truncate_at_sentence,
};

use std::any::Any;
use std::sync::Arc;

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
    fn type_name(&self) -> &'static str {
        (**self).type_name()
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
    fn type_name(&self) -> &'static str {
        (**self).type_name()
    }
}

impl FieldAccess for Arc<dyn Component> {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        Arc::get_mut(self)
            .ok_or_else(|| {
                FieldError::ReadOnly(
                    name.to_string(),
                    "Arc has multiple owners; configure before wrapping".into(),
                )
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

impl Component for Arc<dyn Component> {
    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        Arc::get_mut(self)
            .expect(
                "Arc<dyn Component>::as_any_mut: Arc has multiple owners. \
                 Use ComponentArcExt::try_as_any_mut to handle this case safely.",
            )
            .as_any_mut()
    }
}

#[cfg(test)]
mod tests;
