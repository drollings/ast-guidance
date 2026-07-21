//! The fluent-wvr prelude — import this in every consumer for the 80% case.
//!
//! ```rust
//! use fluent_wvr::prelude::*;
//! ```

pub use crate::impl_component;
pub use crate::wrapper::{retry_call, ComponentAdapter, Instrumented, RetryResult, WithRetry};
pub use crate::{
    Capability, CapabilitySet, Component, ComponentArcExt, Describable, FieldAccess, FieldError,
    FieldSchema, MetadataValue, SchemaProvider, WorkContext, WorkError, WorkOutput, WorkUnit,
};
pub use internment::ArcIntern;
