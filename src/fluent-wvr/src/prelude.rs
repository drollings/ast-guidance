//! The fluent-wvr prelude — import this in every consumer for the 80% case.
//!
//! ```rust
//! use fluent_wvr::prelude::*;
//! ```

pub use crate::wrapper::{
    retry_call, ComponentAdapter, ComponentCascade, ExecuteFn, Instrumented, Middleware,
    MiddlewareChain, Pipeline, SuffixedComponent,
};
pub use crate::{impl_component, impl_fieldless};
pub use crate::{
    Capability, CapabilitySet, Component, Describable, FieldAccess, FieldError, MetadataValue,
    WorkContext, WorkError, WorkOutput, WorkUnit,
};
pub use internment::ArcIntern;
