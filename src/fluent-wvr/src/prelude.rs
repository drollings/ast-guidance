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
    boundary::{
        decode_boundary, decode_boundary_typed, extract_members, repair_boundary, BoundaryError,
        BoundaryOptions,
    },
    Capability, CapabilitySet, Component, Describable, DynamicComponent, DynamicExecutor,
    FieldAccess, FieldError, MetadataValue, OutputStore, WorkContext, WorkError, WorkOutput,
    WorkUnit,
};
pub use internment::ArcIntern;
