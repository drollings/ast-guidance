#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! # memory-plugin
//!
//! Pluggable memory tier for guidance. Provides structured, persistent memory
//! backends that integrate with the `fluent-concurrency` async runtime and
//! `fluent-wvr` polymorphic control plane.
//!
//! ## Architecture
//!
//! Every memory plugin implements `MemoryPlugin`, which composes:
//! - `Component` (from fluent-wvr): `FieldAccess + Describable + WorkUnit + Send + Sync`
//! - `MemoryOps` (from this crate): domain-specific memory lifecycle
//!
//! The `MemoryPluginRegistry` stores type-erased `Arc<dyn MemoryPlugin>` handles.
//! The orchestrator never branches on implementation type.

pub mod capability;
pub mod plugins;
pub mod registry;
pub mod traits;
pub mod types;
pub mod batch;

pub use capability::MemoryCapability;
pub use registry::MemoryPluginRegistry;
pub use traits::{MemoryOps, MemoryPlugin};
pub use types::*;
pub use batch::MemoryBatch;
