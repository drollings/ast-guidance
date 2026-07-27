//! fluent-dag: DAG executor with resolver, middleware, adapter, and work unit
//! abstractions. Orchestrates dependency-driven workflow execution.

pub mod adapter;
pub mod ambiguous_resolver;
pub mod dep_graph;
pub mod error;
pub mod executor;
pub mod middleware;
pub mod resolver;
pub mod target;
pub mod type_inference;
pub mod work_unit;
pub mod wvr;
pub mod yamake_loader;
