//! fluent-dag: DAG executor with resolver, middleware, adapter, and work unit
//! abstractions. Orchestrates dependency-driven workflow execution.
#![forbid(unsafe_code)]

pub mod adapter;
pub(crate) mod closure;
pub mod checkpointed;
pub mod dep_graph;
pub mod error;
pub mod middleware;
pub(crate) mod narrowing;
pub mod resolver;
pub mod target;
pub mod target_work_unit;
pub mod type_inference;
pub mod work_unit;
pub mod wvr;
pub mod yamake_loader;

#[cfg(test)]
#[path = "../tests/mod.rs"]
mod tests;
