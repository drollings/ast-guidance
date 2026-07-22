//! Job Copilot — local-only human-in-the-loop job application copilot.
//!
//! This crate provides the library core for the job-copilot-daemon binary.
//! It exposes JSON-RPC schema types, configuration, sanitization, profile
//! loading, field dispatching, and the server transport.
//!
//! **Dependency policy** (from `AGENTS.md`):
//! This crate may import from `common-core`, `fluent-wvr`,
//! `fluent-concurrency`, `guidance-llm`, `fluent-types`, `dag`,
//! `search-vector`, `memory-plugin`, `content-node`, and the standard
//! library / `tokio` / `reqwest`. It must NOT import from `guidance`,
//! `coral`, `wasm_ipc`, `project-knowledge`, `ontology`, or `rdf`.
//! Domain logic for the copilot lives in `src/job-copilot`; do not add
//! it to any shared crate.

pub mod components;
pub mod config;
pub mod dispatcher;
pub mod error;
pub mod memory;
pub mod profile;
pub mod prompt;
pub mod sanitize;
pub mod schema;
pub mod server;
pub mod similarity;

pub use schema::{
    AnalyzeFormResponse, DaemonHealth, FeedbackAction, FeedbackParams, FieldDescription,
    HistogramSummary, PageAnalyzeFormParams, PreFilledValue, SelectOption, SkippedField,
    SkippedReason, ValueSource,
};

pub use dispatcher::FieldValueDispatcher;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_compiles_and_loads() {
        let _: &str = config::CONFIG_FILE_DEFAULT;
    }
}
