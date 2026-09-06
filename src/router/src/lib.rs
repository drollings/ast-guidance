//! LLM Router & Agent Orchestration Framework
//!
//! ## Modules
//! - `pipeline_types` — `StageDecision`, `PipelineStage`, `StageVerdict`
//! - `types` — `RouterRequest`, `RouterResponse`, `RouterMessage`, etc.
//! - `session` — `StepStatus` re-exported from `fluent_types::ContentNode`
//! - `config` — `RouterConfig` and all sub-config types
//! - `pipeline` — `PipelineOrchestrator`, `PipelineResult`
//! - `stages` — pipeline stage implementations (deterministic, classifier, router)
//! - `transforms` — `TransformStrategy`, transforms (NoTransform, PiiAnonymize, etc.)
//! - `dispatch` — `DispatchBackend` + `OpenAiChatBackend`/`RetryBackend`/`BackendChain`
//! - `kv_cache` — `HotSnapshotIndex`, `ColdSnapshotIndex`, `SnapshotStore`
//! - `summarization` — `ResultScorer`, `ScoredResult`, `Summarizer`
//! - `dag_session` — `DependencySession`, `SessionStep`, `StepResult`, `DagError`,
//!   `SessionRegistry`
//! - `ledger` — `ContentNodeLedger` (canonical `ContentNode` store; LOD0/LOD5
//!   eager, LOD1–4 lazy from LOD0 via `Summarizer`), `CompactionStrategy`,
//!   `RecencyCompaction` (folded in from the deleted `compaction.rs`)

pub mod audit;
pub mod charts;
pub mod cli;
pub mod concept_store_sqlite;
pub mod config;
pub mod dag_session;
pub mod dispatch;
pub mod error;
pub mod filters;
pub mod frontier;
pub mod instances;
pub mod knowledge;
pub mod kv_cache;
pub mod ledger;
pub mod ledger_guard;
pub mod logging;
pub mod metrics;
pub mod node_store;
pub mod normalize;
pub mod routing_context;
pub mod ort;
pub mod overlay;
pub mod pipeline;
pub mod pipeline_types;
pub mod ranking;
pub mod retrieval;
pub mod routes;
pub mod score_matrix;
pub mod server;
pub mod session;
pub mod stages;
pub mod streaming;
pub mod summarization;
pub mod supervisor;
pub mod target_match;
pub mod telemetry;
pub mod transforms;
pub mod types;
pub mod views;

/// Testing utilities — available in all build profiles for use by
/// downstream crates' test code (e.g., E2E tests in coral-context).
pub mod testing;

#[cfg(test)]
#[path = "../tests/server_http_tests.rs"]
mod server_http_tests;
#[cfg(test)]
#[path = "../tests/server_tests.rs"]
mod server_tests;
#[cfg(test)]
#[path = "../tests/stage_tests.rs"]
mod stage_tests;
#[cfg(test)]
#[path = "../tests/config_route_tests.rs"]
mod config_route_tests;
#[cfg(test)]
#[path = "../tests/deprecated_baseline.rs"]
mod deprecated_baseline;
#[cfg(test)]
#[path = "../tests/build_graph.rs"]
mod build_graph;
#[cfg(test)]
#[path = "../tests/supervisor_integration_tests.rs"]
mod supervisor_integration_tests;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
pub(crate) mod test_stubs;
#[cfg(test)]
#[path = "../tests/mod.rs"]
mod tests;
