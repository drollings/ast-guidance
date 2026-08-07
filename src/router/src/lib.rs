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
//! - `dispatch` — `ChatBackend` + `OpenAiChatBackend`/`RetryChatBackend`/`FallbackChatBackend`
//! - `kv_cache` — `HotKvCache`, `ColdKvCache`, `KvCacheManager`
//! - `summarization` — `ResultScorer`, `ScoredResult`, `Summarizer`
//! - `scheduler` — `AffinityScheduler`, `ScheduledTask`, `AgingConfig`
//! - `dag_session` — `DependencySession`, `SessionStep`, `StepResult`, `DagError`,
//!   `SessionRegistry`
//! - `ledger` — `ContentNodeLedger` (canonical `ContentNode` store; LOD0/LOD5
//!   eager, LOD1–4 lazy from LOD0 via `Summarizer`), `CompactionStrategy`,
//!   `RecencyCompaction` (folded in from the deleted `compaction.rs`)

pub mod audit;
pub mod charts;
pub mod config;
pub mod dag_session;
pub mod dispatch;
pub mod error;
pub mod filters;
pub mod frontier;
pub mod hnsw;
pub mod instances;
pub mod knowledge;
pub mod kv_cache;
pub mod ledger;
pub mod ledger_guard;
pub mod logging;
pub mod metrics;
pub mod node_store;
pub mod normalize;
pub mod pipeline;
pub mod pipeline_types;
pub mod routes;
pub mod scheduler;
pub mod score_matrix;
pub mod server;
pub mod session;
pub mod stages;
pub mod streaming;
pub mod summarization;
pub mod target_match;
pub mod telemetry;
pub mod transforms;
pub mod types;
pub mod views;

/// Testing utilities — available in all build profiles for use by
/// downstream crates' test code (e.g., E2E tests in coral-context).
pub mod testing;

#[cfg(test)]
mod server_http_tests;
#[cfg(test)]
mod server_tests;
#[cfg(test)]
mod stage_tests;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
pub(crate) mod test_stubs;
#[cfg(test)]
mod tests;
