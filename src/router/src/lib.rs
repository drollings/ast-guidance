//! LLM Router & Agent Orchestration Framework
//!
//! See `ROADMAP_20260722_CORAL_ROUTER.md` for the full architecture.
//!
//! ## Modules
//! - `pipeline_types` — `StageDecision`, `PipelineStage`, `StageVerdict`
//! - `types` — `RouterRequest`, `RouterResponse`, `RouterMessage`, etc.
//! - `session` — `SessionNode`, `StepStatus`
//! - `config` — `RouterConfig` and all sub-config types
//! - `pipeline` — `PipelineOrchestrator`, `PipelineResult`
//! - `stages` — pipeline stage implementations
//! - `transforms` — `TransformStrategy`, transforms (NoTransform, PiiAnonymize, etc.)
//! - `dispatch` — `FrontierDispatcher`, `AgentDispatcher`
//! - `watchdog` — `WatchdogSet`, `MaxTokenWatchdog`, `WallClockWatchdog`, `RepetitionWatchdog`
//! - `agent` — `AgentRegistry`, `AgentIdentity`, `AgentTask`, `AgentError`
//! - `orchestrator` — `OrchestratorSession`
//! - `compaction` — `CompactionStrategy`, `RecencyCompaction`
//! - `kv_cache` — `HotKvCache`, `ColdKvCache`, `KvCacheManager`
//! - `summarization` — `ResultScorer`, `ScoredResult`, `Summarizer`

pub mod pipeline_types;
pub mod types;
pub mod session;
pub mod config;
pub mod pipeline;
pub mod stages;
pub mod transforms;
pub mod dispatch;
pub mod watchdog;
pub mod agent;
pub mod orchestrator;
pub mod compaction;
pub mod kv_cache;
pub mod summarization;

#[cfg(test)]
pub(crate) mod test_stubs;

#[cfg(test)]
mod stage_tests;

#[cfg(test)]
mod tests;
