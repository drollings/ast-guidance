//! Affinity-aware priority scheduler.
//!
//! Re-exports `AffinityScheduler`, `ScheduledTask`, and `AgingConfig` from
//! `fluent_concurrency::affinity`. Import from there for new code; this
//! module is kept for backward compatibility.
//!
//! Uses `fluent-concurrency::pool::PriorityResultPool` internally — higher
//! priority values are dispatched first; within the same priority, FIFO order
//! is maintained.

pub use fluent_concurrency::affinity::{AffinityScheduler, AgingConfig, ScheduledTask};
