//! # fluent-db — the canonical database-access layer for the workspace
//!
//! **The DB principle**: a single database-access crate owns *connection
//! lifecycle* (open/pool/checkout), *statement execution* (typed query/execute,
//! transactions, batching, retry classification), *schema lifecycle* (DDL init,
//! idempotent migrations, versioning), *reusable store shapes* (single-connection
//! store, pooled store, TTL/LRU cache store, HNSW-backed vector store), *vector
//! math* for embeddings, and *the capability-gated async surface* (the successor
//! to `DbCapability`).
//!
//! Domain semantics — what a "node", "session", "ledger entry", "query cache row",
//! or "workflow chart" *is* — stay in the consumer crate. `fluent-db` is generic,
//! composable, and polymorphic: its components implement the `fluent-wvr`
//! `Component`/`WorkUnit` surface where orchestration needs them, and it composes
//! `tokio` (`Semaphore`, `spawn_blocking`) and `common-core` primitives rather
//! than re-implementing them.
//!
//! ## Import boundary (D2)
//!
//! `fluent-db` may import `common-core` (with `sqlite` feature), `fluent-wvr`,
//! `tokio`, `rusqlite`, `hnsw_rs`, `anndists`, `serde`, `thiserror`,
//! `tracing`. It must NOT import `fluent-concurrency`, `guidance`, `coral`,
//! `fluent-router`, `search-vector`, `knowledge`, `ontology`, `rdf`,
//! `fluent-types`, or `wasm_ipc`. The dependency direction is **acyclic**:
//! `fluent-concurrency` re-exports `fluent-db::capability::DbCapability`
//! behind its `db` feature (never the reverse), and the capability-gating
//! primitives (`CURRENT_CAPS`, `check_capability`, `CapabilityError`) live in
//! `fluent-wvr` so both crates read the same task-local.
//!
//! ## Zero-domain mechanics stay put
//!
//! `common-core::sqlite` remains the zero-domain home of *raw* SQLite helpers
//! (`open_wal`, `open_in_memory`, `run_batch`, `in_clause`,
//! `is_unique_violation`, `make_hnsw`, `embedding_cache` DDL). `fluent-db` is the
//! *policy* layer (pooling, typing, orchestration) above the *mechanics* layer; it
//! composes `common-core`, it does not move or duplicate it.
//!
//! ## Zero-cost and optional
//!
//! The rusqlite surface is feature-gated on `sqlite` (default-on). A consumer
//! that only wants pools/scope/batch pays nothing for the database layer.

#[cfg(feature = "sqlite")]
pub mod error;
#[cfg(feature = "sqlite")]
pub mod hnsw;

#[cfg(feature = "sqlite")]
pub mod cache;
#[cfg(feature = "sqlite")]
pub mod capability;
#[cfg(feature = "sqlite")]
pub mod migrate;
#[cfg(feature = "sqlite")]
pub mod pool;
#[cfg(feature = "sqlite")]
pub mod query;
#[cfg(feature = "sqlite")]
pub mod store;
#[cfg(feature = "sqlite")]
pub mod vector;
#[cfg(feature = "sqlite")]
pub mod wvr;

#[cfg(all(test, feature = "sqlite"))]
#[path = "../tests/mod.rs"]
mod tests;
