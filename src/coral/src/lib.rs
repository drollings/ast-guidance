//! Coral: Context-graph library for guidance.
//!
//! ## Modules
//! - `db` — SQLite-backed node store with KNN search, traversal, and capability filtering
//! - `cache_l1` — L1 (hot) cache for frequently accessed nodes
//! - `cache` — Cache reactor, stats, and tier orchestration
//! - `cache_router` — Multi-tier routing with parallel KNN + traversal
//! - `ingest` — Batch ingestion with deferred transactional flush
//! - `mcp` — JSON-RPC 2.0 server (Model Context Protocol) over STDIO
//! - `wasm_runtime` — WASM plugin bridge implementing `WorkUnit` + `Component`

pub mod cache;
pub mod cache_l1;
pub mod cache_router;
pub mod db;
pub mod error;
pub mod ingest;
pub mod knowledge;
pub mod mcp;
pub mod packer;
pub mod tier_units;
pub mod wasm_runtime;
pub mod wvr;

#[cfg(test)]
pub(crate) mod test_stubs;
