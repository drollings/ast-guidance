//! Shared types for the memory plugin system.

use fluent_types::SessionId;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

/// Initialization context passed to `MemoryOps::initialize`.
#[derive(Debug, Clone)]
pub struct MemoryInitContext {
    /// Current session identifier.
    pub session_id: SessionId,
    /// Workspace root (e.g., `/opt/src/rust/monorepo`).
    pub workspace_root: PathBuf,
    /// Memory storage root (e.g., `~/.guidance/memory/`).
    pub memory_root: PathBuf,
    /// Platform string: `"linux"`, `"macos"`, `"windows"`.
    pub platform: &'static str,
}

/// Per-query context. Carries session identity and capability tokens.
#[derive(Clone)]
pub struct MemoryQueryContext {
    /// Current session identifier.
    pub session_id: SessionId,
    /// Capability set for this query scope.
    pub caps: fluent_wvr::CapabilitySet,
    /// Runtime handle for spawning background work.
    pub rt: std::sync::Arc<dyn fluent_wvr::Runtime>,
}

/// Search request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchRequest {
    /// Free-text query.
    pub query: String,
    /// Optional category filter.
    pub category: Option<String>,
    /// Minimum trust score filter (0.0–1.0).
    pub min_trust: f64,
    /// Maximum results to return.
    pub limit: usize,
    /// Which retrieval strategy to use.
    pub strategy: SearchStrategy,
}

/// Retrieval strategy discriminator.
///
/// Flat enum dispatch — no `dyn Trait` in the hot path. The plugin
/// matches on this enum to select the retrieval pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// FTS5 keyword search (holographic default).
    FtsKeyword,
    /// HRR compositional algebraic probe.
    HrrProbe {
        /// Entity names to probe.
        entities: Vec<String>,
    },
    /// Knowledge graph structural traversal.
    GraphTraversal {
        /// Maximum traversal depth.
        depth: usize,
    },
    /// Hybrid: FTS + Jaccard + HRR (holographic full pipeline).
    Hybrid,
}

/// Scored retrieval result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    /// The fact or entity content.
    pub content: String,
    /// Relevance score (higher = more relevant).
    pub score: f64,
    /// Trust score (0.0–1.0).
    pub trust: f64,
    /// Plugin name that produced this result.
    pub source: String,
    /// Plugin-specific metadata.
    pub metadata: Value,
}

/// A single turn message for session-end extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnMessage {
    /// Role: `"user"`, `"assistant"`, or `"system"`.
    pub role: String,
    /// Message content.
    pub content: String,
    /// Optional Unix timestamp.
    pub timestamp: Option<i64>,
}

/// Tool schema in OpenAI function-calling format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// JSON Schema for parameters.
    pub parameters: Value,
}

/// Memory-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// Plugin is not available (missing config, deps, etc.).
    #[error("not available: {0}")]
    NotAvailable(String),

    /// Initialization failed.
    #[error("initialization failed: {0}")]
    InitFailed(String),

    /// Query/retrieval failed.
    #[error("query failed: {0}")]
    QueryFailed(String),

    /// Ingestion/persistence failed.
    #[error("ingestion failed: {0}")]
    IngestionFailed(String),

    /// Tool dispatch error.
    #[error("tool error: {0}")]
    ToolError(String),

    /// SQLite error.
    #[error("database error: {0}")]
    Database(#[from] common_core::error::SqliteError),

    /// JSON serialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl From<rusqlite::Error> for MemoryError {
    fn from(e: rusqlite::Error) -> Self {
        MemoryError::Database(common_core::error::SqliteError(e))
    }
}
