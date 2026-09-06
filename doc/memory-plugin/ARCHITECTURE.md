# Memory Plugin Architecture — Clean-Room Port from Hermes-Agent

**Status:** Design Specification  
**Author:** opencode (automated architecture synthesis)  
**Date:** 2026-06-14  
**Supersedes:** None (greenfield)  

---

## 1. Executive Summary

This document specifies a clean-room port of the `hermes-agent` memory plugin system into the Rust monorepo, engineered around `fluent-concurrency` (async runtime) and `fluent-wvr` (polymorphic control plane) architectural primitives.

The result is three pluggable memory backends — `holographic`, `hindsight`, `honcho` — that present a uniform `MemoryPlugin` interface to the guidance orchestrator. The orchestrator never branches on implementation type. It iterates over `Arc<dyn MemoryPlugin>` handles and calls trait methods. The compiler enforces the interface at every implementation site.

### Design Mapping (Hermes → Rust)

| Hermes Pattern | Rust Primitive | Rationale |
|---|---|---|
| `MemoryProvider` ABC | `MemoryPlugin: Component + MemoryOps` | Extends the existing `Component` supertrait with memory-specific lifecycle |
| `threading.Lock` / daemon threads | `tokio::sync::Mutex` + `SupervisedBatch` + `Scope` | Structured concurrency replaces ad-hoc threading |
| Plugin discovery (filesystem scan) | Compile-time registration + `PluginDiscovery` trait | Deterministic; no runtime heuristic scanning |
| Single-provider constraint | `MemoryPluginRegistry` with `active` slot | Configurable default, multiple plugins coexist |
| `prefetch` / `sync_turn` hooks | `MemoryOps` trait methods | Direct 1:1 mapping |
| SQLite + FTS5 (holographic) | `rusqlite` (already a workspace dep) | Same embedded engine, Rust-native |
| HRR algebra (numpy) | Pure Rust `HrrAlgebra` | No numpy dependency; `f64` SIMD-free scalar math |
| Knowledge graph (hindsight) | `HindsightGraph` with typed entities | Rust enum dispatch, not stringly-typed |
| Cross-session reasoning (honcho) | `HonchoAnalyzer` with `SupervisedBatch`-managed sessions | Async-native, credit-backpressured ingestion |

---

## 2. Trait Definitions

### 2.1 Core Memory Traits

All memory-specific behavior is captured in two traits: `MemoryOps` (domain operations) and `MemoryPlugin` (the unified component boundary).

```rust
// src/memory/src/traits.rs

use crate::types::*;
use fluent_wvr::{Component, FieldAccess, Describable, WorkUnit, WorkContext, WorkOutput, WorkError};
use internment::ArcIntern;
use serde_json::Value;

// ──────────────────────────────────────────────────────────────
// MemoryOps — domain-specific operations that every memory plugin must support.
// This is NOT a Component sub-trait; it is a separate concern that the
// MemoryPlugin supertrait composes in.
// ──────────────────────────────────────────────────────────────

pub trait MemoryOps: Send + Sync {
    /// Short identifier: "holographic", "hindsight", "honcho"
    fn name(&self) -> &'static str;

    /// Health check — no network calls, only config/deps validation.
    fn is_available(&self) -> bool;

    /// One-time initialization. Called once at guidance startup.
    /// Receives session identity and workspace path.
    fn initialize(&mut self, ctx: &MemoryInitContext) -> Result<(), MemoryError>;

    /// Clean shutdown. Release DB connections, join background tasks.
    fn shutdown(&self) -> impl Future<Output = ()> + Send;

    // ── Retrieval ──────────────────────────────────────────────

    /// Pre-fetch context before each LLM call.
    /// Returns formatted text for injection into the system prompt.
    fn prefetch(&self, query: &str, ctx: &MemoryQueryContext) -> impl Future<Output = String> + Send;

    /// Background prefetch for next turn. Fire-and-forget.
    fn queue_prefetch(&self, query: &str, ctx: &MemoryQueryContext);

    /// Structured search returning scored results.
    fn search(&self, req: &MemorySearchRequest) -> impl Future<Output = Result<Vec<MemoryResult>, MemoryError>> + Send;

    // ── Ingestion ──────────────────────────────────────────────

    /// Persist a completed turn. Non-blocking, may enqueue to background writer.
    fn sync_turn(
        &self,
        user_content: &str,
        assistant_content: &str,
        ctx: &MemoryQueryContext,
    ) -> impl Future<Output = ()> + Send;

    /// End-of-session extraction. Called on /reset, timeout, or exit.
    fn on_session_end(&self, messages: &[TurnMessage]) -> impl Future<Output = ()> + Send;

    // ── Tool dispatch ──────────────────────────────────────────

    /// Handle a tool call from the LLM. Returns JSON string.
    fn handle_tool_call(&self, tool_name: &str, args: &Value) -> Result<String, MemoryError>;

    /// Tool schemas in OpenAI function-calling format.
    fn tool_schemas(&self) -> Vec<ToolSchema>;
}

// ──────────────────────────────────────────────────────────────
// MemoryPlugin — the unified component boundary.
// Any type implementing MemoryPlugin is automatically a Component.
// ──────────────────────────────────────────────────────────────

pub trait MemoryPlugin: Component + MemoryOps {}
impl<T: Component + MemoryOps> MemoryPlugin for T {}
```

### 2.2 Supporting Types

```rust
// src/memory/src/types.rs

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

/// Initialization context passed to MemoryOps::initialize
#[derive(Debug, Clone)]
pub struct MemoryInitContext {
    pub session_id: ArcIntern<str>,
    pub workspace_root: PathBuf,
    pub memory_root: PathBuf,       // ~/.guidance/memory/ or similar
    pub platform: &'static str,     // "linux", "macos", "windows"
}

/// Per-query context (session identity, caps, runtime handle)
#[derive(Debug, Clone)]
pub struct MemoryQueryContext {
    pub session_id: ArcIntern<str>,
    pub caps: fluent_concurrency::CapabilitySet,
    pub rt: std::sync::Arc<dyn fluent_wvr::Runtime>,
}

/// Search request envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchRequest {
    pub query: String,
    pub category: Option<String>,
    pub min_trust: f64,
    pub limit: usize,
    pub strategy: SearchStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// FTS5 keyword search (holographic default)
    FtsKeyword,
    /// HRR compositional algebraic probe
    HrrProbe { entities: Vec<String> },
    /// Knowledge graph structural traversal
    GraphTraversal { depth: usize },
    /// Hybrid: FTS + Jaccard + HRR (holographic full pipeline)
    Hybrid,
}

/// Scored retrieval result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    pub content: String,
    pub score: f64,
    pub trust: f64,
    pub source: &'static str,   // plugin name
    pub metadata: Value,
}

/// Turn message for session-end extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnMessage {
    pub role: String,           // "user" | "assistant" | "system"
    pub content: String,
    pub timestamp: Option<i64>,
}

/// Tool schema (OpenAI function-calling format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value,      // JSON Schema object
}

/// Memory-specific errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("not available: {0}")]
    NotAvailable(String),
    #[error("initialization failed: {0}")]
    InitFailed(String),
    #[error("query failed: {0}")]
    QueryFailed(String),
    #[error("ingestion failed: {0}")]
    IngestionFailed(String),
    #[error("tool error: {0}")]
    ToolError(String),
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
```

### 2.3 FieldAccess and Describable for MemoryPlugin

Each concrete plugin implements `FieldAccess` and `Describable` manually (no derive macros for the plugin boundary — the schema is static per plugin type).

```rust
// Example for HolographicMemory:

impl FieldAccess for HolographicMemory {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        match name {
            "db_path" => {
                self.config.db_path = PathBuf::from(value);
                Ok(())
            }
            "hrr_dim" => {
                self.config.hrr_dim = value.parse::<usize>()
                    .map_err(|e| FieldError::Parse(format!("invalid hrr_dim: {e}")))?;
                Ok(())
            }
            "default_trust" => {
                self.config.default_trust = value.parse::<f64>()
                    .map_err(|e| FieldError::Parse(format!("invalid default_trust: {e}")))?;
                Ok(())
            }
            "auto_extract" => {
                self.config.auto_extract = value.parse::<bool>()
                    .map_err(|e| FieldError::Parse(format!("invalid auto_extract: {e}")))?;
                Ok(())
            }
            _ => Err(FieldError::NotFound(name.into())),
        }
    }

    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "db_path" => Ok(self.config.db_path.display().to_string()),
            "hrr_dim" => Ok(self.config.hrr_dim.to_string()),
            "default_trust" => Ok(self.config.default_trust.to_string()),
            "auto_extract" => Ok(self.config.auto_extract.to_string()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }

    fn field_names(&self) -> &'static [&'static str] {
        &["db_path", "hrr_dim", "default_trust", "auto_extract"]
    }
}

impl Describable for HolographicMemory {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "db_path":       { "type": "string",  "description": "SQLite database path" },
                "hrr_dim":       { "type": "integer", "description": "HRR vector dimensions" },
                "default_trust": { "type": "number",  "description": "Default trust score for new facts" },
                "auto_extract":  { "type": "boolean", "description": "Auto-extract facts at session end" }
            },
            "required": ["db_path"]
        })
    }
}
```

---

## 3. Plugin Registry Architecture

### 3.1 Design Principles

1. **Flat, concrete registry.** No `HashMap<Box<dyn Any>>` — the registry stores `Arc<dyn MemoryPlugin>` keyed by `&'static str` plugin name.
2. **No ambient authority.** Every plugin access goes through the registry, which requires a `&MemoryCapability` token.
3. **Single active plugin, multiple registered.** Only one plugin is "active" (receives prefetch/sync calls), but others can be queried explicitly.
4. **Type erasure happens at registration.** Plugins are wrapped in `Arc<dyn MemoryPlugin>` before insertion. The registry never sees concrete types after registration.

### 3.2 Registry Implementation

```rust
// src/memory/src/registry.rs

use crate::traits::*;
use crate::types::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Type-erased memory plugin entry.
struct PluginEntry {
    plugin: Arc<dyn MemoryPlugin>,
    active: bool,
}

/// The central memory plugin registry.
/// Thread-safe via RwLock; all mutations go through async methods.
pub struct MemoryPluginRegistry {
    plugins: BTreeMap<&'static str, PluginEntry>,
    active_name: Option<&'static str>,
}

impl MemoryPluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: BTreeMap::new(),
            active_name: None,
        }
    }

    /// Register a plugin. Type erasure happens here — the concrete type
    /// is wrapped in Arc<dyn MemoryPlugin> and never seen again.
    pub fn register(&mut self, plugin: Arc<dyn MemoryPlugin>) {
        let name = plugin.name();
        self.plugins.insert(name, PluginEntry {
            plugin,
            active: false,
        });
    }

    /// Set the active plugin by name. Only one can be active.
    pub fn set_active(&mut self, name: &'static str) -> Result<(), MemoryError> {
        if !self.plugins.contains_key(name) {
            return Err(MemoryError::InitFailed(
                format!("plugin '{name}' not registered")
            ));
        }
        // Deactivate current
        if let Some(current) = self.active_name {
            if let Some(entry) = self.plugins.get_mut(current) {
                entry.active = false;
            }
        }
        self.plugins.get_mut(name).unwrap().active = true;
        self.active_name = Some(name);
        Ok(())
    }

    /// Get the active plugin.
    pub fn active(&self) -> Option<Arc<dyn MemoryPlugin>> {
        self.active_name
            .and_then(|name| self.plugins.get(name))
            .map(|entry| Arc::clone(&entry.plugin))
    }

    /// Get a plugin by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn MemoryPlugin>> {
        self.plugins.get(name).map(|entry| Arc::clone(&entry.plugin))
    }

    /// List all registered plugins with availability status.
    pub fn list(&self) -> Vec<(&'static str, bool, bool)> {
        self.plugins.iter()
            .map(|(name, entry)| (*name, entry.active, entry.plugin.is_available()))
            .collect()
    }

    /// Initialize all registered plugins.
    pub async fn initialize_all(&self, ctx: &MemoryInitContext) -> Vec<(&'static str, Result<(), MemoryError>)> {
        let mut results = Vec::new();
        for (name, entry) in &self.plugins {
            let result = {
                let mut plugin = Arc::clone(&entry.plugin);
                // MemoryOps::initialize takes &mut self, so we need get_mut on Arc
                // This is safe because initialize is called during startup (single-threaded)
                Arc::get_mut(&mut plugin.clone())
                    .map(|p| p.initialize(ctx))
                    .unwrap_or(Ok(()))
            };
            results.push((*name, result));
        }
        results
    }

    /// Shutdown all plugins.
    pub async fn shutdown_all(&self) {
        for (_, entry) in &self.plugins {
            entry.plugin.shutdown().await;
        }
    }
}

impl Default for MemoryPluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}
```

### 3.3 Capability Token for Plugin Access

```rust
// src/memory/src/capability.rs

use std::sync::Arc;
use crate::registry::MemoryPluginRegistry;

/// Capability token that grants access to the memory plugin system.
/// Must be passed explicitly — no ambient authority.
#[derive(Clone)]
pub struct MemoryCapability {
    registry: Arc<tokio::sync::RwLock<MemoryPluginRegistry>>,
}

impl MemoryCapability {
    pub fn new(registry: Arc<tokio::sync::RwLock<MemoryPluginRegistry>>) -> Self {
        Self { registry }
    }

    /// Get the active memory plugin. Returns None if no plugin is active.
    pub async fn active_plugin(&self) -> Option<Arc<dyn MemoryPlugin>> {
        let reg = self.registry.read().await;
        reg.active()
    }

    /// Get a specific plugin by name.
    pub async fn get_plugin(&self, name: &str) -> Option<Arc<dyn MemoryPlugin>> {
        let reg = self.registry.read().await;
        reg.get(name)
    }

    /// Execute a search against the active plugin.
    pub async fn search(&self, req: &MemorySearchRequest) -> Result<Vec<MemoryResult>, MemoryError> {
        let plugin = self.active_plugin().await
            .ok_or_else(|| MemoryError::NotAvailable("no active memory plugin".into()))?;
        plugin.search(req).await
    }

    /// Execute a prefetch against the active plugin.
    pub async fn prefetch(&self, query: &str, ctx: &MemoryQueryContext) -> String {
        let plugin = match self.active_plugin().await {
            Some(p) => p,
            None => return String::new(),
        };
        plugin.prefetch(query, ctx).await
    }
}
```

---

## 4. Structural Layout

### 4.1 Crate Structure

```
src/memory/                    # New workspace member: guidance-memory
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Module declarations, re-exports
│   ├── traits.rs              # MemoryOps, MemoryPlugin supertrait
│   ├── types.rs               # MemoryInitContext, MemoryResult, MemoryError, etc.
│   ├── registry.rs            # MemoryPluginRegistry, MemoryCapability
│   ├── capability.rs          # MemoryCapability token
│   ├── batch.rs               # MemoryBatch: ingestion SupervisedBatch wrapper
│   ├── plugins/
│   │   ├── mod.rs             # Plugin module declarations
│   │   ├── holographic/
│   │   │   ├── mod.rs         # HolographicMemory: Component + MemoryOps impl
│   │   │   ├── store.rs       # SQLite store with FTS5
│   │   │   ├── hrr.rs         # HRR algebra (phase vectors, bind/unbind/bundle)
│   │   │   └── retrieval.rs   # Hybrid search pipeline
│   │   ├── hindsight/
│   │   │   ├── mod.rs         # HindsightMemory: Component + MemoryOps impl
│   │   │   ├── graph.rs       # Knowledge graph data structures
│   │   │   ├── entity.rs      # Entity resolution and deduplication
│   │   │   └── retrieval.rs   # Structural traversal retrieval
│   │   └── honcho/
│   │       ├── mod.rs         # HonchoMemory: Component + MemoryOps impl
│   │       ├── analyzer.rs    # Cross-session reasoning compiler
│   │       ├── session.rs     # Session tracking and summary generation
│   │       └── retrieval.rs   # Behavioral memory retrieval
```

### 4.2 Cargo.toml

```toml
[package]
name = "guidance-memory"
version = "0.1.0"
edition = "2021"

[dependencies]
# Workspace deps
fluent-wvr   = { path = "../fluent-wvr" }
fluent-concurrency = { path = "../fluent-concurrency" }
internment   = { workspace = true }
serde        = { workspace = true }
serde_json   = { workspace = true }
tokio        = { workspace = true }
thiserror    = { workspace = true }
tracing      = { workspace = true }
rusqlite     = { workspace = true, features = ["bundled", "backup"] }

# Internal
guidance-types = { path = "../types" }
```

---

## 5. Thread Safety Architecture

### 5.1 Send + Sync Enforcement

Every memory plugin struct must be `Send + Sync` to satisfy the `Component` supertrait bounds. This is enforced at compile time:

```rust
// Compile-time assertion (in each plugin's mod.rs or tests)
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_plugin() {
        assert_send_sync::<HolographicMemory>();
        assert_send_sync::<HindsightMemory>();
        assert_send_sync::<HonchoMemory>();
    }
};
```

### 5.2 Shared State Patterns

| State Pattern | Implementation | Contention Strategy |
|---|---|---|
| SQLite connection | `tokio::sync::Mutex<rusqlite::Connection>` | Single-writer model; reads serialized through mutex but SQLite WAL allows concurrent reads |
| HRR vector cache | `std::sync::RwLock<HashMap<FactId, Vec<f64>>>` | Read-heavy workload; RwLock allows concurrent reads |
| Knowledge graph | `tokio::sync::RwLock<GraphData>` | RwLock for concurrent traversal; exclusive lock only for structural mutations |
| Session summaries | `tokio::sync::Mutex<Vec<SessionSummary>>` | Write-once per session; contention minimal |
| Background ingest queue | `tokio::sync::mpsc::Sender<IngestJob>` | Bounded channel with CreditFlow backpressure |
| Plugin registry | `tokio::sync::RwLock<MemoryPluginRegistry>` | Read-heavy (active plugin lookup); write only at startup |

### 5.3 Credit-Based Backpressure for Ingestion

The ingestion pipeline uses `CreditFlow` from `fluent-concurrency` to prevent unbounded memory accumulation during deep repo syncs:

```rust
// src/memory/src/batch.rs

use fluent_concurrency::{SupervisedBatch, SupervisedBatchConfig, Scope, CreditFlow};
use crate::traits::*;

/// Memory-specific ingestion batch.
/// Wraps the memory pipeline semantics (over `Scope`) with pipeline semantics.
pub struct MemoryBatch {
    batch: SupervisedBatch,
    credit: CreditFlow,
}

impl MemoryBatch {
    /// Create a new memory ingestion batch with bounded backpressure.
    ///
    /// - `max_concurrent`: max simultaneous ingest tasks (prevents SQLite thrashing)
    /// - `credit_limit`: max queued items before producer blocks (prevents heap growth)
    pub fn new(
        rt: std::sync::Arc<dyn fluent_wvr::Runtime>,
        caps: fluent_concurrency::CapabilitySet,
        max_concurrent: usize,
        credit_limit: usize,
    ) -> Self {
        let config = SupervisedBatchConfig {
            max_concurrent_tasks: max_concurrent,
            ..Default::default()
        };
        let (credit, _receiver) = CreditFlow::new(credit_limit);
        Self {
            batch: SupervisedBatch::new(rt, caps, config),
            credit,
        }
    }

    /// Enqueue an ingestion job with backpressure.
    /// Blocks (async) if credit is exhausted.
    pub async fn ingest(&self, job: IngestJob) -> Result<(), MemoryError> {
        self.credit.acquire().await
            .map_err(|_| MemoryError::IngestionFailed("batch shutting down".into()))?;
        // Spawn within the batch's scope — fault containment applies
        self.batch.spawn(Box::new(IngestUnit { job }))
            .map_err(|e| MemoryError::IngestionFailed(e.to_string()))
    }

    /// Close the batch, waiting for all in-flight ingestion to complete.
    pub async fn close(self) {
        self.batch.close().await;
    }
}

/// Concrete ingestion work unit (flat enum dispatch, no dyn Trait in hot path).
pub enum IngestJob {
    SyncTurn { user: String, assistant: String, session: String },
    SessionEnd { messages: Vec<crate::types::TurnMessage> },
    AutoExtract { content: String, category: String },
}

struct IngestUnit {
    job: IngestJob,
}

impl fluent_wvr::WorkUnit for IngestUnit {
    fn name(&self) -> &str {
        match &self.job {
            IngestJob::SyncTurn { .. } => "memory.sync_turn",
            IngestJob::SessionEnd { .. } => "memory.session_end",
            IngestJob::AutoExtract { .. } => "memory.auto_extract",
        }
    }
    fn depends(&self) -> &[ArcIntern<fluent_wvr::ArcIntern<str>>] { &[] }
    fn provides(&self) -> &[ArcIntern<fluent_wvr::ArcIntern<str>>] { &[] }
    fn execute(&self, _ctx: &fluent_wvr::WorkContext) -> Result<fluent_wvr::WorkOutput, fluent_wvr::WorkError> {
        // Ingestion is async — this unit is a bridge that the SupervisedBatch polls
        Ok(fluent_wvr::WorkOutput::ok("ingestion dispatched"))
    }
}
```

### 5.4 Fault Containment

When an ingestion task panics inside the `SupervisedBatch`:
1. The `SupervisedBatch` catches the `JoinError` (panic is caught, not propagated)
2. A `BatchEvent::TaskFailed { name, error }` is emitted
3. Dependent tasks (if any) are cancelled via structural cancellation tokens
4. Independent tasks continue unaffected
5. **No automatic restart** — restart is a deliberate operator action per `fluent-concurrency` philosophy

---

## 6. Plugin Implementations

### 6.1 Holographic (Local SQLite + HRR)

**Responsibility:** Local embedded state with FTS5 search, HRR compositional algebra, trust scoring, entity resolution.

**Hermes → Rust mapping:**

| Hermes | Rust |
|---|---|
| `MemoryStore` (Python, `threading.RLock`) | `HolographicStore` (`tokio::sync::Mutex<Connection>`) |
| `FactRetriever` (Python) | `HolographicRetriever` (Rust, same pipeline) |
| `holographic.py` (numpy phase vectors) | `hrr.rs` (pure Rust `Vec<f64>`, no numpy) |
| FTS5 triggers (Python `executescript`) | Same FTS5 schema, compiled into `store.rs` |
| Entity extraction (regex) | `regex` crate, same patterns |

**HRR Algebra in Pure Rust:**

```rust
// src/memory/src/plugins/holographic/hrr.rs

/// Phase vector HRR algebra. Pure Rust, no numpy dependency.
/// Uses f64 phase vectors with modular arithmetic.

pub const TWO_PI: f64 = std::f64::consts::TAU;

/// Deterministic phase vector from SHA-256 counter blocks.
/// Reproducible across platforms — same input always yields same vector.
pub fn encode_atom(word: &str, dim: usize) -> Vec<f64> {
    use sha2::{Sha256, Digest};

    let values_per_block = 16; // 32 bytes / 2 bytes per u16
    let blocks_needed = (dim + values_per_block - 1) / values_per_block;

    let mut uint16_values: Vec<u16> = Vec::with_capacity(dim);
    for i in 0..blocks_needed {
        let mut hasher = Sha256::new();
        hasher.update(format!("{word}:{i}"));
        let digest = hasher.finalize();
        for chunk in digest.chunks_exact(2) {
            uint16_values.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }

    uint16_values[..dim].iter()
        .map(|&v| (v as f64) * (TWO_PI / 65536.0))
        .collect()
}

/// Circular convolution: element-wise phase addition mod 2π.
pub fn bind(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai + bi) % TWO_PI).collect()
}

/// Circular correlation: element-wise phase subtraction mod 2π.
pub fn unbind(memory: &[f64], key: &[f64]) -> Vec<f64> {
    memory.iter().zip(key.iter()).map(|(&mi, &ki)| (mi - ki) % TWO_PI).collect()
}

/// Superposition via circular mean of complex exponentials.
pub fn bundle(vectors: &[Vec<f64>]) -> Vec<f64> {
    let dim = vectors[0].len();
    let n = vectors.len() as f64;
    (0..dim).map(|j| {
        let sum_re: f64 = vectors.iter().map(|v| v[j].cos()).sum();
        let sum_im: f64 = vectors.iter().map(|v| v[j].sin()).sum();
        sum_im.atan2(sum_re).rem_euclid(TWO_PI)
    }).collect()
}

/// Phase cosine similarity. Range [-1, 1].
pub fn similarity(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let sum: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).cos()).sum();
    sum / n
}

/// Encode a bag-of-words text into a single phase vector.
pub fn encode_text(text: &str, dim: usize) -> Vec<f64> {
    let tokens: Vec<&str> = text.split_whitespace()
        .map(|t| t.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.is_empty() {
        return encode_atom("__hrr_empty__", dim);
    }

    let atoms: Vec<Vec<f64>> = tokens.iter()
        .map(|t| encode_atom(&t.to_lowercase(), dim))
        .collect();
    bundle(&atoms)
}

/// SNR estimate: sqrt(dim / n_items).
pub fn snr_estimate(dim: usize, n_items: usize) -> f64 {
    if n_items == 0 { return f64::INFINITY; }
    (dim as f64 / n_items as f64).sqrt()
}
```

**SQLite Store Schema** (identical to Hermes, compiled into Rust):

```rust
// Embedded in store.rs as a const &str
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
";
```

### 6.2 Hindsight (Structured Knowledge Graph)

**Responsibility:** Multi-strategy retrieval with strict structural entity resolution. Code symbols (functions, structs, modules) map to graph entities. Tracks symbol location/hash shifts during codebase iteration.

**Hermes → Rust mapping:**

| Hermes | Rust |
|---|---|
| Hindsight cloud API | `HindsightGraph` — local in-memory graph with `serde` persistence |
| Server-side entity resolution | `EntityResolver` — Rust-native deduplication with hash + name |
| `observation` recall type | `GraphEntity` enum with typed variants |
| Single-writer thread model | `tokio::sync::mpsc` channel to a single `Scope`-managed writer task |

**Graph Entity Model:**

```rust
// src/memory/src/plugins/hindsight/graph.rs

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Typed entity in the knowledge graph.
/// Flat enum dispatch — no dyn Trait in the hot path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEntity {
    Function {
        name: String,
        module_path: String,
        file_path: String,
        line_start: u32,
        line_end: u32,
        content_hash: String,     // SHA-256 of function body
        visibility: Visibility,
    },
    Struct {
        name: String,
        module_path: String,
        file_path: String,
        fields: Vec<String>,
        content_hash: String,
    },
    Module {
        name: String,
        path: String,
        file_path: String,
        child_symbols: Vec<String>,  // names of contained entities
    },
    Type {
        name: String,
        definition: String,
        file_path: String,
    },
    Constant {
        name: String,
        value_summary: String,
        file_path: String,
    },
    /// Observed fact (cross-session memory)
    Observation {
        content: String,
        category: String,
        trust: f64,
        evidence: Vec<String>,     // entity names that support this observation
        created_at: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility { Public, Crate, Private }

/// A directed edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,      // entity name
    pub to: String,        // entity name
    pub relation: Relation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Relation {
    Contains,      // module -> function
    Calls,         // function -> function
    Uses,          // function -> type/constant
    DependsOn,     // module -> module
    Supports,      // entity -> observation (evidence link)
    Contradicts,   // entity -> observation (conflicting evidence)
}

/// The full knowledge graph, persisted as JSON.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeGraph {
    pub entities: BTreeMap<String, GraphEntity>,
    pub edges: Vec<GraphEdge>,
    pub version: u64,          // incremented on each mutation
    pub content_index: BTreeMap<String, String>,  // content_hash -> entity_name
}
```

### 6.3 Honcho (Cross-Session Reasoning Analyzer)

**Responsibility:** Behavioral memory — compiles cross-session summaries, tracks reasoning paths, maintains persistent conclusions about developer/orchestrator iterations.

**Hermes → Rust mapping:**

| Hermes | Rust |
|---|---|
| Honcho cloud API (`honcho` SDK) | `HonchoAnalyzer` — local-only, no cloud dependency |
| Session management (Python `session.py`) | `SessionStore` with `tokio::sync::Mutex<Vec<SessionSummary>>` |
| Background flush threads | `SupervisedBatch`-managed flush task with `CreditFlow` backpressure |
| Peer card deduplication | `ConclusionDeduplicator` — hash-based dedup of reasoning conclusions |
| Cron guard | `tokio::time::interval` with `Interval`-based gating |

**Session Summary Model:**

```rust
// src/memory/src/plugins/honcho/session.rs

use serde::{Deserialize, Serialize};

/// A compiled summary of one interaction session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub started_at: i64,
    pub ended_at: Option<i64>,
    pub turn_count: usize,
    pub topics: Vec<String>,              // extracted topic labels
    pub decisions: Vec<Decision>,         // conclusions reached
    pub reasoning_paths: Vec<ReasoningPath>,  // how the developer reasoned
    pub tool_usage: Vec<ToolUsageRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub statement: String,
    pub confidence: f64,
    pub evidence_sessions: Vec<String>,   // session IDs that support this
    pub contradicted_by: Vec<String>,     // session IDs with conflicting evidence
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPath {
    pub description: String,
    pub steps: Vec<String>,
    pub outcome: String,
    pub effectiveness: Option<f64>,       // feedback signal
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsageRecord {
    pub tool_name: String,
    pub call_count: usize,
    pub success_rate: f64,
}
```

---

## 7. Hermes Behavior Mapping Table

Complete mapping of every `MemoryProvider` ABC method to the Rust `MemoryOps` trait:

| Hermes Method | Rust Method | Notes |
|---|---|---|
| `name` (property) | `fn name(&self) -> &'static str` | Direct 1:1 |
| `is_available()` | `fn is_available(&self) -> bool` | Direct 1:1 |
| `initialize(session_id, **kwargs)` | `fn initialize(&mut self, ctx: &MemoryInitContext)` | Struct replaces kwargs |
| `get_tool_schemas()` | `fn tool_schemas(&self) -> Vec<ToolSchema>` | Direct 1:1 |
| `system_prompt_block()` | *(removed)* | Orchestration concern, not plugin |
| `prefetch(query, session_id)` | `fn prefetch(&self, query, ctx) -> String` | Direct 1:1 |
| `queue_prefetch(query, session_id)` | `fn queue_prefetch(&self, query, ctx)` | Direct 1:1 |
| `sync_turn(user, assistant, session_id)` | `fn sync_turn(&self, user, assistant, ctx)` | Direct 1:1 |
| `handle_tool_call(tool_name, args)` | `fn handle_tool_call(&self, tool_name, args) -> String` | Direct 1:1 |
| `shutdown()` | `fn shutdown(&self) -> impl Future<Output = ()>` | Now async |
| `on_session_end(messages)` | `fn on_session_end(&self, messages) -> impl Future<Output = ()>` | Now async |
| `on_turn_start(turn, msg)` | *(removed)* | Orchestration concern |
| `on_session_switch(...)` | *(removed)* | Orchestration concern |
| `on_pre_compress(messages)` | *(removed)* | Orchestration concern |
| `on_delegation(task, result)` | *(removed)* | Orchestration concern |
| `on_memory_write(action, target, content)` | *(removed)* | Orchestration concern |
| `get_config_schema()` | *(replaced by `FieldAccess` + `Describable`)* | Trait-based reflection |
| `save_config(values, hermes_home)` | *(replaced by `FieldAccess::set_field`)* | Unified field access |

**Key design decision:** Hermes exposes 15+ lifecycle hooks because Python ABCs encourage "override what you need." In our architecture, the orchestrator owns lifecycle decisions. The plugin only implements `MemoryOps` (8 methods) and `Component` (3 traits). Orchestration hooks like `on_turn_start`, `on_session_switch`, `on_pre_compress`, and `on_delegation` are handled by the guidance query engine, not the memory plugin.

---

## 8. Anti-Patterns Enforced

| Anti-Pattern | Enforcement |
|---|---|
| No ambient `tokio::spawn` | Every background task is spawned inside a `SupervisedBatch` or `Scope` |
| No raw pointer vtables | All type erasure via `Arc<dyn MemoryPlugin>` (standard safe Rust) |
| No `dyn Trait` in per-item loops | HRR algebra uses concrete `Vec<f64>`; graph traversal uses `BTreeMap` keys |
| No `#[async_trait]` proc macros | `MemoryOps` uses `impl Future<Output = ...> + Send` (RPITIT) |
| No `unsafe` code | `#![forbid(unsafe_code)]` at crate root |
| No `serde` in hot loops | `FieldAccess` + `Describable` are `&self` instance methods, not serde deserialization |
| No type erasure after push | Plugins are wrapped in `Arc<dyn MemoryPlugin>` at registration, before insertion |
| No ambient I/O | SQLite access requires `&MemoryCapability` token; no `rusqlite::Connection::open` outside of `initialize` |

---

## 9. Integration with Guidance

The memory plugin system integrates with the guidance query engine at two points:

1. **Prefetch injection:** Before each LLM call, the guidance query engine calls `capability.prefetch(query, ctx)` and injects the result into the system prompt.

2. **Tool dispatch:** When the LLM invokes a memory tool (e.g., `fact_store`, `knowledge_search`), the guidance tool router calls `capability.active_plugin().handle_tool_call(name, args)`.

The `MemoryCapability` token is passed into the guidance `WorkContext` as a capability, ensuring no ambient access:

```rust
// In guidance query_engine.rs (conceptual)
pub struct GuidanceWorkContext {
    pub library: Arc<Library>,
    pub embedder: Arc<dyn EmbeddingProvider>,
    pub memory: MemoryCapability,      // ← explicit capability
    pub config: WorkConfig,
}
```

---

## 10. Implementation Roadmap

| Phase | Deliverable | Est. Effort |
|---|---|---|
| 1 | `guidance-memory` crate skeleton: `traits.rs`, `types.rs`, `registry.rs`, `capability.rs` | 1 day |
| 2 | `HolographicMemory`: store.rs, hrr.rs, retrieval.rs, full `MemoryOps` impl | 3 days |
| 3 | `HindsightMemory`: graph.rs, entity.rs, retrieval.rs | 3 days |
| 4 | `HonchoMemory`: session.rs, analyzer.rs, retrieval.rs | 2 days |
| 5 | `MemoryBatch` integration with `fluent-concurrency` SupervisedBatch | 1 day |
| 6 | Guidance integration: prefetch injection, tool dispatch | 1 day |
| 7 | Tests: unit (HRR algebra, entity resolution), integration (full pipeline), property-based (dedup) | 2 days |

**Total:** ~13 days for a production-ready memory tier.
