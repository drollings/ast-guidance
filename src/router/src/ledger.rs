//! Full-detail content ledger with LOD compaction (decision D6).
//!
//! `ContentNodeLedger` is now a **thin facade** over the shared
//! `NodeStore` (M4): the durable, per-session store of `ContentNode`s (the
//! canonical `fluent_types::ContentNode`) lives in `crate::node_store`, where
//! nodes are shared behind `Arc<RwLock<ContentNode>>` with interned
//! session/role index keys and a durable `content_json` column. The facade
//! keeps the server-facing surface (`record_request`, `record_result`,
//! `record_content_node`, `get_session_nodes`, `get_session_entries`,
//! `ensure_lod`, `compact_session`, `collapse_node`) signature-identical and
//! delegates to the store.
//!
//! The facade also owns the **write-path guard** (M1, decision D1): every
//! write delegate scrubs its text through `crate::ledger_guard::scrub_for_ledger`
//! before reaching `NodeStore`, so the durable ledger can never cache text
//! matching the builtin filter engine (on by default, no config flag). The
//! scrub is irreversible. `NodeStore` itself stays policy-free; the only
//! documented bypass is the `KnowledgeCapability` impl directly on `NodeStore`
//! (`crate::knowledge`), which is a trait-object boundary that cannot route
//! through the facade — production server writes all flow through here.
//!
//! The LOD lifecycle is owned by the store:
//!
//! - **LOD0** (full text) and **LOD5** (label) are guaranteed eager at node
//!   creation.
//! - **LOD1–LOD4** are derived lazily, **always from LOD0 only** (never
//!   chained from a lower tier — VISION), via the `Summarizer` WorkUnit, and
//!   cached on the shared node once derived (at most once across all holders).
//! - Compaction (`CompactionStrategy`/`RecencyCompaction`, formerly the
//!   standalone `compaction.rs`) demotes older nodes to higher LOD levels to
//!   stay within a context budget; the demoted text is filled in lazily.
//!
//! This module keeps the flat `LedgerEntry` audit projection, the `LedgerError`
//! taxonomy, the schema migrations, and the compaction policy types. The
//! schema stores both the flat queryable projection (used by the server's
//! best-effort `record_request`/`record_result` logging) and a `content_json`
//! column holding the full serialized `ContentNode` (single source of truth
//! for LOD/role metadata).

use std::path::PathBuf;
use std::sync::Arc;

use fluent_db::error::DbError;
use fluent_db::migrate::{ensure_column, Migration};
use fluent_types::{ContentNode, NodeId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ledger_guard::scrub_for_ledger;
use crate::node_store::NodeStore;
use crate::summarization::Summarizer;
use crate::views::LedgerView;
use crate::views::{Lod, ParallelLedger};

/// Node-construction helper moved to `NodeStore` (M4). Re-exported here so the
/// facade's tests keep compiling unchanged.
#[cfg(test)]
pub(crate) use crate::node_store::new_node;

/// LOD level of the full text. Always eager at node creation.
pub const LOD0_FULL_TEXT: u8 = 0;
/// LOD level of the label (short descriptor). Always eager at node creation.
pub const LOD5_LABEL: u8 = 5;
/// LOD levels derived lazily from LOD0 via the `Summarizer`.
pub const LAZY_LOD_RANGE: std::ops::RangeInclusive<u8> = 1..=4;

#[derive(Error, Debug)]
pub enum LedgerError {
    #[error("database error: {0}")]
    Db(String),
    #[error("node not found: {0:?}")]
    NotFound(NodeId),
    #[error("summarization error: {0}")]
    Summary(String),
    #[error("no summarizer configured for lazy LOD derivation")]
    NoSummarizer,
    #[error("invalid LOD level: {0} (lazy levels are {LAZY_LOD_RANGE:?})")]
    InvalidLod(u8),
}

/// Full-detail ledger entry recorded before any filter runs (§5).
///
/// Kept as the flat read projection for audit/boundary consumers; the
/// canonical in-memory form is `fluent_types::ContentNode` (see
/// `ContentNodeLedger::get_session_nodes`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub node_id: NodeId,
    pub session_id: String,
    pub request_id: String,
    pub role: String,
    pub content: String,
    pub turn_index: u64,
    pub accepted: bool,
    pub acceptance_score: Option<f64>,
    pub active_lod: u8,
    pub parent_id: Option<NodeId>,
    pub step_id: Option<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
    pub created_at: u64,
}

/// The durable, per-session store of `ContentNode`s — a thin facade over the
/// shared `Arc<NodeStore>` (M4). Every method delegates; the exact public
/// surface is preserved so the server (`ServerDeps.ledger`) is untouched.
pub struct ContentNodeLedger {
    store: Arc<NodeStore>,
}

impl ContentNodeLedger {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, LedgerError> {
        Ok(Self {
            store: Arc::new(NodeStore::open(path)?),
        })
    }

    /// Open an in-memory ledger (tests / ephemeral stores).
    pub fn open_in_memory() -> Result<Self, LedgerError> {
        Ok(Self {
            store: Arc::new(NodeStore::open_in_memory()?),
        })
    }

    /// Attach a `Summarizer` so LOD1–LOD4 can be derived lazily from LOD0.
    /// Without one, `ensure_lod` returns `LedgerError::NoSummarizer`.
    #[must_use]
    pub fn with_summarizer(self, summarizer: Summarizer) -> Self {
        self.store.set_summarizer(summarizer);
        self
    }

    /// The shared store — the new shared/refcounted/interned read path.
    pub fn node_store(&self) -> &Arc<NodeStore> {
        &self.store
    }

    /// Record a user request as a new node. LOD0 (full text) and LOD5 (label)
    /// are written eagerly; LOD1–LOD4 stay empty until derived lazily.
    ///
    /// The write-path guard (M1) scrubs `content` against the builtin filter
    /// engine before persisting; flagged writes emit an audit record.
    pub fn record_request(
        &self,
        session_id: &str,
        request_id: &str,
        content: &str,
    ) -> Result<NodeId, LedgerError> {
        let s = scrub_for_ledger(content);
        if s.flagged {
            emit_write_audit(s.pattern.as_deref());
        }
        self.store.record_request(session_id, request_id, &s.text)
    }

    /// Update the result of a previously recorded request node: acceptance,
    /// score, and final content. Keeps the flat projection and the
    /// `content_json` node in sync (LOD0/LOD5 are recomputed eagerly).
    ///
    /// The write-path guard (M1) scrubs `content` before persisting.
    pub fn record_result(
        &self,
        node_id: NodeId,
        accepted: bool,
        score: Option<f64>,
        content: &str,
    ) -> Result<(), LedgerError> {
        let s = scrub_for_ledger(content);
        if s.flagged {
            emit_write_audit(s.pattern.as_deref());
        }
        self.store.record_result(node_id, accepted, score, &s.text)
    }

    /// Persist an arbitrary origin-typed `ContentNode`. LOD0/LOD5 are
    /// guaranteed present (derived from the node's text when missing).
    ///
    /// The write-path guard (M1) scrubs the node's LOD0 text and clears LOD5
    /// so the store re-derives the label from the scrubbed text.
    pub fn record_content_node(&self, node: &ContentNode) -> Result<NodeId, LedgerError> {
        let mut node = node.clone();
        let s = scrub_for_ledger(node.lod.first().map_or("", String::as_str));
        if s.flagged {
            while node.lod.len() < LOD5_LABEL as usize + 1 {
                node.lod.push(String::new());
            }
            node.lod[LOD0_FULL_TEXT as usize] = s.text;
            node.lod[LOD5_LABEL as usize].clear();
            emit_write_audit(s.pattern.as_deref());
        }
        self.store.record_content_node(&node)
    }

    /// Collapse a node: replace LOD0's content with `summary` and mark it
    /// `active_lod = lod`. Used by compaction.
    pub fn collapse_node(
        &self,
        node_id: NodeId,
        summary: &str,
        lod: u8,
    ) -> Result<(), LedgerError> {
        self.store.collapse_node(node_id, summary, lod)
    }

    /// Derive (or return the cached) LOD level for a node, **from LOD0 only**
    /// via the `Summarizer` — never chained from a lower tier.
    ///
    /// Only LOD1–LOD4 are lazy; LOD0/LOD5 are eager at creation. The derived
    /// tier is cached on the shared node, so a second request from any holder
    /// hits the cache, not the LLM.
    pub fn ensure_lod(&self, node_id: NodeId, level: u8) -> Result<ContentNode, LedgerError> {
        self.store.ensure_lod(node_id, level)
    }

    /// Compact a session: demote older nodes to higher LOD levels via the
    /// `RecencyCompaction` policy. Demotion sets `active_lod`; the actual LOD
    /// text is filled in lazily from LOD0 (never chained). Returns the IDs of
    /// demoted nodes.
    pub fn compact_session(
        &self,
        session_id: &str,
        max_nodes: usize,
    ) -> Result<Vec<NodeId>, LedgerError> {
        self.store.compact_session(session_id, max_nodes)
    }

    /// Fetch a node by ID (canonical `ContentNode`, single source of truth in
    /// the shared store).
    ///
    /// Returns `None` if the node is absent. The pre-LOD flat-column hydration
    /// fallback was retired (M5.7): the canonical read path is single-format.
    pub fn get_node(&self, node_id: NodeId) -> Option<ContentNode> {
        self.store.snapshot(node_id)
    }

    /// All nodes for a session (canonical `ContentNode`s), most recent first.
    pub fn get_session_nodes(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<ContentNode>, LedgerError> {
        self.store.get_session_nodes(session_id, limit)
    }

    pub fn get_session_entries(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<LedgerEntry>, LedgerError> {
        self.store.get_session_entries(session_id, limit)
    }

    /// Render a session through a `ParallelLedger` at `default_lod` (M2). The
    /// view's single `render()` exit. A compacted (collapsed) node renders its
    /// collapsed LOD0 — compaction mutates LOD0, so the view's fidelity policy
    /// never "defeats" compaction.
    pub fn render_session(&self, session_id: &str, default_lod: Lod) -> String {
        let view = ParallelLedger::for_session(Arc::clone(&self.store), session_id)
            .with_default_lod(default_lod);
        view.render()
    }

    /// Panic while holding the durable connection mutex (test-only): exercises
    /// the poison-recovery path in `SqliteStore`'s `common_core::sync::lock`.
    #[cfg(test)]
    fn poison_conn(&self) {
        self.store.poison_conn();
    }
}

/// Emit the M1 write-path audit record: a builtin-filter match flagged on a
/// durable ledger write.
fn emit_write_audit(pattern: Option<&str>) {
    crate::audit::emit(
        "filter",
        serde_json::json!({
            "write_path": true,
            "pattern": pattern,
            "node_scrubbed": true,
        }),
    );
}

/// Versioned ledger schema lifecycle (fluent_db::migrate, M6.2).
///
/// One migration per schema step: the base table + session index, then the
/// three LOD-era columns (`label`, `lod`, `content_json`) that pre-LOD
/// databases lack. Each column migration is a no-op when the column already
/// exists, so a fresh database and an upgraded one converge.
///
/// Shared with `NodeStore` (4B), which owns the durable backing now — the
/// schema stays here as the single source of truth.
pub(crate) fn ledger_migrations() -> [&'static dyn Migration; 4] {
    [
        &LedgerBaseSchema,
        &LedgerLabelColumn,
        &LedgerLodColumn,
        &LedgerContentJsonColumn,
    ]
}

struct LedgerBaseSchema;

impl Migration for LedgerBaseSchema {
    fn version(&self) -> u32 {
        1
    }
    fn name(&self) -> &str {
        "ledger-base-schema"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS ledger (
                node_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                turn_index INTEGER NOT NULL DEFAULT 0,
                accepted INTEGER NOT NULL DEFAULT 1,
                acceptance_score REAL,
                active_lod INTEGER NOT NULL DEFAULT 0,
                parent_id INTEGER,
                step_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                label TEXT NOT NULL DEFAULT '',
                lod TEXT NOT NULL DEFAULT '[]',
                content_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_ledger_session ON ledger(session_id, turn_index);",
        )
        .map_err(DbError::from)?;
        Ok(())
    }
}

struct LedgerLabelColumn;

impl Migration for LedgerLabelColumn {
    fn version(&self) -> u32 {
        2
    }
    fn name(&self) -> &str {
        "ledger-label-column"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        ensure_column(tx, "ledger", "label", "label TEXT NOT NULL DEFAULT ''")
    }
}

struct LedgerLodColumn;

impl Migration for LedgerLodColumn {
    fn version(&self) -> u32 {
        3
    }
    fn name(&self) -> &str {
        "ledger-lod-column"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        ensure_column(tx, "ledger", "lod", "lod TEXT NOT NULL DEFAULT '[]'")
    }
}

struct LedgerContentJsonColumn;

impl Migration for LedgerContentJsonColumn {
    fn version(&self) -> u32 {
        4
    }
    fn name(&self) -> &str {
        "ledger-content-json-column"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        ensure_column(
            tx,
            "ledger",
            "content_json",
            "content_json TEXT NOT NULL DEFAULT '{}'",
        )
    }
}

// ── LOD compaction policy ─────────────────────────────────────────────────
//
// Moved here from the deleted standalone `compaction.rs` (D6): compaction is
// ledger responsibility — it demotes older session nodes to lower detail
// levels to stay within context budget. The interface is a trait so smarter
// policies can be plugged in later.

/// Given a session's nodes, return the LOD level each node should be at.
/// Lower LOD = less detail retained. Higher LOD = more compacted.
/// LOD 0 = full detail, LOD N = progressively more compressed.
pub trait CompactionStrategy: Send + Sync {
    /// Returns LOD levels for each node. The returned `Vec` must have
    /// the same length as `nodes`.
    fn select_lod(&self, nodes: &[ContentNode], max_nodes: usize) -> Vec<u8>;
}

/// Simple recency-based compaction: recent nodes keep high detail,
/// older nodes are progressively demoted.
pub struct RecencyCompaction;

impl CompactionStrategy for RecencyCompaction {
    fn select_lod(&self, nodes: &[ContentNode], max_nodes: usize) -> Vec<u8> {
        let n = nodes.len();
        let mut lods = vec![0u8; n];

        // If we're under the max, everything stays at full detail.
        if n <= max_nodes {
            return lods;
        }

        // Recent nodes keep high detail (LOD 0-1), older nodes drop.
        for (i, lod) in lods.iter_mut().enumerate() {
            let position_from_end = n - i;
            if position_from_end <= max_nodes / 4 {
                *lod = 0; // most recent: full detail
            } else if position_from_end <= max_nodes / 2 {
                *lod = 1; // recent: ~800 chars
            } else if position_from_end <= 3 * max_nodes / 4 {
                *lod = 2; // moderate: ~240 chars
            } else {
                *lod = 3; // older: ~80 chars
            }
        }

        lods
    }
}

/// No-op compaction: leaves all nodes at full detail.
/// Useful for testing or when compaction is disabled.
pub struct NoopCompaction;

impl CompactionStrategy for NoopCompaction {
    fn select_lod(&self, nodes: &[ContentNode], _max_nodes: usize) -> Vec<u8> {
        vec![0u8; nodes.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common_core::hash::uuid_v4;
    use common_core::sync::lock;
    use std::io::Write;
    use std::sync::Mutex;
    use tracing_subscriber::layer::SubscriberExt;

    /// A `MakeWriter` that captures formatted log lines for audit assertions
    /// (mirrors the capture helper in `config::builder` tests).
    #[derive(Clone, Default)]
    struct LogCapture(Arc<Mutex<Vec<String>>>);

    impl Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            lock(&self.0).push(String::from_utf8_lossy(buf).into_owned());
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl tracing_subscriber::fmt::MakeWriter<'_> for LogCapture {
        type Writer = Self;
        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
        let capture = LogCapture::default();
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_writer(capture.clone())
                .with_ansi(false)
                .with_target(true),
        );
        let result = tracing::subscriber::with_default(subscriber, f);
        let logs = lock(&capture.0).clone();
        (result, logs)
    }

    fn temp_ledger() -> ContentNodeLedger {
        let dir = std::env::temp_dir().join(format!("coral-router-ledger-{}", uuid_v4()));
        let ledger = ContentNodeLedger::open(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        ledger
    }

    #[test]
    fn record_and_fetch_roundtrip() {
        let ledger = temp_ledger();
        let id = ledger.record_request("sess-1", "req-1", "hello").unwrap();
        ledger.record_result(id, true, Some(1.0), "reply").unwrap();
        let entries = ledger.get_session_entries("sess-1", 10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "reply");
        assert!(entries[0].accepted);
    }

    #[test]
    fn get_node_single_format_no_flat_hydration_fallback() {
        let ledger = temp_ledger();
        // Simulate a migrated pre-LOD row: the flat projection is populated
        // but `content_json` is still the '{}' placeholder. The canonical
        // read (`get_node`) must return `None` — the dual-format hydration
        // fallback is retired (M5.7); only `get_session_entries` (the flat
        // audit view) reads columns directly. The row is inserted after
        // hydration, so the in-memory maps never see it.
        let store = ledger.node_store().durable().unwrap();
        store
            .execute(
                "INSERT INTO ledger (node_id, session_id, request_id, role, content,
                                     turn_index, accepted, active_lod, metadata, created_at,
                                     label, lod, content_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                rusqlite::params![
                    99_i64,
                    "sess-pre-lod",
                    "req-pre-lod",
                    "user",
                    "legacy flat content",
                    0_i64,
                    1_i64,
                    0_i64,
                    "{}",
                    0_i64,
                    "legacy label",
                    "[]",
                    "{}",
                ],
            )
            .unwrap();

        let id = NodeId::from_int(99);
        assert!(
            ledger.get_node(id).is_none(),
            "unparseable content_json -> None"
        );

        let entries = ledger.get_session_entries("sess-pre-lod", 10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "legacy flat content");
    }

    #[test]
    fn poisoned_db_mutex_recovers() {
        let ledger = temp_ledger();
        // Poison the db mutex by panicking while it is held.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ledger.poison_conn();
        }));
        // Subsequent calls must still succeed via the poison-recovery helper.
        let id = ledger
            .record_request("sess-p", "req-p", "after-poison")
            .unwrap();
        ledger
            .record_result(id, false, Some(0.0), "recovered")
            .unwrap();
        let entries = ledger.get_session_entries("sess-p", 10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "recovered");
    }

    #[test]
    fn lod0_and_lod5_eager_at_creation() {
        let ledger = temp_ledger();
        let id = ledger
            .record_request(
                "sess-1",
                "req-1",
                "Hello world. This is a longer user message.",
            )
            .unwrap();
        let node = ledger.get_node(id).unwrap();

        // LOD0 full text + LOD5 label eager.
        assert_eq!(node.lod[0], "Hello world. This is a longer user message.");
        assert_eq!(node.lod[5], "Hello world.");
        // LOD1–4 stay empty until derived lazily.
        assert!(node.lod[1].is_empty());
        assert!(node.lod[4].is_empty());
        assert_eq!(node.active_lod, Some(LOD0_FULL_TEXT));
    }

    #[test]
    fn record_content_node_stores_canonical_type() {
        let ledger = temp_ledger();
        let mut node = new_node(
            NodeId::from_int(7),
            "sess-2",
            "req-2",
            "assistant",
            "An accepted assistant answer.",
            Some(true),
        );
        node.acceptance_score = Some(0.9);
        node.step_id = Some("step-1".into());
        let id = ledger.record_content_node(&node).unwrap();

        let fetched = ledger.get_node(id).unwrap();
        assert_eq!(fetched.role.as_deref(), Some("assistant"));
        assert_eq!(fetched.acceptance_score, Some(0.9));
        assert_eq!(fetched.step_id.as_deref(), Some("step-1"));
        assert_eq!(fetched.lod[5], "An accepted assistant answer.");
        // LOD0/LOD5 guaranteed even if the caller forgot them.
        assert_eq!(fetched.lod[0], "An accepted assistant answer.");
    }

    #[test]
    fn session_nodes_most_recent_first() {
        let ledger = temp_ledger();
        ledger.record_request("sess-3", "r1", "first").unwrap();
        ledger.record_request("sess-3", "r2", "second").unwrap();

        let nodes = ledger.get_session_nodes("sess-3", 10).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].request_id.as_deref(), Some("r2"));
        assert_eq!(nodes[1].request_id.as_deref(), Some("r1"));
    }

    #[test]
    fn ensure_lod_requires_summarizer() {
        let ledger = temp_ledger();
        let id = ledger.record_request("sess-4", "r1", "some text").unwrap();
        assert!(matches!(
            ledger.ensure_lod(id, 2),
            Err(LedgerError::NoSummarizer)
        ));
        assert!(matches!(
            ledger.ensure_lod(id, 0),
            Err(LedgerError::InvalidLod(0))
        ));
    }

    #[test]
    fn ensure_lod_derives_from_lod0_and_caches() {
        use crate::test_stubs::StubChatBackend;
        use std::sync::Arc;

        let client: Arc<dyn fluent_llm::client::ChatBackend> =
            Arc::new(StubChatBackend::always("lazy LOD summary"));
        let summarizer = Summarizer::new(client, 20);
        let ledger = temp_ledger().with_summarizer(summarizer);

        let id = ledger
            .record_request(
                "sess-5",
                "r1",
                "The full text that must be summarized from LOD0 only.",
            )
            .unwrap();

        let node = ledger.ensure_lod(id, 2).unwrap();
        assert_eq!(node.lod[2], "lazy LOD summary");
        // Cached: a second derivation hits the cache, not the LLM.
        let node2 = ledger.ensure_lod(id, 2).unwrap();
        assert_eq!(node2.lod[2], "lazy LOD summary");
        // LOD0 is untouched by derivation (never chained from a lower tier).
        let node3 = ledger.get_node(id).unwrap();
        assert_eq!(
            node3.lod[0],
            "The full text that must be summarized from LOD0 only."
        );
    }

    #[test]
    fn compact_session_demotes_oldest_nodes() {
        let ledger = temp_ledger();
        for i in 0..5 {
            ledger
                .record_request("sess-6", &format!("r{i}"), &format!("message {i}"))
                .unwrap();
        }

        let demoted = ledger.compact_session("sess-6", 4).unwrap();
        assert!(!demoted.is_empty(), "some nodes must be demoted");
        let nodes = ledger.get_session_nodes("sess-6", 10).unwrap();
        // Newest node stays at full detail; oldest is demoted to LOD3.
        let newest = nodes
            .iter()
            .find(|n| n.request_id.as_deref() == Some("r4"))
            .unwrap();
        assert_eq!(newest.active_lod, Some(0));
        let oldest = nodes
            .iter()
            .find(|n| n.request_id.as_deref() == Some("r0"))
            .unwrap();
        assert_eq!(oldest.active_lod, Some(3));
    }

    #[test]
    fn recency_compaction_under_max() {
        let nodes = make_nodes(3);
        let lods = RecencyCompaction.select_lod(&nodes, 10);
        assert_eq!(lods, vec![0, 0, 0]);
    }

    #[test]
    fn recency_compaction_over_max() {
        let nodes = make_nodes(8);
        let lods = RecencyCompaction.select_lod(&nodes, 4);
        assert_eq!(lods[0], 3);
        assert_eq!(lods[1], 3);
        assert_eq!(lods[7], 0);
    }

    #[test]
    fn noop_compaction() {
        let nodes = make_nodes(100);
        let lods = NoopCompaction.select_lod(&nodes, 10);
        assert!(lods.iter().all(|&l| l == 0));
    }

    fn make_nodes(count: usize) -> Vec<ContentNode> {
        (0..count)
            .map(|i| {
                new_node(
                    NodeId::from_int(i as i64),
                    "test",
                    &format!("req-{i}"),
                    "user",
                    &format!("node {i}"),
                    Some(true),
                )
            })
            .collect()
    }

    // ── M1 write-path guard (facade scrub) ────────────────────────────────

    #[test]
    fn record_request_scrubs_email_and_emits_audit() {
        let ledger = temp_ledger();
        let (id, logs) = capture_logs(|| {
            ledger
                .record_request("sess-guard", "r1", "Contact user@example.com now")
                .unwrap()
        });
        let _ = id;
        let joined = logs.join("\n");
        assert!(
            joined.contains("router.audit")
                && joined.contains("write_path")
                && joined.contains("email"),
            "flagged write must emit a write-path audit, logs:\n{joined}"
        );

        let entries = ledger.get_session_entries("sess-guard", 10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "Contact [REDACTED:email] now");
        assert!(
            !entries[0].content.contains("user@example.com"),
            "durable content must be scrubbed"
        );
    }

    #[test]
    fn record_result_scrubs_phone() {
        let ledger = temp_ledger();
        let id = ledger
            .record_request("sess-guard-r", "r1", "What number?")
            .unwrap();
        ledger
            .record_result(id, true, Some(1.0), "Call 555-123-4567 to reach us.")
            .unwrap();

        let node = ledger.get_node(id).unwrap();
        assert_eq!(node.lod[0], "Call [REDACTED:phone] to reach us.");
        let entries = ledger.get_session_entries("sess-guard-r", 10).unwrap();
        assert_eq!(entries[0].content, "Call [REDACTED:phone] to reach us.");
    }

    #[test]
    fn record_content_node_scrubs_api_key_to_reject_marker() {
        let ledger = temp_ledger();
        let mut node = new_node(
            NodeId::from_int(101),
            "sess-guard-c",
            "r1",
            "assistant",
            "the token is api_key = super_secret_value_123",
            Some(true),
        );
        node.acceptance_score = Some(0.9);
        let id = ledger.record_content_node(&node).unwrap();

        let fetched = ledger.get_node(id).unwrap();
        assert_eq!(fetched.lod[0], "[rejected: api_key]");
        assert_eq!(fetched.acceptance_score, Some(0.9));
    }

    #[test]
    fn clean_write_is_not_flagged() {
        let ledger = temp_ledger();
        let (_, logs) = capture_logs(|| {
            ledger
                .record_request("sess-guard-clean", "r1", "plain text, no pii")
                .unwrap()
        });
        let joined = logs.join("\n");
        assert!(
            !joined.contains("write_path"),
            "clean writes must not emit a write-path audit, logs:\n{joined}"
        );
        let entries = ledger.get_session_entries("sess-guard-clean", 10).unwrap();
        assert_eq!(entries[0].content, "plain text, no pii");
    }

    #[test]
    fn render_session_renders_three_nodes_as_three_lines() {
        let ledger = temp_ledger();
        ledger
            .record_request("sess-render", "r1", "first node text")
            .unwrap();
        ledger
            .record_request("sess-render", "r2", "second node text")
            .unwrap();
        ledger
            .record_request("sess-render", "r3", "third node text")
            .unwrap();

        let rendered = ledger.render_session("sess-render", Lod::LOD0);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 3, "3 nodes -> 3 lines, got: {rendered}");
        assert!(lines.contains(&"first node text"));
        assert!(lines.contains(&"second node text"));
        assert!(lines.contains(&"third node text"));
    }

    #[test]
    fn render_session_renders_collapsed_node_lod0() {
        let ledger = temp_ledger();
        let id = ledger
            .record_request("sess-collapse", "r1", "original long content")
            .unwrap();
        ledger
            .collapse_node(id, "collapsed summary", LOD0_FULL_TEXT)
            .unwrap();

        // Compaction mutates LOD0, so a LOD0 view shows the collapsed text —
        // the fidelity policy never "defeats" compaction.
        assert_eq!(
            ledger.render_session("sess-collapse", Lod::LOD0),
            "collapsed summary"
        );
    }
}
