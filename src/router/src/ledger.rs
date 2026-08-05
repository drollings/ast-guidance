//! Full-detail content ledger with LOD compaction (decision D6).
//!
//! `ContentNodeLedger` is the durable, per-session store of `ContentNode`s
//! (the canonical `fluent_types::ContentNode`). It owns the LOD lifecycle:
//!
//! - **LOD0** (full text) and **LOD5** (label) are guaranteed eager at node
//!   creation.
//! - **LOD1–LOD4** are derived lazily, **always from LOD0 only** (never
//!   chained from a lower tier — VISION), via the `Summarizer` WorkUnit, and
//!   cached on the node once derived.
//! - Compaction (`CompactionStrategy`/`RecencyCompaction`, formerly the
//!   standalone `compaction.rs`) demotes older nodes to higher LOD levels to
//!   stay within a context budget; the demoted text is filled in lazily.
//!
//! The schema stores both the flat queryable projection (used by the server's
//! best-effort `record_request`/`record_result` logging) and a `content_json`
//! column holding the full serialized `ContentNode` (single source of truth
//! for LOD/role metadata).

use std::path::PathBuf;
use std::sync::Mutex;

use common_core::sync::lock;
use fluent_db::error::DbError;
use fluent_db::migrate::{ensure_column, migrate, Migration};
use fluent_db::store::SqliteStore;
use fluent_types::{ContentNode, NodeId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::summarization::Summarizer;

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

pub struct ContentNodeLedger {
    store: SqliteStore,
    next_id: Mutex<i64>,
    summarizer: Option<Summarizer>,
}

/// Deterministic, LLM-free LOD5 label (short descriptor), derived eagerly at
/// node creation. Falls back to the role when no content survives truncation.
fn derive_label(role: &str, content: &str) -> String {
    let sentence = common_core::string::first_sentence(content);
    let snippet = if sentence.is_empty() {
        common_core::string::truncate_utf8(content, 64)
    } else {
        common_core::string::truncate_utf8(&sentence, 64)
    };
    if snippet.is_empty() {
        role.to_string()
    } else {
        snippet
    }
}

impl ContentNodeLedger {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, LedgerError> {
        let db_path = path.into();
        let store = SqliteStore::open(&db_path).map_err(|e| LedgerError::Db(e.to_string()))?;

        // Versioned schema lifecycle (fluent_db::migrate): the base schema
        // migration creates the table; the column migrations upgrade
        // pre-LOD databases idempotently.
        let migrations: [&dyn Migration; 4] = [
            &LedgerBaseSchema,
            &LedgerLabelColumn,
            &LedgerLodColumn,
            &LedgerContentJsonColumn,
        ];
        store
            .with_conn(|conn| migrate(conn, &migrations))
            .map_err(|e| LedgerError::Db(e.to_string()))?;

        Ok(Self {
            store,
            next_id: Mutex::new(1),
            summarizer: None,
        })
    }

    /// Attach a `Summarizer` so LOD1–LOD4 can be derived lazily from LOD0.
    /// Without one, `ensure_lod` returns `LedgerError::NoSummarizer`.
    #[must_use]
    pub fn with_summarizer(mut self, summarizer: Summarizer) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// Record a user request as a new node. LOD0 (full text) and LOD5 (label)
    /// are written eagerly; LOD1–LOD4 stay empty until derived lazily.
    pub fn record_request(
        &self,
        session_id: &str,
        request_id: &str,
        content: &str,
    ) -> Result<NodeId, LedgerError> {
        let mut next = lock(&self.next_id);
        let id = NodeId::from_int(*next);
        *next += 1;

        let node = new_node(
            id, session_id, request_id, "user", content, None, // accepted set by record_result
        );
        self.insert_node(&node)?;
        Ok(id)
    }

    /// Update the result of a previously recorded request node: acceptance,
    /// score, and final content. Keeps the flat projection and the
    /// `content_json` node in sync (LOD0/LOD5 are recomputed eagerly).
    pub fn record_result(
        &self,
        node_id: NodeId,
        accepted: bool,
        score: Option<f64>,
        content: &str,
    ) -> Result<(), LedgerError> {
        self.with_node_mut(node_id, |node| {
            node.accepted = Some(accepted);
            node.acceptance_score = score;
            if !content.is_empty() {
                node.lod[LOD0_FULL_TEXT as usize] = content.to_string();
                node.lod[LOD5_LABEL as usize] =
                    derive_label(&node.role.clone().unwrap_or_default(), content);
                node.active_lod = Some(LOD0_FULL_TEXT);
            }
        })?;
        Ok(())
    }

    /// Persist an arbitrary origin-typed `ContentNode`. LOD0/LOD5 are
    /// guaranteed present (derived from the node's text when missing).
    pub fn record_content_node(&self, node: &ContentNode) -> Result<NodeId, LedgerError> {
        let id = node.id.unwrap_or_else(|| {
            let mut next = lock(&self.next_id);
            let id = NodeId::from_int(*next);
            *next += 1;
            id
        });
        let mut node = node.clone();
        node.id = Some(id);
        ensure_lod_eager(&mut node);
        self.insert_node(&node)?;
        Ok(id)
    }

    /// Record a node under a fixed NodeId (internal — avoids a second
    /// next_id bump when the caller already allocated one).
    fn insert_node(&self, node: &ContentNode) -> Result<(), LedgerError> {
        let metadata = node
            .metadata
            .as_ref()
            .unwrap_or(&serde_json::json!({}))
            .to_string();
        let lod_json =
            serde_json::to_string(&node.lod).map_err(|e| LedgerError::Db(e.to_string()))?;
        let content_json =
            serde_json::to_string(node).map_err(|e| LedgerError::Db(e.to_string()))?;
        let content = node.lod.first().map_or("", String::as_str);
        let label = node
            .lod
            .get(LOD5_LABEL as usize)
            .cloned()
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| derive_label(&node.role.clone().unwrap_or_default(), content));

        self.store
            .execute(
                "INSERT INTO ledger (node_id, session_id, request_id, role, content, turn_index,
                                     accepted, acceptance_score, active_lod, parent_id, step_id,
                                     metadata, created_at, label, lod, content_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                rusqlite::params![
                    node.id.map(NodeId::as_int),
                    node.session_id.clone().unwrap_or_default(),
                    node.request_id.clone().unwrap_or_default(),
                    node.role.clone().unwrap_or_default(),
                    content,
                    node.turn_index.unwrap_or(0) as i64,
                    node.accepted.unwrap_or(true),
                    node.acceptance_score,
                    i64::from(node.active_lod.unwrap_or(LOD0_FULL_TEXT)),
                    node.parent_id.map(NodeId::as_int),
                    node.step_id.as_deref(),
                    metadata,
                    node.created_at.unwrap_or_else(common_core::now_secs) as i64,
                    label,
                    lod_json,
                    content_json,
                ],
            )
            .map_err(|e| LedgerError::Db(e.to_string()))?;
        Ok(())
    }

    /// Apply a mutation to a node and persist both the flat projection and the
    /// `content_json` column. Single place that keeps the two views in sync.
    fn with_node_mut<F>(&self, node_id: NodeId, f: F) -> Result<ContentNode, LedgerError>
    where
        F: FnOnce(&mut ContentNode),
    {
        let mut node = self
            .get_node(node_id)
            .ok_or(LedgerError::NotFound(node_id))?;
        f(&mut node);
        ensure_lod_eager(&mut node);
        self.update_node(&node)?;
        Ok(node)
    }

    /// Persist an updated node (flat projection + `content_json`).
    fn update_node(&self, node: &ContentNode) -> Result<(), LedgerError> {
        let metadata = node
            .metadata
            .as_ref()
            .unwrap_or(&serde_json::json!({}))
            .to_string();
        let lod_json =
            serde_json::to_string(&node.lod).map_err(|e| LedgerError::Db(e.to_string()))?;
        let content_json =
            serde_json::to_string(node).map_err(|e| LedgerError::Db(e.to_string()))?;
        let content = node.lod.first().map_or("", String::as_str);
        let label = node
            .lod
            .get(LOD5_LABEL as usize)
            .cloned()
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| derive_label(&node.role.clone().unwrap_or_default(), content));

        self.store
            .execute(
                "UPDATE ledger SET session_id = ?1, request_id = ?2, role = ?3, content = ?4,
                                   turn_index = ?5, accepted = ?6, acceptance_score = ?7,
                                   active_lod = ?8, parent_id = ?9, step_id = ?10, metadata = ?11,
                                   created_at = ?12, label = ?13, lod = ?14, content_json = ?15
                 WHERE node_id = ?16",
                rusqlite::params![
                    node.session_id.clone().unwrap_or_default(),
                    node.request_id.clone().unwrap_or_default(),
                    node.role.clone().unwrap_or_default(),
                    content,
                    node.turn_index.unwrap_or(0) as i64,
                    node.accepted.unwrap_or(true),
                    node.acceptance_score,
                    i64::from(node.active_lod.unwrap_or(LOD0_FULL_TEXT)),
                    node.parent_id.map(NodeId::as_int),
                    node.step_id.as_deref(),
                    metadata,
                    node.created_at.unwrap_or_else(common_core::now_secs) as i64,
                    label,
                    lod_json,
                    content_json,
                    node.id.map(NodeId::as_int),
                ],
            )
            .map_err(|e| LedgerError::Db(e.to_string()))?;
        Ok(())
    }

    /// Collapse a node: replace LOD0's content with `summary` and mark it
    /// `active_lod = lod`. Used by compaction.
    pub fn collapse_node(
        &self,
        node_id: NodeId,
        summary: &str,
        lod: u8,
    ) -> Result<(), LedgerError> {
        self.with_node_mut(node_id, |node| {
            node.lod[LOD0_FULL_TEXT as usize] = summary.to_string();
            node.lod[LOD5_LABEL as usize] =
                derive_label(&node.role.clone().unwrap_or_default(), summary);
            node.active_lod = Some(lod);
        })?;
        Ok(())
    }

    /// Derive (or return the cached) LOD level for a node, **from LOD0 only**
    /// via the `Summarizer` — never chained from a lower tier.
    ///
    /// Only LOD1–LOD4 are lazy; LOD0/LOD5 are eager at creation.
    pub fn ensure_lod(&self, node_id: NodeId, level: u8) -> Result<ContentNode, LedgerError> {
        if !LAZY_LOD_RANGE.contains(&level) {
            return Err(LedgerError::InvalidLod(level));
        }
        let node = self
            .get_node(node_id)
            .ok_or(LedgerError::NotFound(node_id))?;
        let full_text = node
            .lod
            .first()
            .cloned()
            .ok_or(LedgerError::NotFound(node_id))?;

        let cached = node
            .lod
            .get(level as usize)
            .map_or("", String::as_str)
            .to_string();
        if !cached.is_empty() {
            return Ok(node);
        }

        let summarizer = self.summarizer.as_ref().ok_or(LedgerError::NoSummarizer)?;
        let derived = summarizer
            .summarize_text(&full_text)
            .map_err(|e| LedgerError::Summary(e.to_string()))?;

        self.with_node_mut(node_id, |node| {
            while node.lod.len() <= level as usize {
                node.lod.push(String::new());
            }
            node.lod[level as usize] = derived;
        })
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
        let mut nodes = self.get_session_nodes(session_id, usize::MAX)?;
        // `select_lod` expects chronological (oldest-first) order; the store
        // returns newest-first.
        nodes.reverse();
        let lods = RecencyCompaction.select_lod(&nodes, max_nodes);
        let mut demoted = Vec::new();
        for (node, lod) in nodes.iter().zip(lods) {
            let current = node.active_lod.unwrap_or(LOD0_FULL_TEXT);
            if lod > current {
                if let Some(id) = node.id {
                    self.set_active_lod(id, lod)?;
                    demoted.push(id);
                }
            }
        }
        Ok(demoted)
    }

    /// Set a node's active LOD level (demotion), keeping `content_json` in
    /// sync. The demoted text stays lazy — derived from LOD0 on demand.
    fn set_active_lod(&self, node_id: NodeId, active_lod: u8) -> Result<(), LedgerError> {
        self.with_node_mut(node_id, |node| {
            node.active_lod = Some(active_lod);
        })?;
        Ok(())
    }

    /// Fetch a node by ID (canonical `ContentNode`, hydrated from `content_json`).
    pub fn get_node(&self, node_id: NodeId) -> Option<ContentNode> {
        let json: Option<String> = self
            .store
            .query_row(
                "SELECT content_json FROM ledger WHERE node_id = ?1",
                rusqlite::params![node_id.as_int()],
                |r| r.get(0),
            )
            .ok()
            .flatten();
        if let Some(json) = json {
            let parsed = serde_json::from_str::<ContentNode>(&json).ok();
            if parsed.is_some() {
                return parsed;
            }
        }
        // Pre-LOD rows have '{}': hydrate from the flat projection.
        self.store
            .with_conn(|conn| {
                hydrate_node(conn, node_id).map_err(|e| DbError::Other(e.to_string()))
            })
            .ok()
            .flatten()
    }

    /// All nodes for a session (canonical `ContentNode`s), most recent first.
    pub fn get_session_nodes(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<ContentNode>, LedgerError> {
        let ids = self
            .store
            .query_rows(
                "SELECT node_id FROM ledger WHERE session_id = ?1
                 ORDER BY turn_index DESC LIMIT ?2",
                rusqlite::params![session_id, limit as i64],
                |r| Ok(NodeId::from_int(r.get(0)?)),
            )
            .map_err(|e| LedgerError::Db(e.to_string()))?;

        let mut nodes = Vec::new();
        for id in ids {
            if let Some(node) = self.get_node(id) {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }

    pub fn get_session_entries(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<LedgerEntry>, LedgerError> {
        self.store
            .query_rows(
                "SELECT node_id, session_id, request_id, role, content,
                        turn_index, accepted, acceptance_score, active_lod,
                        parent_id, step_id, metadata, created_at
                 FROM ledger WHERE session_id = ?1
                 ORDER BY turn_index DESC LIMIT ?2",
                rusqlite::params![session_id, limit as i64],
                |row| {
                    Ok(LedgerEntry {
                        node_id: NodeId::from_int(row.get(0)?),
                        session_id: row.get(1)?,
                        request_id: row.get(2)?,
                        role: row.get(3)?,
                        content: row.get(4)?,
                        turn_index: row.get(5)?,
                        accepted: row.get(6)?,
                        acceptance_score: row.get(7)?,
                        active_lod: row.get(8)?,
                        parent_id: row.get::<_, Option<i64>>(9)?.map(NodeId::from_int),
                        step_id: row.get(10)?,
                        metadata: serde_json::from_str(row.get::<_, String>(11)?.as_str())
                            .unwrap_or_default(),
                        created_at: row.get(12)?,
                    })
                },
            )
            .map_err(|e| LedgerError::Db(e.to_string()))
    }

    /// Panic while holding the connection mutex (test-only): exercises the
    /// poison-recovery path in `SqliteStore`'s `common_core::sync::lock`.
    #[cfg(test)]
    fn poison_conn(&self) {
        let _ = self.store.with_conn(|_| -> Result<(), DbError> {
            panic!("simulated panic while holding db mutex")
        });
    }
}

/// Hydrate a `ContentNode` from the flat columns (used for rows written
/// before the `content_json` column existed).
fn hydrate_node(
    db: &rusqlite::Connection,
    node_id: NodeId,
) -> Result<Option<ContentNode>, LedgerError> {
    let row = db
        .query_row(
            "SELECT session_id, request_id, role, content, turn_index, accepted,
                    acceptance_score, active_lod, parent_id, step_id, metadata, created_at
             FROM ledger WHERE node_id = ?1",
            rusqlite::params![node_id.as_int()],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, bool>(5)?,
                    r.get::<_, Option<f64>>(6)?,
                    r.get::<_, i64>(7)?,
                    r.get::<_, Option<i64>>(8)?,
                    r.get::<_, Option<String>>(9)?,
                    r.get::<_, String>(10)?,
                    r.get::<_, i64>(11)?,
                ))
            },
        )
        .map_err(|_| LedgerError::NotFound(node_id))?;

    let content = row.3;
    Ok(Some(ContentNode {
        id: Some(node_id),
        name: format!("node-{}", node_id.as_int()).into(),
        source: "session".into(),
        lod: vec![
            content.clone(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            derive_label(&row.2, &content),
        ],
        embedding: None,
        capabilities: None,
        session_id: Some(row.0),
        request_id: Some(row.1),
        role: Some(row.2),
        turn_index: Some(row.4 as u64),
        accepted: Some(row.5),
        acceptance_score: row.6,
        active_lod: Some(row.7 as u8),
        parent_id: row.8.map(NodeId::from_int),
        step_id: row.9,
        step_status: None,
        metadata: serde_json::from_str(&row.10).ok(),
        created_at: Some(row.11 as u64),
    }))
}

/// Build a fresh `ContentNode` with LOD0 (full text) and LOD5 (label) eager.
fn new_node(
    id: NodeId,
    session_id: &str,
    request_id: &str,
    role: &str,
    content: &str,
    accepted: Option<bool>,
) -> ContentNode {
    let mut node = ContentNode {
        id: Some(id),
        name: format!("{role}-msg-{}", id.as_int()).into(),
        source: "session".into(),
        lod: vec![
            content.to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
        ],
        embedding: None,
        capabilities: None,
        session_id: Some(session_id.to_string()),
        request_id: Some(request_id.to_string()),
        role: Some(role.to_string()),
        // Monotonic per allocation → stable `ORDER BY turn_index DESC` within
        // a session even before a real turn counter exists.
        turn_index: Some(id.as_int() as u64),
        accepted,
        acceptance_score: None,
        active_lod: Some(LOD0_FULL_TEXT),
        parent_id: None,
        step_id: None,
        step_status: None,
        metadata: None,
        created_at: Some(common_core::now_secs()),
    };
    ensure_lod_eager(&mut node);
    node
}

/// Guarantee LOD0 (full text) and LOD5 (label) are present on a node.
fn ensure_lod_eager(node: &mut ContentNode) {
    while node.lod.len() < LOD5_LABEL as usize + 1 {
        node.lod.push(String::new());
    }
    let content = node.lod[LOD0_FULL_TEXT as usize].clone();
    if content.is_empty() {
        // Nothing to derive LOD0 from — nothing to do.
        return;
    }
    if node.lod[LOD5_LABEL as usize].is_empty() {
        let role = node.role.clone().unwrap_or_default();
        node.lod[LOD5_LABEL as usize] = derive_label(&role, &content);
    }
    if node.active_lod.is_none() {
        node.active_lod = Some(LOD0_FULL_TEXT);
    }
}

/// Versioned ledger schema lifecycle (fluent_db::migrate, M6.2).
///
/// One migration per schema step: the base table + session index, then the
/// three LOD-era columns (`label`, `lod`, `content_json`) that pre-LOD
/// databases lack. Each column migration is a no-op when the column already
/// exists, so a fresh database and an upgraded one converge.
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
}
