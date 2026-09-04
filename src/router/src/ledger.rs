//! Full-detail content ledger with LOD compaction.
//!
//! `ContentNodeLedger` is now a **thin facade** over the shared
//! `ContentNodeStore`: the durable, per-session store of `ContentNode`s (the
//! canonical `fluent_types::ContentNode`) lives in `crate::node_store`, where
//! nodes are shared behind `Arc<RwLock<ContentNode>>` with interned
//! session/role index keys and a durable `content_json` column. The facade
//! keeps the server-facing surface (`record_request`, `record_result`,
//! `record_content_node`, `get_session_nodes`, `get_session_entries`,
//! `ensure_lod`, `compact_session`, `collapse_node`) signature-identical and
//! delegates to the store.
//!
//!  The facade also owns the **write-path guard**: every write delegate
//! scrubs its text through `crate::ledger_guard::scrub_for_ledger` before
//! reaching `ContentNodeStore`, so the durable ledger can never cache text
//! matching the builtin filter engine (on by default, no config flag).  The
//! scrub is irreversible.  `ContentNodeStore` itself stays policy-free; the
//! only documented bypass is the `KnowledgeCapability` impl directly on
//! `ContentNodeStore` (`crate::knowledge`), which is a trait-object
//! boundary that cannot route through the facade — production server writes
//! all flow through here.
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

pub mod correction_index;
pub mod span_cache;
pub mod annotations;
pub mod frame_index;
pub mod nlp;
pub mod node_annotation;
pub mod orchestrator;
pub mod overlay;
pub mod overlay_worker;
pub mod prompt;
pub mod tiering;
pub mod workflow;
pub mod workflow_store;

#[cfg(test)]
mod overlay_acceptance;

use std::path::PathBuf;
use std::sync::Arc;

use fluent_db::error::DbError;
use fluent_db::migrate::{ensure_column, Migration};
use fluent_types::{AnnotationClaim, ClaimStatus, ContentNode, NodeId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ledger_guard::scrub_for_ledger;
use crate::node_store::ContentNodeStore;
use crate::summarization::Summarizer;
use crate::views::LedgerView;
use crate::views::{Lod, ParallelLedger};

/// Node-construction helper moved to `ContentNodeStore`. Re-exported here so the
/// facade's tests keep compiling unchanged.
#[cfg(test)]
#[allow(unused_imports)]
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
/// shared `Arc<ContentNodeStore>`. Every method delegates; the exact public
/// surface is preserved so the server (`ServerDeps.ledger`) is untouched.
pub struct ContentNodeLedger {
    store: Arc<ContentNodeStore>,
}

impl ContentNodeLedger {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, LedgerError> {
        Ok(Self {
            store: Arc::new(ContentNodeStore::open(path)?),
        })
    }

    /// Open an in-memory ledger (tests / ephemeral stores).
    pub fn open_in_memory() -> Result<Self, LedgerError> {
        Ok(Self {
            store: Arc::new(ContentNodeStore::open_in_memory()?),
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
    pub fn node_store(&self) -> &Arc<ContentNodeStore> {
        &self.store
    }

    /// Record a user request as a new node. LOD0 (full text) and LOD5 (label)
    /// are written eagerly; LOD1–LOD4 stay empty until derived lazily.
    ///
    /// The write-path guard scrubs `content` against the builtin filter
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
    /// The write-path guard scrubs `content` before persisting.
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
    /// The write-path guard scrubs the node's LOD0 text and clears LOD5
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
    /// fallback was retired: the canonical read path is single-format.
    pub fn get_node(&self, node_id: NodeId) -> Option<ContentNode> {
        self.store.snapshot(node_id)
    }

    /// Update an existing node's `metadata` (the review worker's
    /// `review_status` write, §12.6).
    pub fn update_node_metadata(
        &self,
        node_id: NodeId,
        metadata: serde_json::Value,
    ) -> Result<(), LedgerError> {
        self.store.update_node_metadata(node_id, metadata)
    }

    /// Atomic review write (ROADMAP §12.6/§12.7, C4): the parse node's
    /// `review_status` metadata update, the `parse_review` node (on a miss),
    /// and the `interlingua_index` correction rows all commit in **one SQLite
    /// transaction**, so a crash mid-review never half-applies. See
    /// [`ContentNodeStore::apply_review`] for the parameter semantics.
    pub(crate) fn apply_review(
        &self,
        parse_node_id: NodeId,
        parse_metadata: serde_json::Value,
        review_node: Option<&ContentNode>,
        correction_rows: &[crate::ledger::correction_index::CorrectionRow],
    ) -> Result<Option<NodeId>, LedgerError> {
        self.store
            .apply_review(parse_node_id, parse_metadata, review_node, correction_rows)
    }

    /// Write a tiered annotation claim against a node's **current content
    /// hash** (ROADMAP M4). The public writer surface for the frame / review /
    /// enrichment producers: it resolves the node's hash and routes the claim
    /// to the `AnnotationStore`, so a later content mutation (new hash) makes
    /// the claim unreachable — invalidation is keying, never a scheduler.
    /// `Ok(None)` when the store has no durable backing (fail-open).
    pub fn write_annotation(
        &self,
        node_id: NodeId,
        claim: &AnnotationClaim,
    ) -> Result<Option<ClaimStatus>, LedgerError> {
        self.store.write_annotation(node_id, claim)
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

    /// Render a session through a `ParallelLedger` at `default_lod`. The
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
    #[allow(dead_code)]
    fn poison_conn(&self) {
        self.store.poison_conn();
    }
}

/// Emit the write-path audit record: a builtin-filter match flagged on a
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

/// Versioned ledger schema lifecycle (fluent_db::migrate).
///
/// One migration per schema step: the base table + session index, then the
/// three LOD-era columns (`label`, `lod`, `content_json`) that pre-LOD
/// databases lack, then migration 5 (the interlingua tables). Each column
/// migration is a no-op when the column already exists, so a fresh database
/// and an upgraded one converge.
///
/// Shared with `ContentNodeStore` (4B), which owns the durable backing now — the
/// schema stays here as the single source of truth.
pub(crate) fn ledger_migrations() -> [&'static dyn Migration; 9] {
    [
        &LedgerBaseSchema,
        &LedgerLabelColumn,
        &LedgerLodColumn,
        &LedgerContentJsonColumn,
        &LedgerInterlinguaSchema,
        &LedgerInterlinguaSchemaV2,
        &LedgerOverlayCandidatesSchema,
        &LedgerAnnotationSchema,
        &LedgerSpanCacheHexKey,
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

/// Migration 5 (ROADMAP §14.2/§14.3): the interlingua tables.
///
/// - `interlingua_index` — the router's durable **correction index** (the
///   `CorrectionIndex` impl, §12.5) and the audit of which ids were attached
///   to which parse node. PK `(node_id, interlingua_id, role)`: a parse node
///   references each id at most once per role; the same `InterlinguaId` can
///   appear on many nodes (indexed for cross-node lookup).
/// - `interlingua_concepts` — the `SqliteConceptStore`'s materialized index of
///   the YaGO taxonomy (C3's second home, reconciled with coral at boot).
struct LedgerInterlinguaSchema;

impl Migration for LedgerInterlinguaSchema {
    fn version(&self) -> u32 {
        5
    }
    fn name(&self) -> &str {
        "ledger-interlingua-schema"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS interlingua_index (
                node_id INTEGER NOT NULL,
                interlingua_id INTEGER NOT NULL,
                interlingua_source TEXT NOT NULL DEFAULT 'spacy_lemma',
                role TEXT NOT NULL DEFAULT 'lemma',
                confidence REAL,
                review_status TEXT NOT NULL DEFAULT 'unreviewed',
                corrections TEXT,               -- JSON: pattern-level correction cache (§12.5)
                PRIMARY KEY (node_id, interlingua_id, role)
            );
            CREATE INDEX IF NOT EXISTS idx_interlingua_id
                ON interlingua_index(interlingua_id, interlingua_source);

            CREATE TABLE IF NOT EXISTS interlingua_concepts (
                id INTEGER PRIMARY KEY,          -- InterlinguaId.as_i64()
                namespace INTEGER NOT NULL,      -- u16
                canonical_name TEXT NOT NULL,
                yago_iri TEXT,
                yago_class_iri TEXT,
                label TEXT,
                node_id INTEGER,                 -- full 64-bit NodeId (F5)
                parent_class_id INTEGER,         -- InterlinguaId of the rdfs:subClassOf parent
                UNIQUE (namespace, canonical_name)
            );
            CREATE INDEX IF NOT EXISTS idx_concepts_namespace
                ON interlingua_concepts(namespace);",
        )
        .map_err(DbError::from)?;
        Ok(())
    }
}

/// Migration 6 (red-team round 4, M2/M3): rebuild the two interlingua tables
/// so columns mean what they say and a truncated-id collision no longer drops
/// a canonical.
///
/// - **`interlingua_index`** gains a real `entity_id` column and the PK
///   becomes `(node_id, interlingua_id, role, entity_id)`. The old schema
///   overloaded `review_status` to hold the cache-scoping entity id as a
///   string on the pattern-cache rows (`node_id = 0`, `role = 'correction'`);
///   those are mapped to the real column with `review_status = 'cached'`.
/// - **`interlingua_concepts`** keys on `(namespace, canonical_name)` (the
///   canonical string, which resolves collisions) and `id` becomes a plain
///   indexed column — two canonicals that collide on the 48-bit local id are
///   both stored, matching the in-memory store's first-wins-with-both-canonicals
///   semantics.
///
/// Table-rebuild (create-new → copy → drop → rename) so existing rows are
/// preserved.
struct LedgerInterlinguaSchemaV2;

impl Migration for LedgerInterlinguaSchemaV2 {
    fn version(&self) -> u32 {
        6
    }
    fn name(&self) -> &str {
        "ledger-interlingua-schema-v2"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        tx.execute_batch(
            "-- Index names are schema-global in SQLite: drop the migration-5
             -- indexes before recreating them on the rebuilt tables.
             DROP INDEX IF EXISTS idx_interlingua_id;
             DROP INDEX IF EXISTS idx_concepts_namespace;

             CREATE TABLE interlingua_index_v2 (
                node_id INTEGER NOT NULL,
                interlingua_id INTEGER NOT NULL,
                interlingua_source TEXT NOT NULL DEFAULT 'spacy_lemma',
                role TEXT NOT NULL DEFAULT 'lemma',
                entity_id INTEGER NOT NULL DEFAULT 0,   -- 0 = not entity-scoped
                confidence REAL,
                review_status TEXT NOT NULL DEFAULT 'unreviewed',
                corrections TEXT,
                PRIMARY KEY (node_id, interlingua_id, role, entity_id)
            );
            CREATE INDEX idx_interlingua_id
                ON interlingua_index_v2(interlingua_id, interlingua_source);

            -- Pattern-cache rows stored the entity id as a string in
            -- `review_status`; map it to the real column and mark them cached.
            INSERT INTO interlingua_index_v2
                (node_id, interlingua_id, interlingua_source, role, entity_id,
                 confidence, review_status, corrections)
            SELECT node_id, interlingua_id, interlingua_source, role,
                   CASE WHEN node_id = 0 AND role = 'correction'
                        THEN CAST(COALESCE(NULLIF(review_status, ''), '0') AS INTEGER)
                        ELSE 0 END,
                   confidence,
                   CASE WHEN node_id = 0 AND role = 'correction'
                        THEN 'cached'
                        ELSE review_status END,
                   corrections
            FROM interlingua_index;
            DROP TABLE interlingua_index;
            ALTER TABLE interlingua_index_v2 RENAME TO interlingua_index;

            CREATE TABLE interlingua_concepts_v2 (
                id INTEGER NOT NULL,                -- InterlinguaId.as_i64()
                namespace INTEGER NOT NULL,         -- u16
                canonical_name TEXT NOT NULL,
                yago_iri TEXT,
                yago_class_iri TEXT,
                label TEXT,
                node_id INTEGER,                    -- full 64-bit NodeId (F5)
                parent_class_id INTEGER,            -- InterlinguaId of the rdfs:subClassOf parent
                PRIMARY KEY (namespace, canonical_name)
            );
            CREATE INDEX idx_concepts_id ON interlingua_concepts_v2(id);
            CREATE INDEX idx_concepts_namespace ON interlingua_concepts_v2(namespace);

            INSERT INTO interlingua_concepts_v2
                (id, namespace, canonical_name, yago_iri, yago_class_iri, label,
                 node_id, parent_class_id)
            SELECT id, namespace, canonical_name, yago_iri, yago_class_iri, label,
                   node_id, parent_class_id
            FROM interlingua_concepts;
            DROP TABLE interlingua_concepts;
            ALTER TABLE interlingua_concepts_v2 RENAME TO interlingua_concepts;",
        )
        .map_err(DbError::from)?;
        Ok(())
    }
}

/// Migration 7 (ROADMAP_20260827_ORT §6.1): the overlay/candidate plane.
///
/// `overlay_candidates` is the durable surface for async overlay outputs —
/// entity links, PII-shaped spans, parse corrections, concept summaries. The
/// async entity-link worker (M6.2) and future overlays write **candidates**
/// here, never a runtime write to `TokenRecord.interlingua_entity_id` /
/// `concept_ids`. It mirrors the `interlingua_index` conventions:
///
/// - **First-wins** on `(node_id, span_start, kind, entity_id)` — `INSERT OR
///   IGNORE` keeps the first candidate for a given span/kind/entity, so a
///   duplicate overlay pass never overwrites an accepted candidate.
/// - **Id-membership**: a non-zero `entity_id` must resolve in
///   `interlingua_concepts` (reconciled at boot — see `ledger/overlay.rs`).
///
/// `entity_id` is `0` when the candidate is not entity-shaped (e.g. a PII or
/// parse-correction candidate), matching the `interlingua_index.entity_id`
/// default-0 sentinel.
struct LedgerOverlayCandidatesSchema;

impl Migration for LedgerOverlayCandidatesSchema {
    fn version(&self) -> u32 {
        7
    }
    fn name(&self) -> &str {
        "ledger-overlay-candidates-schema"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS overlay_candidates (
                node_id INTEGER NOT NULL,
                span_start INTEGER NOT NULL,
                span_end INTEGER NOT NULL,
                kind TEXT NOT NULL,
                entity_id INTEGER NOT NULL DEFAULT 0,   -- 0 = not entity-shaped
                score REAL,
                source TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                PRIMARY KEY (node_id, span_start, kind, entity_id)
            );
            CREATE INDEX IF NOT EXISTS idx_overlay_node_kind
                ON overlay_candidates(node_id, kind);
            CREATE INDEX IF NOT EXISTS idx_overlay_status
                ON overlay_candidates(status, entity_id);",
        )
        .map_err(DbError::from)?;
        Ok(())
    }
}

/// Migration 8 (ROADMAP M4): the tiered annotation table.
///
/// `ledger_annotations` is the durable surface for `AnnotationStore` claims —
/// every annotation keyed to the node's `content_hash` (the version identity)
/// and stamped with a producing `ProvenanceTier` and a `ClaimStatus`. One row
/// per claim per node version (`claim_id`), so a higher-tier claim supersedes
/// (never deletes) its predecessor and a hash change makes the old rows
/// unreachable — no staleness scheduler.
struct LedgerAnnotationSchema;

impl Migration for LedgerAnnotationSchema {
    fn version(&self) -> u32 {
        8
    }
    fn name(&self) -> &str {
        "ledger-annotations-schema"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS ledger_annotations (
                content_hash INTEGER NOT NULL,   -- the node version identity (LOD0 hash)
                claim_key TEXT NOT NULL,         -- the claim's identity within a version
                claim_id INTEGER NOT NULL,       -- monotonic version within (hash, key)
                tier TEXT NOT NULL,              -- ProvenanceTier (snake_case)
                status TEXT NOT NULL,            -- ClaimStatus (provisional|confirmed|superseded)
                payload TEXT NOT NULL,           -- the claim value (JSON)
                produced_by TEXT NOT NULL,       -- legible producer label
                produced_at INTEGER NOT NULL,    -- unix seconds
                PRIMARY KEY (content_hash, claim_key, claim_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ledger_annotations_active
                ON ledger_annotations(content_hash, claim_key, claim_id);",
        )
        .map_err(DbError::from)?;
        Ok(())
    }
}

/// Migration 9 (R10): span-cache key as fixed-width hex TEXT (F7).
///
/// Stores the `u64` span key as `span_key TEXT = format!("{:016x}", key)` for
/// `role='span_cache'` rows — the `i64` cast (`key as i64`) silently corrupts
/// the upper half of the `u64` space (negative `interlingua_id`). Migration
/// rebuilds `interlingua_index` to include `span_key` in the primary key so
/// span-cache rows are keyed by hex (no `i64` truncation) while other roles
/// keep `span_key = ''`.
struct LedgerSpanCacheHexKey;

impl Migration for LedgerSpanCacheHexKey {
    fn version(&self) -> u32 {
        9
    }
    fn name(&self) -> &str {
        "ledger-span-cache-hex-key"
    }
    fn up(&self, tx: &mut rusqlite::Transaction<'_>) -> Result<(), DbError> {
        tx.execute_batch(
            "DROP INDEX IF EXISTS idx_interlingua_id;
             DROP INDEX IF EXISTS idx_interlingua_span_key;
             DROP INDEX IF EXISTS idx_interlingua_span_cache;
             CREATE TABLE interlingua_index_new (
                 node_id INTEGER NOT NULL,
                 interlingua_id INTEGER NOT NULL DEFAULT 0,
                 interlingua_source TEXT NOT NULL DEFAULT 'spacy_lemma',
                 role TEXT NOT NULL DEFAULT 'lemma',
                 entity_id INTEGER NOT NULL DEFAULT 0,
                 confidence REAL,
                 review_status TEXT NOT NULL DEFAULT 'unreviewed',
                 corrections TEXT,
                 span_key TEXT NOT NULL DEFAULT '',
                 PRIMARY KEY (node_id, interlingua_id, span_key, role, entity_id)
             );
             CREATE INDEX idx_interlingua_id
                 ON interlingua_index_new(interlingua_id, interlingua_source);
             CREATE INDEX idx_interlingua_span_key
                 ON interlingua_index_new(span_key);
             CREATE UNIQUE INDEX idx_interlingua_span_cache
                 ON interlingua_index_new(node_id, span_key, role, entity_id)
                 WHERE role = 'span_cache';
             INSERT INTO interlingua_index_new
                 (node_id, interlingua_id, interlingua_source, role, entity_id,
                  confidence, review_status, corrections, span_key)
             SELECT node_id, interlingua_id, interlingua_source, role, entity_id,
                    confidence, review_status, corrections,
                    CASE WHEN role = 'span_cache' THEN printf('%016x', interlingua_id) ELSE '' END
             FROM interlingua_index;
             DROP TABLE interlingua_index;
             ALTER TABLE interlingua_index_new RENAME TO interlingua_index;",
        )
        .map_err(DbError::from)?;
        Ok(())
    }
}

// ── LOD compaction policy ─────────────────────────────────────────────────
//
// Moved here from the deleted standalone `compaction.rs`: compaction is
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
#[path = "../tests/ledger.rs"]
mod tests;
