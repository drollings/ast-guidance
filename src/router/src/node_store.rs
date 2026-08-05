//! NodeStore — the shared, reference-counted, interned, durable ContentNode
//! store (M4).
//!
//! This is the surgical successor to `ContentNodeLedger`'s per-process
//! `SqliteStore`+`Mutex<i64>` storage: nodes live once, behind
//! `Arc<RwLock<ContentNode>>`, so every holder (parallel/filtered ledger
//! views in later milestones) shares the same node object and a lazily-derived
//! LOD tier is computed **at most once** and visible to all holders (VISION).
//! Session and role indices are keyed by `ArcIntern<str>` so sharing a node
//! across N views costs a refcount bump, not a string copy.
//!
//! Durability is preserved: every mutation writes the canonical `content_json`
//! column in the existing `ledger` table (schema + migrations owned by
//! `crate::ledger`), and the maps are **hydrated on open**. Reads prefer the
//! in-memory maps.
//!
//! The LOD lifecycle (LOD0/LOD5 eager, LOD1–LOD4 lazy from LOD0 only via the
//! `Summarizer`, recency compaction) is ported here unchanged from the ledger.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use common_core::sync::{lock, lock_read, lock_write};
use fluent_db::migrate::migrate;
use fluent_db::store::SqliteStore;
use fluent_db::vector::knn_brute_force;
use fluent_types::{ContentNode, KnnHit, NodeId};
use fluent_wvr::ArcIntern;

use crate::ledger::{
    CompactionStrategy, RecencyCompaction, LOD0_FULL_TEXT, LOD5_LABEL, LAZY_LOD_RANGE, LedgerEntry,
    LedgerError,
};
use crate::summarization::Summarizer;

#[cfg(test)]
use fluent_db::error::DbError;

/// The shared ContentNode store: refcounted nodes + interned indices + durable
/// backing.
pub struct NodeStore {
    /// Node id → shared node. The `Arc<RwLock<ContentNode>>` is the sharing
    /// primitive: LOD derivation mutates the shared node, so every holder
    /// observes it.
    nodes: RwLock<HashMap<NodeId, Arc<RwLock<ContentNode>>>>,
    /// session_id (interned) → node ids in insertion order.
    by_session: RwLock<HashMap<ArcIntern<str>, Vec<NodeId>>>,
    /// role (interned) → node ids in insertion order.
    by_role: RwLock<HashMap<ArcIntern<str>, Vec<NodeId>>>,
    /// Monotonic id allocator. Seeded from `MAX(node_id) + 1` at hydration so
    /// a restart never re-issues ids that collide with persisted rows.
    next_id: AtomicI64,
    /// The existing `ledger` table. `None` for a pure in-memory store that
    /// skips durability.
    durable: Option<SqliteStore>,
    /// Lazy LOD derivation. `Mutex` for interior mutability so the summarizer
    /// can be attached after the store is `Arc`-shared
    /// (`ContentNodeLedger::with_summarizer`).
    summarizer: Mutex<Option<Summarizer>>,
}

impl NodeStore {
    /// Open (or create) the durable store at `path`, run the ledger schema
    /// migrations, and hydrate the in-memory maps from every row.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, LedgerError> {
        let db_path = path.into();
        let store = SqliteStore::open(&db_path).map_err(|e| LedgerError::Db(e.to_string()))?;
        Self::open_with_store(Some(store))
    }

    /// Open an in-memory (non-durable-across-processes) store. Used by tests
    /// and ephemeral ledgers.
    pub fn open_in_memory() -> Result<Self, LedgerError> {
        let store = SqliteStore::open_in_memory().map_err(|e| LedgerError::Db(e.to_string()))?;
        Self::open_with_store(Some(store))
    }

    /// A store with no durable backing at all (pure in-memory, no ledger
    /// table). Writes are memory-only.
    pub fn ephemeral() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            by_session: RwLock::new(HashMap::new()),
            by_role: RwLock::new(HashMap::new()),
            next_id: AtomicI64::new(1),
            durable: None,
            summarizer: Mutex::new(None),
        }
    }

    fn open_with_store(durable: Option<SqliteStore>) -> Result<Self, LedgerError> {
        if let Some(ref store) = durable {
            let migrations = crate::ledger::ledger_migrations();
            store
                .with_conn(|conn| migrate(conn, &migrations))
                .map_err(|e| LedgerError::Db(e.to_string()))?;
        }
        let store_obj = Self {
            nodes: RwLock::new(HashMap::new()),
            by_session: RwLock::new(HashMap::new()),
            by_role: RwLock::new(HashMap::new()),
            next_id: AtomicI64::new(1),
            durable,
            summarizer: Mutex::new(None),
        };
        store_obj.hydrate()?;
        Ok(store_obj)
    }

    /// Attach a `Summarizer` so LOD1–LOD4 can be derived lazily from LOD0.
    /// Without one, `ensure_lod` returns `LedgerError::NoSummarizer`. Interior
    /// mutable so it can be attached after the store is `Arc`-shared.
    #[must_use]
    pub fn with_summarizer(self, summarizer: Summarizer) -> Self {
        self.set_summarizer(summarizer);
        self
    }

    /// Set the summarizer on an already-shared store (the facade's
    /// `with_summarizer` uses this through the `Arc`).
    pub fn set_summarizer(&self, summarizer: Summarizer) {
        *lock(&self.summarizer) = Some(summarizer);
    }

    /// Access the durable backing (for the facade's flat view and the poison
    /// test helper). `None` for an ephemeral store.
    #[cfg(test)]
    pub(crate) fn durable(&self) -> Option<&SqliteStore> {
        self.durable.as_ref()
    }

    /// Load every persisted row into the maps, seeding `next_id` from
    /// `MAX(node_id) + 1` (pre-existing restart-collision bug fix, M4).
    fn hydrate(&self) -> Result<(), LedgerError> {
        let Some(ref store) = self.durable else {
            return Ok(());
        };
        let rows: Vec<(i64, String)> = store
            .query_rows("SELECT node_id, content_json FROM ledger", &[], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .map_err(|e| LedgerError::Db(e.to_string()))?;

        let mut max_id = 0i64;
        {
            let mut nodes = lock_write(&self.nodes);
            let mut by_session = lock_write(&self.by_session);
            let mut by_role = lock_write(&self.by_role);
            for (id, json) in rows {
                let Ok(node) = serde_json::from_str::<ContentNode>(&json) else {
                    continue;
                };
                let node_id = NodeId::from_int(id);
                let session_key = node.session_id.as_deref().map(ArcIntern::from);
                let role_key = node.role.as_deref().map(ArcIntern::from);
                max_id = max_id.max(id);
                nodes.insert(node_id, Arc::new(RwLock::new(node)));
                if let Some(key) = session_key {
                    by_session.entry(key).or_default().push(node_id);
                }
                if let Some(key) = role_key {
                    by_role.entry(key).or_default().push(node_id);
                }
            }
        }
        self.next_id.store(max_id + 1, Ordering::SeqCst);
        Ok(())
    }

    /// Fetch the shared node handle (zero-copy internal path). Every holder of
    /// the store shares the same `Arc`, so LOD derivation on one holder is
    /// visible to all.
    pub fn get_node(&self, node_id: NodeId) -> Option<Arc<RwLock<ContentNode>>> {
        lock_read(&self.nodes).get(&node_id).cloned()
    }

    /// Clone a node for serde/persistence (the `snapshot` helper).
    pub fn snapshot(&self, node_id: NodeId) -> Option<ContentNode> {
        self.get_node(node_id).map(|arc| lock_read(&arc).clone())
    }

    /// Record a user request as a new node. LOD0 and LOD5 are written eagerly;
    /// LOD1–LOD4 stay empty until derived lazily.
    pub fn record_request(
        &self,
        session_id: &str,
        request_id: &str,
        content: &str,
    ) -> Result<NodeId, LedgerError> {
        let id = NodeId::from_int(self.next_id.fetch_add(1, Ordering::SeqCst));
        let node = new_node(id, session_id, request_id, "user", content, None);
        self.insert_node(&node)?;
        Ok(id)
    }

    /// Update the result of a previously recorded request node: acceptance,
    /// score, and final content. Keeps the shared node and the `content_json`
    /// column in sync (LOD0/LOD5 recomputed eagerly).
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
    /// guaranteed present (derived from the node's text when missing). An id
    /// is allocated when `node.id` is `None`.
    pub fn record_content_node(&self, node: &ContentNode) -> Result<NodeId, LedgerError> {
        let mut node = node.clone();
        let id = if let Some(id) = node.id {
            id
        } else {
            let id = NodeId::from_int(self.next_id.fetch_add(1, Ordering::SeqCst));
            node.id = Some(id);
            id
        };
        ensure_lod_eager(&mut node);
        self.insert_node(&node)?;
        Ok(id)
    }

    /// Insert a node into the shared maps (and the durable `content_json`
    /// column). The canonical write path: every mutation funnels through here.
    pub fn insert_node(&self, node: &ContentNode) -> Result<NodeId, LedgerError> {
        let mut node = node.clone();
        let node_id = if let Some(id) = node.id {
            id
        } else {
            let id = NodeId::from_int(self.next_id.fetch_add(1, Ordering::SeqCst));
            node.id = Some(id);
            id
        };
        ensure_lod_eager(&mut node);
        let arc = Arc::new(RwLock::new(node));
        lock_write(&self.nodes).insert(node_id, Arc::clone(&arc));
        self.index_node(node_id);
        self.persist_insert(&arc)?;
        Ok(node_id)
    }

    /// Maintain the interned session/role indices for a node.
    fn index_node(&self, node_id: NodeId) {
        let (session, role) = match self.get_node(node_id) {
            Some(arc) => {
                let guard = lock_read(&arc);
                (guard.session_id.clone(), guard.role.clone())
            }
            None => (None, None),
        };
        if let Some(session) = session {
            let key = ArcIntern::from(session.as_str());
            lock_write(&self.by_session).entry(key).or_default().push(node_id);
        }
        if let Some(role) = role {
            let key = ArcIntern::from(role.as_str());
            lock_write(&self.by_role).entry(key).or_default().push(node_id);
        }
    }

    /// Write the node's canonical row (flat projection + `content_json`).
    fn persist_insert(&self, node: &Arc<RwLock<ContentNode>>) -> Result<(), LedgerError> {
        let Some(ref store) = self.durable else {
            return Ok(());
        };
        let node = lock_read(node);
        let metadata = node
            .metadata
            .as_ref()
            .unwrap_or(&serde_json::json!({}))
            .to_string();
        let lod_json = serde_json::to_string(&node.lod).map_err(|e| LedgerError::Db(e.to_string()))?;
        let content_json = serde_json::to_string(&*node).map_err(|e| LedgerError::Db(e.to_string()))?;
        let content = node.lod.first().map_or("", String::as_str);
        let label = node
            .lod
            .get(LOD5_LABEL as usize)
            .cloned()
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| derive_label(&node.role.clone().unwrap_or_default(), content));

        store
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

    /// Persist an updated node (flat projection + `content_json`).
    fn persist_update(&self, node: &ContentNode) -> Result<(), LedgerError> {
        let Some(ref store) = self.durable else {
            return Ok(());
        };
        let metadata = node
            .metadata
            .as_ref()
            .unwrap_or(&serde_json::json!({}))
            .to_string();
        let lod_json = serde_json::to_string(&node.lod).map_err(|e| LedgerError::Db(e.to_string()))?;
        let content_json = serde_json::to_string(node).map_err(|e| LedgerError::Db(e.to_string()))?;
        let content = node.lod.first().map_or("", String::as_str);
        let label = node
            .lod
            .get(LOD5_LABEL as usize)
            .cloned()
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| derive_label(&node.role.clone().unwrap_or_default(), content));

        store
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

    /// Apply a mutation to the shared node and persist both the shared node and
    /// the `content_json` column. The single place that keeps the views in
    /// sync.
    fn with_node_mut<F>(&self, node_id: NodeId, f: F) -> Result<ContentNode, LedgerError>
    where
        F: FnOnce(&mut ContentNode),
    {
        let arc = self.get_node(node_id).ok_or(LedgerError::NotFound(node_id))?;
        let mut guard = lock_write(&arc);
        f(&mut guard);
        ensure_lod_eager(&mut guard);
        let node = guard.clone();
        drop(guard);
        self.persist_update(&node)?;
        Ok(node)
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
    /// Only LOD1–LOD4 are lazy; LOD0/LOD5 are eager at creation. The derived
    /// tier is cached on the shared node, so a second request from any holder
    /// hits the cache, not the LLM.
    pub fn ensure_lod(&self, node_id: NodeId, level: u8) -> Result<ContentNode, LedgerError> {
        if !LAZY_LOD_RANGE.contains(&level) {
            return Err(LedgerError::InvalidLod(level));
        }
        let arc = self.get_node(node_id).ok_or(LedgerError::NotFound(node_id))?;

        let (full_text, cached) = {
            let guard = lock_read(&arc);
            let full_text = guard
                .lod
                .first()
                .cloned()
                .ok_or(LedgerError::NotFound(node_id))?;
            let cached = guard
                .lod
                .get(level as usize)
                .map_or("", String::as_str)
                .to_string();
            (full_text, cached)
        };
        if !cached.is_empty() {
            return Ok(lock_read(&arc).clone());
        }

        let derived = {
            let summarizer = lock(&self.summarizer);
            let summarizer = summarizer.as_ref().ok_or(LedgerError::NoSummarizer)?;
            summarizer
                .summarize_text(&full_text)
                .map_err(|e| LedgerError::Summary(e.to_string()))?
        };

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

    /// All nodes for a session (canonical `ContentNode`s), most recent first,
    /// capped at `limit`.
    pub fn get_session_nodes(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<ContentNode>, LedgerError> {
        let key = ArcIntern::from(session_id);
        let ids: Vec<NodeId> = lock_read(&self.by_session)
            .get(&key)
            .cloned()
            .unwrap_or_default();
        let mut nodes = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(node) = self.snapshot(id) {
                nodes.push(node);
            }
        }
        // Most recent first, mirroring the flat store's `ORDER BY turn_index
        // DESC`.
        nodes.sort_by(|a, b| {
            let ta = a
                .turn_index
                .or_else(|| a.id.map(|i| i.as_int() as u64))
                .unwrap_or(0);
            let tb = b
                .turn_index
                .or_else(|| b.id.map(|i| i.as_int() as u64))
                .unwrap_or(0);
            tb.cmp(&ta)
        });
        nodes.truncate(limit);
        Ok(nodes)
    }

    /// The flat `LedgerEntry` audit projection for a session, read from the
    /// durable table (most recent first, capped at `limit`).
    pub fn get_session_entries(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<LedgerEntry>, LedgerError> {
        let Some(ref store) = self.durable else {
            return Ok(Vec::new());
        };
        store
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

    /// All node ids for a role (interned index), in insertion order.
    pub fn nodes_for_role(&self, role: &str) -> Vec<NodeId> {
        let key = ArcIntern::from(role);
        lock_read(&self.by_role)
            .get(&key)
            .cloned()
            .unwrap_or_default()
    }

    /// Cosine KNN search over node embeddings (brute force over the shared
    /// nodes) — the single similarity path, `fluent_db::vector`'s
    /// `knn_brute_force`.
    pub fn knn_search(&self, embedding: &[f32], k: usize) -> Vec<KnnHit> {
        // Snapshot embeddings under the map read lock (cloning the vecs out)
        // so no borrow escapes a node's guard.
        let candidates: Vec<(NodeId, Vec<f32>)> = {
            let nodes = lock_read(&self.nodes);
            nodes
                .iter()
                .filter_map(|(&node_id, arc)| {
                    let guard = lock_read(arc);
                    guard.embedding.clone().map(|emb| (node_id, emb))
                })
                .collect()
        };
        let hits = knn_brute_force(
            embedding,
            candidates.iter().map(|(id, emb)| (*id, emb.as_slice())),
            k,
        );
        let mut results = Vec::with_capacity(hits.len());
        for (node_id, distance) in hits {
            let name = self
                .snapshot(node_id)
                .map(|n| n.name)
                .unwrap_or_default();
            results.push(KnnHit {
                node_id,
                distance,
                name,
            });
        }
        results
    }

    /// Panic while holding the durable connection mutex (test-only): exercises
    /// the poison-recovery path in `SqliteStore`'s `common_core::sync::lock`.
    #[cfg(test)]
    pub(crate) fn poison_conn(&self) {
        if let Some(ref store) = self.durable {
            let _ = store.with_conn(|_| -> Result<(), DbError> {
                panic!("simulated panic while holding db mutex")
            });
        }
    }
}

/// Build a fresh `ContentNode` with LOD0 (full text) and LOD5 (label) eager.
/// Re-exported from `crate::ledger` so the facade's tests keep compiling
/// unchanged.
pub(crate) fn new_node(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_stubs::StubChatBackend;

    fn temp_store() -> NodeStore {
        let dir = std::env::temp_dir().join(format!(
            "coral-router-nodestore-{}",
            common_core::hash::uuid_v4()
        ));
        let store = NodeStore::open(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        store
    }

    #[test]
    fn same_id_returns_same_arc_identity() {
        let store = temp_store();
        let id = store.record_request("s", "r1", "hello").unwrap();
        let a = store.get_node(id).unwrap();
        let b = store.get_node(id).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "two lookups must share one Arc");
    }

    #[test]
    fn ensure_lod_computed_once_across_views() {
        let client: Arc<dyn fluent_llm::client::ChatBackend> =
            Arc::new(StubChatBackend::always("lazy LOD summary"));
        let summarizer = Summarizer::new(client, 20);
        let store = temp_store().with_summarizer(summarizer);
        let id = store
            .record_request("s", "r1", "The full text that must be summarized once.")
            .unwrap();

        // "Two concurrent views" hold the same Arc — derive once, then both see
        // the cached tier without a second LLM call.
        let v1 = store.get_node(id).unwrap();
        let v2 = store.get_node(id).unwrap();
        let node = store.ensure_lod(id, 2).unwrap();
        assert_eq!(node.lod[2], "lazy LOD summary");
        assert_eq!(lock_read(&v1).lod[2], "lazy LOD summary");
        assert_eq!(lock_read(&v2).lod[2], "lazy LOD summary");
    }

    #[test]
    fn interned_session_and_role_indices_return_correct_sets() {
        let store = temp_store();
        store.record_request("sess-a", "r1", "one").unwrap();
        store.record_request("sess-a", "r2", "two").unwrap();
        store.record_request("sess-b", "r3", "three").unwrap();

        let sess_a = store.get_session_nodes("sess-a", 10).unwrap();
        assert_eq!(sess_a.len(), 2);
        assert_eq!(sess_a[0].request_id.as_deref(), Some("r2"));
        assert_eq!(sess_a[1].request_id.as_deref(), Some("r1"));
        assert!(store.get_session_nodes("sess-b", 10).unwrap().len() == 1);
        assert!(store.get_session_nodes("absent", 10).unwrap().is_empty());

        // role index (interned): all recorded requests carry role "user".
        let user_ids = store.nodes_for_role("user");
        assert_eq!(user_ids.len(), 3);
        assert!(store.nodes_for_role("assistant").is_empty());
    }

    #[test]
    fn hydration_round_trip_preserves_data_and_continues_next_id() {
        let dir = std::env::temp_dir().join(format!(
            "coral-router-nodestore-rt-{}",
            common_core::hash::uuid_v4()
        ));
        let path = dir.clone();
        {
            let store = NodeStore::open(&path).unwrap();
            store.record_request("s", "r1", "first").unwrap();
            store.record_request("s", "r2", "second").unwrap();
        } // drop
        {
            let store = NodeStore::open(&path).unwrap();
            let nodes = store.get_session_nodes("s", 10).unwrap();
            assert_eq!(nodes.len(), 2, "data must survive reopen");
            assert_eq!(nodes[0].request_id.as_deref(), Some("r2"));

            // next_id continues past the hydrated max: the next allocation
            // must not collide with the persisted ids.
            let id = store.record_request("s", "r3", "third").unwrap();
            assert!(id.as_int() > 2, "next id must be past the hydrated max");
            assert!(store.get_node(id).is_some());
            assert_eq!(store.get_session_nodes("s", 10).unwrap().len(), 3);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn knn_search_delegates_to_brute_force_over_embeddings() {
        let store = temp_store();
        let mut node = new_node(
            NodeId::from_int(0),
            "s",
            "r1",
            "assistant",
            "embedding target",
            Some(true),
        );
        node.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
        let id = store.record_content_node(&node).unwrap();

        let hits = store.knn_search(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].node_id, id);

        let no_hits = store.knn_search(&[0.0, 1.0, 0.0, 0.0], 1);
        assert_eq!(no_hits.len(), 1, "orthogonal but still nearest");
        assert_eq!(no_hits[0].node_id, id);
    }

    #[test]
    fn ephemeral_store_needs_no_durable() {
        let store = NodeStore::ephemeral();
        let id = store.record_request("s", "r1", "x").unwrap();
        assert!(store.get_node(id).is_some());
        assert!(store.get_session_entries("s", 10).unwrap().is_empty());
    }
}
