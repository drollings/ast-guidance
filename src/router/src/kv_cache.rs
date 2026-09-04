//! KV cache snapshot management - two-tier index (hot RAM + cold metadata).
//!
//! The fork owns the KV cache bytes: it loads one shared weight pool and serves
//! many instances, and it persists snapshots under `--slot-save-path` itself
//! (see `LLAMA_CPP_SERVER_INSTANCES.md`). This module is the router's *index*
//! into those snapshots - it never reads or writes the raw KV bytes.
//!
//! # Hot tier
//! In-process, RAM-resident. Tracks which sessions have KV cache state actively
//! loaded in a fork slot. Stores metadata only.
//!
//! # Cold tier
//! Records snapshot *metadata* (name, instance, n_ctx_seq, size, mtime) keyed by
//! `(model, adapter, session)` so a rewind can find which fork snapshot to
//! switch into a slot. It does not copy KV bytes into its own tree - the fork's
//! `--slot-save-path` owns the bytes. The derived `file_path`
//! (`<slot_save_path>/<model_key>/<snapshot_name>.bin`) matches the fork's
//! layout byte-for-byte.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use common_core::cache::LoadCache;
use common_core::sync::lock;
use common_core::now_secs;
use thiserror::Error;

/// The snapshot-index key: `(model, adapter, session_id)`.
type SnapshotKey = (String, Option<String>, String);

/// Errors produced by KV cache operations.
#[derive(Error, Debug)]
pub enum KvCacheError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("snapshot not found: {0}")]
    NotFound(String),
    #[error("version mismatch: {0}")]
    VersionMismatch(String),
    #[error("model hash mismatch: {0}")]
    ModelHashMismatch(String),
    #[error("quant mismatch: {0}")]
    QuantMismatch(String),
    #[error("database error: {0}")]
    Db(String),
}

/// Sanitize a `base:variant` model id into the fork's `<model_key>` directory
/// segment: both `/` and `:` become `_`. The fork namespaces snapshots per
/// `(base, variant)` weight pool so they never collide across quantizations.
pub fn model_key(model: &str) -> String {
    model.replace(['/', ':'], "_")
}

/// The fork's on-disk snapshot path for `(model, snapshot)`:
/// `<slot_save_path>/<model_key>/<snapshot_name>.bin`. The router derives this
/// so its metadata and the server's layout agree; it never writes the bytes.
pub fn kv_snapshot_path(slot_save_path: &Path, model: &str, snapshot_name: &str) -> PathBuf {
    slot_save_path
        .join(model_key(model))
        .join(format!("{snapshot_name}.bin"))
}

/// A KV cache snapshot - metadata and the fork-layout filesystem path only.
///
/// The actual KV bytes live in the fork's slot memory and on disk under
/// `--slot-save-path`; the router records this metadata record pointing at the
/// resulting file. Snapshot identity for the index is `(model, adapter,
/// session_id)`; the fork-facing identity is `(snapshot_name, instance)` which
/// the next dispatch sends as the `snapshot` / `instance` request fields.
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    pub model: String,
    pub adapter: Option<String>,
    pub session_id: String,
    /// Fork-side snapshot name - the `snapshot` request field and
    /// `<snapshot_name>.bin`.
    pub snapshot_name: String,
    /// Instance whose slot owns the snapshot - the `instance` request field.
    pub instance: Option<String>,
    /// Derived path `<slot_save_path>/<model_key>/<snapshot_name>.bin`.
    pub file_path: PathBuf,
    /// Token count, when the caller records it. `None` where the value is
    /// unknowable - never a fabricated default.
    pub token_count: Option<usize>,
    pub created_at: u64,
    pub last_used_at: u64,
    /// llama.cpp build version, when recorded. `None` where unknowable.
    pub llama_cpp_version: Option<String>,
    pub model_quant: Option<String>,
    /// Base-model hash, when recorded. `None` where unknowable.
    pub base_model_hash: Option<String>,
    /// Monotonic per-session turn sequence (M3). `None` for legacy rows.
    pub turn_seq: Option<u64>,
}

/// Hot tier: in-process, RAM-resident LRU cache of recently-used snapshot
/// metadata. Entries represent sessions with KV cache state actively loaded in
/// a fork slot. Stores metadata only - no raw bytes.
pub struct HotSnapshotIndex {
    snapshots: LoadCache<(String, Option<String>, String), Arc<KvSnapshot>, KvCacheError>,
    max_mb: usize,
}

impl HotSnapshotIndex {
    /// Creates a new hot cache with the given capacity (number of entries).
    /// The `max_mb` limit is a soft budget - individual snapshot metadata is
    /// tiny (hundreds of bytes), so `max_mb` is primarily informational for
    /// the fork's actual slot memory budget.
    pub fn new(capacity: usize, max_mb: usize) -> Self {
        let snapshots = LoadCache::new(
            capacity.max(1),
            |_: &(String, Option<String>, String)| -> Result<Arc<KvSnapshot>, KvCacheError> {
                Err(KvCacheError::NotFound(
                    "hot tier is write-through; load-on-miss is never invoked".into(),
                ))
            },
        )
        .expect("hot tier capacity is non-zero");
        Self { snapshots, max_mb }
    }

    fn key(
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
    ) -> (String, Option<String>, String) {
        (
            model.to_string(),
            adapter.map(String::from),
            session_id.to_string(),
        )
    }

    /// Insert snapshot metadata into the hot cache.
    pub fn put(&self, snapshot: KvSnapshot) {
        let key = Self::key(
            &snapshot.model,
            snapshot.adapter.as_deref(),
            &snapshot.session_id,
        );
        self.insert_arc(key, Arc::new(snapshot));
    }

    /// Insert an already-`Arc`-wrapped snapshot (used by the two-tier promote
    /// path in `SnapshotStore::retrieve`).
    fn insert_arc(&self, key: (String, Option<String>, String), snapshot: Arc<KvSnapshot>) {
        self.snapshots.insert(key, snapshot);
    }

    /// Get snapshot metadata from the hot cache.
    pub fn get(
        &self,
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
    ) -> Option<Arc<KvSnapshot>> {
        self.snapshots.get(&Self::key(model, adapter, session_id))
    }

    /// Remove snapshot metadata from the hot cache.
    pub fn remove(&self, model: &str, adapter: Option<&str>, session_id: &str) {
        self.snapshots
            .remove(&Self::key(model, adapter, session_id));
    }

    /// Current number of entries in the hot cache.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns true if the hot cache is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// The max RAM budget in MB.
    pub fn max_mb(&self) -> usize {
        self.max_mb
    }
}

/// Cold tier: the durable snapshot *index*. Keyed by `(model, adapter,
/// session)`, each entry records snapshot metadata (name, instance, size,
/// mtime) and a `file_path` derived to the fork's
/// `<slot_save_path>/<model_key>/` layout. The fork owns the bytes; this tier
/// only records which snapshot a session's KV was saved under so a rewind can
/// switch it back into a slot.
///
/// # Eviction story (M10)
///
/// The workspace has one documented eviction story with three distinct,
/// deliberately-ununified pieces:
///
/// - **Hot tier** (`HotSnapshotIndex`): shared `common_core::cache::LoadCache`
///   (in-process LRU, capacity-bounded).
/// - **Cold tier** (this type): a named, in-memory TTL predicate sweep
///   (`evict` retains entries with `age < ttl_secs`). It is *not* delegated
///   to `db::cache::TtlCache`: that type is SQLite-backed with string
///   key→payload rows and `max_entries` LRU, while this tier holds structured
///   `KvSnapshot` values keyed by `(model, adapter, session)` with no entry
///   cap and fork-layout path derivation. Delegating would change durability,
///   key/value shapes, and capacity behavior. The one-second boundary
///   difference (`age == ttl` evicts here; `TtlCache::get` keeps while
///   `now <= ts + ttl`) is preserved verbatim — unifying it would be a
///   behavior change.
/// - **Residency ordering** (`instances::pool`): shared
///   `common_core::cache::{eviction_order, evict_until_fit}` (footprint ×
///   coldness, largest-coldest first). The cold-tier TTL sweep is a
///   *predicate* filter, not a byte-budget eviction, so it intentionally does
///   not use that engine (see `common_core::cache` docs).
pub struct ColdSnapshotIndex {
    /// The fork's `--slot-save-path`. When `None`, snapshots are recorded as
    /// metadata only (no server-owned store) - never a crash.
    slot_save_path: Option<PathBuf>,
    /// Metadata index keyed by `(model, adapter, session)`.
    entries: Mutex<HashMap<SnapshotKey, KvSnapshot>>,
    ttl_secs: u64,
}

impl ColdSnapshotIndex {
    /// Creates a metadata index rooted at `slot_save_path` (the fork's
    /// `--slot-save-path`). `max_mb` is informational; the fork owns the bytes.
    /// `ttl_secs` governs metadata eviction.
    ///
    /// M10.2: the former `_eviction: EvictionPolicy` parameter was removed —
    /// it was accepted and dropped at every call site (always `Lru`), so the
    /// cold tier never honored it. Eviction here is always the TTL predicate
    /// sweep in [`ColdSnapshotIndex::evict`]; byte-budget ordering lives in
    /// `common_core::cache` and is composed by `instances::pool`, not here.
    pub fn new(
        slot_save_path: impl Into<PathBuf>,
        _max_mb: usize,
        ttl_secs: u64,
    ) -> Self {
        Self {
            slot_save_path: Some(slot_save_path.into()),
            entries: Mutex::new(HashMap::new()),
            ttl_secs,
        }
    }

    /// A metadata-only index with no server-owned store. Snapshot `file_path`
    /// stays empty and restores degrade to logged, not dispatched.
    pub fn metadata_only(ttl_secs: u64) -> Self {
        Self {
            slot_save_path: None,
            entries: Mutex::new(HashMap::new()),
            ttl_secs,
        }
    }

    fn derive_path(&self, snapshot: &KvSnapshot) -> PathBuf {
        match &self.slot_save_path {
            Some(base) => kv_snapshot_path(base, &snapshot.model, &snapshot.snapshot_name),
            None => PathBuf::new(),
        }
    }

    /// Record a snapshot's metadata in the cold tier. The fork owns the KV
    /// bytes; this only records the fork-facing identity and derives the
    /// `file_path` to the fork's layout.
    pub fn save(&self, snapshot: &KvSnapshot) -> Result<(), KvCacheError> {
        let mut stored = snapshot.clone();
        stored.file_path = self.derive_path(snapshot);
        let key = (
            snapshot.model.clone(),
            snapshot.adapter.clone(),
            snapshot.session_id.clone(),
        );
        lock(&self.entries).insert(key, stored);
        Ok(())
    }

    /// Load snapshot metadata from the cold tier. Does not read the KV bytes -
    /// callers pass the returned `snapshot_name`/`instance` to the next dispatch
    /// as request fields for the fork to switch in.
    pub fn load(
        &self,
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
    ) -> Result<KvSnapshot, KvCacheError> {
        let key = (model.to_string(), adapter.map(String::from), session_id.to_string());
        lock(&self.entries)
            .get(&key)
            .cloned()
            .ok_or_else(|| KvCacheError::NotFound(format!("no snapshot for session '{session_id}'")))
    }

    /// Evict stale metadata. Returns the number evicted.
    pub fn evict(&self) -> Result<usize, KvCacheError> {
        let now = now_secs();
        let mut entries = lock(&self.entries);
        let before = entries.len();
        entries.retain(|_, snap| {
            let age = now.saturating_sub(snap.last_used_at);
            age < self.ttl_secs
        });
        Ok(before.saturating_sub(entries.len()))
    }

    /// List all recorded snapshots for a session.
    pub fn list_snapshots(&self, session_id: &str) -> Vec<KvSnapshot> {
        lock(&self.entries)
            .values()
            .filter(|s| s.session_id == session_id)
            .cloned()
            .collect()
    }

    /// Remove a session's metadata record.
    pub fn remove(&self, model: &str, adapter: Option<&str>, session_id: &str) {
        let key = (model.to_string(), adapter.map(String::from), session_id.to_string());
        lock(&self.entries).remove(&key);
    }

    /// Max `turn_seq` for `session_id` across all entries (M3).
    pub fn seed_seq_for(&self, session_id: &str) -> u64 {
        lock(&self.entries)
            .values()
            .filter(|s| s.session_id == session_id)
            .filter_map(|s| s.turn_seq)
            .max()
            .unwrap_or(0)
    }
}

impl SnapshotStore {
    /// Max `turn_seq` for `session_id` (M3).
    pub fn seed_seq_for(&self, session_id: &str) -> u64 {
        self.cold.seed_seq_for(session_id)
    }
}

/// Two-tier KV cache index: checks hot tier first, falls back to cold tier.
///
/// Hot tier tracks sessions with actively-loaded fork slots (metadata only).
/// Cold tier is the durable metadata index keyed by `(model, adapter, session)`.
/// On a cold-tier hit, the metadata is promoted to the hot tier.
///
/// Clone is cheap (both tiers are `Arc`-shared), so a single manager can be
/// attached to many `DependencySession`s.
#[derive(Clone)]
pub struct SnapshotStore {
    hot: Arc<HotSnapshotIndex>,
    cold: Arc<ColdSnapshotIndex>,
    /// Optional fork management client. When present, snapshot
    /// save/list/delete round-trip through the fork's management API; without
    /// one they degrade to metadata-only no-ops (never a crash).
    fork: Option<Arc<crate::instances::InstanceClient>>,
}

impl SnapshotStore {
    pub fn new(hot: Arc<HotSnapshotIndex>, cold: Arc<ColdSnapshotIndex>) -> Self {
        Self {
            hot,
            cold,
            fork: None,
        }
    }

    /// Attach the fork management client so `save_snapshot`/`list_snapshots`
    /// /`delete_snapshot` round-trip through the fork's snapshot API. Optional,
    /// post-construction; without it the manager keeps today's metadata-only
    /// behavior. Reuses the single `InstanceClient` (no second HTTP client).
    #[must_use]
    pub fn with_fork_io(mut self, fork: Arc<crate::instances::InstanceClient>) -> Self {
        self.fork = Some(fork);
        self
    }

    /// Record snapshot metadata in both tiers and promote to the hot tier.
    /// Synchronous: both tiers are in-memory indices (the fork owns the KV
    /// bytes).
    pub fn store(&self, snapshot: KvSnapshot) -> Result<(), KvCacheError> {
        self.cold.save(&snapshot)?;
        self.hot.put(snapshot);
        Ok(())
    }

    /// Retrieve snapshot metadata: hot tier first, cold tier as fallback.
    /// On cold-tier hit, the metadata is promoted to the hot tier.
    ///
    /// Returns metadata only - callers pass the returned `snapshot_name` and
    /// `instance` to the next dispatch as the fork's `snapshot`/`instance`
    /// request fields.
    pub fn retrieve(
        &self,
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
    ) -> Result<Arc<KvSnapshot>, KvCacheError> {
        if let Some(snapshot) = self.hot.get(model, adapter, session_id) {
            return Ok(snapshot);
        }

        let snapshot = self.cold.load(model, adapter, session_id)?;
        let arc = Arc::new(snapshot);
        self.hot.insert_arc(
            HotSnapshotIndex::key(model, adapter, session_id),
            Arc::clone(&arc),
        );
        Ok(arc)
    }

    /// Evict from cold tier based on policy.
    pub fn evict(&self) -> Result<usize, KvCacheError> {
        self.cold.evict()
    }

    /// Record (server-side) that a snapshot named `name` was saved on `instance`
    /// for the given `(model, adapter, session)` key.
    ///
    /// With an attached fork handle this round-trips the fork's
    /// `POST /instances/:instance/snapshot` (through the shared
    /// `common_core::runtime::block_on` bridge - never a hand-rolled runtime)
    /// and then records the metadata in both tiers so a rewind finds it. Without
    /// a fork handle it is a metadata-only no-op returning `Ok(())` (today's
    /// behavior). A fork failure logs and returns `KvCacheError::Db` - it never
    /// panics.
    pub fn save_snapshot(
        &self,
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
        name: &str,
        instance: &str,
    ) -> Result<(), KvCacheError> {
        self.save_snapshot_with_seq(model, adapter, session_id, name, instance, None)
    }

    pub fn save_snapshot_with_seq(
        &self,
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
        name: &str,
        instance: &str,
        turn_seq: Option<u64>,
    ) -> Result<(), KvCacheError> {
        let Some(fork) = &self.fork else {
            return Ok(()); // no fork handle: metadata-only no-op (today's behavior)
        };
        let fork = Arc::clone(fork);
        let instance = instance.to_string();
        let name = name.to_string();
        let fork_instance = instance.clone();
        let fork_name = name.clone();
        let result = common_core::runtime::block_on(async move {
            fork.save_snapshot(&fork_instance, &fork_name).await
        });
        if let Err(e) = result {
            tracing::warn!(
                target: "router.kv_cache",
                instance = %instance,
                name = %name,
                error = %e,
                "kv snapshot save on fork failed - degrading to metadata-only",
            );
            return Err(KvCacheError::Db(format!("fork snapshot save failed: {e}")));
        }
        // Record the metadata index (both tiers) under the session key so a
        // rewind can restore it.
        let snapshot = KvSnapshot {
            model: model.to_string(),
            adapter: adapter.map(String::from),
            session_id: session_id.to_string(),
            snapshot_name: name.clone(),
            instance: Some(instance),
            // Derived to the fork layout by the cold tier's `store`.
            file_path: PathBuf::new(),
            token_count: None,
            created_at: now_secs(),
            last_used_at: now_secs(),
            llama_cpp_version: None,
            model_quant: None,
            base_model_hash: None,
            turn_seq,
        };
        self.store(snapshot)
    }

    /// List the fork's snapshots for `instance`. With an attached fork handle
    /// this delegates to `GET /instances/:instance/snapshots` and maps each
    /// `SnapshotInfo` to a `KvSnapshot` (best effort, `file_path` empty). Without
    /// a handle it returns `vec![]` (today's behavior).
    pub fn list_snapshots(&self, instance: &str) -> Vec<KvSnapshot> {
        let Some(fork) = &self.fork else {
            return vec![];
        };
        let fork = Arc::clone(fork);
        let instance = instance.to_string();
        let fork_instance = instance.clone();
        match common_core::runtime::block_on(async move {
            fork.list_snapshots(&fork_instance).await
        })
        {
            Ok(infos) => infos
                .into_iter()
                .map(|info| KvSnapshot {
                    model: instance.clone(),
                    adapter: None,
                    session_id: instance.clone(),
                    snapshot_name: info.name,
                    instance: Some(instance.clone()),
                    file_path: PathBuf::new(),
                    token_count: if info.size > 0 {
                        usize::try_from(info.size).ok()
                    } else {
                        None
                    },
                    created_at: 0,
                    last_used_at: 0,
                    llama_cpp_version: None,
                    model_quant: None,
                    base_model_hash: None,
                    turn_seq: None,
                })
                .collect(),
            Err(e) => {
                tracing::warn!(
                    target: "router.kv_cache",
                    instance = %instance,
                    error = %e,
                    "kv snapshot list on fork failed - returning empty",
                );
                vec![]
            }
        }
    }

    /// Delete a fork snapshot `name` for `instance`. With an attached fork
    /// handle this delegates to `DELETE /instances/:instance/snapshot/:name`;
    /// without one it is a no-op returning `Ok(())` - never a crash.
    pub fn delete_snapshot(
        &self,
        instance: &str,
        name: &str,
    ) -> Result<(), KvCacheError> {
        let Some(fork) = &self.fork else {
            return Ok(());
        };
        let fork = Arc::clone(fork);
        let instance = instance.to_string();
        let name = name.to_string();
        let fork_instance = instance.clone();
        let fork_name = name.clone();
        match common_core::runtime::block_on(async move {
            fork.delete_snapshot(&fork_instance, &fork_name).await
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                tracing::warn!(
                    target: "router.kv_cache",
                    instance = %instance,
                    name = %name,
                    error = %e,
                    "kv snapshot delete on fork failed",
                );
                Err(KvCacheError::Db(format!(
                    "fork snapshot delete failed: {e}"
                )))
            }
        }
    }
}

#[cfg(test)]
#[path = "../tests/kv_cache.rs"]
mod tests;
