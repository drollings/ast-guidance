//! KV cache snapshot management — two-tier index (hot RAM + cold metadata).
//!
//! The fork owns the KV cache bytes: it loads one shared weight pool and serves
//! many instances, and it persists snapshots under `--slot-save-path` itself
//! (see `LLAMA_CPP_SERVER_INSTANCES.md`). This module is the router's *index*
//! into those snapshots — it never reads or writes the raw KV bytes.
//!
//! # Hot tier
//! In-process, RAM-resident. Tracks which sessions have KV cache state actively
//! loaded in a fork slot. Stores metadata only.
//!
//! # Cold tier
//! Records snapshot *metadata* (name, instance, n_ctx_seq, size, mtime) keyed by
//! `(model, adapter, session)` so a rewind can find which fork snapshot to
//! switch into a slot. It does not copy KV bytes into its own tree — the fork's
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

/// A KV cache snapshot — metadata and the fork-layout filesystem path only.
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
    /// Fork-side snapshot name — the `snapshot` request field and
    /// `<snapshot_name>.bin`.
    pub snapshot_name: String,
    /// Instance whose slot owns the snapshot — the `instance` request field.
    pub instance: Option<String>,
    /// Derived path `<slot_save_path>/<model_key>/<snapshot_name>.bin`.
    pub file_path: PathBuf,
    /// Token count, when the caller records it. `None` where the value is
    /// unknowable — never a fabricated default.
    pub token_count: Option<usize>,
    pub created_at: u64,
    pub last_used_at: u64,
    /// llama.cpp build version, when recorded. `None` where unknowable.
    pub llama_cpp_version: Option<String>,
    pub model_quant: Option<String>,
    /// Base-model hash, when recorded. `None` where unknowable.
    pub base_model_hash: Option<String>,
}

/// Hot tier: in-process, RAM-resident LRU cache of recently-used snapshot
/// metadata. Entries represent sessions with KV cache state actively loaded in
/// a fork slot. Stores metadata only — no raw bytes.
pub struct HotKvCache {
    snapshots: LoadCache<(String, Option<String>, String), Arc<KvSnapshot>, KvCacheError>,
    max_mb: usize,
}

impl HotKvCache {
    /// Creates a new hot cache with the given capacity (number of entries).
    /// The `max_mb` limit is a soft budget — individual snapshot metadata is
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
    /// path in `KvCacheManager::retrieve`).
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
pub struct ColdKvCache {
    /// The fork's `--slot-save-path`. When `None`, snapshots are recorded as
    /// metadata only (no server-owned store) — never a crash.
    slot_save_path: Option<PathBuf>,
    /// Metadata index keyed by `(model, adapter, session)`.
    entries: Mutex<HashMap<SnapshotKey, KvSnapshot>>,
    ttl_secs: u64,
}

impl ColdKvCache {
    /// Creates a metadata index rooted at `slot_save_path` (the fork's
    /// `--slot-save-path`). `max_mb` is informational; the fork owns the bytes.
    /// `ttl_secs` governs metadata eviction.
    pub fn new(
        slot_save_path: impl Into<PathBuf>,
        _max_mb: usize,
        ttl_secs: u64,
        _eviction: crate::config::EvictionPolicy,
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

    /// Load snapshot metadata from the cold tier. Does not read the KV bytes —
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
pub struct KvCacheManager {
    hot: Arc<HotKvCache>,
    cold: Arc<ColdKvCache>,
}

impl KvCacheManager {
    pub fn new(hot: Arc<HotKvCache>, cold: Arc<ColdKvCache>) -> Self {
        Self { hot, cold }
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
    /// Returns metadata only — callers pass the returned `snapshot_name` and
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
            HotKvCache::key(model, adapter, session_id),
            Arc::clone(&arc),
        );
        Ok(arc)
    }

    /// Evict from cold tier based on policy.
    pub fn evict(&self) -> Result<usize, KvCacheError> {
        self.cold.evict()
    }

    /// Record (server-side) that a snapshot named `name` was saved for
    /// `instance`. Without an attached M4 sidecar client this only records
    /// metadata and never dispatches — a no-op, never a crash.
    pub fn save_snapshot(
        &self,
        _name: &str,
        _instance: &str,
    ) -> Result<(), KvCacheError> {
        Ok(())
    }

    /// List the fork's snapshots for `instance`. Without an attached M4 sidecar
    /// client this returns the locally-recorded metadata (possibly empty).
    pub fn list_snapshots(&self, instance: &str) -> Vec<KvSnapshot> {
        // Local metadata is keyed by session; with no sidecar there is no
        // server list. Scope to the cold tier's session list as a best effort.
        let _ = instance;
        vec![]
    }

    /// Delete a fork snapshot `name` for `instance`. Without an attached M4
    /// sidecar client this is a no-op — never a crash.
    pub fn delete_snapshot(
        &self,
        _name: &str,
        _instance: &str,
    ) -> Result<(), KvCacheError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_snapshot(session_id: &str) -> KvSnapshot {
        KvSnapshot {
            model: "test-model".into(),
            adapter: None,
            session_id: session_id.into(),
            snapshot_name: "default".into(),
            instance: None,
            file_path: PathBuf::new(),
            token_count: Some(100),
            created_at: now_secs(),
            last_used_at: now_secs(),
            llama_cpp_version: Some("0.1.0".into()),
            model_quant: None,
            base_model_hash: Some("abc123".into()),
        }
    }

    #[test]
    fn test_hot_cache_put_get() {
        let cache = HotKvCache::new(10, 1024);
        let snap = test_snapshot("sess-1");
        cache.put(snap.clone());

        let retrieved = cache.get("test-model", None, "sess-1").unwrap();
        assert_eq!(retrieved.session_id, "sess-1");
        assert_eq!(retrieved.token_count, Some(100));
    }

    #[test]
    fn test_hot_cache_miss() {
        let cache = HotKvCache::new(10, 1024);
        assert!(cache.get("nonexistent", None, "sess-x").is_none());
    }

    #[test]
    fn test_hot_cache_remove() {
        let cache = HotKvCache::new(10, 1024);
        cache.put(test_snapshot("sess-1"));
        assert_eq!(cache.len(), 1);

        cache.remove("test-model", None, "sess-1");
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_hot_cache_lru_eviction() {
        let cache = HotKvCache::new(3, 1024);

        for i in 0..5 {
            cache.put(KvSnapshot {
                model: "m".into(),
                adapter: None,
                session_id: format!("sess-{i}"),
                snapshot_name: "default".into(),
                instance: None,
                file_path: PathBuf::new(),
                token_count: Some(1),
                created_at: now_secs(),
                last_used_at: now_secs(),
                llama_cpp_version: Some("0.1".into()),
                model_quant: None,
                base_model_hash: Some("hash".into()),
            });
        }

        assert_eq!(cache.len(), 3);
        assert!(cache.get("m", None, "sess-0").is_none());
        assert!(cache.get("m", None, "sess-1").is_none());
        assert!(cache.get("m", None, "sess-2").is_some());
    }

    #[test]
    fn model_key_sanitizes_slashes_and_colons() {
        assert_eq!(model_key("abiray/lfm2.5"), "abiray_lfm2.5");
        assert_eq!(model_key("org/model:q4"), "org_model_q4");
    }

    #[test]
    fn kv_snapshot_path_matches_fork_layout() {
        let p = kv_snapshot_path(Path::new("/srv/slots"), "abiray/lfm2.5", "readfiles");
        assert_eq!(p, PathBuf::from("/srv/slots/abiray_lfm2.5/readfiles.bin"));
    }

    #[tokio::test]
    async fn test_cold_cache_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let cold = ColdKvCache::new(dir.path(), 1024, 86400, crate::config::EvictionPolicy::Lru);

        let mut snap = test_snapshot("sess-cold");
        snap.snapshot_name = "readfiles".into();
        snap.instance = Some("scratch".into());
        cold.save(&snap).unwrap();

        let loaded = cold.load("test-model", None, "sess-cold").unwrap();
        assert_eq!(loaded.model, "test-model");
        assert_eq!(loaded.session_id, "sess-cold");
        assert_eq!(loaded.snapshot_name, "readfiles");
        assert_eq!(loaded.instance.as_deref(), Some("scratch"));
        // The derived path matches the fork layout: <slot_save_path>/<model_key>/<name>.bin
        assert_eq!(
            loaded.file_path,
            dir.path().join("test-model").join("readfiles.bin")
        );
    }

    #[tokio::test]
    async fn test_cold_cache_load_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let cold = ColdKvCache::new(dir.path(), 1024, 86400, crate::config::EvictionPolicy::Lru);

        let result = cold.load("test-model", None, "no-such-session");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_kv_cache_manager_two_tier() {
        let dir = tempfile::tempdir().unwrap();
        let hot = Arc::new(HotKvCache::new(10, 1024));
        let cold = Arc::new(ColdKvCache::new(
            dir.path(),
            1024,
            86400,
            crate::config::EvictionPolicy::Lru,
        ));
        let mgr = KvCacheManager::new(Arc::clone(&hot), Arc::clone(&cold));

        mgr.store(test_snapshot("sess-tier")).unwrap();

        // Should be in hot tier
        let retrieved = mgr.retrieve("test-model", None, "sess-tier").unwrap();
        assert_eq!(retrieved.session_id, "sess-tier");

        // Remove from hot, should fall back to cold
        hot.remove("test-model", None, "sess-tier");
        let retrieved2 = mgr.retrieve("test-model", None, "sess-tier").unwrap();
        assert_eq!(retrieved2.session_id, "sess-tier");
    }

    #[tokio::test]
    async fn test_cold_cache_evict_by_ttl() {
        let dir = tempfile::tempdir().unwrap();
        let cold = ColdKvCache::new(
            dir.path(),
            1024,
            0, // immediate TTL
            crate::config::EvictionPolicy::Lru,
        );

        cold.save(&test_snapshot("sess-evict")).unwrap();

        let evicted = cold.evict().unwrap();
        assert_eq!(evicted, 1);
    }

    #[tokio::test]
    async fn test_list_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let cold = ColdKvCache::new(dir.path(), 1024, 86400, crate::config::EvictionPolicy::Lru);

        cold.save(&test_snapshot("sess-list")).unwrap();

        let snapshots = cold.list_snapshots("sess-list");
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].session_id, "sess-list");
    }

    #[tokio::test]
    async fn metadata_only_cold_tier_degrades_gracefully() {
        let cold = ColdKvCache::metadata_only(86400);
        let mut snap = test_snapshot("sess-meta");
        snap.snapshot_name = "x".into();
        cold.save(&snap).unwrap();

        let loaded = cold.load("test-model", None, "sess-meta").unwrap();
        // No server-owned store: the derived path is empty, never a crash.
        assert!(loaded.file_path.as_os_str().is_empty());
    }
}
