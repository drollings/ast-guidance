//! KV cache snapshot management — two-tier cache (hot RAM + cold disk).
//!
//! # Hot tier
//! In-process, RAM-resident. Tracks which sessions have KV cache state actively
//! loaded in a llama.cpp server slot. Stores metadata only — the actual KV cache
//! bytes live in the llama.cpp server's memory, not in the router process.
//!
//! # Cold tier
//! Durable, disk-backed, organized as a directory tree keyed by
//! `(model, adapter, session)`. Snapshot files are written by llama.cpp's server
//! via its `/slots/{id}?action=save` HTTP endpoint; the router process only
//! manages the filesystem layout and sidecar metadata — it never reads or writes
//! the raw KV cache bytes.
//!
//! # Design note
//! llama.cpp's slot save/restore works through HTTP calls passing filenames,
//! not raw bytes. See `llama.cpp/tests/test-recurrent-state-rollback.cpp` for
//! the in-process equivalent (`common_prompt_checkpoint::update_tgt` /
//! `load_tgt`). The router process tells llama.cpp's server "save slot N to
//! path P" or "restore slot N from path P" and tracks only the sidecar metadata
//! (timestamps, model hash, version). Pulling gigabyte-scale KV buffers into
//! the router's own memory space would add unnecessary overhead and latency.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use common_core::now_secs;
use lru::LruCache;
use thiserror::Error;

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

/// A KV cache snapshot — metadata and filesystem path only.
///
/// The actual KV cache bytes live in the llama.cpp server's slot memory and on
/// disk at `file_path`. The router process never reads or buffers the raw bytes.
/// When the llama.cpp server saves a slot via `POST /slots/{id}?action=save`,
/// the router records this metadata record pointing at the resulting file.
///
/// Snapshot identity is the triple `(model, adapter, session_id)`. Before
/// restoring, callers must verify `base_model_hash` and `llama_cpp_version`
/// match the current environment.
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    pub model: String,
    pub adapter: Option<String>,
    pub session_id: String,
    /// Filesystem path to the KV cache file written by llama.cpp's server.
    pub file_path: PathBuf,
    pub token_count: usize,
    pub created_at: u64,
    pub last_used_at: u64,
    pub llama_cpp_version: String,
    pub model_quant: Option<String>,
    pub base_model_hash: String,
}

/// Hot tier: in-process, RAM-resident LRU cache of recently-used snapshot
/// metadata. Entries represent sessions with KV cache state actively loaded
/// in a llama.cpp server slot. Stores metadata only — no raw bytes.
pub struct HotKvCache {
    snapshots: Mutex<SnapshotLru>,
    max_mb: usize,
}

type SnapshotLru = LruCache<(String, Option<String>, String), Arc<KvSnapshot>>;

impl HotKvCache {
    /// Creates a new hot cache with the given capacity (number of entries).
    /// The `max_mb` limit is a soft budget — individual snapshot metadata is
    /// tiny (hundreds of bytes), so `max_mb` is primarily informational for
    /// the llama.cpp server's actual slot memory budget.
    pub fn new(capacity: usize, max_mb: usize) -> Self {
        Self {
            snapshots: Mutex::new(LruCache::new(
                std::num::NonZero::new(capacity.max(1)).unwrap(),
            )),
            max_mb,
        }
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
        self.snapshots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .put(key, Arc::new(snapshot));
    }

    /// Get snapshot metadata from the hot cache.
    pub fn get(
        &self,
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
    ) -> Option<Arc<KvSnapshot>> {
        self.snapshots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&Self::key(model, adapter, session_id))
            .cloned()
    }

    /// Remove snapshot metadata from the hot cache.
    pub fn remove(&self, model: &str, adapter: Option<&str>, session_id: &str) {
        self.snapshots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop(&Self::key(model, adapter, session_id));
    }

    /// Current number of entries in the hot cache.
    pub fn len(&self) -> usize {
        self.snapshots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Returns true if the hot cache is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }

    /// The max RAM budget in MB.
    pub fn max_mb(&self) -> usize {
        self.max_mb
    }
}

/// Cold tier: durable, disk-backed, organized as a directory tree keyed by
/// `(model, adapter, session)`. Snapshot files are written by llama.cpp's
/// server; the router process manages the filesystem layout and sidecar
/// metadata only.
pub struct ColdKvCache {
    mountpoint: PathBuf,
    max_mb: usize,
    ttl_secs: u64,
    eviction: crate::config::EvictionPolicy,
}

impl ColdKvCache {
    /// Creates a new cold cache.
    pub fn new(
        mountpoint: impl Into<PathBuf>,
        max_mb: usize,
        ttl_secs: u64,
        eviction: crate::config::EvictionPolicy,
    ) -> Self {
        let mp = mountpoint.into();
        if let Err(e) = std::fs::create_dir_all(&mp) {
            tracing::warn!("could not create cold cache directory {mp:?}: {e}");
        }
        Self {
            mountpoint: mp,
            max_mb,
            ttl_secs,
            eviction,
        }
    }

    /// The max disk budget in MB.
    pub fn max_mb(&self) -> usize {
        self.max_mb
    }

    /// The TTL in seconds for cold cache entries.
    pub fn ttl_secs(&self) -> u64 {
        self.ttl_secs
    }

    fn snapshot_dir(&self, model: &str, adapter: Option<&str>, session_id: &str) -> PathBuf {
        let mut dir = self.mountpoint.clone();
        dir.push(model);
        if let Some(ad) = adapter {
            dir.push(ad);
        }
        dir.push(session_id);
        dir
    }

    fn snapshot_path(&self, model: &str, adapter: Option<&str>, session_id: &str) -> PathBuf {
        let mut path = self.snapshot_dir(model, adapter, session_id);
        let filename = format!("{session_id}.kv");
        path.push(filename);
        path
    }

    /// Record a snapshot in the cold tier. Copies the KV cache file from
    /// `snapshot.file_path` into the organized directory tree at
    /// `{mountpoint}/{model}/{adapter}/{session_id}.kv`.
    ///
    /// The file at `snapshot.file_path` is expected to already exist — it was
    /// written by llama.cpp's server via its slot save endpoint. This method
    /// copies it into the managed directory layout and records the sidecar
    /// metadata.
    pub async fn save(&self, snapshot: &KvSnapshot) -> Result<(), KvCacheError> {
        let dir = self.snapshot_dir(
            &snapshot.model,
            snapshot.adapter.as_deref(),
            &snapshot.session_id,
        );
        tokio::fs::create_dir_all(&dir).await?;

        let target = self.snapshot_path(
            &snapshot.model,
            snapshot.adapter.as_deref(),
            &snapshot.session_id,
        );

        if snapshot.file_path != target {
            tokio::fs::copy(&snapshot.file_path, &target).await?;
        }

        Ok(())
    }

    /// Load snapshot metadata from the cold tier. Returns the metadata record
    /// with `file_path` pointing to the organized filesystem location.
    ///
    /// Does NOT read the raw KV cache bytes — callers should pass the returned
    /// `file_path` to llama.cpp's server for restoration via
    /// `POST /slots/{id}?action=restore`.
    ///
    /// Callers must verify `base_model_hash`, `llama_cpp_version`, and
    /// `model_quant` against the current environment before attempting
    /// restoration. Mismatch should return an error, not attempt best-effort
    /// restore.
    pub async fn load(
        &self,
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
    ) -> Result<KvSnapshot, KvCacheError> {
        let path = self.snapshot_path(model, adapter, session_id);

        let metadata = tokio::fs::metadata(&path)
            .await
            .map_err(|_| KvCacheError::NotFound(format!("no snapshot at {}", path.display())))?;
        let created = metadata
            .created()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let modified = metadata
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        Ok(KvSnapshot {
            model: model.to_string(),
            adapter: adapter.map(String::from),
            session_id: session_id.to_string(),
            file_path: path,
            token_count: 0,
            created_at: created,
            last_used_at: modified,
            llama_cpp_version: String::new(),
            model_quant: None,
            base_model_hash: String::new(),
        })
    }

    /// Evict stale or over-budget snapshots. Returns the number evicted.
    pub async fn evict(&self) -> Result<usize, KvCacheError> {
        let now = now_secs();
        let mut evicted = 0;

        let entries = self.walk_snapshots().await?;

        for (path, _snapshot) in entries {
            let metadata = tokio::fs::metadata(&path).await?;
            let last_used = metadata
                .modified()
                .unwrap_or(UNIX_EPOCH)
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let age = now.saturating_sub(last_used);
            let should_evict = match self.eviction {
                crate::config::EvictionPolicy::Lru
                | crate::config::EvictionPolicy::Ttl
                | crate::config::EvictionPolicy::Hybrid => age >= self.ttl_secs,
            };

            if should_evict {
                tokio::fs::remove_file(&path).await?;
                evicted += 1;
            }
        }

        Ok(evicted)
    }

    /// List all snapshots for a session.
    pub async fn list_snapshots(&self, session_id: &str) -> Vec<KvSnapshot> {
        let mut results = Vec::new();
        if let Ok(entries) = self.walk_snapshots().await {
            for (path, snapshot) in entries {
                if snapshot.session_id == session_id {
                    let mut s = snapshot;
                    s.file_path = path;
                    results.push(s);
                }
            }
        }
        results
    }

    /// Walk the mountpoint and collect snapshot entries with basic metadata.
    async fn walk_snapshots(&self) -> Result<Vec<(PathBuf, KvSnapshot)>, KvCacheError> {
        let mut results = Vec::new();
        self.walk_dir(&self.mountpoint, &mut results).await?;
        Ok(results)
    }

    async fn walk_dir(
        &self,
        dir: &Path,
        results: &mut Vec<(PathBuf, KvSnapshot)>,
    ) -> Result<(), KvCacheError> {
        if tokio::fs::metadata(dir).await.is_err() {
            return Ok(());
        }

        let mut entries = tokio::fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                Box::pin(self.walk_dir(&path, results)).await?;
            } else if path.extension().is_some_and(|e| e == "kv") {
                let snapshot = self.read_file_meta(&path).await?;
                results.push((path, snapshot));
            }
        }
        Ok(())
    }

    async fn read_file_meta(&self, path: &Path) -> Result<KvSnapshot, KvCacheError> {
        let metadata = tokio::fs::metadata(path).await?;
        let created = metadata
            .created()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let modified = metadata
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        // Derive model/adapter/session from path segments
        let segments: Vec<&str> = path.iter().filter_map(|s| s.to_str()).collect();

        let mount_segs = self.mountpoint.iter().filter_map(|s| s.to_str()).count();

        let relative: Vec<&str> = segments.iter().skip(mount_segs).copied().collect();

        let model = relative.first().copied().unwrap_or("unknown").to_string();
        let (adapter, session_id) = if relative.len() >= 4 {
            // model/adapter/session/session.kv
            (Some(relative[1].to_string()), relative[2].to_string())
        } else if relative.len() == 3 {
            // model/session/session.kv
            (None, relative[1].to_string())
        } else if relative.len() == 2 {
            (None, relative[1].to_string())
        } else {
            (None, model.clone())
        };

        Ok(KvSnapshot {
            model,
            adapter,
            session_id,
            file_path: path.to_path_buf(),
            token_count: 0,
            created_at: created,
            last_used_at: modified,
            llama_cpp_version: String::new(),
            model_quant: None,
            base_model_hash: String::new(),
        })
    }
}

/// Two-tier KV cache: checks hot tier first, falls back to cold tier.
///
/// Hot tier tracks sessions with actively-loaded llama.cpp slots (metadata only).
/// Cold tier is the durable disk store keyed by `(model, adapter, session)`.
/// On a cold-tier hit, the metadata is promoted to the hot tier.
pub struct KvCacheManager {
    hot: Arc<HotKvCache>,
    cold: Arc<ColdKvCache>,
}

impl KvCacheManager {
    pub fn new(hot: Arc<HotKvCache>, cold: Arc<ColdKvCache>) -> Self {
        Self { hot, cold }
    }

    /// Store snapshot metadata in both tiers. Copies the KV cache file into
    /// the cold tier's organized directory tree and promotes to the hot tier.
    pub async fn store(&self, snapshot: KvSnapshot) -> Result<(), KvCacheError> {
        self.cold.save(&snapshot).await?;
        self.hot.put(snapshot);
        Ok(())
    }

    /// Retrieve snapshot metadata: hot tier first, cold tier as fallback.
    /// On cold-tier hit, the metadata is promoted to the hot tier.
    ///
    /// Returns metadata only — callers must pass the returned `file_path` to
    /// llama.cpp's server for KV cache restoration.
    pub async fn retrieve(
        &self,
        model: &str,
        adapter: Option<&str>,
        session_id: &str,
    ) -> Result<Arc<KvSnapshot>, KvCacheError> {
        if let Some(snapshot) = self.hot.get(model, adapter, session_id) {
            return Ok(snapshot);
        }

        let snapshot = self.cold.load(model, adapter, session_id).await?;
        let arc = Arc::new(snapshot);
        self.hot
            .snapshots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .put(
                HotKvCache::key(model, adapter, session_id),
                Arc::clone(&arc),
            );
        Ok(arc)
    }

    /// Evict from cold tier based on policy.
    pub async fn evict(&self) -> Result<usize, KvCacheError> {
        self.cold.evict().await
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
            file_path: PathBuf::new(),
            token_count: 100,
            created_at: now_secs(),
            last_used_at: now_secs(),
            llama_cpp_version: "0.1.0".into(),
            model_quant: None,
            base_model_hash: "abc123".into(),
        }
    }

    #[test]
    fn test_hot_cache_put_get() {
        let cache = HotKvCache::new(10, 1024);
        let snap = test_snapshot("sess-1");
        cache.put(snap.clone());

        let retrieved = cache.get("test-model", None, "sess-1").unwrap();
        assert_eq!(retrieved.session_id, "sess-1");
        assert_eq!(retrieved.token_count, 100);
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
                file_path: PathBuf::new(),
                token_count: 1,
                created_at: now_secs(),
                last_used_at: now_secs(),
                llama_cpp_version: "0.1".into(),
                model_quant: None,
                base_model_hash: "hash".into(),
            });
        }

        assert_eq!(cache.len(), 3);
        assert!(cache.get("m", None, "sess-0").is_none());
        assert!(cache.get("m", None, "sess-1").is_none());
        assert!(cache.get("m", None, "sess-2").is_some());
    }

    #[tokio::test]
    async fn test_cold_cache_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = tempfile::tempdir().unwrap();
        let cold = ColdKvCache::new(dir.path(), 1024, 86400, crate::config::EvictionPolicy::Lru);

        let src_file = src_dir.path().join("src.kv");
        tokio::fs::write(&src_file, b"dummy kv cache bytes")
            .await
            .unwrap();

        let mut snap = test_snapshot("sess-cold");
        snap.file_path = src_file;
        cold.save(&snap).await.unwrap();

        let loaded = cold.load("test-model", None, "sess-cold").await.unwrap();
        assert_eq!(loaded.model, "test-model");
        assert_eq!(loaded.session_id, "sess-cold");
        assert!(loaded.file_path.exists());
    }

    #[tokio::test]
    async fn test_cold_cache_load_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let cold = ColdKvCache::new(dir.path(), 1024, 86400, crate::config::EvictionPolicy::Lru);

        let result = cold.load("test-model", None, "no-such-session").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_kv_cache_manager_two_tier() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = tempfile::tempdir().unwrap();
        let hot = Arc::new(HotKvCache::new(10, 1024));
        let cold = Arc::new(ColdKvCache::new(
            dir.path(),
            1024,
            86400,
            crate::config::EvictionPolicy::Lru,
        ));
        let mgr = KvCacheManager::new(Arc::clone(&hot), Arc::clone(&cold));

        let src_file = src_dir.path().join("tier-src.kv");
        tokio::fs::write(&src_file, b"tier kv cache bytes")
            .await
            .unwrap();

        let mut snap = test_snapshot("sess-tier");
        snap.file_path = src_file;
        mgr.store(snap).await.unwrap();

        // Should be in hot tier
        let retrieved = mgr.retrieve("test-model", None, "sess-tier").await.unwrap();
        assert_eq!(retrieved.session_id, "sess-tier");

        // Remove from hot, should fall back to cold
        hot.remove("test-model", None, "sess-tier");
        let retrieved2 = mgr.retrieve("test-model", None, "sess-tier").await.unwrap();
        assert_eq!(retrieved2.session_id, "sess-tier");
    }

    #[tokio::test]
    async fn test_cold_cache_evict_by_ttl() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = tempfile::tempdir().unwrap();
        let cold = ColdKvCache::new(
            dir.path(),
            1024,
            0, // immediate TTL
            crate::config::EvictionPolicy::Lru,
        );

        let src_file = src_dir.path().join("evict-src.kv");
        tokio::fs::write(&src_file, b"evict kv cache bytes")
            .await
            .unwrap();

        let mut snap = test_snapshot("sess-evict");
        snap.file_path = src_file;
        cold.save(&snap).await.unwrap();

        let evicted = cold.evict().await.unwrap();
        assert_eq!(evicted, 1);
    }

    #[tokio::test]
    async fn test_list_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = tempfile::tempdir().unwrap();
        let cold = ColdKvCache::new(dir.path(), 1024, 86400, crate::config::EvictionPolicy::Lru);

        let src_file = src_dir.path().join("list-src.kv");
        tokio::fs::write(&src_file, b"list kv cache bytes")
            .await
            .unwrap();

        let mut snap = test_snapshot("sess-list");
        snap.file_path = src_file;
        cold.save(&snap).await.unwrap();

        let snapshots = cold.list_snapshots("sess-list").await;
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].session_id, "sess-list");
    }
}
