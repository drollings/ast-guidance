//! TTL/LRU query cache delegating to `fluent_db::cache::TtlCache` (D7).
//!
//! The eviction/expiry logic and `query_cache` table now live in `fluent-db`;
//! this module keeps the knowledge-specific lowercase-`fnv1a64` key
//! derivation and the domain `QueryCache`/`Entry` types.

use common_core::hash::fnv1a64;
use fluent_db::cache::TtlCache;
use fluent_db::error::DbError;
use std::path::Path;

#[derive(Debug)]
pub struct Entry {
    pub query: String,
    pub result_summary: String,
    pub timestamp: u64,
    pub ttl_seconds: u64,
}

/// Knowledge-specific key derivation: lowercase the query, then hash with
/// `fnv1a64` and hex-encode.
fn query_key(query: &str) -> String {
    let hash = fnv1a64(query.to_lowercase().as_bytes());
    format!("{hash:016x}")
}

pub struct QueryCache {
    inner: TtlCache,
}

impl QueryCache {
    pub fn new(db_path: &Path, default_ttl_seconds: u64) -> Result<Self, DbError> {
        Self::with_max_entries(db_path, default_ttl_seconds, 4096)
    }

    pub fn with_max_entries(
        db_path: &Path,
        default_ttl_seconds: u64,
        max_entries: usize,
    ) -> Result<Self, DbError> {
        Ok(Self {
            inner: TtlCache::open_with_key(db_path, default_ttl_seconds, max_entries, query_key)?,
        })
    }

    pub fn new_in_memory(default_ttl_seconds: u64) -> Result<Self, DbError> {
        Ok(Self {
            inner: TtlCache::open_in_memory_with_key(default_ttl_seconds, query_key)?,
        })
    }

    pub fn get(&self, query: &str) -> Result<Option<Entry>, DbError> {
        Ok(self.inner.get(query)?.map(|e| Entry {
            query: e.key,
            result_summary: e.result_summary,
            timestamp: e.timestamp,
            ttl_seconds: e.ttl_seconds,
        }))
    }

    pub fn put(&self, query: &str, result: &str) -> Result<(), DbError> {
        self.inner.put(query, result)
    }

    /// Remove all expired entries. Returns the number removed.
    pub fn evict_expired(&self) -> Result<usize, DbError> {
        self.inner.evict_expired()
    }

    /// Remove every entry. Returns the number removed.
    pub fn clear(&self) -> Result<usize, DbError> {
        self.inner.clear()
    }

    pub fn stats(&self) -> Result<(usize, u64, usize), DbError> {
        self.inner.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_wvr_testutil::tempdir;

    fn setup() -> (QueryCache, std::path::PathBuf) {
        let dir = tempdir();
        let db_path = dir.path().join("cache.db");
        let cache = QueryCache::new(&db_path, 3600).unwrap();
        (cache, dir.keep())
    }

    #[test]
    fn put_and_get() {
        let (cache, _dir) = setup();
        cache.put("test query", "test result").unwrap();
        let entry = cache.get("test query").unwrap().unwrap();
        assert_eq!(entry.result_summary, "test result");
    }

    #[test]
    fn stats_work() {
        let (cache, _dir) = setup();
        let (count, ttl, _expired) = cache.stats().unwrap();
        assert_eq!(count, 0);
        assert_eq!(ttl, 3600);
        cache.put("q", "r").unwrap();
        let (count, _, _) = cache.stats().unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn clear_works() {
        let (cache, _dir) = setup();
        cache.put("q", "r").unwrap();
        cache.clear().unwrap();
        assert!(cache.get("q").unwrap().is_none());
    }

    #[test]
    fn lru_eviction_works() {
        let dir = tempdir();
        let db_path = dir.path().join("cache.db");
        let cache = QueryCache::with_max_entries(&db_path, 3600, 3).unwrap();

        cache.put("q1", "r1").unwrap();
        cache.put("q2", "r2").unwrap();
        cache.put("q3", "r3").unwrap();
        cache.put("q4", "r4").unwrap(); // should evict q1

        let (count, _, _) = cache.stats().unwrap();
        assert!(
            count <= 3,
            "expected <= 3 entries after LRU eviction, got {count}"
        );
    }

    #[test]
    fn expired_entry_returns_none() {
        let dir = tempdir();
        let db_path = dir.path().join("cache.db");
        // TTL of 0 seconds means immediately expired
        let cache = QueryCache::with_max_entries(&db_path, 0, 4096).unwrap();

        cache.put("expired", "data").unwrap();
        let entry = cache.get("expired").unwrap();
        assert!(entry.is_none(), "expired entry should return None");
    }

    #[test]
    fn evict_expired_works() {
        let dir = tempdir();
        let db_path = dir.path().join("cache.db");
        let cache = QueryCache::with_max_entries(&db_path, 0, 4096).unwrap();

        cache.put("a", "1").unwrap();
        cache.put("b", "2").unwrap();
        let evicted = cache.evict_expired().unwrap();
        assert!(evicted >= 2, "should evict expired entries");

        let (count, _, _) = cache.stats().unwrap();
        assert_eq!(count, 0);
    }
}
