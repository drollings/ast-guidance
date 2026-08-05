//! Generic TTL/LRU key-value cache store (D7).
//!
//! `TtlCache` is the canonical home for the TTL/LRU SQLite cache shape that
//! `knowledge::QueryCache` hand-rolled (open, get-with-expiry, put,
//! evict_expired, evict_lru, stats). The `query_cache` table DDL and all
//! eviction SQL move here verbatim so existing `.db` files keep working
//! unchanged and behavior stays byte-identical.
//!
//! Key derivation is parameterized via `hash_key: fn(&str) -> String` so
//! consumer-specific key schemes (e.g. knowledge's lowercase-`fnv1a64`)
//! remain configurable; the default is the identity function.

use std::path::Path;

use common_core::time::now_secs;

use crate::error::DbError;
use crate::store::SqliteStore;

/// Table DDL — kept under the historical `query_cache` name so databases
/// written by the old in-crate implementation keep working unchanged.
const QUERY_CACHE_DDL: &str = "CREATE TABLE IF NOT EXISTS query_cache (
    key TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    ttl_seconds INTEGER NOT NULL
)";

/// A single cached entry.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Stored key (post-`hash_key` derivation).
    pub key: String,
    /// The cached result payload.
    pub result_summary: String,
    /// Unix timestamp of insertion.
    pub timestamp: u64,
    /// Time-to-live in seconds (`0` means immediately expired).
    pub ttl_seconds: u64,
}

/// Default key derivation: store the raw key.
fn identity_key(key: &str) -> String {
    key.to_string()
}

/// A TTL/LRU key-value cache over a single-connection SQLite store.
pub struct TtlCache {
    store: SqliteStore,
    default_ttl_seconds: u64,
    max_entries: usize,
    hash_key: fn(&str) -> String,
}

impl std::fmt::Debug for TtlCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtlCache")
            .field("default_ttl_seconds", &self.default_ttl_seconds)
            .field("max_entries", &self.max_entries)
            .finish_non_exhaustive()
    }
}

impl TtlCache {
    /// Open (or create) a cache at `path` with the default (identity) key
    /// derivation. Expired entries are evicted on open.
    pub fn open(
        path: &Path,
        default_ttl_seconds: u64,
        max_entries: usize,
    ) -> Result<Self, DbError> {
        Self::open_with_key(path, default_ttl_seconds, max_entries, identity_key)
    }

    /// Like [`TtlCache::open`] with a consumer-provided key derivation.
    pub fn open_with_key(
        path: &Path,
        default_ttl_seconds: u64,
        max_entries: usize,
        hash_key: fn(&str) -> String,
    ) -> Result<Self, DbError> {
        let store = SqliteStore::open(path)?;
        store.init_schema(QUERY_CACHE_DDL)?;
        let cache = Self {
            store,
            default_ttl_seconds,
            max_entries,
            hash_key,
        };
        cache.evict_expired()?;
        Ok(cache)
    }

    /// Open an in-memory cache with the default (identity) key derivation.
    pub fn open_in_memory(default_ttl_seconds: u64) -> Result<Self, DbError> {
        Self::open_in_memory_with_key(default_ttl_seconds, identity_key)
    }

    /// Like [`TtlCache::open_in_memory`] with a consumer-provided key
    /// derivation.
    pub fn open_in_memory_with_key(
        default_ttl_seconds: u64,
        hash_key: fn(&str) -> String,
    ) -> Result<Self, DbError> {
        let store = SqliteStore::open_in_memory()?;
        store.init_schema(QUERY_CACHE_DDL)?;
        Ok(Self {
            store,
            default_ttl_seconds,
            max_entries: 4096,
            hash_key,
        })
    }

    /// Fetch a live entry for `key`. An expired entry (TTL `0`, or
    /// `now > timestamp + ttl`) is deleted and treated as a miss.
    pub fn get(&self, key: &str) -> Result<Option<Entry>, DbError> {
        let hashed = (self.hash_key)(key);
        let entry = self.store.query_row(
            "SELECT key, result_json, timestamp, ttl_seconds FROM query_cache WHERE key = ?1",
            rusqlite::params![hashed],
            |row| {
                Ok(Entry {
                    key: row.get(0)?,
                    result_summary: row.get(1)?,
                    timestamp: row.get(2)?,
                    ttl_seconds: row.get(3)?,
                })
            },
        )?;
        match entry {
            Some(entry) => {
                let now = now_secs();
                if entry.ttl_seconds == 0 || now > entry.timestamp + entry.ttl_seconds {
                    self.store.execute(
                        "DELETE FROM query_cache WHERE key = ?1",
                        rusqlite::params![entry.key],
                    )?;
                    return Ok(None);
                }
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    /// Insert (or replace) `result` under `key`, then evict LRU entries when
    /// over capacity.
    pub fn put(&self, key: &str, result: &str) -> Result<(), DbError> {
        let hashed = (self.hash_key)(key);
        let now = now_secs();
        self.store.execute(
            "INSERT OR REPLACE INTO query_cache (key, result_json, timestamp, ttl_seconds) \
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![hashed, result, now, self.default_ttl_seconds],
        )?;
        self.evict_lru()?;
        Ok(())
    }

    /// Remove all expired entries (TTL `0`, or `now > timestamp + ttl`).
    /// Returns the number of rows deleted.
    pub fn evict_expired(&self) -> Result<usize, DbError> {
        let now = now_secs();
        self.store.execute(
            "DELETE FROM query_cache WHERE ttl_seconds = 0 OR ?1 > timestamp + ttl_seconds",
            rusqlite::params![now],
        )
    }

    /// Evict the oldest entries (LRU by timestamp) when over capacity.
    /// Returns the number of rows deleted.
    pub fn evict_lru(&self) -> Result<usize, DbError> {
        let count: usize = self
            .store
            .query_row("SELECT COUNT(*) FROM query_cache", &[], |row| row.get(0))?
            .unwrap_or(0);
        if count > self.max_entries {
            let excess = count - self.max_entries;
            self.store.execute(
                "DELETE FROM query_cache WHERE key IN (
                    SELECT key FROM query_cache ORDER BY timestamp ASC LIMIT ?1
                )",
                rusqlite::params![excess],
            )
        } else {
            Ok(0)
        }
    }

    /// Remove every entry. Returns the number of rows deleted.
    pub fn clear(&self) -> Result<usize, DbError> {
        self.store.execute("DELETE FROM query_cache", &[])
    }

    /// `(total entries, default ttl seconds, currently-expired entries)`.
    pub fn stats(&self) -> Result<(usize, u64, usize), DbError> {
        let count: usize = self
            .store
            .query_row("SELECT COUNT(*) FROM query_cache", &[], |row| row.get(0))?
            .unwrap_or(0);
        let expired: usize = {
            let now = now_secs();
            self.store
                .query_row(
                    "SELECT COUNT(*) FROM query_cache WHERE ?1 > timestamp + ttl_seconds",
                    rusqlite::params![now],
                    |row| row.get(0),
                )?
                .unwrap_or(0)
        };
        Ok((count, self.default_ttl_seconds, expired))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn put_and_get() {
        let dir = tempdir();
        let cache = TtlCache::open(&dir.path().join("cache.db"), 3600, 4096).unwrap();
        cache.put("test query", "test result").unwrap();
        let entry = cache.get("test query").unwrap().unwrap();
        assert_eq!(entry.result_summary, "test result");
    }

    #[test]
    fn stats_work() {
        let dir = tempdir();
        let cache = TtlCache::open(&dir.path().join("cache.db"), 3600, 4096).unwrap();
        let (count, ttl, _expired) = cache.stats().unwrap();
        assert_eq!(count, 0);
        assert_eq!(ttl, 3600);
        cache.put("q", "r").unwrap();
        let (count, _, _) = cache.stats().unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn clear_works() {
        let dir = tempdir();
        let cache = TtlCache::open(&dir.path().join("cache.db"), 3600, 4096).unwrap();
        cache.put("q", "r").unwrap();
        cache.clear().unwrap();
        assert!(cache.get("q").unwrap().is_none());
    }

    #[test]
    fn lru_eviction_works() {
        let dir = tempdir();
        let cache = TtlCache::open(&dir.path().join("cache.db"), 3600, 3).unwrap();
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
        // TTL of 0 seconds means immediately expired.
        let cache = TtlCache::open(&dir.path().join("cache.db"), 0, 4096).unwrap();
        cache.put("expired", "data").unwrap();
        let entry = cache.get("expired").unwrap();
        assert!(entry.is_none(), "expired entry should return None");
    }

    #[test]
    fn evict_expired_works() {
        let dir = tempdir();
        let cache = TtlCache::open(&dir.path().join("cache.db"), 0, 4096).unwrap();
        cache.put("a", "1").unwrap();
        cache.put("b", "2").unwrap();
        let evicted = cache.evict_expired().unwrap();
        assert!(evicted >= 2, "should evict expired entries");
        let (count, _, _) = cache.stats().unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn open_in_memory_round_trips() {
        let cache = TtlCache::open_in_memory(3600).unwrap();
        cache.put("mem", "value").unwrap();
        let entry = cache.get("mem").unwrap().unwrap();
        assert_eq!(entry.result_summary, "value");
    }

    #[test]
    fn hash_key_is_configurable() {
        let dir = tempdir();
        fn upper_key(key: &str) -> String {
            key.to_uppercase()
        }
        let cache =
            TtlCache::open_with_key(&dir.path().join("cache.db"), 3600, 4096, upper_key).unwrap();
        cache.put("query", "r").unwrap();
        // Key derivation is applied symmetrically: any input that derives to
        // the same key resolves the entry, and the stored key is the derived
        // form, not the raw query.
        let entry = cache.get("QUERY").unwrap().unwrap();
        assert_eq!(entry.key, "QUERY");
        assert_eq!(entry.result_summary, "r");
        let entry2 = cache.get("query").unwrap().unwrap();
        assert_eq!(entry2.result_summary, "r");
    }
}
