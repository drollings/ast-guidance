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
#[path = "../tests/query_cache.rs"]
mod tests;
