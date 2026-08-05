use common_core::hash::fnv1a64;
use common_core::time::now_secs;
use rusqlite::{params, Connection};
use std::path::Path;

#[derive(Debug)]
pub struct Entry {
    pub query: String,
    pub result_summary: String,
    pub timestamp: u64,
    pub ttl_seconds: u64,
}

pub struct QueryCache {
    db: Connection,
    default_ttl_seconds: u64,
    max_entries: usize,
}

impl QueryCache {
    pub fn new(db_path: &Path, default_ttl_seconds: u64) -> rusqlite::Result<Self> {
        Self::with_max_entries(db_path, default_ttl_seconds, 4096)
    }

    pub fn with_max_entries(
        db_path: &Path,
        default_ttl_seconds: u64,
        max_entries: usize,
    ) -> rusqlite::Result<Self> {
        let db = common_core::sqlite::open_wal(db_path)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS query_cache (
                key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                ttl_seconds INTEGER NOT NULL
            )",
        )?;
        let cache = Self {
            db,
            default_ttl_seconds,
            max_entries,
        };
        cache.evict_expired()?;
        Ok(cache)
    }

    pub fn new_in_memory(default_ttl_seconds: u64) -> rusqlite::Result<Self> {
        let db = common_core::sqlite::open_in_memory()?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS query_cache (
                key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                ttl_seconds INTEGER NOT NULL
            )",
        )?;
        Ok(Self {
            db,
            default_ttl_seconds,
            max_entries: 4096,
        })
    }

    fn query_key(query: &str) -> String {
        let hash = fnv1a64(query.to_lowercase().as_bytes());
        format!("{hash:016x}")
    }

    pub fn get(&self, query: &str) -> rusqlite::Result<Option<Entry>> {
        let key = Self::query_key(query);
        let mut stmt = self.db.prepare(
            "SELECT key, result_json, timestamp, ttl_seconds FROM query_cache WHERE key = ?1",
        )?;
        let result = stmt.query_row(params![key], |row| {
            Ok(Entry {
                query: row.get(0)?,
                result_summary: row.get(1)?,
                timestamp: row.get(2)?,
                ttl_seconds: row.get(3)?,
            })
        });
        match result {
            Ok(entry) => {
                let now = now_secs();
                // TTL=0 means always expired; otherwise check timestamp
                if entry.ttl_seconds == 0 || now > entry.timestamp + entry.ttl_seconds {
                    // Expired — remove it
                    self.db
                        .execute("DELETE FROM query_cache WHERE key = ?1", params![key])?;
                    return Ok(None);
                }
                Ok(Some(entry))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn put(&self, query: &str, result: &str) -> rusqlite::Result<()> {
        let key = Self::query_key(query);
        let now = now_secs();
        self.db.execute(
            "INSERT OR REPLACE INTO query_cache (key, result_json, timestamp, ttl_seconds) VALUES (?1, ?2, ?3, ?4)",
            params![key, result, now, self.default_ttl_seconds],
        )?;

        // Evict if over capacity
        self.evict_lru()?;

        Ok(())
    }

    /// Remove all expired entries (TTL=0 or timestamp+ttl < now).
    pub fn evict_expired(&self) -> rusqlite::Result<usize> {
        let now = now_secs();
        let count = self.db.execute(
            "DELETE FROM query_cache WHERE ttl_seconds = 0 OR ?1 > timestamp + ttl_seconds",
            params![now],
        )?;
        Ok(count)
    }

    /// Evict oldest entries when over capacity (LRU by timestamp).
    fn evict_lru(&self) -> rusqlite::Result<()> {
        let count: usize = self
            .db
            .query_row("SELECT COUNT(*) FROM query_cache", [], |row| row.get(0))?;

        if count > self.max_entries {
            let excess = count - self.max_entries;
            self.db.execute(
                "DELETE FROM query_cache WHERE key IN (
                    SELECT key FROM query_cache ORDER BY timestamp ASC LIMIT ?1
                )",
                params![excess],
            )?;
        }
        Ok(())
    }

    pub fn clear(&self) -> rusqlite::Result<()> {
        self.db.execute("DELETE FROM query_cache", [])?;
        Ok(())
    }

    pub fn stats(&self) -> rusqlite::Result<(usize, u64, usize)> {
        let count: usize = self
            .db
            .query_row("SELECT COUNT(*) FROM query_cache", [], |row| row.get(0))?;
        let expired: usize = {
            let now = now_secs();
            self.db.query_row(
                "SELECT COUNT(*) FROM query_cache WHERE ?1 > timestamp + ttl_seconds",
                params![now],
                |row| row.get(0),
            )?
        };
        Ok((count, self.default_ttl_seconds, expired))
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
