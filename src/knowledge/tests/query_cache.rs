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
