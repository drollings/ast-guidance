use super::*;
use spacy_rs::CorrectionField;

fn open() -> (Arc<SqliteStore>, SqliteSpanCache) {
    let store = Arc::new(SqliteStore::open_in_memory().expect("in-memory"));
    store
        .with_conn(|conn| fluent_db::migrate::migrate(conn, &crate::ledger::ledger_migrations()))
        .expect("migrate");
    let cache = SqliteSpanCache::new(Arc::clone(&store));
    (store, cache)
}

fn corr() -> Vec<Correction> {
    vec![Correction {
        token_index: 1,
        field: CorrectionField::Pos,
        old_value: String::new(),
        new_value: "verb".into(),
    }]
}

#[test]
fn put_get_invalidate_roundtrip() {
    let (_store, cache) = open();
    let key = 0x1234_5678_9abc_def0u64;
    assert!(cache.get(key).is_none());
    cache.put(key, corr());
    assert_eq!(cache.get(key).unwrap(), corr());
    assert_eq!(cache.len(), 1);
    cache.invalidate(key);
    assert!(cache.get(key).is_none());
    assert_eq!(cache.len(), 0);
}

#[test]
fn key_zero_is_noop() {
    let (_store, cache) = open();
    cache.put(0, corr());
    assert_eq!(cache.len(), 0);
    assert!(cache.get(0).is_none());
}

#[test]
fn span_cache_high_key_roundtrip() {
    let (_store, cache) = open();
    let key = 0xFFFF_FFFF_FFFF_FFFFu64;
    cache.put(key, corr());
    assert_eq!(cache.get(key).unwrap(), corr());
    assert_eq!(cache.len(), 1);
    let key2 = 0x8000_0000_0000_0001u64;
    cache.put(key2, corr());
    assert_eq!(cache.get(key2).unwrap(), corr());
    assert_eq!(cache.len(), 2);
}

#[test]
fn span_cache_hex_is_fixed_width() {
    assert_eq!(format!("{:016x}", 1u64), "0000000000000001");
    assert_eq!(super::hex_key(1u64), "0000000000000001");
    assert_eq!(super::hex_key(0xFFFF_FFFF_FFFF_FFFFu64), "ffffffffffffffff");
}

#[test]
fn in_memory_double_roundtrips() {
    // The hermetic double moved with the trait (M2b): same contract, no sqlite.
    let cache = InMemorySpanCache::new();
    assert!(cache.is_empty());
    let key = 42u64;
    assert!(cache.get(key).is_none());
    cache.put(key, corr());
    assert_eq!(cache.get(key).unwrap(), corr());
    assert_eq!(cache.len(), 1);
    cache.invalidate(key);
    assert!(cache.get(key).is_none());
    assert!(cache.is_empty());
    // Key 0 is never stored.
    cache.put(0, corr());
    assert!(cache.is_empty());
}

#[test]
fn seam_adapter_shares_the_backend() {
    // M2b seam proof: the router adapts its `Arc<dyn SpanCache>` into the
    // dependency-free spacy-rs seam; writes through one handle are visible
    // through the other (no behavior change for ladder callers).
    let (_store, sqlite) = open();
    let backend: Arc<dyn SpanCache> = Arc::new(sqlite);
    let seam = span_cache_seam(&backend);
    let key = 0x1234_5678_9abc_def1u64;
    assert!(seam.get(key).is_none());
    // Write via the seam → visible via the trait.
    seam.put(key, corr());
    assert_eq!(backend.get(key).unwrap(), corr());
    // Invalidate via the seam → gone via the trait (the
    // CorrectionIndex-invalidation contract).
    seam.invalidate(key);
    assert!(backend.get(key).is_none());
    // And the reverse direction: trait write → seam read.
    backend.put(key, corr());
    assert_eq!(seam.get(key).unwrap(), corr());
    seam.invalidate(key);
    assert!(seam.get(key).is_none());
}

#[test]
fn seam_adapter_shares_the_memory_backend() {
    let backend: Arc<dyn SpanCache> = Arc::new(InMemorySpanCache::new());
    let seam = span_cache_seam(&backend);
    let key = 7u64;
    seam.put(key, corr());
    assert_eq!(backend.get(key).unwrap(), corr());
    backend.invalidate(key);
    assert!(seam.get(key).is_none());
}
