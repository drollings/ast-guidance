use super::*;
use crate::vector::vec_to_bytes;

#[test]
fn empty_index_is_not_built() {
    let idx = HnswIndex::new();
    assert!(!idx.is_built());
    assert_eq!(idx.len(), 0);
    assert!(idx.search(&[1.0, 0.0], 5).is_empty());
}

#[test]
fn insert_then_search_returns_nearest() {
    let idx = HnswIndex::new();
    let id_a = idx.insert(101, &[1.0, 0.0, 0.0]);
    let id_b = idx.insert(202, &[0.0, 1.0, 0.0]);
    assert_eq!(id_a, 0);
    assert_eq!(id_b, 1);
    assert!(idx.is_built());
    assert_eq!(idx.len(), 2);
    assert_eq!(idx.id_map_snapshot(), vec![101, 202]);

    let nearest = idx.search(&[1.0, 0.1, 0.0], 1);
    assert_eq!(nearest.len(), 1);
    let (external, _dist) = nearest[0];
    let node_id = idx.id_map_snapshot()[external];
    assert_eq!(node_id, 101);
}

#[test]
fn rebuild_from_round_trips() {
    let idx = HnswIndex::new();
    let rows = vec![
        (7, vec_to_bytes(&[1.0, 0.0])),
        (8, vec_to_bytes(&[0.0, 1.0])),
        (9, vec_to_bytes(&[0.8, 0.2])),
    ];
    let count = idx
        .rebuild_from(rows.into_iter(), |b| Some(crate::vector::bytes_to_vec(b)))
        .unwrap();
    assert_eq!(count, 3);
    assert_eq!(idx.len(), 3);
    assert_eq!(idx.id_map_snapshot(), vec![7, 8, 9]);

    let nearest = idx.search(&[1.0, 0.05], 1);
    let external = nearest[0].0;
    assert_eq!(idx.id_map_snapshot()[external], 7);
}

#[test]
fn rebuild_from_skips_bad_blobs() {
    let idx = HnswIndex::new();
    let rows = vec![
        (1, vec_to_bytes(&[1.0, 0.0])),
        (2, vec![0u8; 3]), // not a multiple of 4 -> decode None
        (3, Vec::new()),   // empty -> skipped
    ];
    let count = idx
        .rebuild_from(rows.into_iter(), crate::vector::try_bytes_to_vec)
        .unwrap();
    assert_eq!(count, 3, "total rows examined");
    assert_eq!(idx.len(), 1, "only the valid blob is indexed");
    assert_eq!(idx.id_map_snapshot(), vec![1]);
}

#[test]
fn insert_after_rebuild_extends_id_map() {
    let idx = HnswIndex::new();
    idx.insert(1, &[1.0, 0.0]);
    idx.rebuild_from(std::iter::once((5, vec_to_bytes(&[0.0, 1.0]))), |b| {
        Some(crate::vector::bytes_to_vec(b))
    })
    .unwrap();
    assert_eq!(idx.id_map_snapshot(), vec![5]);
    idx.insert(9, &[1.0, 0.0]);
    assert_eq!(idx.id_map_snapshot(), vec![5, 9]);
    assert_eq!(idx.len(), 2);
}

#[test]
fn poisoned_hnsw_rwlock_still_serves_insert_and_search() {
    // The hnsw `RwLock` must recover from poison via
    // `common_core::sync::lock_write` / `lock_read`. A panic while holding
    // a write guard obtained via `.write().unwrap()` poisons the lock; a
    // subsequent `insert`/`search` must still work.
    let idx = HnswIndex::new();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = idx.hnsw.write().unwrap();
        panic!("boom");
    }));
    assert!(panic.is_err(), "expected the closure to panic");
    idx.insert(101, &[1.0, 0.0, 0.0]);
    let nearest = idx.search(&[1.0, 0.1, 0.0], 1);
    assert_eq!(nearest.len(), 1);
    assert_eq!(idx.id_map_snapshot()[nearest[0].0], 101);
    assert!(idx.is_built());
    assert_eq!(idx.len(), 1);
}

#[test]
fn adaptive_hnsw_boundary_is_strict() {
    // M6: the one threshold source is `DEFAULT_HNSW_THRESHOLD`; the boundary
    // is strict `>`: len == threshold stays brute-force, threshold+1 uses HNSW.
    use common_core::constants::DEFAULT_HNSW_THRESHOLD;
    let policy = AdaptiveHnsw::default();
    assert_eq!(policy.threshold, DEFAULT_HNSW_THRESHOLD);
    assert!(!policy.should_use_built(DEFAULT_HNSW_THRESHOLD));
    assert!(policy.should_use_built(DEFAULT_HNSW_THRESHOLD + 1));
    assert!(!policy.should_use_built(0));
}

#[test]
fn adaptive_hnsw_dispatch_needs_built_and_scale() {
    // M6: query-time gate — unbuilt probes nothing at any scale; a built
    // index below the threshold still routes to brute force.
    let policy = AdaptiveHnsw::default();
    let t = policy.threshold;
    assert!(!policy.dispatch(false, t + 1), "unbuilt → brute force at any scale");
    assert!(!policy.dispatch(false, 0));
    assert!(!policy.dispatch(true, t), "built but at/below threshold → brute force");
    assert!(policy.dispatch(true, t + 1));
    // A custom threshold is honored; it is still a single source (the field).
    let custom = AdaptiveHnsw::new(10);
    assert_eq!(custom.threshold, 10);
    assert!(!custom.should_use_built(10));
    assert!(custom.should_use_built(11));
    assert!(custom.dispatch(true, 11));
}

#[test]
fn hnsw_lookup_unbuilt_is_none() {
    // M5.3: store-independent tests — the fallback signal contract.
    let idx = HnswIndex::new();
    assert_eq!(hnsw_lookup(&idx, &[1.0, 0.0], 5), None);
}

#[test]
fn hnsw_lookup_empty_query_is_none() {
    // M0: an empty probe is malformed → the caller must fall back
    // (mirrors the `k == 0` arm of the lookup contract).
    let idx = HnswIndex::new();
    idx.insert(101, &[1.0, 0.0, 0.0]);
    idx.insert(202, &[0.0, 1.0, 0.0]);
    assert_eq!(hnsw_lookup(&idx, &[], 5), None);
}

#[test]
fn poisoned_hnsw_rwlock_still_serves_hnsw_lookup() {
    // M0: mirrors `poisoned_hnsw_rwlock_still_serves_insert_and_search` —
    // `hnsw_lookup` (is_built → search → id_map_snapshot) must also recover
    // via `common_core::sync::lock_read` / `lock`.
    let idx = HnswIndex::new();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = idx.hnsw.write().unwrap();
        panic!("boom");
    }));
    assert!(panic.is_err(), "expected the closure to panic");
    idx.insert(101, &[1.0, 0.0, 0.0]);
    let hits = hnsw_lookup(&idx, &[1.0, 0.1, 0.0], 1).expect("lookup recovers after poison");
    assert_eq!(hits[0].0, 101, "caller key, not the external idx");
}

#[test]
fn hnsw_lookup_resolves_keys_with_distances() {
    let idx = HnswIndex::new();
    idx.insert(101, &[1.0, 0.0, 0.0]);
    idx.insert(202, &[0.0, 1.0, 0.0]);
    let hits = hnsw_lookup(&idx, &[1.0, 0.1, 0.0], 1).expect("built → Some");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 101, "caller key, not the external idx");
    assert!(hits[0].1 >= 0.0, "cosine distance");
    // k == 0 probes nothing → caller falls back.
    assert_eq!(hnsw_lookup(&idx, &[1.0, 0.0, 0.0], 0), None);
    // k beyond len returns a nonempty subset of what exists (no padding, no
    // panic): the graph is approximate, so overfetch may resolve fewer than
    // len — but every resolved key is a caller key, never an external idx.
    let all = hnsw_lookup(&idx, &[1.0, 0.0, 0.0], 10).expect("overfetch → Some");
    assert!(!all.is_empty(), "overfetch resolves at least one neighbour");
    assert!(all.iter().all(|(key, _)| *key == 101 || *key == 202));
}
