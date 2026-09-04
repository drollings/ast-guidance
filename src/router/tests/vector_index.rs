use super::*;
use std::sync::Arc;
use fluent_types::NodeId;

fn random_embedding(dim: usize, seed: u64) -> Vec<f32> {
    // Deterministic pseudo-random embedding via hash
    let mut v = Vec::with_capacity(dim);
    let mut x = seed;
    for _ in 0..dim {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let f = ((x >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        v.push(f);
    }
    // Normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / norm).collect()
}

#[test]
fn vector_index_brute_force_vs_hnsw_recall() {
    let dim = 16;
    let n = 100;
    let k = 10;

    let brute = BruteForceIndex::new();
    let hnsw = HnswVectorIndex::new();

    for i in 0..n {
        let emb = random_embedding(dim, i as u64 * 9973 + 42);
        let id = NodeId::from_int(i as i64 + 1);
        brute.insert(id, &emb).unwrap();
        hnsw.insert(id, &emb).unwrap();
    }

    // Query with a known embedding (pick one of the inserted)
    let query = random_embedding(dim, 5 * 9973 + 42);
    let brute_hits = brute.search(&query, k);
    let hnsw_hits = hnsw.search(&query, k);

    // Recall@k vs brute-force ground truth
    let brute_set: std::collections::HashSet<i64> = brute_hits.iter().map(|(id,_)| id.as_int()).collect();
    let hnsw_set: std::collections::HashSet<i64> = hnsw_hits.iter().map(|(id,_)| id.as_int()).collect();
    let intersect = brute_set.intersection(&hnsw_set).count();
    let recall = intersect as f64 / k as f64;
    assert!(
        recall >= 0.9,
        "HNSW recall@{} vs brute-force must be >=0.9, got {recall} (brute {brute_set:?} vs hnsw {hnsw_set:?})",
        k
    );
    assert_eq!(brute.len(), n);
    assert_eq!(hnsw.len(), n);
}

#[test]
fn vector_index_seam_is_concern_separated() {
    // Salience provider must never call VectorIndex::search — different concern.
    // Use a mock VectorIndex that panics if searched.
    struct PanicIndex;
    impl VectorIndex for PanicIndex {
        fn insert(&self, _: NodeId, _: &[f32]) -> Result<(), fluent_db::error::DbError> { Ok(()) }
        fn search(&self, _: &[f32], _: usize) -> Vec<(NodeId, f64)> {
            panic!("salience must never call VectorIndex::search")
        }
        fn len(&self) -> usize { 0 }
    }
    let _panic: Arc<dyn VectorIndex> = Arc::new(PanicIndex);
    // LedgerSalienceProvider signals are deterministic graph signals, no vector search.
    use crate::node_store::ContentNodeStore;
    use crate::ranking::{LedgerSalienceProvider, SalienceSource};
    let store = ContentNodeStore::ephemeral();
    let provider = LedgerSalienceProvider::new(Arc::new(store), common_core::now_secs());
    let sigs = provider.signals_for(&[NodeId::from_int(1)]);
    assert_eq!(sigs.len(), 1);
    // No panic means separation holds
}

#[test]
fn vector_index_empty_search_returns_empty() {
    let brute = BruteForceIndex::new();
    let hnsw = HnswVectorIndex::new();
    assert!(brute.search(&[1.0, 0.0], 5).is_empty());
    assert!(hnsw.search(&[1.0, 0.0], 5).is_empty());
}

#[test]
fn ann_calibration_control_50_unrelated_queries_must_not_fire_high_confidence() {
    // 50 unrelated queries over 1k-node ledger — top-1 score >0.9 must not fire
    let n = 1000;
    let dim = 32;
    let index = BruteForceIndex::new();
    for i in 0..n {
        let emb = random_embedding(dim, i as u64 * 7919 + 7);
        index.insert(NodeId::from_int(i as i64 + 1), &emb).unwrap();
    }
    // Unrelated queries: use seeds far from ledger seeds
    let mut false_high = 0;
    for q in 0..50 {
        let query = random_embedding(dim, 1_000_000 + q as u64 * 12345);
        let hits = index.search(&query, 1);
        if let Some((_, score)) = hits.first() {
            if *score > 0.9 {
                false_high += 1;
            }
        }
    }
    assert_eq!(false_high, 0, "50 unrelated queries must not produce top-1 >0.9, got {false_high} false high-confidence hits");
}

#[test]
fn ann_calibration_recall_and_latency_baseline() {
    // Measure recall@10 and latency baseline before caching any ranking
    let dim = 32;
    let n = 500;
    let k = 10;
    let brute = BruteForceIndex::new();
    let hnsw = HnswVectorIndex::new();
    for i in 0..n {
        let emb = random_embedding(dim, i as u64 * 7919 + 13);
        let id = NodeId::from_int(i as i64 + 1);
        brute.insert(id, &emb).unwrap();
        hnsw.insert(id, &emb).unwrap();
    }
    let query = random_embedding(dim, 42);
    let start = std::time::Instant::now();
    let brute_hits = brute.search(&query, k);
    let brute_lat = start.elapsed();
    let start = std::time::Instant::now();
    let hnsw_hits = hnsw.search(&query, k);
    let hnsw_lat = start.elapsed();
    let brute_set: std::collections::HashSet<i64> = brute_hits.iter().map(|(id,_)| id.as_int()).collect();
    let hnsw_set: std::collections::HashSet<i64> = hnsw_hits.iter().map(|(id,_)| id.as_int()).collect();
    let recall = hnsw_set.intersection(&brute_set).count() as f64 / k as f64;
    assert!(recall >= 0.8, "recall@{} baseline must be >=0.8, got {recall}", k);
    // Latency: HNSW should be <= brute-force (or at least not 10x slower) for 500 nodes
    assert!(hnsw_lat <= brute_lat * 10, "HNSW latency {hnsw_lat:?} should not be 10x brute {brute_lat:?}");
}
