use super::*;

#[test]
fn test_vec_bytes_round_trip() {
    let v = vec![1.5, -2.5, 3.0, 0.0, -0.5];
    let restored = bytes_to_vec(&vec_to_bytes(&v));
    for (a, b) in v.iter().zip(restored.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}
#[test]
fn test_try_bytes_to_vec_valid() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = vec_to_bytes(&v);
    let restored = try_bytes_to_vec(&bytes).unwrap();
    assert_eq!(restored.len(), 4);
}
#[test]
fn test_try_bytes_to_vec_invalid_length() {
    assert!(try_bytes_to_vec(&[0u8; 3]).is_none());
}
#[test]
fn test_quantize_round_trip() {
    let original = vec![0.5, -0.3, 0.8, -0.1, 0.0, 1.0, -1.0];
    let q = QuantizedEmbedding::from_f32(&original);
    let restored = q.to_f32();
    for (a, b) in original.iter().zip(restored.iter()) {
        assert!((a - b).abs() < 0.02);
    }
}
#[test]
fn test_q8_cosine_similarity() {
    let a = QuantizedEmbedding::from_f32(&[1.0, 0.0, 0.0]);
    let b = QuantizedEmbedding::from_f32(&[1.0, 0.0, 0.0]);
    assert!((cosine_similarity_q8(&a, &b) - 1.0).abs() < 0.02);
}
#[test]
#[allow(clippy::float_cmp)]
fn test_q8_cosine_empty() {
    assert_eq!(
        cosine_similarity_q8(
            &QuantizedEmbedding::from_f32(&[]),
            &QuantizedEmbedding::from_f32(&[])
        ),
        0.0
    );
}

#[test]
fn knn_brute_force_returns_top_k() {
    let query = vec![1.0, 0.0, 0.0];
    let candidates = [
        (1u32, vec![1.0, 0.0, 0.0]),
        (2, vec![0.0, 1.0, 0.0]),
        (3, vec![0.9, 0.1, 0.0]),
    ];
    let results = knn_brute_force(
        &query,
        candidates.iter().map(|(id, e)| (*id, e.as_slice())),
        2,
    );
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
    assert!((results[0].1 - 0.0).abs() < 1e-5);
    assert_eq!(results[1].0, 3);
}

#[test]
fn knn_brute_force_skips_mismatched_dimensions() {
    let query = vec![1.0, 0.0];
    let candidates = [(1u32, vec![1.0, 0.0]), (2, vec![1.0, 0.0, 0.0])];
    let results = knn_brute_force(
        &query,
        candidates.iter().map(|(id, e)| (*id, e.as_slice())),
        10,
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
}

#[test]
fn knn_brute_force_empty_query_returns_empty() {
    let candidates = [(1u32, vec![1.0])];
    let results = knn_brute_force(&[], candidates.iter().map(|(id, e)| (*id, e.as_slice())), 5);
    assert!(results.is_empty());
}

#[test]
fn knn_brute_force_zero_k_returns_empty() {
    let query = vec![1.0, 0.0];
    let candidates = [(1u32, vec![1.0, 0.0])];
    let results = knn_brute_force(
        &query,
        candidates.iter().map(|(id, e)| (*id, e.as_slice())),
        0,
    );
    assert!(results.is_empty());
}

#[test]
fn knn_brute_force_empty_candidates() {
    let query = vec![1.0, 0.0];
    let results: Vec<(u32, f32)> = knn_brute_force(&query, std::iter::empty(), 5);
    assert!(results.is_empty());
}

#[test]
fn knn_brute_force_stable_tie_order_and_truncate() {
    // M0: mirrors GuidanceDb's `brute_force_order_truncation_and_tie_winner`.
    // Identical embeddings tie; the stable sort keeps insertion order and
    // truncate(k) keeps the head.
    let query = vec![1.0, 0.0, 0.0];
    let candidates = [
        (1u32, vec![1.0, 0.0, 0.0]),
        (2, vec![1.0, 0.0, 0.0]),
        (3, vec![0.0, 1.0, 0.0]),
    ];
    let all = knn_brute_force(
        &query,
        candidates.iter().map(|(id, e)| (*id, e.as_slice())),
        10,
    );
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].0, 1, "tie: insertion order wins");
    assert_eq!(all[1].0, 2);
    assert_eq!(all[2].0, 3);
    assert!(all[0].1 <= all[1].1);
    assert!(all[1].1 <= all[2].1);

    // k truncation keeps the head in order.
    let top2 = knn_brute_force(
        &query,
        candidates.iter().map(|(id, e)| (*id, e.as_slice())),
        2,
    );
    assert_eq!(vec![top2[0].0, top2[1].0], vec![1, 2]);
}

#[test]
fn distance_to_similarity_identity() {
    // M1: canonical mapping is `1 - d`, unclamped (preserves raw-distance shapes).
    assert!((distance_to_similarity(0.0) - 1.0).abs() < 1e-6);
    assert!((distance_to_similarity(0.25) - 0.75).abs() < 1e-6);
    assert!((distance_to_similarity(1.0) - 0.0).abs() < 1e-6);
    assert!((distance_to_similarity(1.5) - -0.5).abs() < 1e-6);
}

#[test]
fn distance_to_similarity_clamped_floors_at_zero() {
    // M1: the chart path that already clamps — no new semantics.
    assert!((distance_to_similarity_clamped(0.25) - 0.75).abs() < 1e-6);
    assert_eq!(distance_to_similarity_clamped(1.5), 0.0);
    assert_eq!(distance_to_similarity_clamped(1.0), 0.0);
}

#[test]
fn distance_to_similarity_nan() {
    // Unclamped propagates NaN per IEEE-754; clamped floors via `f32::max`
    // semantics (NaN → 0.0) — documented, not new behavior.
    assert!(distance_to_similarity(f32::NAN).is_nan());
    assert_eq!(distance_to_similarity_clamped(f32::NAN), 0.0);
}

#[test]
fn scored_hits_maps_scores_preserving_ids_and_order() {
    // M1: thin `hnsw_lookup` post-mapper — ids and order untouched.
    let sim = scored_hits(vec![(7i64, 0.25f32), (8, 1.5)], distance_to_similarity);
    assert_eq!(sim.len(), 2);
    assert_eq!(sim[0].0, 7);
    assert!((sim[0].1 - 0.75).abs() < 1e-6);
    assert_eq!(sim[1].0, 8);
    assert!((sim[1].1 - -0.5).abs() < 1e-6);

    let clamped = scored_hits(vec![(7i64, 0.25f32), (8, 1.5)], distance_to_similarity_clamped);
    assert!((clamped[0].1 - 0.75).abs() < 1e-6);
    assert_eq!(clamped[1].1, 0.0);

    assert!(scored_hits(Vec::<(i64, f32)>::new(), distance_to_similarity).is_empty());
}

#[test]
fn rrf_merge_single_result() {
    let kw = vec![(1i64, "foo")];
    let vec_results: Vec<(i64, &str)> = Vec::new();
    let merged = rrf_merge(kw, vec_results, 60.0);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].1, "foo");
}

#[test]
fn rrf_merge_boosts_shared_results() {
    let kw = vec![(1i64, "shared"), (2, "kw_only")];
    let vec_results = vec![(1i64, "shared"), (3, "vec_only")];
    let merged = rrf_merge(kw, vec_results, 60.0);
    // shared (id=1) should be ranked first since it appears in both lists.
    assert!(merged.len() >= 2);
    assert_eq!(merged[0].1, "shared");
    assert!(merged[0].0 > merged[1].0);
}

#[test]
fn rrf_merge_deduplicates() {
    let kw = vec![(1i64, "dup")];
    let vec_results = vec![(1i64, "dup")];
    let merged = rrf_merge(kw.clone(), vec_results, 60.0);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].1, "dup");
}

#[test]
fn rrf_merge_empty_inputs() {
    let merged: Vec<(f64, &str)> = rrf_merge(Vec::new(), Vec::new(), 60.0);
    assert!(merged.is_empty());
}
