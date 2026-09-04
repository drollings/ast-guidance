use super::*;
use common_core::vector_math::cosine_similarity_f32;

#[cfg(feature = "onnx")]
use crate::tokenizer::LfmEncoding;

#[test]
fn l2_normalize_unit_vector() {
    let mut v = vec![3.0, 4.0];
    l2_normalize(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
}

#[test]
fn l2_normalize_zero_stays_zero() {
    let mut v = vec![0.0, 0.0, 0.0];
    l2_normalize(&mut v);
    assert_eq!(v, vec![0.0, 0.0, 0.0]);
}

#[test]
fn cosine_similarity_identical_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity_f32(&a, &b) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_similarity_orthogonal_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert!((cosine_similarity_f32(&a, &b)).abs() < 1e-6);
}

#[test]
fn cosine_similarity_opposite_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    assert!((cosine_similarity_f32(&a, &b) - (-1.0)).abs() < 1e-6);
}

#[test]
fn cosine_similarity_zero_vector() {
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 0.0];
    assert!((cosine_similarity_f32(&a, &b)).abs() < 1e-6);
}

#[test]
fn maxsim_score_perfect_match() {
    // One query token, one doc token — identical → score = 1.0
    let q: Vec<&[f32]> = vec![&[1.0, 0.0]];
    let d: Vec<&[f32]> = vec![&[1.0, 0.0]];
    assert!((maxsim_score(&q, &d) - 1.0).abs() < 1e-6);
}

#[test]
fn maxsim_score_picks_best_match() {
    // Query token [1,0] should match doc token [1,0] (sim=1) over [0,1] (sim=0)
    let q: Vec<&[f32]> = vec![&[1.0, 0.0]];
    let d: Vec<&[f32]> = vec![&[0.0, 1.0], &[1.0, 0.0]];
    assert!((maxsim_score(&q, &d) - 1.0).abs() < 1e-6);
}

#[test]
fn maxsim_score_averages_over_query_tokens() {
    // Two query tokens: [1,0] matches [1,0] (1.0), [0,1] matches [0,1] (1.0) → avg 1.0
    let q: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0]];
    let d: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0]];
    assert!((maxsim_score(&q, &d) - 1.0).abs() < 1e-6);
}

#[test]
fn maxsim_score_empty_query() {
    let q: Vec<&[f32]> = vec![];
    let d: Vec<&[f32]> = vec![&[1.0, 0.0]];
    assert!((maxsim_score(&q, &d)).abs() < 1e-6);
}

#[test]
fn maxsim_score_empty_doc() {
    let q: Vec<&[f32]> = vec![&[1.0, 0.0]];
    let d: Vec<&[f32]> = vec![];
    assert!((maxsim_score(&q, &d)).abs() < 1e-6);
}

#[test]
fn maxsim_score_symmetric_with_relevance() {
    // Two query tokens, two doc tokens. Each query matches its best doc.
    let q: Vec<&[f32]> = vec![&[0.9, 0.1], &[0.1, 0.9]];
    let d: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0]];
    let score = maxsim_score(&q, &d);
    // Each query token matches its best doc token with ~0.995 similarity
    assert!(score > 0.9, "expected high score, got {score}");
}

#[test]
fn maxsim_score_normalized_range() {
    let q: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0]];
    let d: Vec<&[f32]> = vec![&[-1.0, 0.0], &[0.0, -1.0]];
    let norm = maxsim_score_normalized(&q, &d);
    assert!(norm >= 0.0 && norm <= 1.0, "normalized score out of range: {norm}");
}

#[test]
fn cached_colbert_lru_eviction() {
    let cache = CachedColbert::new(2);
    cache.insert("a".into(), vec![vec![1.0]]);
    cache.insert("b".into(), vec![vec![2.0]]);
    cache.insert("c".into(), vec![vec![3.0]]); // evicts "a"
    assert!(cache.get("a").is_none());
    assert!(cache.get("b").is_some());
    assert!(cache.get("c").is_some());
    assert_eq!(cache.len(), 2);
}

#[test]
fn cached_colbert_update_existing_key() {
    let cache = CachedColbert::new(2);
    cache.insert("a".into(), vec![vec![1.0]]);
    cache.insert("a".into(), vec![vec![2.0]]); // update, not eviction
    assert_eq!(cache.len(), 1);
    let tokens = cache.get("a").unwrap();
    assert_eq!(tokens, vec![vec![2.0]]);
}

#[test]
fn cached_colbert_empty_initially() {
    let cache = CachedColbert::new(10);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
#[cfg(feature = "onnx")]
fn strip_special_tokens_removes_cls_sep_pad() {
    let tokens = vec![
        vec![0.1, 0.2], // CLS (id=1)
        vec![0.3, 0.4], // "hello" (id=4)
        vec![0.5, 0.6], // "world" (id=5)
        vec![0.7, 0.8], // SEP (id=2)
    ];
    let encoding = LfmEncoding {
        ids: vec![1, 4, 5, 2],
        attention_mask: vec![1, 1, 1, 1],
        offsets: vec![(0, 0), (0, 5), (6, 11), (0, 0)],
    };
    let stripped = strip_special_tokens(tokens, &encoding);
    assert_eq!(stripped.len(), 2);
    assert_eq!(stripped[0], vec![0.3, 0.4]);
    assert_eq!(stripped[1], vec![0.5, 0.6]);
}

#[test]
#[cfg(feature = "onnx")]
fn strip_special_tokens_keeps_pad_masked() {
    let tokens = vec![
        vec![0.1, 0.2], // CLS (id=1, mask=1)
        vec![0.3, 0.4], // hello (id=4, mask=1)
        vec![0.5, 0.6], // PAD (id=0, mask=0)
    ];
    let encoding = LfmEncoding {
        ids: vec![1, 4, 0],
        attention_mask: vec![1, 1, 0],
        offsets: vec![(0, 0), (0, 5), (0, 0)],
    };
    let stripped = strip_special_tokens(tokens, &encoding);
    assert_eq!(stripped.len(), 1);
    assert_eq!(stripped[0], vec![0.3, 0.4]);
}

#[test]
fn entity_similarity_empty_index_returns_nothing() {
    let index = EntitySimilarityIndex::empty(0.5);
    let q: Vec<&[f32]> = vec![&[1.0, 0.0]];
    assert!(index.lookup(&q).is_empty());
}

#[test]
fn entity_similarity_lookup_above_threshold() {
    let entries = vec![ConceptEncoding {
        namespace: "YagoEntity".into(),
        canonical: "schema:Person".into(),
        token_embeddings: vec![vec![1.0, 0.0]],
    }];
    let index = EntitySimilarityIndex::new(entries, 0.8);
    let q: Vec<&[f32]> = vec![&[0.99, 0.14]];
    let hits = index.lookup(&q);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].canonical, "schema:Person");
    assert!(hits[0].score >= 0.8);
}

#[test]
fn entity_similarity_lookup_below_threshold() {
    let entries = vec![ConceptEncoding {
        namespace: "YagoEntity".into(),
        canonical: "schema:Person".into(),
        token_embeddings: vec![vec![1.0, 0.0]],
    }];
    let index = EntitySimilarityIndex::new(entries, 0.99);
    let q: Vec<&[f32]> = vec![&[0.5, 0.87]];
    let hits = index.lookup(&q);
    assert!(hits.is_empty());
}

#[test]
fn entity_similarity_sorted_by_score() {
    let entries = vec![
        ConceptEncoding {
            namespace: "A".into(),
            canonical: "a".into(),
            token_embeddings: vec![vec![1.0, 0.0]],
        },
        ConceptEncoding {
            namespace: "B".into(),
            canonical: "b".into(),
            token_embeddings: vec![vec![0.0, 1.0]],
        },
    ];
    let index = EntitySimilarityIndex::new(entries, 0.0);
    let q: Vec<&[f32]> = vec![&[0.9, 0.44]];
    let hits = index.lookup(&q);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].canonical, "a");
    assert_eq!(hits[1].canonical, "b");
}
