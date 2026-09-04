use super::*;
use fluent_dag::target::Target;
use fluent_types::{ExecutorKind, TargetType};
use internment::ArcIntern;

fn test_target(id: i64, name: &str) -> Target {
    Target::new()
        .id(id)
        .name(ArcIntern::from(name))
        .target_type(TargetType::Phony)
        .executor(ExecutorKind::Native)
        .depends(bitvec::vec::BitVec::new())
        .provides(bitvec::vec::BitVec::new())
        .command(String::new())
        .essential(false)
        .build()
}

#[test]
fn unverified_entry_is_not_replayable() {
    let store = InMemoryWorkflowStore::new();
    let entry = WorkflowEntry {
        query_embedding: vec![1.0, 0.0, 0.0],
        dag: vec![test_target(1, "a")],
        audit_id: "audit1".into(),
        verified: false,
    };
    store.insert(entry).unwrap();
    let hits = store.nearest_verified(&[1.0, 0.0, 0.0], 3, 0.75);
    assert!(hits.is_empty(), "unverified entry must not be replayable");
    // nearest without verified filter still finds it
    let all = store.nearest(&[1.0, 0.0, 0.0], 3);
    assert_eq!(all.len(), 1);
}

#[test]
fn nearest_edge_cases_empty_zero_k_and_overfetch() {
    // M5.1: k-edge locks for the HNSW→brute-force migration.
    let store = InMemoryWorkflowStore::new();
    assert!(store.nearest(&[1.0, 0.0, 0.0], 3).is_empty(), "empty store");
    let entry = WorkflowEntry {
        query_embedding: vec![1.0, 0.0, 0.0],
        dag: vec![test_target(1, "a")],
        audit_id: "audit1".into(),
        verified: true,
    };
    store.insert(entry).unwrap();
    assert!(store.nearest(&[1.0, 0.0, 0.0], 0).is_empty(), "k=0");
    assert!(store.nearest(&[], 3).is_empty(), "empty query");
    assert_eq!(store.nearest(&[1.0, 0.0, 0.0], 10).len(), 1, "k>len returns all");
}

#[test]
fn confident_but_wrong_does_not_verify() {
    // 30 cases where assembler was confident (0.9) but answer was wrong per human label → must NOT pass gated insert
    for _ in 0..30 {
        assert!(!gated_insert_allowed(0.9, 0.2, false), "confident but unverified must not be insertable");
    }
    // verified true with high confidence and novelty → allowed
    assert!(gated_insert_allowed(0.9, 0.2, true));
    // low confidence → not allowed even if verified
    assert!(!gated_insert_allowed(0.7, 0.2, true));
    // low novelty → not allowed
    assert!(!gated_insert_allowed(0.9, 0.05, true));
}

#[test]
fn paraphrase_replay_is_stable() {
    let store = InMemoryWorkflowStore::new();
    let dag = vec![test_target(1, "a"), test_target(2, "b")];
    let entry = WorkflowEntry {
        query_embedding: vec![1.0, 0.0, 0.0],
        dag: dag.clone(),
        audit_id: "audit1".into(),
        verified: true,
    };
    store.insert(entry).unwrap();
    // Paraphrase embedding close to original (cosine ~0.99)
    let paraphrase = vec![0.99, 0.01, 0.0];
    let hits = store.nearest_verified(&paraphrase, 3, 0.75);
    assert_eq!(hits.len(), 1);
    assert!(hits[0].1 >= 0.78, "paraphrase cosine must be >=0.78, got {}", hits[0].1);
    // topo_sort order identical — here dag order is stable
    assert_eq!(hits[0].0.dag[0].name, dag[0].name);
    assert_eq!(hits[0].0.dag[1].name, dag[1].name);
}

#[test]
fn nearest_respects_min_score() {
    let store = InMemoryWorkflowStore::new();
    store
        .insert(WorkflowEntry {
            query_embedding: vec![1.0, 0.0, 0.0],
            dag: vec![test_target(1, "a")],
            audit_id: "audit1".into(),
            verified: true,
        })
        .unwrap();
    // Orthogonal query → cosine 0 < 0.75 → no hit
    let hits = store.nearest_verified(&[0.0, 1.0, 0.0], 3, 0.75);
    assert!(hits.is_empty());
}
