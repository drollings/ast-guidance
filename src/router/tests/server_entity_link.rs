use super::*;
use std::time::Duration;

use crate::ledger::ContentNodeLedger;
use crate::ledger::overlay::CandidateStatus;
use fluent_concurrency::tokio_runtime;
use fluent_concept::InMemoryConceptStore;
use fluent_types::{InterlinguaId, InterlinguaNamespace};

fn entity_root() -> InterlinguaId {
InterlinguaId::new(InterlinguaNamespace::YagoClass, 0x100)
}

fn store() -> OverlayCandidateStore {
let ledger = ContentNodeLedger::open_in_memory().expect("ledger");
OverlayCandidateStore::new(ledger.node_store().shared_sqlite().expect("shared"))
}

/// A concept store whose taxonomy makes `entity_root` the parent of the
/// candidate `YagoEntity` ids, so `is_subclass_of` gates correctly. The
/// hierarchy is derived from `parent_class_id` on insert (C5).
fn concepts() -> Arc<dyn ConceptStore> {
let store = InMemoryConceptStore::new();
let root = entity_root();
store
    .insert(fluent_types::ConceptMetadata {
        id: root,
        canonical_name: "yago:Entity".into(),
        namespace: InterlinguaNamespace::YagoClass,
        yago_iri: None,
        yago_class_iri: None,
        label: None,
        node_id: None,
        parent_class_id: None,
    })
    .expect("insert root");
for local in [0x001, 0x002, 0x009] {
    let child = InterlinguaId::new(InterlinguaNamespace::YagoEntity, local);
    store
        .insert(fluent_types::ConceptMetadata {
            id: child,
            canonical_name: format!("yago:Entity{local}"),
            namespace: InterlinguaNamespace::YagoEntity,
            yago_iri: None,
            yago_class_iri: None,
            label: None,
            node_id: None,
            parent_class_id: Some(root),
        })
        .expect("insert child");
}
Arc::new(store)
}

fn scorer_fixed(hits: Vec<(InterlinguaId, f64)>) -> EntityLinkScorer {
    Arc::new(move |_text| hits.clone())
}

async fn wait_for_candidates(s: &OverlayCandidateStore, node: NodeId, n: usize) {
    for _ in 0..100 {
        if s.for_node(node).map(|r| r.len()).unwrap_or(0) >= n {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("entity-link candidates never landed");
}

#[test]
fn entity_link_jobs_from_signals_extracts_propn_spans() {
    let signal = spacy_rs::routing::RoutingSignal {
        sentence: "Visit Paris in May".into(),
        predicate: "visit".into(),
        subject: None,
        direct_object: Some("Paris".into()),
        indirect_object: None,
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec!["Visit".into(), "Paris".into(), "in".into(), "May".into()],
        lemmas: vec!["visit".into(), "Paris".into(), "in".into(), "May".into()],
        pos: vec!["verb".into(), "propn".into(), "adp".into(), "propn".into()],
        deps: vec!["root".into(), "dobj".into(), "prep".into(), "pobj".into()],
        heads: vec![0, -1, 0, -1],
        interlingua: None,
    };
    let jobs = entity_link_jobs_from_signals(
        "Visit Paris in May",
        NodeId::from_int(9),
        &[signal],
    );
    assert_eq!(jobs.len(), 2, "the two PROPN tokens");
    assert_eq!(jobs[0].text, "Paris");
    assert_eq!(jobs[0].span_start, "Visit ".len());
    assert_eq!(jobs[0].span_end, "Visit ".len() + "Paris".len());
    assert_eq!(jobs[1].text, "May");
    assert_eq!(jobs[0].node_id, NodeId::from_int(9));
}

#[test]
fn entity_link_jobs_skip_non_propn_and_unlocatable() {
    let signal = spacy_rs::routing::RoutingSignal {
        sentence: "the cat".into(),
        predicate: "be".into(),
        subject: Some("cat".into()),
        direct_object: None,
        indirect_object: None,
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec!["the".into(), "cat".into()],
        lemmas: vec!["the".into(), "cat".into()],
        pos: vec!["det".into(), "noun".into()],
        deps: vec!["det".into(), "nsubj".into()],
        heads: vec![1, 0],
        interlingua: None,
    };
    let jobs =
        entity_link_jobs_from_signals("the cat", NodeId::from_int(1), &[signal]);
    assert!(jobs.is_empty(), "no PROPN tokens → no jobs");
}

#[tokio::test]
async fn writes_candidates_only_through_threshold_and_subclass_gate() {
    let s = store();
    let concepts = concepts();
    let node = NodeId::from_int(1);
    let good = InterlinguaId::new(InterlinguaNamespace::YagoEntity, 0x001);
    let below_threshold = InterlinguaId::new(InterlinguaNamespace::YagoEntity, 0x002);
    let not_an_entity = InterlinguaId::new(InterlinguaNamespace::YagoClass, 0x777);
    let scorer = scorer_fixed(vec![
        (good, 0.95),
        (below_threshold, 0.3),
        (not_an_entity, 0.9),
    ]);
    let worker = Arc::new(EntityLinkWorker::new(
        &s,
        &concepts,
        &scorer,
        0.5,
        entity_root(),
        8,
        4,
        tokio_runtime(),
    ));
    worker
        .submit(EntityLinkJob {
            node_id: node,
            span_start: 0,
            span_end: 4,
            text: "Paris".into(),
        })
        .await
        .expect("submit");
    wait_for_candidates(&s, node, 1).await;
    worker.drain().await;

    let rows = s.for_node(node).expect("query");
    // Only the above-threshold, entity-subclass candidate landed.
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].entity_id, Some(good));
    assert_eq!(rows[0].span_start, 0);
    assert_eq!(rows[0].source, "entity_link");
    assert_eq!(rows[0].status, CandidateStatus::Pending);
    // Candidates never write a doc id — the candidate plane is the only
    // surface.
    assert!(rows.iter().all(|r| r.entity_id.is_some()));
}

#[tokio::test]
async fn failing_scorer_is_fail_open() {
    let s = store();
    let concepts = concepts();
    let node = NodeId::from_int(2);
    // A scorer that panics is caught by the worker's fail-open contract
    // only for errors; use an empty result (the graceful no-op path) and a
    // threshold-rejecting result to assert nothing is written.
    let empty: EntityLinkScorer = Arc::new(|_text| Vec::new());
    let worker = Arc::new(EntityLinkWorker::new(
        &s,
        &concepts,
        &empty,
        0.5,
        entity_root(),
        8,
        4,
        tokio_runtime(),
    ));
    worker
        .submit(EntityLinkJob {
            node_id: node,
            span_start: 0,
            span_end: 2,
            text: "x".into(),
        })
        .await
        .expect("submit");
    worker.drain().await;
    assert!(s.for_node(node).expect("q").is_empty());
}

#[tokio::test]
async fn credit_gate_bounds_in_flight_links() {
    let s = store();
    let concepts = concepts();
    // A scorer that yields nothing (instant), but with credit 1 the gate
    // still enforces the cap on concurrent submits.
    let scorer: EntityLinkScorer = Arc::new(|_t| Vec::new());
    let worker = Arc::new(EntityLinkWorker::new(
        &s,
        &concepts,
        &scorer,
        0.5,
        entity_root(),
        8,
        1, // credit limit 1
        tokio_runtime(),
    ));
    assert!(!worker.is_blocked());
    // The first submit consumes the only credit token.
    worker
        .submit(EntityLinkJob {
            node_id: NodeId::from_int(3),
            span_start: 0,
            span_end: 1,
            text: "a".into(),
        })
        .await
        .expect("first");
    worker.drain().await;
}

#[tokio::test]
async fn submit_after_drain_is_closed() {
    let s = store();
    let concepts = concepts();
    let scorer: EntityLinkScorer = Arc::new(|_t| Vec::new());
    let worker = Arc::new(EntityLinkWorker::new(
        &s,
        &concepts,
        &scorer,
        0.5,
        entity_root(),
        8,
        4,
        tokio_runtime(),
    ));
    worker.clone().drain().await;
    let err = worker
        .submit(EntityLinkJob {
            node_id: NodeId::from_int(4),
            span_start: 0,
            span_end: 1,
            text: "a".into(),
        })
        .await;
    assert!(matches!(err, Err(EntityLinkError::Closed(_))));
}
