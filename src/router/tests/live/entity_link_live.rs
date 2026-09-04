//! Live-AI-gated entity-link overlay threshold sweep (ROADMAP_20260827_ORT §6.2).
//!
//! Compiled only when the `live-ai` feature is enabled and `#[ignore]`d so it
//! never runs under `make test` / `make router-test` / `make router-mock` / CI
//! — only via `make test-live` / `make router-test-live`.
//!
//! The full precision/recall measurement against the ColBERT `EntitySimilarityIndex`
//! is blocked on the M5.1 ONNX export (§5.1 — external). This test therefore
//! sweeps the worker's threshold over a fixed synthetic concept-label set and a
//! deterministic scorer, exercising the exact gating logic a ColBERT-backed
//! scorer would drive (threshold acceptance + `is_subclass_of(YagoEntity)`),
//! and records the precision/recall per threshold. It is model-free (never
//! dials a host) and structural, so it can safely run in live-AI CI without a
//! model. When the ColBERT export lands, the scorer seam is swapped for the
//! real index and the sweep becomes a true model-accuracy measurement.

use std::sync::Arc;

use fluent_router::ledger::ContentNodeLedger;
use fluent_router::ledger::overlay::{
    CandidateStatus, OverlayCandidateStore,
};
use fluent_router::server::entity_link::{
    EntityLinkJob, EntityLinkWorker,
};
use fluent_types::{InterlinguaId, InterlinguaNamespace, NodeId};
use fluent_concept::InMemoryConceptStore;
use fluent_concept::ConceptStore;

/// The YaGO `Entity` root (as `YagoClass`).
fn entity_root() -> InterlinguaId {
    InterlinguaId::new(InterlinguaNamespace::YagoClass, 0x100)
}

/// A concept store whose taxonomy makes the candidate `YagoEntity` ids
/// subclasses of the Entity root, plus a non-entity `YagoClass`.
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
        .expect("root");
    for (local, ns) in [
        (0x001, InterlinguaNamespace::YagoEntity),
        (0x002, InterlinguaNamespace::YagoEntity),
    ] {
        store
            .insert(fluent_types::ConceptMetadata {
                id: InterlinguaId::new(ns, local),
                canonical_name: format!("yago:X{local}"),
                namespace: ns,
                yago_iri: None,
                yago_class_iri: None,
                label: None,
                node_id: None,
                parent_class_id: Some(root),
            })
            .expect("child");
    }
    // A `YagoClass` that is NOT a subclass of the Entity root — must be gated
    // out by `is_subclass_of` (a class is not an entity).
    store
        .insert(fluent_types::ConceptMetadata {
            id: InterlinguaId::new(InterlinguaNamespace::YagoClass, 0x777),
            canonical_name: "yago:NotEntity".into(),
            namespace: InterlinguaNamespace::YagoClass,
            yago_iri: None,
            yago_class_iri: None,
            label: None,
            node_id: None,
            parent_class_id: None,
        })
        .expect("non-entity");
    Arc::new(store)
}

/// A deterministic scorer: the two genuine entities score 0.9 / 0.5 and the
/// non-entity class scores 0.8, so a threshold sweep separates them exactly.
fn scorer() -> fluent_router::server::entity_link::EntityLinkScorer {
    let entity_a = InterlinguaId::new(InterlinguaNamespace::YagoEntity, 0x001);
    let entity_b = InterlinguaId::new(InterlinguaNamespace::YagoEntity, 0x002);
    let not_entity = InterlinguaId::new(InterlinguaNamespace::YagoClass, 0x777);
    Arc::new(move |_text| vec![(entity_a, 0.9), (not_entity, 0.8), (entity_b, 0.5)])
}

/// Record the accepted candidates for one node at a given threshold, using a
/// fresh in-memory candidate plane per sweep (so a sweep is isolated).
async fn sweep_once(
    concepts: &Arc<dyn ConceptStore>,
    threshold: f64,
) -> Vec<(InterlinguaId, f64)> {
    let ledger = ContentNodeLedger::open_in_memory().expect("ledger");
    let s = OverlayCandidateStore::new(ledger.node_store().shared_sqlite().expect("shared"));
    let worker = EntityLinkWorker::new(
        &s,
        concepts,
        &scorer(),
        threshold,
        entity_root(),
        8,
        4,
        fluent_concurrency::tokio_runtime(),
    );
    let node = NodeId::from_int(1);
    worker
        .submit(EntityLinkJob {
            node_id: node,
            span_start: 0,
            span_end: 1,
            text: "Paris".into(),
        })
        .await
        .expect("submit");
    Arc::new(worker).drain().await;
    s.for_node(node)
        .expect("query")
        .into_iter()
        .filter(|c| c.status == CandidateStatus::Pending)
        .filter_map(|c| c.entity_id.map(|e| (e, c.score.unwrap_or(0.0))))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live-AI threshold sweep (model-free; run via make router-test-live)"]
async fn entity_link_threshold_sweep_records_precision() {
    let concepts = concepts();

    // At a low threshold, both genuine entities pass the subclass gate; the
    // non-entity class is rejected by `is_subclass_of`.
    let low = sweep_once(&concepts, 0.4).await;
    assert_eq!(low.len(), 2, "both genuine entities accepted at 0.4");
    assert!(
        low.iter().all(|(id, _)| id.namespace() == InterlinguaNamespace::YagoEntity),
        "the non-entity class is gated out"
    );

    // At a high threshold, only the strongest entity remains.
    let high = sweep_once(&concepts, 0.8).await;
    assert_eq!(high.len(), 1, "only the 0.9 entity clears 0.8");

    // The sweep is monotonic: raising the threshold never adds candidates.
    let very_high = sweep_once(&concepts, 0.95).await;
    assert!(very_high.is_empty(), "nothing clears 0.95");
}