use super::*;
use fluent_concurrency::tokio_runtime;
use fluent_db::migrate::migrate;
use spacy_rs::concept_store_mem::InMemoryConceptStore;
use std::sync::atomic::{AtomicUsize, Ordering};

fn open_index() -> (Arc<SqliteStore>, SqlitePreferredSenseIndex) {
    let store = Arc::new(SqliteStore::open_in_memory().expect("in-memory"));
    store
        .with_conn(|conn| migrate(conn, &crate::ledger::ledger_migrations()))
        .expect("migrate");
    let idx = SqlitePreferredSenseIndex::new(Arc::clone(&store));
    (store, idx)
}

fn pid(v: i64) -> InterlinguaId {
    InterlinguaId::new(fluent_types::InterlinguaNamespace::SpacyLemma, v)
}

fn res(id: i64) -> Resolution {
    Resolution {
        chosen_candidate_id: pid(id),
        detail: "chose sense".into(),
    }
}

#[test]
fn record_and_query_roundtrip_per_ambiguity_kind() {
    let (_store, idx) = open_index();
    let p = pid(7);
    assert!(idx.preferred_sense(p, AmbiguityKind::PredicatePolysemy).is_none());
    idx.record_preferred_sense(p, AmbiguityKind::PredicatePolysemy, res(1))
        .expect("record");
    let back = idx
        .preferred_sense(p, AmbiguityKind::PredicatePolysemy)
        .expect("query");
    assert_eq!(back, res(1));
    // A different ambiguity kind is a distinct pattern.
    assert!(idx.preferred_sense(p, AmbiguityKind::AttachmentNearTie).is_none());
    // A different predicate is a distinct pattern.
    assert!(idx.preferred_sense(pid(8), AmbiguityKind::PredicatePolysemy).is_none());
}

#[test]
fn sense_rows_do_not_collide_with_correction_rows() {
    let (store, idx) = open_index();
    let p = pid(9);
    idx.record_preferred_sense(p, AmbiguityKind::PredicatePolysemy, res(2))
        .expect("sense");

    // A correction-cache row for the same predicate uses role='correction'
    // and must not be visible as a sense resolution.
    use crate::ledger::correction_index::upsert_correction_row;
    use crate::ledger::correction_index::CorrectionRow;
    store
        .with_conn(|conn| {
            upsert_correction_row(
                conn,
                &CorrectionRow {
                    lemma_id: p.as_i64(),
                    entity_id: 0,
                    corrections_json: "[]".into(),
                },
            )
        })
        .expect("correction row");

    assert!(idx.preferred_sense(p, AmbiguityKind::PredicatePolysemy).is_some());
    // The role columns are distinct — no accidental collision.
    let count: i64 = store
        .query_row(
            "SELECT COUNT(*) FROM interlingua_index WHERE node_id = ?1 AND interlingua_id = ?2",
            params![PATTERN_NODE, p.as_i64()],
            |r| r.get(0),
        )
        .expect("count")
        .expect("row");
    assert_eq!(count, 2, "one sense row + one correction row");
}

#[test]
fn record_upserts_instead_of_duplicating() {
    let (store, idx) = open_index();
    let p = pid(3);
    idx.record_preferred_sense(p, AmbiguityKind::NegationModalScope, res(4))
        .expect("first");
    idx.record_preferred_sense(p, AmbiguityKind::NegationModalScope, res(5))
        .expect("second");
    let count: i64 = store
        .query_row(
            "SELECT COUNT(*) FROM interlingua_index WHERE node_id = ?1 AND role = ?2",
            params![PATTERN_NODE, SENSE_ROLE],
            |r| r.get(0),
        )
        .expect("count")
        .expect("row");
    assert_eq!(count, 1, "upsert never duplicates");
}

#[tokio::test]
async fn wave_fires_one_fetch_and_promotes() {
    let (_store, idx) = open_index();
    let index: Arc<dyn PreferredSenseIndex> = Arc::new(idx);
    let fetched = Arc::new(AtomicUsize::new(0));
    let f = Arc::clone(&fetched);
    let fetch: FrameResolutionFetch = Arc::new(move |_reqs| {
        f.fetch_add(1, Ordering::SeqCst);
        Ok(r#"[
            {"chosen_candidate_id": 281474976710656, "detail": "a"},
            {"chosen_candidate_id": 281474976710657, "detail": "b"}
        ]"#
        .into())
    });

    let worker = Arc::new(FrameResolutionWorker::new(
        &index,
        &fetch,
        "frame-model".into(),
        8,
        4,
        tokio_runtime(),
    ));

    let p1 = pid(1);
    let p2 = pid(2);
    let job = FrameResolutionJob {
        requests: vec![
            FrameResolutionRequest {
                predicate_lemma_id: p1,
                ambiguity_kind: AmbiguityKind::PredicatePolysemy,
                detail: "p1 polysemy".into(),
                candidate_ids: vec![],
            },
            FrameResolutionRequest {
                predicate_lemma_id: p2,
                ambiguity_kind: AmbiguityKind::AttachmentNearTie,
                detail: "p2 tie".into(),
                candidate_ids: vec![],
            },
        ],
    };
    worker.enqueue(job).await.expect("enqueue");
    // Give the worker a moment to process the (single, batched) job.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    worker.drain().await;

    assert_eq!(fetched.load(Ordering::SeqCst), 1, "N frames → one fetch");
    assert!(
        index.preferred_sense(p1, AmbiguityKind::PredicatePolysemy).is_some(),
        "wave resolved + promoted p1"
    );
    assert!(
        index.preferred_sense(p2, AmbiguityKind::AttachmentNearTie).is_some(),
        "wave resolved + promoted p2"
    );
}

#[tokio::test]
async fn fully_resolved_wave_skips_the_fetch() {
    let (_store, idx) = open_index();
    let index: Arc<dyn PreferredSenseIndex> = Arc::new(idx);
    let p = pid(1);
    index
        .record_preferred_sense(p, AmbiguityKind::PredicatePolysemy, res(9))
        .expect("seed");

    let fetched = Arc::new(AtomicUsize::new(0));
    let f = Arc::clone(&fetched);
    let fetch: FrameResolutionFetch = Arc::new(move |_r| {
        f.fetch_add(1, Ordering::SeqCst);
        Err("should not be called".into())
    });
    let worker = Arc::new(FrameResolutionWorker::new(
        &index,
        &fetch,
        "frame-model".into(),
        8,
        4,
        tokio_runtime(),
    ));

    worker
        .enqueue(FrameResolutionJob {
            requests: vec![FrameResolutionRequest {
                predicate_lemma_id: p,
                ambiguity_kind: AmbiguityKind::PredicatePolysemy,
                detail: "already resolved".into(),
                candidate_ids: vec![],
            }],
        })
        .await
        .expect("enqueue");
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    worker.drain().await;
    assert_eq!(
        fetched.load(Ordering::SeqCst),
        0,
        "a fully-resolved wave never fires the fetch"
    );
}

#[test]
fn parse_naive_handles_single_object_and_garbage() {
    let single = parse_naive(r#"{"chosen_candidate_id":5,"detail":"x"}"#, 3);
    assert_eq!(single.len(), 3);
    assert_eq!(single[0].chosen_candidate_id.as_i64(), 5);
    let garbage = parse_naive("nonsense", 2);
    assert_eq!(garbage.len(), 2);
}

#[test]
fn trait_is_implementable_over_hermetic_store() {
    // Compile-check: the router impl satisfies the spacy-rs trait.
    let (_store, idx) = open_index();
    let _: &dyn PreferredSenseIndex = &idx;
    // And a hermetic ConceptStore still works (the trait's error type).
    let _mem = InMemoryConceptStore::new();
}
