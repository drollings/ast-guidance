use super::*;

fn open() -> (Arc<SqliteStore>, SqliteCorrectionIndex) {
    let store = Arc::new(SqliteStore::open_in_memory().expect("in-memory"));
    store
        .with_conn(|conn| {
            fluent_db::migrate::migrate(conn, &crate::ledger::ledger_migrations())
        })
        .expect("migrate");
    let idx = SqliteCorrectionIndex::new(Arc::clone(&store));
    (store, idx)
}

fn corrections() -> Vec<Correction> {
    vec![Correction {
        token_index: 1,
        field: spacy_rs::CorrectionField::Dep,
        old_value: "dep".into(),
        new_value: "nsubj".into(),
    }]
}

#[test]
fn record_and_query_roundtrip() {
    let (_store, idx) = open();
    let lemma = InterlinguaId::from_u64(0x0300_0000_0000_0001);
    assert!(idx.query_previous_corrections(lemma, None).is_none());
    idx.record_correction(lemma, None, &corrections())
        .expect("record");
    let back = idx.query_previous_corrections(lemma, None).expect("query");
    assert_eq!(back, corrections());
    // A different lemma has no cached correction.
    assert!(idx
        .query_previous_corrections(
            InterlinguaId::from_u64(0x0300_0000_0000_0002),
            None
        )
        .is_none());
}

#[test]
fn entity_scoped_patterns_do_not_collide() {
    let (_store, idx) = open();
    let lemma = InterlinguaId::from_u64(0x0300_0000_0000_0001);
    let entity_a = Some(InterlinguaId::from_u64(0x0100_0000_0000_0001));
    let entity_b = Some(InterlinguaId::from_u64(0x0100_0000_0000_0002));
    idx.record_correction(lemma, entity_a, &corrections()).expect("a");
    assert!(idx.query_previous_corrections(lemma, entity_b).is_none());
    assert!(idx.query_previous_corrections(lemma, entity_a).is_some());
}

#[test]
fn review_status_is_status_only_and_entity_id_is_real() {
    let (store, idx) = open();
    let lemma = InterlinguaId::from_u64(0x0300_0000_0000_0001);
    let entity = Some(InterlinguaId::from_u64(0x0100_0000_0000_0009));
    idx.record_correction(lemma, entity, &corrections()).expect("record");
    let (entity_id, status): (i64, String) = store
        .query_row(
            "SELECT entity_id, review_status FROM interlingua_index \
             WHERE node_id = ?1 AND interlingua_id = ?2 AND role = ?3",
            params![PATTERN_NODE, lemma.as_i64(), CORRECTION_ROLE],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .expect("row")
        .expect("some");
    assert_eq!(entity_id, entity.unwrap().as_i64(), "entity id in its own column");
    assert_eq!(status, "cached", "review_status is status-only, never an id");
}

#[test]
fn record_upserts_instead_of_duplicating() {
    let (_store, idx) = open();
    let lemma = InterlinguaId::from_u64(0x0300_0000_0000_0001);
    idx.record_correction(lemma, None, &corrections()).expect("first");
    idx.record_correction(lemma, None, &corrections()).expect("second");
    let count: i64 = _store
        .query_row(
            "SELECT COUNT(*) FROM interlingua_index WHERE node_id = ?1 AND role = ?2",
            params![PATTERN_NODE, CORRECTION_ROLE],
            |r| r.get(0),
        )
        .expect("count")
        .expect("row");
    assert_eq!(count, 1);
}
