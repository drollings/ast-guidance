use super::*;

fn open() -> (Arc<SqliteStore>, SqliteConceptStore) {
    let store = Arc::new(SqliteStore::open_in_memory().expect("in-memory"));
    // Run the router's migrations so the interlingua tables exist.
    store
        .with_conn(|conn| {
            fluent_db::migrate::migrate(conn, &crate::ledger::ledger_migrations())
        })
        .expect("migrate");
    let cs = SqliteConceptStore::new(Arc::clone(&store));
    (store, cs)
}

fn meta(id: InterlinguaId, name: &str) -> ConceptMetadata {
    ConceptMetadata {
        id,
        canonical_name: name.to_string(),
        namespace: id.namespace(),
        yago_iri: Some(format!("iri:{name}")),
        yago_class_iri: None,
        label: Some(name.to_string()),
        node_id: Some(NodeId::from_int(id.local_id())),
        parent_class_id: None,
    }
}

#[test]
fn crud_roundtrip_and_lookup() {
    let (_store, cs) = open();
    let id = InterlinguaId::new(InterlinguaNamespace::YagoClass, 0x42);
    cs.insert(meta(id, "schema:Person")).expect("insert");
    assert_eq!(cs.get(id).expect("get").canonical_name, "schema:Person");
    assert_eq!(cs.resolve_name("schema:Person").expect("name"), id);
    assert_eq!(
        cs.resolve_yago_iri("iri:schema:Person").expect("iri"),
        id
    );
    assert!(cs.contains(id));
    assert_eq!(cs.iter_ids().count(), 1);
    assert!(matches!(
        cs.get(InterlinguaId::new(InterlinguaNamespace::YagoClass, 999)),
        Err(ConceptStoreError::NotFound(_))
    ));
}

#[test]
fn insert_is_idempotent() {
    let (_store, cs) = open();
    let id = InterlinguaId::new(InterlinguaNamespace::YagoClass, 7);
    cs.insert(meta(id, "schema:Place")).expect("first");
    cs.insert(meta(id, "schema:Place")).expect("second");
    assert_eq!(cs.iter_ids().count(), 1);
}

#[test]
fn colliding_canonicals_are_both_stored() {
    let (_store, cs) = open();
    // Two canonicals that collide on the truncated 48-bit local id.
    let local = 0x1234_5678_9abc_i64;
    let a = InterlinguaId::new(InterlinguaNamespace::YagoClass, local);
    let b = InterlinguaId::new(InterlinguaNamespace::YagoClass, local);
    cs.insert(meta(a, "schema:One")).expect("insert a");
    cs.insert(meta(b, "schema:Two")).expect("insert b");
    // Both canonicals are stored (the PK is (namespace, canonical_name));
    // `iter_ids` dedupes to the single shared id bucket.
    assert_eq!(cs.iter_ids().count(), 1, "one id, one bucket");
    assert_eq!(cs.resolve_name("schema:One").expect("one"), a);
    assert_eq!(cs.resolve_name("schema:Two").expect("two"), b);
    // `get` returns the first-wins canonical deterministically (rowid order).
    assert_eq!(cs.get(a).expect("get").canonical_name, "schema:One");
    // The class count counts stored canonicals, not distinct ids.
    assert_eq!(cs.yago_class_count().expect("count"), 2);
}

#[test]
fn ancestors_and_is_subclass_through_hydrated_hierarchy() {
    let (_store, cs) = open();
    let animal = InterlinguaId::new(InterlinguaNamespace::YagoClass, 1);
    let mammal = InterlinguaId::new(InterlinguaNamespace::YagoClass, 2);
    let dog = InterlinguaId::new(InterlinguaNamespace::YagoClass, 3);
    for (id, name, parent) in [
        (animal, "schema:Animal", None),
        (mammal, "schema:Mammal", Some(animal)),
        (dog, "schema:Dog", Some(mammal)),
    ] {
        let mut m = meta(id, name);
        m.label = None;
        m.parent_class_id = parent;
        // The only write path: `insert` persists the parent edge from the
        // same metadata field the loader fills (C5 — one source of edges).
        cs.insert(m).expect("insert");
    }
    cs.hydrate_hierarchy().expect("hydrate");
    assert_eq!(cs.ancestors_of(dog), vec![mammal, animal]);
    assert!(cs.is_subclass_of(dog, animal));
    assert!(cs.is_subclass_of(dog, dog));
    assert!(!cs.is_subclass_of(animal, dog));
    assert_eq!(cs.yago_class_count().expect("count"), 3);
}

#[test]
fn from_sql_truncates_high_bits_to_48() {
    // Characterization (M1a): `from_sql` masks the stored id to the 48-bit
    // local (`LOCAL_MASK`); a value with high bits set round-trips truncated.
    let (_store, cs) = open();
    let id = InterlinguaId::new(InterlinguaNamespace::YagoClass, 0x42);
    cs.insert(meta(id, "schema:HighBits")).expect("insert");
    // Directly widen the stored id with high bits, then read it back.
    let raw: i64 = 0x1234_0000_0000_0042_i64;
    cs.store
        .with_conn(|conn| {
            conn.execute(
                "UPDATE interlingua_concepts SET id = ?1 WHERE canonical_name = 'schema:HighBits'",
                rusqlite::params![raw],
            )
            .map(|_| ())
            .map_err(fluent_db::error::DbError::from)
        })
        .expect("widen");
    let got: i64 = cs
        .store
        .query_row(
            "SELECT id FROM interlingua_concepts WHERE canonical_name = 'schema:HighBits'",
            &[],
            |row| row.get(0),
        )
        .expect("read raw")
        .expect("row");
    assert_eq!(got, raw);
    // The read path masks through `local_id_of`: high bits do not survive.
    let masked = cs
        .get(InterlinguaId::from_sql(0x0100, raw))
        .expect_err("widened row unreachable via masked id");
    let _ = masked;
    // Restore the canonical id and prove the masked round-trip is clean.
    cs.store
        .with_conn(|conn| {
            conn.execute(
                "UPDATE interlingua_concepts SET id = ?1 WHERE canonical_name = 'schema:HighBits'",
                rusqlite::params![id.as_i64()],
            )
            .map(|_| ())
            .map_err(fluent_db::error::DbError::from)
        })
        .expect("restore");
    let back = cs.get(id).expect("get");
    assert_eq!(back.id, id);
    assert_eq!(back.id, InterlinguaId::from_sql(0x0100, raw));
    assert_eq!(back.id.local_id() & !fluent_types::LOCAL_MASK as i64, 0);
}
