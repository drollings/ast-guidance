use super::*;
use fluent_types::{local_id_of, NodeId, InterlinguaNamespace};

fn cid(ns: InterlinguaNamespace, h: i64) -> InterlinguaId {
    InterlinguaId::new(ns, local_id_of(h))
}

fn meta(id: InterlinguaId, name: &str, iri: Option<&str>) -> ConceptMetadata {
    ConceptMetadata {
        id,
        canonical_name: name.to_string(),
        namespace: id.namespace(),
        yago_iri: iri.map(ToString::to_string),
        yago_class_iri: None,
        label: Some(name.to_string()),
        node_id: Some(NodeId::from_int(id.local_id() ^ 0x8000_0000_0000)),
        parent_class_id: None,
    }
}

#[test]
fn insert_get_roundtrip_and_indexes() {
    let store = InMemoryConceptStore::new();
    let id = cid(InterlinguaNamespace::YagoClass, 0x42);
    let m = meta(id, "schema:Person", Some("http://yago-knowledge.org/resource/Person"));
    store.insert(m.clone()).expect("insert");
    assert_eq!(store.get(id).expect("get").id, id);
    assert_eq!(store.resolve_name("schema:Person").expect("name"), id);
    assert_eq!(
        store
            .resolve_yago_iri("http://yago-knowledge.org/resource/Person")
            .expect("iri"),
        id
    );
    assert!(store.contains(id));
    assert_eq!(store.iter_ids().count(), 1);
}

#[test]
fn first_wins_keeps_both_canonicals_under_one_id() {
    let store = InMemoryConceptStore::new();
    let shared = cid(InterlinguaNamespace::SpacyLemma, 7);
    store.insert(meta(shared, "lead_verb", None)).expect("first");
    store.insert(meta(shared, "lead_metal", None)).expect("second");

    // Both canonicals are resolvable under the same bucket id.
    assert_eq!(store.resolve_name("lead_verb").expect("a"), shared);
    assert_eq!(store.resolve_name("lead_metal").expect("b"), shared);
    // The incumbent canonical is unchanged (first-wins).
    assert_eq!(store.get(shared).expect("get").canonical_name, "lead_verb");
    assert_eq!(store.iter_ids().count(), 1, "one bucket id");
}

#[test]
fn unknown_lookups_error() {
    let store = InMemoryConceptStore::new();
    assert!(matches!(
        store.get(cid(InterlinguaNamespace::YagoClass, 1)),
        Err(ConceptStoreError::NotFound(_))
    ));
    assert!(matches!(
        store.resolve_name("nope"),
        Err(ConceptStoreError::NotFound(_))
    ));
}

#[test]
fn hierarchy_derives_from_insert_metadata() {
    // No explicit `set_hierarchy` — the hierarchy must come from the same
    // `parent_class_id` metadata field the loader fills (C5).
    let store = InMemoryConceptStore::new();
    let animal = cid(InterlinguaNamespace::YagoClass, 1);
    let mammal = cid(InterlinguaNamespace::YagoClass, 2);
    let dog = cid(InterlinguaNamespace::YagoClass, 3);
    for (id, name, parent) in [
        (animal, "schema:Animal", None),
        (mammal, "schema:Mammal", Some(animal)),
        (dog, "schema:Dog", Some(mammal)),
    ] {
        let mut m = meta(id, name, None);
        m.parent_class_id = parent;
        store.insert(m).expect("insert");
    }
    assert_eq!(store.ancestors_of(dog), vec![mammal, animal]);
    assert!(store.is_subclass_of(dog, animal));
    assert!(store.is_subclass_of(dog, dog));
    assert!(!store.is_subclass_of(animal, dog));
}

#[test]
fn ancestors_and_is_subclass_backed_by_hierarchy() {
    let store = InMemoryConceptStore::new();
    let animal = cid(InterlinguaNamespace::YagoClass, 1);
    let mammal = cid(InterlinguaNamespace::YagoClass, 2);
    let dog = cid(InterlinguaNamespace::YagoClass, 3);
    store.insert(meta(animal, "schema:Animal", None)).expect("animal");
    store.insert(meta(mammal, "schema:Mammal", None)).expect("mammal");
    store.insert(meta(dog, "schema:Dog", None)).expect("dog");

    let hierarchy = TaxonomyHierarchy::from_edges(&[(mammal, animal), (dog, mammal)])
        .expect("hierarchy");
    store.set_hierarchy(hierarchy);

    assert_eq!(store.ancestors_of(dog), vec![mammal, animal]);
    assert!(store.is_subclass_of(dog, animal));
    assert!(store.is_subclass_of(dog, dog));
    assert!(!store.is_subclass_of(animal, dog));
}
