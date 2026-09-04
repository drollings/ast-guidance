//! ROADMAP M11.8 — the YaGO taxonomy through fluent-concept's ConceptStore +
//! the spacy-rs resolver.
//!
//! Verifies the interlingua bridge end to end: `YaGoLoader` produces
//! `ConceptMetadata` with deterministic ids, registration through
//! `InMemoryConceptStore` makes `schema:Person` resolvable, and the pure
//! `InterlinguaResolver` finds it (with the full `node_id` cross-reference,
//! F5). Dev-only dependency: spacy-rs never depends on ontology (§15).

use std::sync::Arc;

use fluent_concept::{ConceptStore, InMemoryConceptStore, TaxonomyHierarchy};
use fluent_types::{property_id_for_iri, yago_class_id_for_iri, InterlinguaNamespace};
use guidance_ontology::yago_loader::YaGoLoader;
use spacy_rs::interlingua::InterlinguaResolver;

#[test]
fn schema_person_resolves_through_the_store_and_resolver() {
    let mut loader = YaGoLoader::new();
    let stats = loader.load_embedded().expect("embedded registry");
    assert!(stats.classes >= 7, "reference classes present");

    let store = Arc::new(InMemoryConceptStore::new());
    let edges = loader.subclass_edges().to_vec();
    let concepts = loader.into_concepts();
    for meta in &concepts {
        store.insert(meta.clone()).expect("insert");
    }
    // The subclass hierarchy mirrors the loader edges (C5).
    let hierarchy = TaxonomyHierarchy::from_edges(&edges).expect("hierarchy");
    store.set_hierarchy(hierarchy);

    // The resolver (pure, no state) sits over the same store.
    let strings = Arc::new(spacy_rs::StringStore::new());
    let resolver = InterlinguaResolver::new(
        Arc::clone(&store) as Arc<dyn ConceptStore>,
        Arc::clone(&strings),
    );

    // `schema:Person` resolves by canonical name.
    let person_id = store.resolve_name("schema:Person").expect("schema:Person");
    assert!(person_id.is_yago());
    assert_eq!(person_id.namespace(), InterlinguaNamespace::YagoClass);

    // The resolver returns the metadata with the full node_id cross-ref (F5).
    let meta = resolver.metadata(person_id).expect("metadata");
    assert_eq!(meta.canonical_name, "schema:Person");
    let node_id = meta.node_id.expect("node_id");
    assert_eq!(
        node_id.as_int(),
        guidance_rdf::normalize::hash_iri("http://schema.org/Person")
    );
    assert_ne!(node_id.as_int(), person_id.local_id());

    // Hierarchy: Dog → Mammal → Animal → Entity; Person → Agent → Entity.
    let dog = yago_class_id_for_iri("http://yago-knowledge.org/resource/Dog");
    let animal = yago_class_id_for_iri("http://yago-knowledge.org/resource/Animal");
    let entity = yago_class_id_for_iri("http://yago-knowledge.org/resource/Entity");
    assert!(store.is_subclass_of(dog, animal));
    assert!(store.is_subclass_of(dog, entity));
    assert!(store.is_subclass_of(person_id, entity));
    assert_eq!(store.ancestors_of(dog).len(), 3, "Mammal, Animal, Entity");
}

#[test]
fn loader_and_yago_helpers_agree_on_ids() {
    // C1 parity: the ontology loader's manual construction and the shared
    // `types` helper must derive the same id for the same input.
    let loader = YaGoLoader::new();
    let mut l = loader;
    l.load_embedded().expect("embedded registry");
    let concepts = l.into_concepts();
    for c in &concepts {
        let expected = yago_class_id_for_iri(c.yago_class_iri.as_deref().unwrap_or(&c.canonical_name));
        assert_eq!(c.id, expected, "loader id == helper id for {}", c.canonical_name);
    }
}

#[test]
fn yago_property_and_class_helpers_parity() {
    // The `property_interlingua_id` function and the shared helper agree.
    let subclass = guidance_ontology::yago::PROP_SUBCLASS;
    let manual = guidance_ontology::yago::property_interlingua_id(&subclass);
    let helper = property_id_for_iri(subclass.iri);
    assert_eq!(manual, helper);
    // Same for a class id used in the loader.
    assert_eq!(
        yago_class_id_for_iri("http://schema.org/Person"),
        fluent_types::yago_class_id_for_iri("http://schema.org/Person")
    );
}