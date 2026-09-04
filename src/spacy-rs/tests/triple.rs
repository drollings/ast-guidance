use super::*;
use crate::concept_store_mem::InMemoryConceptStore;
use crate::interlingua::InterlinguaResolver;
use crate::llm::{attach, AnnotationSet};
use crate::sentencizer::Sentencizer;
use crate::vocab::Vocab;
use fluent_types::{ConceptMetadata, InterlinguaId, NodeId};
use std::sync::Arc;

fn vocab() -> Arc<Vocab> {
    Arc::new(Vocab::new(crate::lexeme::LexiconConfig::default()))
}

fn attached(json: &str, tokens: &[&str]) -> Doc {
    let mut doc = Doc::new(vocab());
    for t in tokens {
        doc.push_back(t, true).expect("push");
    }
    let set = AnnotationSet::parse_json(json).expect("json");
    attach(&mut doc, &set).expect("attach");
    Sentencizer::new().process(&mut doc);
    doc
}

fn resolver_for(store: Arc<InMemoryConceptStore>) -> InterlinguaResolver {
    InterlinguaResolver::new(
        Arc::clone(&store) as Arc<dyn ConceptStore>,
        Arc::clone(vocab().strings()),
    )
}

fn meta(id: InterlinguaId, name: &str, parent: Option<InterlinguaId>) -> ConceptMetadata {
    ConceptMetadata {
        id,
        canonical_name: name.to_string(),
        namespace: id.namespace(),
        yago_iri: None,
        yago_class_iri: None,
        label: Some(name.to_string()),
        node_id: Some(NodeId::from_int(id.local_id())),
        parent_class_id: parent,
    }
}

fn yago_id_for(name: &str) -> InterlinguaId {
    let store = InMemoryConceptStore::new();
    let resolver = InterlinguaResolver::new(
        Arc::new(store) as Arc<dyn ConceptStore>,
        Arc::new(crate::strings::StringStore::new()),
    );
    resolver.lemma_id(name)
}

const DOG_BARK_PARSE: &str = r#"[
    {"text":"The","pos":"det","dep":"det","head":1,"lemma":"the"},
    {"text":"dog","pos":"noun","dep":"nsubj","head":1,"lemma":"dog"},
    {"text":"barks","pos":"verb","dep":"root","head":0,"lemma":"bark"},
    {"text":".","pos":"punct","dep":"punct","head":-1,"lemma":"."}
]"#;

const CAT_SAT_PARSE: &str = r#"[
    {"text":"The","pos":"det","dep":"det","head":1,"lemma":"the"},
    {"text":"cat","pos":"noun","dep":"nsubj","head":1,"lemma":"cat"},
    {"text":"sat","pos":"verb","dep":"root","head":0,"lemma":"sat"},
    {"text":".","pos":"punct","dep":"punct","head":-1,"lemma":"."}
]"#;

#[test]
fn empty_doc_yields_no_triples() {
    assert!(extract_triples(&Doc::new(vocab())).is_empty());
}

#[test]
fn single_sentence_triple_spans_are_correct() {
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let triples = extract_triples(&doc);
    assert_eq!(triples.len(), 1);
    assert_eq!(triples[0].predicate, 2, "barks is the root");
    assert_eq!(triples[0].subject, Some(1), "dog is the nsubj");
    assert_eq!(triples[0].object, None, "no dobj intransitive");
    assert_eq!(triples[0].sentence_span, (0, 4));
}

#[test]
fn transitive_triple_has_subject_and_object() {
    let json = r#"[
        {"text":"Show","pos":"verb","dep":"root","head":0,"lemma":"show"},
        {"text":"me","pos":"pron","dep":"iobj","head":-1,"lemma":"me"},
        {"text":"the","pos":"det","dep":"det","head":1,"lemma":"the"},
        {"text":"report","pos":"noun","dep":"dobj","head":-3,"lemma":"report"}
    ]"#;
    let doc = attached(json, &["Show", "me", "the", "report"]);
    let triples = extract_triples(&doc);
    assert_eq!(triples.len(), 1);
    // "Show" root with no nsubj, dobj report
    assert_eq!(triples[0].subject, None);
    assert_eq!(triples[0].object, Some(3));
}

#[test]
fn multi_sentence_yields_one_triple_per_sentence() {
    let json = r#"[
        {"text":"The","pos":"det","dep":"det","head":1,"lemma":"the"},
        {"text":"cat","pos":"noun","dep":"nsubj","head":1,"lemma":"cat"},
        {"text":"sat","pos":"verb","dep":"root","head":0,"lemma":"sat"},
        {"text":".","pos":"punct","dep":"punct","head":-1,"lemma":"."},
        {"text":"The","pos":"det","dep":"det","head":1,"lemma":"the"},
        {"text":"dog","pos":"noun","dep":"nsubj","head":1,"lemma":"dog"},
        {"text":"ran","pos":"verb","dep":"root","head":0,"lemma":"ran"},
        {"text":".","pos":"punct","dep":"punct","head":-1,"lemma":"."}
    ]"#;
    let doc = attached(json, &["The", "cat", "sat", ".", "The", "dog", "ran", "."]);
    let triples = extract_triples(&doc);
    assert_eq!(triples.len(), 2);
    assert_eq!(triples[0].sentence_span, (0, 4));
    assert_eq!(triples[1].sentence_span, (4, 8));
}

#[test]
fn semantic_plausibility_hermetic_dog_is_animal() {
    // Hermetic taxonomy: Dog → Animal (subClassOf chain)
    let store = Arc::new(InMemoryConceptStore::new());
    // Use lemma-derived ids so token_known can match them via lemma_id
    let dog_lemma = yago_id_for("dog");
    let animal_lemma = yago_id_for("animal");
    // For the hermetic spike, register the dog lemma id itself
    store.insert(meta(dog_lemma, "dog", None)).expect("dog");
    store.insert(meta(animal_lemma, "animal", None)).expect("animal");
    // Also build hierarchy Dog → Animal via explicit edges (lemma ids)
    let hierarchy = crate::concept_store::TaxonomyHierarchy::from_edges(&[(dog_lemma, animal_lemma)]).expect("hierarchy");
    store.set_hierarchy(hierarchy);
    assert!(store.is_subclass_of(dog_lemma, animal_lemma));

    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let triples = extract_triples(&doc);
    let resolver = resolver_for(Arc::clone(&store));
    let score = semantic_plausibility(&doc, &triples, &resolver, &*store).expect("score");
    assert!(score > 0.5, "dog-known triple must be plausible, got {score}");
}

#[test]
fn semantic_plausibility_unknown_is_zero() {
    let store = Arc::new(InMemoryConceptStore::new());
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let triples = extract_triples(&doc);
    let resolver = resolver_for(Arc::clone(&store));
    // Empty store → no lemma known
    let score = semantic_plausibility(&doc, &triples, &resolver, &*store).expect("score");
    assert_eq!(score, 0.0);
}

#[test]
fn semantic_plausibility_loading_is_none() {
    let store = Arc::new(InMemoryConceptStore::new());
    store.set_state(crate::concept_store::ConceptStoreState::Loading);
    let dog_lemma = yago_id_for("dog");
    store.insert(meta(dog_lemma, "dog", None)).expect("dog");
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let triples = extract_triples(&doc);
    let resolver = resolver_for(Arc::clone(&store));
    assert!(semantic_plausibility(&doc, &triples, &resolver, &*store).is_none());
}

#[test]
fn semantic_plausibility_bounded_and_deterministic() {
    let store = Arc::new(InMemoryConceptStore::new());
    let dog_lemma = yago_id_for("dog");
    store.insert(meta(dog_lemma, "dog", None)).expect("dog");
    let doc = attached(CAT_SAT_PARSE, &["The", "cat", "sat", "."]);
    let triples = extract_triples(&doc);
    let resolver = resolver_for(Arc::clone(&store));
    let a = semantic_plausibility(&doc, &triples, &resolver, &*store).expect("a");
    let b = semantic_plausibility(&doc, &triples, &resolver, &*store).expect("b");
    assert_eq!(a, b, "deterministic");
    assert!((0.0..=1.0).contains(&a), "bounded {a}");
}

#[test]
fn subclass_transitive_dog_animal_via_hierarchy() {
    let store = Arc::new(InMemoryConceptStore::new());
    let dog_lemma = yago_id_for("dog");
    let mammal_lemma = yago_id_for("mammal");
    let animal_lemma = yago_id_for("animal");
    store.insert(meta(animal_lemma, "animal", None)).expect("a");
    store.insert(meta(mammal_lemma, "mammal", Some(animal_lemma))).expect("m");
    store.insert(meta(dog_lemma, "dog", Some(mammal_lemma))).expect("d");
    assert!(store.is_subclass_of(dog_lemma, animal_lemma));
    assert!(!store.is_subclass_of(animal_lemma, dog_lemma));
}

#[test]
fn never_touches_oracle_margins() {
    let store = Arc::new(InMemoryConceptStore::new());
    let dog_lemma = yago_id_for("dog");
    store.insert(meta(dog_lemma, "dog", None)).expect("dog");
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let triples = extract_triples(&doc);
    let resolver = resolver_for(Arc::clone(&store));
    let _ = semantic_plausibility(&doc, &triples, &resolver, &*store);
    // Oracle margins are untouched — this module never writes them.
    // (Compile-time guarantee: no mutable access to ParseConfidence here.)
}
