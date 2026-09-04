//! Moved with the scoring (M5.5): the hermetic `Dog→Animal` plausibility
//! suite. spacy-rs produces the input (`extract_triples` +
//! `build_plausibility_inputs` over an attached `Doc`), the knowledge owner
//! (`score_plausibility`) consumes it — no behavior change for callers.

use super::*;
use fluent_concept::{ConceptStore, ConceptStoreState, InMemoryConceptStore};
use fluent_types::{ConceptMetadata, InterlinguaId, NodeId};
use std::sync::Arc;

fn vocab() -> Arc<spacy_rs::Vocab> {
    Arc::new(spacy_rs::Vocab::new(spacy_rs::LexiconConfig::default()))
}

fn attached(json: &str, tokens: &[&str]) -> spacy_rs::Doc {
    let mut doc = spacy_rs::Doc::new(vocab());
    for t in tokens {
        doc.push_back(t, true).expect("push");
    }
    let set = spacy_rs::AnnotationSet::parse_json(json).expect("json");
    spacy_rs::llm::attach(&mut doc, &set).expect("attach");
    spacy_rs::Sentencizer::new().process(&mut doc);
    doc
}

fn lemma_id_for(lemma: &str) -> InterlinguaId {
    let store = InMemoryConceptStore::new();
    let resolver = spacy_rs::InterlinguaResolver::new(
        Arc::new(store) as Arc<dyn ConceptStore>,
        Arc::new(spacy_rs::StringStore::new()),
    );
    resolver.lemma_id(lemma)
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

fn scored_inputs(
    doc: &spacy_rs::Doc,
    resolver: &spacy_rs::InterlinguaResolver,
) -> Vec<fluent_concept::PlausibilityTriple> {
    let triples = spacy_rs::extract_triples(doc);
    spacy_rs::build_plausibility_inputs(doc, &triples, resolver)
}

fn resolver_for(store: Arc<InMemoryConceptStore>) -> spacy_rs::InterlinguaResolver {
    spacy_rs::InterlinguaResolver::new(
        Arc::clone(&store) as Arc<dyn ConceptStore>,
        Arc::clone(vocab().strings()),
    )
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
fn semantic_plausibility_hermetic_dog_is_animal() {
    // Hermetic taxonomy: Dog → Animal (subClassOf chain)
    let store = Arc::new(InMemoryConceptStore::new());
    // Use lemma-derived ids so the kernel can match them via lemma id
    let dog_lemma = lemma_id_for("dog");
    let animal_lemma = lemma_id_for("animal");
    // For the hermetic spike, register the dog lemma id itself
    store.insert(meta(dog_lemma, "dog", None)).expect("dog");
    store.insert(meta(animal_lemma, "animal", None)).expect("animal");
    // Also build hierarchy Dog → Animal via explicit edges (lemma ids)
    let hierarchy = fluent_concept::TaxonomyHierarchy::from_edges(&[(dog_lemma, animal_lemma)]).expect("hierarchy");
    store.set_hierarchy(hierarchy);
    assert!(store.is_subclass_of(dog_lemma, animal_lemma));

    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let resolver = resolver_for(Arc::clone(&store));
    let inputs = scored_inputs(&doc, &resolver);
    let score = score_plausibility(&inputs, &*store).expect("score");
    assert!(score > 0.5, "dog-known triple must be plausible, got {score}");
}

#[test]
fn semantic_plausibility_unknown_is_zero() {
    let store = Arc::new(InMemoryConceptStore::new());
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let resolver = resolver_for(Arc::clone(&store));
    let inputs = scored_inputs(&doc, &resolver);
    // Empty store → no lemma known
    let score = score_plausibility(&inputs, &*store).expect("score");
    assert_eq!(score, 0.0);
}

#[test]
fn semantic_plausibility_loading_is_none() {
    let store = Arc::new(InMemoryConceptStore::new());
    store.set_state(ConceptStoreState::Loading);
    let dog_lemma = lemma_id_for("dog");
    store.insert(meta(dog_lemma, "dog", None)).expect("dog");
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let resolver = resolver_for(Arc::clone(&store));
    let inputs = scored_inputs(&doc, &resolver);
    assert!(score_plausibility(&inputs, &*store).is_none());
}

#[test]
fn semantic_plausibility_bounded_and_deterministic() {
    let store = Arc::new(InMemoryConceptStore::new());
    let dog_lemma = lemma_id_for("dog");
    store.insert(meta(dog_lemma, "dog", None)).expect("dog");
    let doc = attached(CAT_SAT_PARSE, &["The", "cat", "sat", "."]);
    let resolver = resolver_for(Arc::clone(&store));
    let inputs = scored_inputs(&doc, &resolver);
    let a = score_plausibility(&inputs, &*store).expect("a");
    let b = score_plausibility(&inputs, &*store).expect("b");
    assert_eq!(a, b, "deterministic");
    assert!((0.0..=1.0).contains(&a), "bounded {a}");
}

#[test]
fn empty_inputs_yield_none() {
    let store = InMemoryConceptStore::new();
    let empty: Vec<fluent_concept::PlausibilityTriple> = Vec::new();
    assert!(score_plausibility(&empty, &store).is_none());
}

#[test]
fn never_folds_into_oracle_margins() {
    // E7 regression at the new home (M5.3): the kernel returns a bare score
    // for the separate `semantic_plausibility` field; `oracle_margins` are
    // never an input or an output of scoring, regardless of residence.
    let store = Arc::new(InMemoryConceptStore::new());
    let dog_lemma = lemma_id_for("dog");
    store.insert(meta(dog_lemma, "dog", None)).expect("dog");
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let resolver = resolver_for(Arc::clone(&store));
    let inputs = scored_inputs(&doc, &resolver);
    let score = score_plausibility(&inputs, &*store);
    let mut pc = spacy_rs::ParseConfidence::compute(&[0.9], &[0.0, 0.25], 1.0);
    let margins_before = pc.oracle_margins.clone();
    pc.semantic_plausibility = score;
    assert_eq!(pc.oracle_margins, margins_before);
    assert_eq!(pc.oracle_margins, vec![0.0, 0.25]);
}
