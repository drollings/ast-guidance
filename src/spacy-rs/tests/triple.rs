use super::*;
use fluent_concept::{ConceptStore, InMemoryConceptStore};
use crate::interlingua::InterlinguaResolver;
use crate::llm::{attach, AnnotationSet};
use crate::sentencizer::Sentencizer;
use crate::vocab::Vocab;
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

// ── M5 seam tests: spacy-rs produces the input, the knowledge owner scores ──

#[test]
fn plausibility_inputs_carry_ids_and_lemmas() {
    let store = Arc::new(InMemoryConceptStore::new());
    let resolver = resolver_for(Arc::clone(&store));
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let triples = extract_triples(&doc);
    let inputs = build_plausibility_inputs(&doc, &triples, &resolver);
    assert_eq!(inputs.len(), 1);
    let t = &inputs[0];
    let subj = t.subject.as_ref().expect("dog subject");
    assert_eq!(subj.lemma, "dog");
    assert_eq!(subj.id, resolver.lemma_id("dog"));
    assert_eq!(t.predicate.lemma, "bark");
    assert!(t.object.is_none());
}

#[test]
fn via_fetch_returns_stub_score_and_is_deterministic() {
    let store = Arc::new(InMemoryConceptStore::new());
    let resolver = resolver_for(Arc::clone(&store));
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let triples = extract_triples(&doc);
    // Stub knowledge half: the seam round-trips the text-half input.
    let fetch: PlausibilityFetch = Arc::new(|inputs| {
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].predicate.lemma, "bark");
        Some(0.75)
    });
    let a = semantic_plausibility_via_fetch(&doc, &triples, &resolver, Some(&fetch));
    let b = semantic_plausibility_via_fetch(&doc, &triples, &resolver, Some(&fetch));
    assert_eq!(a, Some(0.75));
    assert_eq!(a, b, "deterministic");
}

#[test]
fn via_fetch_without_fetch_is_fail_closed_none() {
    let store = Arc::new(InMemoryConceptStore::new());
    let resolver = resolver_for(Arc::clone(&store));
    let doc = attached(DOG_BARK_PARSE, &["The", "dog", "barks", "."]);
    let triples = extract_triples(&doc);
    assert!(semantic_plausibility_via_fetch(&doc, &triples, &resolver, None).is_none());
}

#[test]
fn via_fetch_empty_triples_is_none_even_with_fetch() {
    let store = Arc::new(InMemoryConceptStore::new());
    let resolver = resolver_for(Arc::clone(&store));
    let fetch: PlausibilityFetch = Arc::new(|_| Some(1.0));
    let empty: Vec<Triple> = Vec::new();
    let doc = attached(CAT_SAT_PARSE, &["The", "cat", "sat", "."]);
    assert!(semantic_plausibility_via_fetch(&doc, &empty, &resolver, Some(&fetch)).is_none());
}
