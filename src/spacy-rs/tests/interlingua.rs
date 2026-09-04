use super::*;
use crate::concept_store::ConceptStore;
use crate::concept_store_mem::InMemoryConceptStore;
use crate::vocab::Vocab;
use fluent_types::{local_id_of, NodeId};

fn resolver() -> (Arc<InMemoryConceptStore>, Arc<StringStore>, InterlinguaResolver) {
    let store = Arc::new(InMemoryConceptStore::new());
    let strings = Arc::new(StringStore::new());
    let r = InterlinguaResolver::new(Arc::clone(&store) as Arc<dyn ConceptStore>, Arc::clone(&strings));
    (store, strings, r)
}

fn meta(id: InterlinguaId, name: &str) -> ConceptMetadata {
    ConceptMetadata {
        id,
        canonical_name: name.to_string(),
        namespace: id.namespace(),
        yago_iri: None,
        yago_class_iri: None,
        label: None,
        node_id: Some(NodeId::from_int(id.local_id())),
        parent_class_id: None,
    }
}

#[test]
fn lemma_id_is_deterministic_and_pure() {
    let (store, strings, r1) = resolver();
    let r2 = InterlinguaResolver::new(Arc::clone(&store) as Arc<dyn ConceptStore>, Arc::clone(&strings));
    let a = r1.lemma_id("run");
    let b = r2.lemma_id("run");
    assert_eq!(a, b, "order-independence across resolver instances");
    assert!(a.is_spacy_lemma());
    assert_eq!(a.local_id(), local_id_of(hash_utf8("run") as i64));
    // C1 parity: the resolver's manual construction and the shared `types`
    // helper must agree.
    assert_eq!(a, fluent_types::lemma_id_for_str("run"));
}

#[test]
fn resolve_hash_no_collision_when_absent() {
    let (store, _, r) = resolver();
    // No concept registered under "run" → no collision note.
    let h = hash_utf8("run");
    let (id, note) = r.resolve_hash(h, "run");
    assert_eq!(id, r.lemma_id("run"));
    assert_eq!(note, CollisionNote::None);
    store.insert(meta(id, "run")).expect("insert");
    let (_, note) = r.resolve_hash(h, "run");
    assert_eq!(note, CollisionNote::None, "same canonical → no collision");
}

#[test]
fn resolve_hash_flags_collision_for_second_canonical() {
    let (store, _, r) = resolver();
    let id = r.lemma_id("lead");

    // Simulate a collision: a concept already claims `lead`'s id under a
    // *different* canonical (real 48-bit hash collisions are too rare to
    // construct deterministically, so we register the second canonical
    // directly — the store keeps both under the shared bucket id, §2.3).
    store.insert(meta(id, "first_canonical")).expect("first canonical");

    let (id2, note) = r.resolve_hash(hash_utf8("lead"), "lead");
    assert_eq!(id2, id);
    match note {
        CollisionNote::Collision { prior_canonical, .. } => {
            assert_eq!(prior_canonical, "first_canonical");
        }
        CollisionNote::None => panic!("expected a collision note"),
    }
    // Both canonicals remain resolvable.
    assert_eq!(r.canonical(id), Some("first_canonical".into()));
}

#[test]
fn canonical_and_metadata_roundtrip() {
    let (store, _, r) = resolver();
    let id = r.lemma_id("dog");
    store.insert(meta(id, "dog")).expect("insert");
    assert_eq!(r.canonical(id), Some("dog".into()));
    assert_eq!(r.metadata(id).map(|m| m.canonical_name), Some("dog".into()));
    assert!(r.metadata(r.lemma_id("absent")).is_none());
    assert!(r.canonical(r.lemma_id("absent")).is_none());
}

#[test]
fn resolve_doc_stamps_lemma_ids_and_confidence_readonly() {
    let (store, _, r) = resolver();
    let vocab = Arc::new(Vocab::new(crate::lang::en::lexicon_config()));
    let tokenizer = crate::lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    let mut doc = tokenizer.tokenize("The cat sat.").expect("tokenize");

    // Simulate an attached parse: intern real lemma hashes so the string
    // store reverse-lookup resolves them.
    let lemmas = ["the", "cat", "sit", "."];
    for (i, l) in lemmas.iter().enumerate() {
        doc.token_mut(i).lemma = vocab.strings().add(l);
    }

    let conf: Vec<f64> = vec![0.9, 0.8, 0.7, 1.0];
    let notes = r.resolve_doc(&mut doc, Some(&conf));

    assert!(notes.is_empty(), "no concepts registered → no collisions");
    for (i, l) in lemmas.iter().enumerate() {
        let t = doc.token(i);
        assert_eq!(t.interlingua_lemma_id, Some(r.lemma_id(l)), "token {i} stamped");
        assert_eq!(t.confidence, Some(conf[i]));
    }
    // The store was never written by the resolver (boot-only invariant).
    assert_eq!(store.iter_ids().count(), 0);
}

#[test]
fn resolve_doc_skips_uninterned_lemmas() {
    let (store, _, r) = resolver();
    let vocab = Arc::new(Vocab::new(crate::lang::en::lexicon_config()));
    let tokenizer = crate::lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    let mut doc = tokenizer.tokenize("The cat sat.").expect("tokenize");

    let lemmas = ["the", "cat", "sit", "."];
    for (i, l) in lemmas.iter().enumerate() {
        doc.token_mut(i).lemma = vocab.strings().add(l);
    }
    // Mark one lemma hash as never interned (the store has no reverse
    // mapping) — that token must be skipped, not stamped.
    let uninterned = crate::hash::hash_utf8("zzz_not_in_store");
    doc.token_mut(1).lemma = uninterned;
    assert!(vocab.strings().get(uninterned).is_none());

    let notes = r.resolve_doc(&mut doc, None);
    assert!(notes.is_empty());
    assert!(doc.token(0).interlingua_lemma_id.is_some());
    assert!(doc.token(1).interlingua_lemma_id.is_none(), "uninterned lemma skipped");
    assert!(doc.token(2).interlingua_lemma_id.is_some());
    // Store untouched.
    assert_eq!(store.iter_ids().count(), 0);
}

#[test]
fn resolve_doc_shorter_confidence_slice_skips_not_panics() {
    // L5: a caller-bug confidence vector shorter than the doc must leave
    // the trailing tokens' confidence unset — never index-panic the hot
    // path.
    let (store, _, r) = resolver();
    let vocab = Arc::new(Vocab::new(crate::lang::en::lexicon_config()));
    let tokenizer = crate::lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    let mut doc = tokenizer.tokenize("The cat sat.").expect("tokenize");
    let lemmas = ["the", "cat", "sit", "."];
    for (i, l) in lemmas.iter().enumerate() {
        doc.token_mut(i).lemma = vocab.strings().add(l);
    }

    let short = vec![0.9, 0.8];
    let _ = r.resolve_doc(&mut doc, Some(&short));

    assert_eq!(doc.token(0).confidence, Some(0.9));
    assert_eq!(doc.token(1).confidence, Some(0.8));
    assert_eq!(doc.token(2).confidence, None, "out-of-range → unset, not panic");
    assert_eq!(doc.token(3).confidence, None);
    // Ids are still stamped for every resolvable token.
    assert!(doc.token(2).interlingua_lemma_id.is_some());
    assert_eq!(store.iter_ids().count(), 0);
}
