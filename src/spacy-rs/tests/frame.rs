use super::*;
use fluent_concept::InMemoryConceptStore;
use crate::llm::{attach, AnnotationSet};
use crate::sentencizer::Sentencizer;
use crate::vocab::Vocab;
use fluent_types::ConceptMetadata;
use std::sync::Arc;

/// A single shared vocab for the whole test module so the resolver's
/// string store and every doc's vocab agree (the same sharing the pipeline
/// establishes between its resolver and the docs it processes).
fn vocab() -> Arc<Vocab> {
    static VOCAB: std::sync::OnceLock<Arc<Vocab>> = std::sync::OnceLock::new();
    Arc::clone(
        VOCAB.get_or_init(|| Arc::new(Vocab::new(crate::lexeme::LexiconConfig::default()))),
    )
}

fn doc_for(tokens: &[&str]) -> Doc {
    let mut doc = Doc::new(vocab());
    for t in tokens {
        doc.push_back(t, true).expect("push");
    }
    doc
}

/// Attach a full UD parse and re-run the sentencizer so the doc carries
/// sentence boundaries (the state `process_sync` produces).
fn attached(text_json: &str, tokens: &[&str]) -> Doc {
    let mut doc = doc_for(tokens);
    let set = AnnotationSet::parse_json(text_json).expect("parse json");
    attach(&mut doc, &set).expect("attach");
    Sentencizer::new().process(&mut doc);
    doc
}

fn extractor(store: Arc<InMemoryConceptStore>) -> FrameExtractor {
    let resolver = Arc::new(InterlinguaResolver::new(
        Arc::clone(&store) as Arc<dyn ConceptStore>,
        Arc::clone(vocab().strings()),
    ));
    FrameExtractor::new(resolver, Arc::clone(&store) as Arc<dyn ConceptStore>)
}

const FULL_PARSE: &str = r#"[
    {"text":"Show","pos":"verb","dep":"root","head":0,"lemma":"show"},
    {"text":"me","pos":"pron","dep":"iobj","head":-1,"lemma":"me"},
    {"text":"the","pos":"det","dep":"det","head":2,"lemma":"the"},
    {"text":"sales","pos":"noun","dep":"compound","head":1,"lemma":"sales"},
    {"text":"report","pos":"noun","dep":"dobj","head":-4,"lemma":"report"},
    {"text":"for","pos":"adp","dep":"prep","head":-5,"lemma":"for"},
    {"text":"yesterday","pos":"noun","dep":"pobj","head":-1,"lemma":"yesterday"},
    {"text":"please","pos":"adv","dep":"advmod","head":-7,"lemma":"please"}
]"#;

const NEG_MODAL_PARSE: &str = r#"[
    {"text":"The","pos":"det","dep":"det","head":1,"lemma":"the"},
    {"text":"dog","pos":"noun","dep":"nsubj","head":3,"lemma":"dog"},
    {"text":"will","pos":"aux","dep":"aux","head":2,"lemma":"will"},
    {"text":"not","pos":"adv","dep":"neg","head":1,"lemma":"not"},
    {"text":"sleep","pos":"verb","dep":"root","head":0,"lemma":"sleep"}
]"#;

fn meta(id: InterlinguaId, name: &str) -> ConceptMetadata {
    ConceptMetadata {
        id,
        canonical_name: name.to_string(),
        namespace: id.namespace(),
        yago_iri: None,
        yago_class_iri: None,
        label: Some(name.to_string()),
        node_id: None,
        parent_class_id: None,
    }
}

#[test]
fn extracts_roles_polarity_and_modality() {
    let store = Arc::new(InMemoryConceptStore::new());
    let doc = attached(FULL_PARSE, &["Show", "me", "the", "sales", "report", "for", "yesterday", "please"]);
    let ex = extractor(Arc::clone(&store));
    let analysis = ex.extract(&doc, None);

    assert_eq!(analysis.frames.len(), 1, "one sentence");
    let frame = &analysis.frames[0];
    assert!(frame.predicate_lemma_id.is_spacy_lemma());
    assert_eq!(frame.sentence_span, (0, 8));
    assert_eq!(frame.polarity, Polarity::Positive);
    assert_eq!(frame.modality, Modality::Declarative);

    // Roles by dep: iobj=me, compound=sales, dobj=report, prep+pobj for
    // the "for yesterday" argument. The det "the" and the advmod "please"
    // are not frame argument slots.
    let kinds: Vec<RoleType> = frame.roles.iter().map(|(r, _, _)| *r).collect();
    assert!(kinds.contains(&RoleType::IndirectObject));
    assert!(kinds.contains(&RoleType::Modifier));
    assert!(kinds.contains(&RoleType::DirectObject));
    assert!(kinds.contains(&RoleType::Argument));
    assert_eq!(frame.roles.len(), 5);
}

#[test]
fn ambiguity_free_frame_mints_permanent_key() {
    let store = Arc::new(InMemoryConceptStore::new());
    let doc = attached(FULL_PARSE, &["Show", "me", "the", "sales", "report", "for", "yesterday", "please"]);
    let ex = extractor(Arc::clone(&store));
    let analysis = ex.extract(&doc, None);
    assert!(analysis.ambiguities.is_empty(), "no ambiguity on a clean parse");
    let keys = ex.keys(&doc, &analysis);
    assert_eq!(keys.len(), 1);
    assert!(!keys[0].provisional, "ambiguity-free frame → permanent");
    assert_eq!(keys[0].predicate_lemma_id, analysis.frames[0].predicate_lemma_id);
}

#[test]
fn near_tie_margin_marks_attachment_ambiguity_and_provisional_key() {
    let store = Arc::new(InMemoryConceptStore::new());
    let doc = attached(FULL_PARSE, &["Show", "me", "the", "sales", "report", "for", "yesterday", "please"]);
    let ex = extractor(Arc::clone(&store));
    // A near-zero margin → the oracle doubted an attachment.
    let margins = [0.0f64, 1.0, 0.5];
    let analysis = ex.extract(&doc, Some(&margins));
    assert!(analysis
        .ambiguities
        .iter()
        .any(|a| a.kind == AmbiguityKind::AttachmentNearTie));
    let keys = ex.keys(&doc, &analysis);
    assert!(keys[0].provisional, "near-tie frame → provisional key");
    // A clean margin set leaves the frame permanent.
    let clean = ex.extract(&doc, Some(&[1.0, 0.7, 0.9]));
    let clean_keys = ex.keys(&doc, &clean);
    assert!(!clean_keys[0].provisional);
}

#[test]
fn two_candidate_predicate_lemma_marks_polysemy() {
    let store = Arc::new(InMemoryConceptStore::new());
    let doc = attached(FULL_PARSE, &["Show", "me", "the", "sales", "report", "for", "yesterday", "please"]);
    let ex = extractor(Arc::clone(&store));

    // Register two canonicals under the same predicate lemma id bucket —
    // the store keeps both, so `candidates` yields 2 → polysemy.
    let pred_id = ex
        .resolver
        .lemma_id("show");
    store.insert(meta(pred_id, "show_present")).expect("a");
    store.insert(meta(pred_id, "show_display")).expect("b");

    let analysis = ex.extract(&doc, None);
    let polysemy = analysis
        .ambiguities
        .iter()
        .find(|a| a.kind == AmbiguityKind::PredicatePolysemy)
        .expect("polysemy ambiguity");
    assert_eq!(polysemy.candidate_ids.len(), 2);
    assert!(polysemy.candidate_ids.contains(&pred_id));
    assert!(analysis.has_open_ambiguity(0));
    let keys = ex.keys(&doc, &analysis);
    assert!(keys[0].provisional);
}

#[test]
fn neg_plus_modal_marks_scope_conflict() {
    let store = Arc::new(InMemoryConceptStore::new());
    let doc = attached(
        NEG_MODAL_PARSE,
        &["The", "dog", "will", "not", "sleep"],
    );
    let ex = extractor(Arc::clone(&store));
    let analysis = ex.extract(&doc, None);

    let frame = &analysis.frames[0];
    assert_eq!(frame.polarity, Polarity::Negative);
    assert_eq!(frame.modality, Modality::Possibility);
    assert!(analysis
        .ambiguities
        .iter()
        .any(|a| a.kind == AmbiguityKind::NegationModalScope));
    assert!(analysis.has_open_ambiguity(0));
}

#[test]
fn interrogative_punct_is_modality() {
    let store = Arc::new(InMemoryConceptStore::new());
    let json = r#"[
        {"text":"What","pos":"pron","dep":"nsubj","head":1,"lemma":"what"},
        {"text":"happened","pos":"verb","dep":"root","head":0,"lemma":"happen"},
        {"text":"?","pos":"punct","dep":"punct","head":-1,"lemma":"?"}
    ]"#;
    let doc = attached(json, &["What", "happened", "?"]);
    let ex = extractor(Arc::clone(&store));
    let analysis = ex.extract(&doc, None);
    assert_eq!(analysis.frames[0].modality, Modality::Interrogative);
}

#[test]
fn serde_roundtrip_frame_and_key() {
    let store = Arc::new(InMemoryConceptStore::new());
    let doc = attached(FULL_PARSE, &["Show", "me", "the", "sales", "report", "for", "yesterday", "please"]);
    let ex = extractor(Arc::clone(&store));
    let analysis = ex.extract(&doc, None);
    let keys = ex.keys(&doc, &analysis);

    let fjson = serde_json::to_string(&analysis).expect("serialize frames");
    let back: FrameAnalysis = serde_json::from_str(&fjson).expect("deserialize frames");
    assert_eq!(back, analysis);

    let kjson = serde_json::to_string(&keys[0]).expect("serialize key");
    let kback: FrameKey = serde_json::from_str(&kjson).expect("deserialize key");
    assert_eq!(kback, keys[0]);
    assert!(!kback.provisional);
}

#[test]
fn empty_doc_yields_no_frames() {
    let store = Arc::new(InMemoryConceptStore::new());
    let ex = extractor(Arc::clone(&store));
    let analysis = ex.extract(&Doc::new(vocab()), None);
    assert!(analysis.frames.is_empty());
    assert!(analysis.ambiguities.is_empty());
}

#[test]
fn loading_gate_stamps_provisional_even_when_unambiguous() {
    use fluent_concept::ConceptStoreState;
    struct LoadingStore(InMemoryConceptStore);
    impl fluent_concept::ConceptStore for LoadingStore {
        fn get(&self, id: fluent_types::InterlinguaId) -> Result<fluent_types::ConceptMetadata, fluent_concept::ConceptStoreError> { self.0.get(id) }
        fn resolve_name(&self, n: &str) -> Result<fluent_types::InterlinguaId, fluent_concept::ConceptStoreError> { self.0.resolve_name(n) }
        fn resolve_yago_iri(&self, i: &str) -> Result<fluent_types::InterlinguaId, fluent_concept::ConceptStoreError> { self.0.resolve_yago_iri(i) }
        fn insert(&self, m: fluent_types::ConceptMetadata) -> Result<(), fluent_concept::ConceptStoreError> { self.0.insert(m) }
        fn contains(&self, id: fluent_types::InterlinguaId) -> bool { self.0.contains(id) }
        fn iter_ids(&self) -> Box<dyn Iterator<Item=fluent_types::InterlinguaId> + '_> { self.0.iter_ids() }
        fn ancestors_of(&self, id: fluent_types::InterlinguaId) -> Vec<fluent_types::InterlinguaId> { self.0.ancestors_of(id) }
        fn is_subclass_of(&self, c: fluent_types::InterlinguaId, p: fluent_types::InterlinguaId) -> bool { self.0.is_subclass_of(c,p) }
        fn state(&self) -> ConceptStoreState { ConceptStoreState::Loading }
    }
    let store = Arc::new(LoadingStore(InMemoryConceptStore::new()));
    let doc = attached(FULL_PARSE, &["Show", "me", "the", "sales", "report", "for", "yesterday", "please"]);
    let ex = crate::frame::FrameExtractor::new(
        Arc::new(crate::interlingua::InterlinguaResolver::new(Arc::clone(&store) as Arc<dyn fluent_concept::ConceptStore>, Arc::clone(vocab().strings()))),
        Arc::clone(&store) as Arc<dyn fluent_concept::ConceptStore>,
    );
    let analysis = ex.extract(&doc, None);
    assert!(analysis.ambiguities.is_empty(), "grammatically unambiguous");
    let keys = ex.keys(&doc, &analysis);
    assert!(keys[0].provisional, "Loading state must stamp provisional even when unambiguous");
}

#[test]
fn property_random_parses_extract_frames_and_keep_one_root() {
    // 100+ random POS with frame extraction still yields valid trees
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); seed >> 33 };
    let pos_words: &[(crate::labels::Upos, &str)] = &[
        (crate::labels::Upos::Noun, "cat"), (crate::labels::Upos::Verb, "run"), (crate::labels::Upos::Det, "the"),
        (crate::labels::Upos::Adp, "of"), (crate::labels::Upos::Propn, "NASA"), (crate::labels::Upos::Punct, "."),
    ];
    let vocab = vocab();
    let annotator = crate::arc_eager::ArcEagerAnnotator::en_default(Arc::clone(&vocab));
    let store = Arc::new(InMemoryConceptStore::new());
    let ex = extractor(Arc::clone(&store));
    for _ in 0..110 {
        let len = 1 + (next() % 14) as usize;
        let mut doc = crate::doc::Doc::new(Arc::clone(&vocab));
        for _ in 0..len {
            let (_, w) = pos_words[(next() as usize) % pos_words.len()];
            doc.push_back(w, true).expect("push");
        }
        let (result, _pc) = annotator.annotate_with_confidence(&doc).expect("parse");
        crate::llm::attach(&mut doc, result.records()).expect("attach");
        crate::sentencizer::Sentencizer::new().process(&mut doc);
        let analysis = ex.extract(&doc, result.oracle_margins.as_deref());
        for k in ex.keys(&doc, &analysis) {
            let _ = k;
        }
        // validator still passes exactly one ROOT
        assert!(doc.len() == 0 || crate::validate::AnnotationValidator::new().validate(&doc, result.records()).is_ok());
    }
}
