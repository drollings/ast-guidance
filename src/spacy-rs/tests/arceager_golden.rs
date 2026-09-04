//! The deterministic parser's golden corpus (ROADMAP §8.7).
//!
//! Adversarial and positive cases for the heuristic POS rules and the
//! verb-governs-argument structure the parser must lift the ladder to:
//!
//! - sentence-initial common noun → **NOT** PROPN ("Dogs bark loudly."),
//! - Title Case marketing phrase → **NOT** PROPN ("Big Data is a trend."),
//! - ALL-CAPS → **IS** PROPN ("NASA launched HTML5."),
//! - `nsubj`/`dobj` extraction,
//! - `prep` + `pobj` argument frames,
//! - multi-sentence BREAK + exactly one ROOT per sentence.
//!
//! Everything is hermetic: no model, no network, deterministic by
//! construction.

use std::sync::Arc;

use spacy_rs::{ArcEagerAnnotator, Doc, Vocab};

fn en_vocab() -> Arc<Vocab> {
    Arc::new(Vocab::new(spacy_rs::lang::en::lexicon_config()))
}

fn parse(text: &str) -> (Doc, spacy_rs::AnnotationSet) {
    let vocab = en_vocab();
    let tokenizer = spacy_rs::lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    let doc = tokenizer.tokenize(text).expect("tokenize");
    let annotator = ArcEagerAnnotator::en_default(vocab);
    let (result, _conf) = annotator.annotate_with_confidence(&doc).expect("parse");
    (doc, result.records.clone())
}

fn token_texts(doc: &Doc) -> Vec<String> {
    (0..doc.len()).map(|i| doc.token_text(i)).collect()
}

fn deps(set: &spacy_rs::AnnotationSet) -> Vec<String> {
    set.0.iter().map(|r| r.dep.clone()).collect()
}

fn pos_of(set: &spacy_rs::AnnotationSet) -> Vec<String> {
    set.0.iter().map(|r| r.pos.clone()).collect()
}

#[test]
fn sentence_initial_common_noun_is_not_propn() {
    let (doc, set) = parse("Dogs bark loudly.");
    let pos = pos_of(&set);
    let dogs = token_texts(&doc)[0].clone();
    assert_eq!(dogs, "Dogs");
    // "Dogs" (sentence-initial, Title Case) must NOT be PROPN.
    assert_eq!(pos[0], "noun", "pos: {pos:?}");
    // "bark" is a closed verb → the root; "Dogs" is its nsubj.
    assert_eq!(deps(&set)[1], "root");
    assert_eq!(deps(&set)[0], "nsubj");
    assert_eq!(set.0[0].head, 1, "Dogs points at bark (relative)");
}

#[test]
fn title_case_marketing_phrase_is_not_propn() {
    let (doc, set) = parse("Big Data is a trend.");
    let pos = pos_of(&set);
    let texts = token_texts(&doc);
    let big = texts.iter().position(|t| t == "Big").expect("Big");
    // "Big" and "Data" are Title Case → NOUN, never PROPN.
    assert_eq!(pos[big], "noun", "pos: {pos:?}");
    assert_eq!(pos[big + 1], "noun", "pos: {pos:?}");
}

#[test]
fn allcaps_is_propn_positive() {
    let (doc, set) = parse("NASA launched HTML5.");
    let pos = pos_of(&set);
    let texts = token_texts(&doc);
    let nasa = texts.iter().position(|t| t == "NASA").expect("NASA");
    let html5 = texts.iter().position(|t| t == "HTML5").expect("HTML5");
    assert_eq!(pos[nasa], "propn", "NASA is PROPN (ALL-CAPS): {pos:?}");
    assert_eq!(pos[html5], "propn", "HTML5 is PROPN (ALL-CAPS + digit): {pos:?}");
}

#[test]
fn verb_governs_argument_extraction() {
    let (_doc, set) = parse("NASA launched HTML5.");
    let deps = deps(&set);
    let launched = set.0.iter().position(|r| r.text == "launched").expect("verb");
    let nasa = set.0.iter().position(|r| r.text == "NASA").expect("nasa");
    let html5 = set.0.iter().position(|r| r.text == "HTML5").expect("html5");
    assert_eq!(deps[launched], "root");
    assert_eq!(deps[nasa], "nsubj");
    assert_eq!(deps[html5], "dobj");
    assert_eq!(nasa as i32 + set.0[nasa].head, launched as i32);
    assert_eq!(html5 as i32 + set.0[html5].head, launched as i32);
}

#[test]
fn prep_pobj_argument_frame() {
    let (_doc, set) = parse("The cat sat on the mat.");
    let deps = deps(&set);
    let sat = set.0.iter().position(|r| r.text == "sat").expect("sat");
    let on = set.0.iter().position(|r| r.text == "on").expect("on");
    let mat = set.0.iter().position(|r| r.text == "mat").expect("mat");
    assert_eq!(deps[sat], "root");
    assert_eq!(deps[on], "prep");
    assert_eq!(deps[mat], "pobj");
    // "on" depends on "sat"; "mat" depends on "on".
    assert_eq!(on as i32 + set.0[on].head, sat as i32);
    assert_eq!(mat as i32 + set.0[mat].head, on as i32);
}

#[test]
fn one_root_per_sentence_multi_sentence_break() {
    let (_doc, set) = parse("Dogs bark. Cats meow.");
    let roots = set
        .0
        .iter()
        .enumerate()
        .filter(|(_, r)| r.dep == "root")
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    assert_eq!(roots.len(), 2, "one ROOT per sentence: {roots:?}");
    // Every non-root head resolves to a token within the doc.
    for (i, r) in set.0.iter().enumerate() {
        let abs = i as i32 + r.head;
        assert!((0..set.len() as i32).contains(&abs));
    }
}

#[test]
fn every_output_is_relatively_headed_and_valid() {
    for text in [
        "The cat sat on the mat.",
        "NASA launched HTML5.",
        "Dogs bark loudly.",
        "show me the sales report",
        "Big Data is a trend.",
    ] {
        let (doc, set) = parse(text);
        assert_eq!(set.len(), doc.len());
        let roots = set.0.iter().filter(|r| r.dep == "root").count();
        assert_eq!(roots, 1, "{text:?} has exactly one root");
        for (i, r) in set.0.iter().enumerate() {
            let abs = i as i32 + r.head;
            assert!((0..set.len() as i32).contains(&abs), "{text:?} head {i}");
            if r.dep == "root" {
                assert_eq!(r.head, 0);
            } else {
                assert_ne!(r.head, 0);
            }
        }
    }
}