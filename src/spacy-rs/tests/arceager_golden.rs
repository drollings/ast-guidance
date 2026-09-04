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
fn every_output_is_relatively_headed_and_valid() {    for text in [
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

#[test]
fn aux_neg_bare_infinitive_help_is_verb() {
    // "help" is outside the closed verb list; the aux + n't context is the
    // only evidence. Refs (UD): help → verb/root, them → dobj.
    let (_doc, set) = parse("Don't help them.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let help = set.0.iter().position(|r| r.text == "help").expect("help");
    let them = set.0.iter().position(|r| r.text == "them").expect("them");
    assert_eq!(pos[help], "verb", "pos: {pos:?}");
    assert_eq!(deps[help], "root");
    assert_eq!(deps[them], "dobj");
    assert_eq!(them as i32 + set.0[them].head, help as i32);
}

#[test]
fn aux_neg_bare_infinitive_answer_is_verb() {
    // Tokenizer splits "won't" into wo/n't; "answer" is outside the closed
    // verb list. Refs (UD): answer → verb/root. "calls" stays over-lexed as
    // VERB by the closed list (its dobj demotion is a separate error class,
    // deliberately not asserted here) — but it must no longer steal root.
    let (_doc, set) = parse("She won't answer calls.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let answer = set.0.iter().position(|r| r.text == "answer").expect("answer");
    let calls = set.0.iter().position(|r| r.text == "calls").expect("calls");
    assert_eq!(pos[answer], "verb", "pos: {pos:?}");
    assert_eq!(deps[answer], "root");
    assert_eq!(set.0[answer].head, 0);
    assert_ne!(deps[calls], "root", "calls must not steal root: {deps:?}");
}

#[test]
fn true_nominal_help_after_verb_stays_noun() {
    // Must-NOT-fire: "help" after a lexical verb is a true nominal direct
    // object and must stay NOUN.
    let (_doc, set) = parse("I need help.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let help = set.0.iter().position(|r| r.text == "help").expect("help");
    assert_eq!(pos[help], "noun", "pos: {pos:?}");
    assert_eq!(deps[help], "dobj");
}

#[test]
fn possessive_clitic_does_not_trigger_verb() {
    // Must-NOT-fire: "'s" is a possessive clitic here, not an aux — the
    // following noun must stay NOUN.
    let (doc, set) = parse("Explain Bell's theorem.");
    let pos = pos_of(&set);
    let texts = token_texts(&doc);
    let theorem = texts.iter().position(|t| t == "theorem").expect("theorem");
    assert_eq!(pos[theorem], "noun", "pos: {pos:?}");
}

#[test]
fn neg_attaches_to_governing_verb() {
    // Refs (UD): did → aux, n't → part/neg → see, see → root, We → nsubj,
    // it → dobj. Before the fix the X-tagged n't blocked the stack and every
    // pre-verbal token fell to the repair-dep fallback.
    let (_doc, set) = parse("We didn't see it.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (we, did, nt, see, it) = (at("We"), at("did"), at("n't"), at("see"), at("it"));
    assert_eq!(pos[did], "aux", "pos: {pos:?}");
    assert_eq!(pos[nt], "part", "pos: {pos:?}");
    assert_eq!(deps[did], "aux", "deps: {deps:?}");
    assert_eq!(deps[nt], "neg", "deps: {deps:?}");
    assert_eq!(nt as i32 + set.0[nt].head, see as i32);
    assert_eq!(deps[see], "root");
    assert_eq!(deps[we], "nsubj");
    assert_eq!(deps[it], "dobj");
}

#[test]
fn contracted_be_progressive_frame() {
    // Refs (UD): 's → aux, raining → verb/root, It → nsubj.
    // "hard" (advmod) belongs to adverb detection — deliberately unasserted.
    let (_doc, set) = parse("It's raining hard.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (it, s, raining) = (at("It"), at("'s"), at("raining"));
    assert_eq!(pos[s], "aux", "pos: {pos:?}");
    assert_eq!(pos[raining], "verb", "pos: {pos:?}");
    assert_eq!(deps[s], "aux", "deps: {deps:?}");
    assert_eq!(deps[raining], "root");
    assert_eq!(deps[it], "nsubj");
    assert_eq!(it as i32 + set.0[it].head, raining as i32);
}

#[test]
fn possessive_s_is_not_aux() {
    // Must-NOT-fire: possessive 's (host is a noun, not a pronoun) keeps its
    // non-aux tag; the UD case-marker reading is future work, but aux is
    // wrong and must not leak from the contracted-be rule.
    let (doc, set) = parse("Explain Bell's theorem.");
    let pos = pos_of(&set);
    let texts = token_texts(&doc);
    let s = texts.iter().position(|t| t == "'s").expect("'s");
    assert_ne!(pos[s], "aux", "pos: {pos:?}");
}

#[test]
fn full_be_participial_adjective_stays_non_verb() {
    // Must-NOT-fire: "surprising" after full-form "were" is a participial
    // adjective (ref: adj/root) — the clitic-hosted participle rule must not
    // fire on full be-forms.
    let (doc, set) = parse("The results were surprising.");
    let pos = pos_of(&set);
    let texts = token_texts(&doc);
    let surprising = texts.iter().position(|t| t == "surprising").expect("surprising");
    assert_ne!(pos[surprising], "verb", "pos: {pos:?}");
}