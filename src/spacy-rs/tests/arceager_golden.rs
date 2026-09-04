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

fn lemmas(set: &spacy_rs::AnnotationSet) -> Vec<String> {
    set.0.iter().map(|r| r.lemma.clone()).collect()
}

#[test]
fn cant_splinter_lemmas() {
    // UD pins ca → can, n't → not (sibling refs -02/-03; spaCy parity).
    let (_doc, set) = parse("I can't go today.");
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let lem = lemmas(&set);
    assert_eq!(lem[at("ca")], "can", "lemmas: {lem:?}");
    assert_eq!(lem[at("n't")], "not", "lemmas: {lem:?}");
}

#[test]
fn wont_splinter_lemmas() {
    // "wo" is the bound allomorph of modal "will" (never *"do"); n't → not.
    let (_doc, set) = parse("She won't answer calls.");
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let lem = lemmas(&set);
    assert_eq!(lem[at("wo")], "will", "lemmas: {lem:?}");
    assert_eq!(lem[at("n't")], "not", "lemmas: {lem:?}");
}

#[test]
fn contracted_be_splinter_lemma() {
    // Aux-classified 's resolves to be (ref contraction-05).
    let (_doc, set) = parse("It's raining hard.");
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let lem = lemmas(&set);
    assert_eq!(lem[at("'s")], "be", "lemmas: {lem:?}");
}

#[test]
fn did_lemma_is_do() {
    // Refs (UD): did → do (contraction-04, question-06).
    let (_doc, set) = parse("Did the report arrive?");
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let lem = lemmas(&set);
    assert_eq!(lem[at("Did")], "do", "lemmas: {lem:?}");
}

#[test]
fn possessive_s_lemma_untouched() {
    // Must-NOT-fire: possessive 's (PART/case, ref command-06 pins "'s")
    // keeps its surface lemma — the be-mapping is aux-gated.
    let (_doc, set) = parse("Explain Bell's theorem.");
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let lem = lemmas(&set);
    assert_eq!(lem[at("'s")], "'s", "lemmas: {lem:?}");
}

#[test]
fn ambiguous_d_contraction_untouched() {
    // Must-NOT-fire: 'd is would/had — lexically underdetermined, so the
    // closed map leaves it alone rather than guessing.
    let (_doc, set) = parse("I'd go today.");
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let lem = lemmas(&set);
    assert_eq!(lem[at("'d")], "'d", "lemmas: {lem:?}");
}

#[test]
fn directive_initial_verb_with_det_object() {
    // Refs (UD): Close → verb/root, books → dobj. "Close" is outside the
    // closed verb list; the sentence-initial DET+NOUN frame is the evidence.
    let (_doc, set) = parse("Close your books.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (close, books) = (at("Close"), at("books"));
    assert_eq!(pos[close], "verb", "pos: {pos:?}");
    assert_eq!(deps[close], "root");
    assert_eq!(deps[books], "dobj", "deps: {deps:?}");
    assert_eq!(books as i32 + set.0[books].head, close as i32);
}

#[test]
fn directive_initial_verb_command_frame() {
    // Refs (UD): Solve → verb/root, equation → dobj.
    let (_doc, set) = parse("Solve this equation.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (solve, equation) = (at("Solve"), at("equation"));
    assert_eq!(pos[solve], "verb", "pos: {pos:?}");
    assert_eq!(deps[solve], "root");
    assert_eq!(deps[equation], "dobj", "deps: {deps:?}");
    assert_eq!(equation as i32 + set.0[equation].head, solve as i32);
}

#[test]
fn subject_noun_before_noun_is_not_directive() {
    // Must-NOT-fire: "Anna" is a subject, not a directive verb — the
    // lookahead demands DET+NOUN, and "finished"(NOUN) blocks it.
    let (doc, set) = parse("Anna finished her lunch.");
    let pos = pos_of(&set);
    let texts = token_texts(&doc);
    let anna = texts.iter().position(|t| t == "Anna").expect("Anna");
    assert_eq!(pos[anna], "noun", "pos: {pos:?}");
}

#[test]
fn conjunction_second_is_not_directive() {
    // Must-NOT-fire: sentence-initial noun followed by a conjunction is a
    // subject ("Dogs ... nap"), never a directive verb.
    let (doc, set) = parse("Dogs and cats nap.");
    let pos = pos_of(&set);
    let texts = token_texts(&doc);
    let dogs = texts.iter().position(|t| t == "Dogs").expect("Dogs");
    assert_eq!(pos[dogs], "noun", "pos: {pos:?}");
}

#[test]
fn pronoun_subject_governs_matrix_and_embedded_verb() {
    // Refs (UD): stayed → verb/root, snowed → verb, We → nsubj,
    // it → nsubj → snowed, snowed heads to stayed. The SCONJ "because" needs
    // a mark arm (future work) and is deliberately unasserted.
    let (_doc, set) = parse("We stayed because it snowed.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (we, stayed, it, snowed) = (at("We"), at("stayed"), at("it"), at("snowed"));
    assert_eq!(pos[stayed], "verb", "pos: {pos:?}");
    assert_eq!(pos[snowed], "verb", "pos: {pos:?}");
    assert_eq!(deps[stayed], "root");
    assert_eq!(deps[we], "nsubj", "deps: {deps:?}");
    assert_eq!(deps[it], "nsubj", "deps: {deps:?}");
    assert_eq!(it as i32 + set.0[it].head, snowed as i32);
    assert_eq!(snowed as i32 + set.0[snowed].head, stayed as i32);
}

#[test]
fn pronoun_subject_governs_sconj_clause_verb() {
    // Refs (UD): snores → verb/root, sleeps → verb heading to snores, He →
    // nsubj. The post-verbal "he" currently misattaches as dobj (the
    // subordinate-subject attachment gap) and is deliberately unasserted.
    let (_doc, set) = parse("He snores when he sleeps.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (he, snores, sleeps) = (at("He"), at("snores"), at("sleeps"));
    assert_eq!(pos[snores], "verb", "pos: {pos:?}");
    assert_eq!(pos[sleeps], "verb", "pos: {pos:?}");
    assert_eq!(deps[snores], "root");
    assert_eq!(deps[he], "nsubj", "deps: {deps:?}");
    assert_eq!(sleeps as i32 + set.0[sleeps].head, snores as i32);
}

#[test]
fn object_pronoun_does_not_govern_verb() {
    // Must-NOT-fire: "me" is an object pronoun, not a nominative subject —
    // "later" stays NOUN.
    let (_doc, set) = parse("Call me later.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("later")], "noun", "pos: {pos:?}");
}

#[test]
fn aux_prev_does_not_govern_participle() {
    // Must-NOT-fire: "coming" follows an AUX ("are"), not a pronoun subject
    // — the full-be progressive stays for copular handling.
    let (_doc, set) = parse("They aren't coming today.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("coming")], "noun", "pos: {pos:?}");
}

#[test]
fn initial_noun_subject_governs_verb() {
    // Refs (UD): ended → verb/root, game → nsubj, scored → verb heading to
    // ended. The "after she" PP-frame (ADP-lexed "after") is future work and
    // deliberately unasserted.
    let (_doc, set) = parse("The game ended after she scored.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (game, ended, scored) = (at("game"), at("ended"), at("scored"));
    assert_eq!(pos[ended], "verb", "pos: {pos:?}");
    assert_eq!(pos[scored], "verb", "pos: {pos:?}");
    assert_eq!(deps[ended], "root");
    assert_eq!(deps[game], "nsubj", "deps: {deps:?}");
    assert_eq!(scored as i32 + set.0[scored].head, ended as i32);
}

#[test]
fn bare_initial_noun_subject_governs_verb() {
    // Refs (UD): passed → verb/root, storm → nsubj. The medial "floods
    // stayed" (noun-subject, non-initial) belongs to later work.
    let (_doc, set) = parse("The storm passed but floods stayed.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (storm, passed) = (at("storm"), at("passed"));
    assert_eq!(pos[passed], "verb", "pos: {pos:?}");
    assert_eq!(deps[passed], "root");
    assert_eq!(deps[storm], "nsubj", "deps: {deps:?}");
}

#[test]
fn bare_verb_object_initials_are_not_subject_verbs() {    // Must-NOT-fire: "Translate hello" / "Define photosynthesis" are
    // verb–object initials, POS-identical to subject–verb ones — the
    // initial-noun rule covers DET-led subjects only, so both stay NOUN.
    let (_doc, set) = parse("Define photosynthesis.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("photosynthesis")], "noun", "pos: {pos:?}");
    let (_doc, set) = parse("Translate hello to French.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("hello")], "noun", "pos: {pos:?}");
}

#[test]
fn relativizer_where_is_not_a_verb() {
    // Must-NOT-fire: "where" in DET+NOUN+where position is the relativizer
    // (ref: adp/mark) — the initial-noun-subject rule excludes WH-forms.
    let (_doc, set) = parse("The store where we met closed.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(pos[at("where")], "verb", "pos: {pos:?}");
}

#[test]
fn medial_compound_noun_is_not_a_verb() {
    // Must-NOT-fire: medial DET+NOUN+NOUN ("the sales report") is a compound
    // nominal — the rule is sentence-initial only.
    let (_doc, set) = parse("Show me the sales report.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("sales")], "noun", "pos: {pos:?}");
}

#[test]
fn conjoined_clause_verb_after_cc_subject() {
    // Refs (UD): rose → verb, spirits → nsubj → rose. The "yet" marker
    // (sconj-mark) needs its own arm; rose steals root until the NOUN matrix
    // verb upgrades (bare-initial work) — its own head is unasserted.
    let (_doc, set) = parse("Grades dropped yet spirits rose.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (spirits, rose) = (at("spirits"), at("rose"));
    assert_eq!(pos[rose], "verb", "pos: {pos:?}");
    assert_eq!(deps[spirits], "nsubj", "deps: {deps:?}");
    assert_eq!(spirits as i32 + set.0[spirits].head, rose as i32);
}

#[test]
fn conjoined_np_subject_governs_final_verb() {
    // Refs (UD): nap → verb/root, Dogs heads to nap (repair-head into the
    // root). The "cats" conj label belongs to coordination-attachment work.
    let (_doc, set) = parse("Dogs and cats nap.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (dogs, nap) = (at("Dogs"), at("nap"));
    assert_eq!(pos[nap], "verb", "pos: {pos:?}");
    assert_eq!(deps[nap], "root");
    assert_eq!(dogs as i32 + set.0[dogs].head, nap as i32);
}

#[test]
fn clause_final_conjoined_word_after_cc_stays_open() {
    // Must-NOT-fire: CC + NOUN at the very end ("and eggs") is a conjoined
    // object nominal — the rule needs a third token to govern.
    let (_doc, set) = parse("Buy milk and eggs.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("eggs")], "noun", "pos: {pos:?}");
}

#[test]
fn clause_final_conjoined_verb_needs_subject_tracking() {
    // Must-NOT-fire (boundary): "failed" after "but" at the end is a
    // conjoined verb, but so is "eggs" a conjoined object in the same shape
    // — disambiguating them needs clause-subject tracking, not this rule.
    let (_doc, set) = parse("She studied but failed.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("failed")], "noun", "pos: {pos:?}");
}

#[test]
fn sconj_marker_attaches_to_clause_verb() {
    // Refs (UD): because → mark → snowed. Before, the marker sat on the
    // stack with no arm and Reduce popped it into repair-dep.
    let (_doc, set) = parse("We stayed because it snowed.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (because, snowed) = (at("because"), at("snowed"));
    assert_eq!(deps[because], "mark", "deps: {deps:?}");
    assert_eq!(because as i32 + set.0[because].head, snowed as i32);
}

#[test]
fn when_marker_attaches_to_clause_verb() {
    // Refs (UD): when → mark → works.
    let (_doc, set) = parse("She sings when she works.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (when, works) = (at("when"), at("works"));
    assert_eq!(deps[when], "mark", "deps: {deps:?}");
    assert_eq!(when as i32 + set.0[when].head, works as i32);
}

#[test]
fn adjective_headed_sconj_does_not_mark() {
    // Must-NOT-fire: "hungry" is a NOUN-tagged adjective (adjective gap) —
    // with no verb on the buffer, the marker must not invent a mark arc.
    let (_doc, set) = parse("Although hungry, he shared lunch.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(deps[at("Although")], "mark", "deps: {deps:?}");
}

#[test]
fn dual_class_after_stays_prepositional() {
    // Must-NOT-fire (boundary): "after" lexes ADP, so the SCONJ-keyed mark
    // arm cannot fire — after-disambiguation is its own rule.
    let (_doc, set) = parse("The game ended after she scored.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("after")], "adp", "pos: {pos:?}");
}

#[test]
fn copular_be_predicate_frame() {
    // Refs (UD): low → adj/root, is → cop → low, fee → nsubj → low.
    let (_doc, set) = parse("Your fee is low.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (fee, is, low) = (at("fee"), at("is"), at("low"));
    assert_eq!(pos[low], "adj", "pos: {pos:?}");
    assert_eq!(deps[low], "root");
    assert_eq!(deps[is], "cop", "deps: {deps:?}");
    assert_eq!(is as i32 + set.0[is].head, low as i32);
    assert_eq!(deps[fee], "nsubj", "deps: {deps:?}");
    assert_eq!(fee as i32 + set.0[fee].head, low as i32);
}

#[test]
fn negated_copular_predicate_frame() {
    // Refs (UD): ready → adj/root, is → cop → ready, She → nsubj → ready.
    // The n't-bridge and cop/nsubj arms compose; n't/yet are unasserted.
    let (_doc, set) = parse("She isn't ready yet.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (she, is, ready) = (at("She"), at("is"), at("ready"));
    assert_eq!(pos[ready], "adj", "pos: {pos:?}");
    assert_eq!(deps[ready], "root");
    assert_eq!(deps[is], "cop", "deps: {deps:?}");
    assert_eq!(is as i32 + set.0[is].head, ready as i32);
    assert_eq!(deps[she], "nsubj", "deps: {deps:?}");
    assert_eq!(she as i32 + set.0[she].head, ready as i32);
}

#[test]
fn np_predicate_noun_is_not_adjective() {
    // Must-NOT-fire: "doctor" follows a DET, not be — the predicate-adjective
    // rule needs a be-AUX (or n't-bridged be) immediately before.
    let (_doc, set) = parse("She is a doctor.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("doctor")], "noun", "pos: {pos:?}");
}

#[test]
fn initial_be_question_subject_is_not_adjective() {
    // Must-NOT-fire (boundary): sentence-initial be is interrogative AUX, not
    // a copula — its subject ("lunch") stays nominal.
    let (_doc, set) = parse("Is lunch ready?");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("lunch")], "noun", "pos: {pos:?}");
}

#[test]
fn comma_delimited_appos_attaches_to_anchor() {
    // Refs (UD): doctor → appos → brother. The anchor must survive the
    // determiner (noun-det wait) for the pair to meet.
    let (_doc, set) = parse("My brother, a doctor, lives here.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (brother, doctor) = (at("brother"), at("doctor"));
    assert_eq!(deps[doctor], "appos", "deps: {deps:?}");
    assert_eq!(doctor as i32 + set.0[doctor].head, brother as i32);
}

#[test]
fn punct_separated_verbs_attach_parataxis() {
    // Refs (UD): called → parataxis → texted. The (Verb,Verb) pair needs a
    // punctuation + nominal between them (see controls).
    let (_doc, set) = parse("She texted; he called.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (texted, called) = (at("texted"), at("called"));
    assert_eq!(deps[called], "parataxis", "deps: {deps:?}");
    assert_eq!(called as i32 + set.0[called].head, texted as i32);
}

#[test]
fn adjacent_clause_verbs_are_not_parataxis() {
    // Must-NOT-fire: "called left" are adjacent (relative clause) — no
    // punctuation/nominal between, so no parataxis arm may fire.
    let (_doc, set) = parse("The man who called left.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(deps[at("left")], "parataxis", "deps: {deps:?}");
    assert_ne!(deps[at("called")], "parataxis", "deps: {deps:?}");
}

#[test]
fn participial_parenthetical_is_not_parataxis() {
    // Must-NOT-fire: "smiling, took" has punctuation but no nominal between
    // (participle + matrix verb, one clause) — parataxis must not fire.
    let (_doc, set) = parse("The CEO, smiling, took questions.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(deps[at("took")], "parataxis", "deps: {deps:?}");
}

#[test]
fn medial_compound_pair_is_not_appos() {
    // Must-NOT-fire: "sales report" has no punctuation between — the pair
    // stays on the compound path, never appos.
    let (_doc, set) = parse("Show me the sales report.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(deps[at("report")], "appos", "deps: {deps:?}");
}

#[test]
fn comma_framed_initial_ly_adverbial_is_adv() {
    // Refs (UD parenthetical-12): Sadly → adv/advmod → ended. The tagger has
    // no ADV path, so comma-framed -ly adverbials fall through to NOUN (and
    // even steal root). Attachment needs an (Adv, Aux/Verb) clause arm —
    // later work, deliberately unasserted beyond connectivity.
    let (_doc, set) = parse("Sadly, the trip ended early.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("Sadly")], "adv", "pos: {pos:?}");
    let abs = at("Sadly") as i32 + set.0[at("Sadly")].head;
    assert!((0..set.len() as i32).contains(&abs), "Sadly headed");
}

#[test]
fn comma_framed_medial_ly_adverbial_is_adv() {
    // Refs (UD parenthetical-02): frankly → adv/advmod → was.
    let (_doc, set) = parse("The report, frankly, was late.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("frankly")], "adv", "pos: {pos:?}");
}

#[test]
fn month_ly_noun_after_prep_stays_noun() {
    // Must-NOT-fire: "July" ends in -ly and precedes a comma, but its host
    // is a preposition (temporal nominal), not a clause edge — stays NOUN.
    let (_doc, set) = parse("In July, we met.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("July")], "noun", "pos: {pos:?}");
    assert_eq!(pos[at("met")], "verb", "pos: {pos:?}");
}

#[test]
fn det_ly_noun_stays_noun() {
    // Must-NOT-fire: "family" ends in -ly but is a determiner-headed nominal
    // — stays NOUN.
    let (_doc, set) = parse("The family ate dinner.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("family")], "noun", "pos: {pos:?}");
    assert_eq!(deps[at("family")], "nsubj", "deps: {deps:?}");
}

#[test]
fn relative_matrix_verb_is_root() {
    // Refs (UD relative-01): left → verb/root. The leftmost verb is the
    // relcl predicate (called), which the root ladder must skip — the
    // matrix verb closes the relative clause. The relcl attachment itself
    // (called → relcl → man) needs its own oracle arm: later work,
    // deliberately unasserted.
    let (_doc, set) = parse("The man who called left.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (called, left) = (at("called"), at("left"));
    assert_eq!(deps[left], "root", "deps: {deps:?}");
    assert_eq!(set.0[left].head, 0);
    assert_ne!(deps[called], "root", "deps: {deps:?}");
}

#[test]
fn interrogative_who_does_not_shift_root() {
    // Must-NOT-fire: sentence-initial "who" has no nominal head, so it is
    // not a relativizer — the leftmost verb stays root.
    let (_doc, set) = parse("Who called earlier?");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("called")], "root", "deps: {deps:?}");
}

#[test]
fn complementizer_that_does_not_shift_root() {
    // Must-NOT-fire: "that" headed by a verb (know) is a complementizer —
    // the leftmost verb stays root.
    let (_doc, set) = parse("I know that they play soccer.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("know")], "root", "deps: {deps:?}");
}

#[test]
fn relcl_predicate_attaches_to_anchor() {
    // Refs (UD relative-01): called → relcl → man, man → nsubj → left.
    // The candidate set offers no relcl arc, so the clause verb strands
    // into repair-dep; the anchor must additionally survive its relativizer
    // (Reduce outbids Shift) for the pair to meet.
    let (_doc, set) = parse("The man who called left.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (man, called, left) = (at("man"), at("called"), at("left"));
    assert_eq!(deps[called], "relcl", "deps: {deps:?}");
    assert_eq!(called as i32 + set.0[called].head, man as i32);
    assert_eq!(deps[man], "nsubj", "deps: {deps:?}");
    assert_eq!(man as i32 + set.0[man].head, left as i32);
}

#[test]
fn complement_clause_verb_is_not_relcl() {
    // Must-NOT-fire: verb-headed "that" is a complementizer — the embedded
    // verb must not take relcl even though a that + verb frame is present.
    let (_doc, set) = parse("She said that he left.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("said")], "root", "deps: {deps:?}");
    assert_ne!(deps[at("left")], "relcl", "deps: {deps:?}");
}

#[test]
fn nominal_headed_that_is_pronoun_and_heads_clause_verb() {
    // Refs (UD relative-11): that → pron/nsubj → cried, cried → verb/relcl
    // → baby, slept → verb/root. The closed map lexes every "that" as DET,
    // so the relativizer (and its DET-headed clause verb, which no
    // subject-rule sees) strand. Two sequenced upgrades in one frame pass,
    // same shape as the contracted-be rule.
    let (_doc, set) = parse("The baby that cried slept.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (baby, that, cried, slept) = (at("baby"), at("that"), at("cried"), at("slept"));
    assert_eq!(pos[that], "pron", "pos: {pos:?}");
    assert_eq!(pos[cried], "verb", "pos: {pos:?}");
    assert_eq!(deps[that], "nsubj", "deps: {deps:?}");
    assert_eq!(that as i32 + set.0[that].head, cried as i32);
    assert_eq!(deps[cried], "relcl", "deps: {deps:?}");
    assert_eq!(cried as i32 + set.0[cried].head, baby as i32);
    assert_eq!(deps[slept], "root", "deps: {deps:?}");
}

#[test]
fn complement_that_stays_det() {
    // Must-NOT-fire: verb-headed "that" is a complementizer — stays DET.
    let (_doc, set) = parse("She said that he left.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("that")], "det", "pos: {pos:?}");
}

#[test]
fn demonstrative_that_stays_det() {
    // Must-NOT-fire: sentence-initial demonstrative "that" has no nominal
    // head — stays DET.
    let (_doc, set) = parse("That book is heavy.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("That")], "det", "pos: {pos:?}");
}

#[test]
fn post_punct_clause_subject_waits_for_its_verb() {
    // Refs (UD multiclause-05): she → nsubj → cleans, cleans → parataxis →
    // cooks. The (Verb, nominal) Right arcs fire across the comma, so the
    // second clause's subject misattaches as dobj to the first verb instead
    // of shifting into Left-nsubj from its own predicate.
    let (_doc, set) = parse("He cooks, she cleans.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (she, cleans) = (at("she"), at("cleans"));
    assert_eq!(deps[she], "nsubj", "deps: {deps:?}");
    assert_eq!(she as i32 + set.0[she].head, cleans as i32);
}

#[test]
fn postverbal_pronoun_without_comma_stays_dobj() {
    // Must-NOT-fire: with no clause boundary between verb and pronoun, the
    // dobj reading stands.
    let (_doc, set) = parse("Call me later.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("me")], "dobj", "deps: {deps:?}");
}

#[test]
fn nominal_headed_where_is_sconj_mark() {
    // Refs (UD relative-10): where → mark → met. The tagger leaves `where`
    // as NOUN (the §8.2-adjacent false negative), so the existing mark arm
    // — which is Sconj-keyed, like `when`/`because` — never fires. The POS
    // stays divergent (refs pin ADP, following the `as`-precedent's SCONJ
    // functional reading instead); the attachment is the UD-substantive
    // half and is asserted exactly.
    let (_doc, set) = parse("The store where we met closed.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (where_, met) = (at("where"), at("met"));
    assert_eq!(pos[where_], "sconj", "pos: {pos:?}");
    assert_eq!(deps[where_], "mark", "deps: {deps:?}");
    assert_eq!(where_ as i32 + set.0[where_].head, met as i32);
}

#[test]
fn interrogative_where_stays_noun() {
    // Must-NOT-fire: sentence-initial interrogative `where` has no nominal
    // head — stays NOUN (its ADV reading belongs to the adverb-lexicon
    // gap, explicitly out of scope).
    let (_doc, set) = parse("Where is my bag?");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("Where")], "noun", "pos: {pos:?}");
}

#[test]
fn object_relative_anchor_survives_embedded_subject() {
    // Refs (UD relative-02): bought → relcl → book, book → nsubj →
    // vanished. The anchor wait covers the marker ([book], that) but the
    // anchor is popped again on the embedded subject ([book], I) — Reduce
    // outbids Shift twice, so the relcl pair never meets and everything
    // strands into repair-dep.
    let (_doc, set) = parse("The book that I bought vanished.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (book, bought, vanished) = (at("book"), at("bought"), at("vanished"));
    assert_eq!(deps[bought], "relcl", "deps: {deps:?}");
    assert_eq!(bought as i32 + set.0[bought].head, book as i32);
    assert_eq!(deps[book], "nsubj", "deps: {deps:?}");
    assert_eq!(book as i32 + set.0[book].head, vanished as i32);
}

#[test]
fn matrix_first_relative_frame_is_stable() {
    // Must-NOT-fire (invariance): a matrix-first verb with a trailing
    // subject-relative keeps its root and its relcl arc — the extended
    // anchor wait must not reroute it.
    let (_doc, set) = parse("I know the man who called.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("know")], "root", "deps: {deps:?}");
    assert_eq!(deps[at("called")], "relcl", "deps: {deps:?}");
}

#[test]
fn final_manner_adverbial_is_adv() {
    // Refs (UD svo-02): loudly → adv/advmod → barks. Closed-class
    // time/manner adverbials in sentence-final position fall through to
    // NOUN; with the verb directly on the stack the existing Right-advmod
    // arm attaches them.
    let (_doc, set) = parse("The dog barks loudly.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (barks, loudly) = (at("barks"), at("loudly"));
    assert_eq!(pos[loudly], "adv", "pos: {pos:?}");
    assert_eq!(deps[loudly], "advmod", "deps: {deps:?}");
    assert_eq!(loudly as i32 + set.0[loudly].head, barks as i32);
}

#[test]
fn coordinated_adverbial_candidate_stays_noun() {
    // Must-NOT-fire (boundary): "daily" before a conjunction ("Run daily
    // or quit") sits in a coordination frame, not a final/comma adverbial
    // slot — CC-disambiguation is its own rule, so it stays NOUN even
    // though the sibling ref reads ADV.
    let (_doc, set) = parse("Run daily or quit.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("daily")], "noun", "pos: {pos:?}");
}

#[test]
fn temporal_today_stays_noun() {
    // Must-NOT-fire (boundary): "today" reads NOUN/advmod in its frozen ref
    // ("Send the invoice today", the UD npadvmod analysis) while a sibling
    // ref reads ADV — the refs are irreconcilable, so the word stays out of
    // the adverbial set until they reconcile.
    let (_doc, set) = parse("Send the invoice today.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("today")], "noun", "pos: {pos:?}");
}

#[test]
fn preverb_aux_attaches_to_predicate() {
    // Refs (UD question-04/10/12): Can → aux → help. The Left-aux arm
    // exists, but Reduce pops the auxiliary on its subject (no arc pairs
    // Aux with a nominal), so it strands into repair-dep with the right
    // head and the wrong label. The anchor-style wait lets the pair meet.
    let (_doc, set) = parse("Can you help me?");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (can, help) = (at("Can"), at("help"));
    assert_eq!(deps[can], "aux", "deps: {deps:?}");
    assert_eq!(can as i32 + set.0[can].head, help as i32);
}

#[test]
fn modal_neg_aux_coexists() {
    // Must-NOT-fire (invariance): "wo" keeps its aux arc to the predicate
    // with "n't" taking neg — the new Left-aux arm must route through the
    // same head the repair fallback found, not disturb the neg attachment.
    let (_doc, set) = parse("She won't answer calls.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (wo, answer, nt) = (at("wo"), at("answer"), at("n't"));
    assert_eq!(deps[wo], "aux", "deps: {deps:?}");
    assert_eq!(wo as i32 + set.0[wo].head, answer as i32);
    assert_eq!(deps[nt], "neg", "deps: {deps:?}");
}

#[test]
fn copular_be_keeps_cop() {
    // Must-NOT-fire: "is" before a predicate adjective stays cop — the aux
    // arm scores below cop and never sees the (Aux, Adj) pair anyway.
    let (_doc, set) = parse("Your fee is low.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("is")], "cop", "deps: {deps:?}");
}

#[test]
fn sensory_linking_frame_is_verbal() {
    // Refs (UD copular-12): smells → verb/root, Dinner → nsubj → smells,
    // great → adj/acomp → smells. Bare-initial sensory verbs fall outside
    // every subject rule (the bare-initial boundary names weather and
    // achievement verbs, never sensory perception with a following
    // predicate nominal), and their complements strand as nominal objects.
    // Two sequenced upgrades in one frame pass (contracted-be shape).
    let (_doc, set) = parse("Dinner smells great.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (dinner, smells, great) = (at("Dinner"), at("smells"), at("great"));
    assert_eq!(pos[smells], "verb", "pos: {pos:?}");
    assert_eq!(deps[smells], "root", "deps: {deps:?}");
    assert_eq!(deps[dinner], "nsubj", "deps: {deps:?}");
    assert_eq!(dinner as i32 + set.0[dinner].head, smells as i32);
    assert_eq!(pos[great], "adj", "pos: {pos:?}");
    assert_eq!(deps[great], "acomp", "deps: {deps:?}");
    assert_eq!(great as i32 + set.0[great].head, smells as i32);
}

#[test]
fn transitive_sensory_object_stays_noun() {
    // Must-NOT-fire: a determiner-led object ("the soup") is transitive,
    // not a predicate complement — stays NOUN even after a sensory verb.
    // (Bare transitives like "smells smoke" stay a documented boundary:
    // no bench instance distinguishes them positionally.)
    let (_doc, set) = parse("Taste the soup.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("soup")], "noun", "pos: {pos:?}");
}

#[test]
fn clause_initial_verb_after_comma_is_verbal() {
    // Refs (UD parenthetical-08): scared → verb/root, us → dobj → scared.
    // Matrix verbs opening after a parenthetical-closing comma fall through
    // to NOUN (no subject rule sees them) and strand their objects.
    let (_doc, set) = parse("The test, honestly, scared us.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (scared, us) = (at("scared"), at("us"));
    assert_eq!(pos[scared], "verb", "pos: {pos:?}");
    assert_eq!(deps[scared], "root", "deps: {deps:?}");
    assert_eq!(deps[us], "dobj", "deps: {deps:?}");
    assert_eq!(us as i32 + set.0[us].head, scared as i32);
}

#[test]
fn post_comma_true_subject_stays_noun() {
    // Must-NOT-fire: "schools" after a subordinate-clause comma is the next
    // clause's subject, not a predicate — stays NOUN. The parenthetical
    // opener (nominal + comma, no verb before the target) is absent here.
    let (_doc, set) = parse("If it snows, schools close.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("schools")], "noun", "pos: {pos:?}");
}

#[test]
fn coordinated_predicate_adjective_stays_nonverbal() {
    // Must-NOT-fire: "red" in "red and fast" is a coordinated predicate
    // adjective (ADJ gap, explicitly out of scope) — the CC-next guard
    // keeps it off the verb path as well as the noun path is wrong.
    let (_doc, set) = parse("Her car, red and fast, won.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(pos[at("red")], "verb", "pos: {pos:?}");
}

#[test]
fn shifted_det_noun_verb_after_boundary_is_verbal() {
    // Refs (UD parenthetical-06): failed → verb/root, plan → nsubj →
    // failed. The DET+NOUN+initial frame shifted past a leading clause
    // boundary (the initial-noun rule only sees positions 0-2, shielded
    // here by the sentence-initial adverbial). Only the clause-final
    // predicate qualifies — an appositive nominal (`an old brick,` with
    // more clause to come) never matches.
    let (_doc, set) = parse("Truthfully, the plan failed.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (plan, failed) = (at("plan"), at("failed"));
    assert_eq!(pos[failed], "verb", "pos: {pos:?}");
    assert_eq!(deps[failed], "root", "deps: {deps:?}");
    assert_eq!(deps[plan], "nsubj", "deps: {deps:?}");
    assert_eq!(plan as i32 + set.0[plan].head, failed as i32);
}

#[test]
fn comma_free_det_noun_pair_stays_nominal() {
    // Must-NOT-fire: without a clause boundary before the determiner
    // ("Show me the sales report"), DET+NOUN+NOUN is a plain nominal —
    // the shifted frame needs its comma. (Uses "sales": "report" itself
    // is a closed-list verb over-fire, documented separately.)
    let (_doc, set) = parse("Show me the sales report.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("sales")], "noun", "pos: {pos:?}");
}

#[test]
fn comment_clause_as_is_sconj_mark() {
    // Refs (UD subordinate-01): As → sconj/mark → know, you → nsubj →
    // know. Sentence-initial comment clauses (`As you know`) read through
    // the closed ADP map, so the marker arm never fires and the subject
    // misattaches as a prepositional object.
    let (_doc, set) = parse("As you know, your fee is low.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (as_, you, know) = (at("As"), at("you"), at("know"));
    assert_eq!(pos[as_], "sconj", "pos: {pos:?}");
    assert_eq!(deps[as_], "mark", "deps: {deps:?}");
    assert_eq!(as_ as i32 + set.0[as_].head, know as i32);
    assert_eq!(deps[you], "nsubj", "deps: {deps:?}");
    assert_eq!(you as i32 + set.0[you].head, know as i32);
}

#[test]
fn comment_clause_everybody_is_subject() {
    // Refs (UD subordinate-12): As → sconj/mark → knows, everybody →
    // pron/nsubj → knows. Same frame with an indefinite-pronoun subject
    // (closed-map gap alongside the As gap); the matrix-clause
    // parataxis/root half belongs to later work and is unasserted.
    let (_doc, set) = parse("As everybody knows, lunch ends early.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (as_, everybody, knows) = (at("As"), at("everybody"), at("knows"));
    assert_eq!(pos[as_], "sconj", "pos: {pos:?}");
    assert_eq!(pos[everybody], "pron", "pos: {pos:?}");
    assert_eq!(deps[as_], "mark", "deps: {deps:?}");
    assert_eq!(deps[everybody], "nsubj", "deps: {deps:?}");
    assert_eq!(everybody as i32 + set.0[everybody].head, knows as i32);
}

#[test]
fn medial_as_frame_stays_prepositional() {
    // Must-NOT-fire: medial "as" (`Paris, as always, …`) is not a comment
    // clause — the As rule is sentence-initial only.
    let (_doc, set) = parse("Paris, as always, charmed us.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("as")], "adp", "pos: {pos:?}");
}

#[test]
fn inverted_copular_predicate_is_adjectival() {
    // Refs (UD question-11): ready → adj/root, lunch → nsubj → ready,
    // Is → cop → ready. The be-predicate rule only sees AUX-adjacent
    // complements, so an inverted copular (overt subject between be and
    // predicate) strands — while the predicate-nominal control below
    // keeps subject-less complements (`a doctor`) shut.
    let (_doc, set) = parse("Is lunch ready?");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (is, lunch, ready) = (at("Is"), at("lunch"), at("ready"));
    assert_eq!(pos[ready], "adj", "pos: {pos:?}");
    assert_eq!(deps[ready], "root", "deps: {deps:?}");
    assert_eq!(deps[lunch], "nsubj", "deps: {deps:?}");
    assert_eq!(lunch as i32 + set.0[lunch].head, ready as i32);
    assert_eq!(deps[is], "cop", "deps: {deps:?}");
    assert_eq!(is as i32 + set.0[is].head, ready as i32);
}

#[test]
fn predicate_nominal_after_be_stays_noun() {
    // Must-NOT-fire: "a doctor" is a predicate nominal, not a predicate
    // adjective — with only a bare determiner between be and target there
    // is no overt subject, so the inverted frame stays shut.
    let (_doc, set) = parse("She is a doctor.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("doctor")], "noun", "pos: {pos:?}");
}

#[test]
fn object_relative_pronoun_is_obj() {
    // Refs (UD relative-02/08): that → obj → bought. The candidate set
    // offers only subject-Left for (Pron, Verb), so an object relativizer
    // with an overt subject strands into repair-dep — head and label both
    // wrong. The obj arm fires only with a nominative pronoun visibly
    // between marker and verb (the subject that makes s the object).
    let (_doc, set) = parse("The book that I bought vanished.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (that, bought) = (at("that"), at("bought"));
    assert_eq!(deps[that], "obj", "deps: {deps:?}");
    assert_eq!(that as i32 + set.0[that].head, bought as i32);
}

#[test]
fn subject_relative_pronoun_stays_nsubj() {
    // Must-NOT-fire: "that" directly heading its clause verb ("The dog
    // that barked") is the subject — no intervening pronoun, no obj arm.
    let (_doc, set) = parse("The dog that barked ran off.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (that, barked) = (at("that"), at("barked"));
    assert_eq!(deps[that], "nsubj", "deps: {deps:?}");
    assert_eq!(that as i32 + set.0[that].head, barked as i32);
}

#[test]
fn relative_matrix_verb_after_relcl_is_verb() {
    // Refs (UD relative-01): left → verb/root, man → nsubj → left. "left"
    // is outside the closed verb list, so the matrix verb falls through to
    // NOUN and the relcl verb steals root. This rule restores the verb tag;
    // the root/relcl attachment half belongs to later work and is
    // deliberately unasserted (only connectivity + no parataxis invent).
    let (_doc, set) = parse("The man who called left.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (called, left) = (at("called"), at("left"));
    assert_eq!(pos[called], "verb", "pos: {pos:?}");
    assert_eq!(pos[left], "verb", "pos: {pos:?}");
    assert_ne!(deps[left], "parataxis", "deps: {deps:?}");
    let abs = left as i32 + set.0[left].head;
    assert!((0..set.len() as i32).contains(&abs), "left headed: {deps:?}");
}

#[test]
fn relative_matrix_verb_that_clause_is_verb() {
    // Refs (UD relative-02): vanished → verb/root. "that" lexes DET with a
    // nominal head (book) — the relative frame, not a complementizer.
    let (_doc, set) = parse("The book that I bought vanished.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("vanished")], "verb", "pos: {pos:?}");
}

#[test]
fn complement_that_object_stays_noun() {
    // Must-NOT-fire: "that" headed by a VERB (know) is a complementizer,
    // not a relativizer — "soccer" is a true nominal object and stays NOUN.
    let (_doc, set) = parse("I know that they play soccer.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("soccer")], "noun", "pos: {pos:?}");
    assert_eq!(deps[at("soccer")], "dobj", "deps: {deps:?}");
}

#[test]
fn relative_frame_titlecase_final_stays_noun() {
    // Must-NOT-fire: a title-case sentence-final nominal ("Anna") is the
    // §8.2 proper-noun class, not a finite verb — the lowercase guard keeps
    // it NOUN even inside a relative frame.
    let (_doc, set) = parse("The man who called Anna.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("Anna")], "noun", "pos: {pos:?}");
}