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
fn inversion_subject_withheld_from_infinitive_upgrade() {
    // Refs (UD wh-aux-02): a finite verb later in the clause proves the
    // post-host nominal is the inverted subject, not the infinitive —
    // `photosynthesis` stays NOUN/nsubj→work, `does` aux→work, `work`
    // crowns. (The WH-word itself stays NOUN/nsubj: the adverb-lexicon gap
    // is explicitly out of scope; only its head is claimed.) Determined
    // frame: no oracle tie.
    let (_doc, set) = parse("How does photosynthesis work?");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (how, does, subj, verb) = (at("How"), at("does"), at("photosynthesis"), at("work"));
    assert_eq!(pos[subj], "noun", "pos: {pos:?}");
    assert_eq!(deps[subj], "nsubj", "deps: {deps:?}");
    assert_eq!(subj as i32 + set.0[subj].head, verb as i32);
    assert_eq!(deps[does], "aux", "deps: {deps:?}");
    assert_eq!(does as i32 + set.0[does].head, verb as i32);
    assert_eq!(deps[verb], "root", "deps: {deps:?}");
    assert_eq!(how as i32 + set.0[how].head, verb as i32, "deps: {deps:?}");
    let (conf, _, _) = ambiguity_of("How does photosynthesis work?");
    assert_eq!(conf.oracle_tie_count, 0, "determined frame must not tie: {conf:?}");
}

#[test]
fn later_sform_object_does_not_block_infinitive_upgrade() {
    // Must-NOT-fire (boundary of the inversion gate): a clause-final s-form
    // is a plural object noun, not the clause predicate — `calls` must not
    // read as the later verb, so `answer` still upgrades (see
    // `aux_neg_bare_infinitive_answer_is_verb` for the attachment pins).
    let (_doc, set) = parse("She won't answer calls.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("answer")], "verb", "pos: {pos:?}");
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
    // (cconj-cc, adversative coordination like "but") needs its own arm; rose steals root until the NOUN matrix
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
fn clause_final_conjoined_verb_resolves_by_first_conjunct() {
    // Refs (UD coordination-08): failed → verb/conj → studied. The
    // predecessor of this test documented the blocker — "failed" and
    // "eggs" share the CC+NOUN-final shape, disambiguable only by
    // clause-subject tracking. That tracker now exists: a VERB two back
    // (studied) proves a verbal second conjunct, a nominal two back
    // (milk) a nominal one. The (Verb, Verb) conj arm (overt-subject or
    // shared-subject shape) does the rest.
    let (_doc, set) = parse("She studied but failed.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (studied, failed) = (at("studied"), at("failed"));
    assert_eq!(pos[failed], "verb", "pos: {pos:?}");
    assert_eq!(deps[failed], "conj", "deps: {deps:?}");
    assert_eq!(failed as i32 + set.0[failed].head, studied as i32);
}

#[test]
fn adverbial_modified_coordination_stays_nominal() {
    // Must-NOT-fire (boundary): "quit" follows a conjunction, but the
    // first conjunct's head is shielded by the pinned-NOUN adverbial
    // "daily" — the agreement tracker reads the adjacent token only, so
    // adverbial-modified coordinations ("Run daily or quit", "Run fast or
    // lose") stay nominal pending daily-disambiguation.
    let (_doc, set) = parse("Run daily or quit.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("quit")], "noun", "pos: {pos:?}");
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
fn clausal_after_is_sconj_mark() {
    // Refs (UD subordinate-05/10): after → sconj/mark → spoke|scored,
    // he|she → nsubj → spoke|scored. The closed map lexes every `after`
    // as ADP, so the clausal reading never reaches the mark arm and the
    // subject misattaches as pobj.
    let (_doc, set) = parse("The game ended after she scored.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (after, she, scored) = (at("after"), at("she"), at("scored"));
    assert_eq!(pos[after], "sconj", "pos: {pos:?}");
    assert_eq!(deps[after], "mark", "deps: {deps:?}");
    assert_eq!(after as i32 + set.0[after].head, scored as i32);
    assert_eq!(deps[she], "nsubj", "deps: {deps:?}");
    assert_eq!(she as i32 + set.0[she].head, scored as i32);
}

#[test]
fn nominal_after_stays_prepositional() {
    // Must-NOT-fire: nominal `after` (`after lunch`) is a true preposition
    // — no clause follows, so the SCONJ upgrade must leave it alone.
    let (_doc, set) = parse("The meeting ended after lunch.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (after, lunch) = (at("after"), at("lunch"));
    assert_eq!(pos[after], "adp", "pos: {pos:?}");
    assert_eq!(deps[after], "prep", "deps: {deps:?}");
    assert_eq!(deps[lunch], "pobj", "deps: {deps:?}");
    assert_eq!(lunch as i32 + set.0[lunch].head, after as i32);
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
fn interrogative_where_is_adverbial() {
    // Refs (UD question-02/08, wh-aux-04, wh-copula-03): clause-initial
    // interrogative `where` is ADV/advmod — unanimously across all four
    // refs — so it reads from the WH-adverbial bit, not the adverb-lexicon
    // gap. (This retires the old NOUN pin: the gap doctrine stands for open
    // adverbs, but closed WH-forms are data, and every ref agrees.)
    // Medial `where` keeps its own dynamics (see `nominal_headed_where`).
    for text in [
        "Where is my bag?",
        "Where is the station?",
        "Where did she go?",
        "Where is the invoice?",
    ] {
        let (_doc, set) = parse(text);
        let pos = pos_of(&set);
        let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
        assert_eq!(pos[at("Where")], "adv", "{text}: pos: {pos:?}");
    }
}

#[test]
fn where_be_question_crowns_aux() {
    // Refs (UD question-02/08): is → aux/root, station/bag → noun/dobj →
    // is, the/my → det → station/bag, Where → adv/advmod → is. The Where-be
    // root rung crowns the be-AUX, the gated be-dobj arm lands the
    // complement, and the WH-adverbial arm lands Where on the crowned be.
    // No oracle tie: the frame is determined, so Track B stays quiet.
    for (text, det, comp) in [
        ("Where is the station?", "the", "station"),
        ("Where is my bag?", "my", "bag"),
    ] {
        let (_doc, set) = parse(text);
        let pos = pos_of(&set);
        let deps = deps(&set);
        let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
        let (where_, is, det_, comp_) = (at("Where"), at("is"), at(det), at(comp));
        assert_eq!(pos[is], "aux", "{text}: pos: {pos:?}");
        assert_eq!(deps[is], "root", "{text}: deps: {deps:?}");
        assert_eq!(set.0[is].head, 0, "{text}: root head");
        assert_eq!(deps[comp_], "dobj", "{text}: deps: {deps:?}");
        assert_eq!(comp_ as i32 + set.0[comp_].head, is as i32, "{text}");
        assert_eq!(deps[det_], "det", "{text}: deps: {deps:?}");
        assert_eq!(det_ as i32 + set.0[det_].head, comp_ as i32, "{text}");
        assert_eq!(pos[where_], "adv", "{text}: pos: {pos:?}");
        assert_eq!(deps[where_], "advmod", "{text}: deps: {deps:?}");
        let abs = where_ as i32 + set.0[where_].head;
        assert_eq!(abs, is as i32, "{text}: Where headed by be");
        let (conf, _, _) = ambiguity_of(text);
        assert_eq!(conf.oracle_tie_count, 0, "{text}: determined frame must not tie: {conf:?}");
    }
}

#[test]
fn where_aux_with_verb_keeps_verb_root() {
    // Must-NOT-fire: `Where did she go` has a finite verb — the VERB rung
    // wins before the Where-be rung is consulted, and `did` is not a
    // be-form anyway. go stays root with did → aux → go and she → nsubj
    // → go; the gated be-dobj arm never sees the pair. Where itself reads
    // ADV/advmod via the generic (Adv, Verb) arm.
    let (_doc, set) = parse("Where did she go?");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (did, she, go) = (at("did"), at("she"), at("go"));
    assert_eq!(pos[go], "verb", "pos: {pos:?}");
    assert_eq!(deps[go], "root", "deps: {deps:?}");
    assert_eq!(deps[did], "aux", "deps: {deps:?}");
    assert_eq!(did as i32 + set.0[did].head, go as i32);
    assert_eq!(deps[she], "nsubj", "deps: {deps:?}");
    assert_eq!(she as i32 + set.0[she].head, go as i32);
    assert_eq!(pos[at("Where")], "adv", "pos: {pos:?}");
    assert_eq!(deps[at("Where")], "advmod", "deps: {deps:?}");
}

#[test]
fn initial_when_with_inversion_is_adverbial() {
    // Refs (UD wh-aux-03): clause-initial `when` before an AUX is the
    // interrogative adverbial (ADV/advmod → close), not the subordinator —
    // while medial `when` keeps SCONJ/mark (see
    // `when_marker_attaches_to_clause_verb`). Determined frame, no tie.
    let (_doc, set) = parse("When does the store close?");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (when, close) = (at("When"), at("close"));
    assert_eq!(pos[when], "adv", "pos: {pos:?}");
    assert_eq!(deps[when], "advmod", "deps: {deps:?}");
    assert_eq!(when as i32 + set.0[when].head, close as i32);
    assert_eq!(deps[close], "root", "deps: {deps:?}");
    let (conf, _, _) = ambiguity_of("When does the store close?");
    assert_eq!(conf.oracle_tie_count, 0, "determined frame must not tie: {conf:?}");
}

#[test]
fn fronted_when_without_inversion_keeps_subordinator() {
    // Must-NOT-fire: initial `when` NOT followed by an AUX is a fronted
    // subordinate marker (`When available, call me back`) — SCONJ stands,
    // the inversion upgrade never sees the pair.
    let (_doc, set) = parse("When available, call me back.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("When")], "sconj", "pos: {pos:?}");
}

#[test]
fn why_how_initial_read_adverbial() {
    // Refs (UD question-05, wh-aux-02): clause-initial `why`/`how` are
    // ADV — the same closed-class exception as interrogative `where`
    // (the open-adverb gap doctrine is untouched). Attachments follow
    // their predicates: Why → advmod → blue via the WH-adjective arm,
    // How → advmod → work via the generic (Adv, Verb) arm.
    for (text, wh, pred) in [
        ("Why is the sky blue?", "Why", "blue"),
        ("How does photosynthesis work?", "How", "work"),
    ] {
        let (_doc, set) = parse(text);
        let pos = pos_of(&set);
        let deps = deps(&set);
        let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
        let (wh_, pred_) = (at(wh), at(pred));
        assert_eq!(pos[wh_], "adv", "{text}: pos: {pos:?}");
        assert_eq!(deps[wh_], "advmod", "{text}: deps: {deps:?}");
        assert_eq!(wh_ as i32 + set.0[wh_].head, pred_ as i32, "{text}");
    }
}

#[test]
fn where_invoice_convention_split_is_stable() {
    // Must-NOT-fire (boundary documentation): wh-copula-03 pins the
    // competing UD convention (invoice → noun/root, is → cop → invoice)
    // for a POS-identical frame, so the Where-be rung crowns `is` there
    // against that ref. Only the convention-agreed tokens are pinned
    // (the → det → invoice, ? → punct → invoice); the split itself is
    // pinned stable with no oracle tie rather than silently re-headed.
    let (_doc, set) = parse("Where is the invoice?");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (the, invoice, q) = (at("the"), at("invoice"), at("?"));
    assert_eq!(deps[the], "det", "deps: {deps:?}");
    assert_eq!(the as i32 + set.0[the].head, invoice as i32);
    assert_eq!(deps[q], "punct", "deps: {deps:?}");
    assert_eq!(q as i32 + set.0[q].head, invoice as i32, "deps: {deps:?}");
    let (conf, _, _) = ambiguity_of("Where is the invoice?");
    assert_eq!(conf.oracle_tie_count, 0, "split frame must not tie: {conf:?}");
}

#[test]
fn non_where_be_nominal_keeps_nominal_root() {
    // Must-NOT-fire: the Where-be rung is clause-initial-Where-gated —
    // `She is a doctor` (be + DET + nominal, no Where) keeps its nominal
    // root and never sees the be-dobj arm.
    let (_doc, set) = parse("She is a doctor.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("doctor")], "root", "deps: {deps:?}");
    assert_ne!(deps[at("is")], "root", "deps: {deps:?}");
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
    // Refs (UD subverbobj-02): loudly → adv/advmod → barks. Closed-class
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
    // Bare-subject frame: no DET between be and predicate, so the
    // copular-complement category tie never fires — determined and quiet.
    let (conf, _, _) = ambiguity_of("Is lunch ready?");
    assert_eq!(conf.oracle_tie_count, 0, "bare-subject copular must not tie: {conf:?}");
    let (reason, should) = refine_of("Is lunch ready?");
    assert_eq!(
        reason,
        spacy_rs::RefineReason::NoTrigger,
        "bare-subject copular must not refine, got {reason:?}"
    );
    assert!(!should, "should_refine must be false");
}

#[test]
fn det_led_copular_complement_emits_category_tie() {
    // Track B (positive): "What is the water cycle" has a DET-led nominal
    // between be and the predicate — the same shape reads as an
    // overt-subject predicate adjective (`is the sky blue`) or a modified
    // predicate nominal, and only lexicon knowledge splits them. The oracle
    // must record a near-tie (→ RefineReason::Confidence(Ties)) and the
    // frame stage an AttachmentNearTie with a provisional key — never a
    // confident cop. Heads/labels are unchanged (cop wins ties by stable
    // order); only the margin drops.
    let (conf, analysis, keys) = ambiguity_of("What is the water cycle?");
    assert!(
        conf.oracle_tie_count >= 1,
        "DET-led copular complement must tie: {conf:?}"
    );
    assert!(
        conf.oracle_margins
            .iter()
            .any(|m| m.abs() <= spacy_rs::TIE_MARGIN_EPSILON),
        "near-zero margin expected: {conf:?}"
    );
    assert!(
        analysis
            .ambiguities
            .iter()
            .any(|a| a.kind == spacy_rs::AmbiguityKind::AttachmentNearTie),
        "AttachmentNearTie expected: {analysis:?}"
    );
    assert!(
        keys.iter().any(|k| k.provisional),
        "ambiguous frame mints a provisional key"
    );
    let (reason, should) = refine_of("What is the water cycle?");
    assert_eq!(
        reason,
        spacy_rs::RefineReason::Confidence(spacy_rs::ConfidenceReason::Ties),
        "DET-led copular complement must refine on Ties, got {reason:?}"
    );
    assert!(should, "should_refine must be true for {reason:?}");
    // Parse-stability pin: the tie must not re-head anything (is keeps
    // its cop arc to the crowned predicate).
    let (_doc, set) = parse("What is the water cycle?");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (is, cycle) = (at("is"), at("cycle"));
    assert_eq!(deps[is], "cop", "deps: {deps:?}");
    assert_eq!(is as i32 + set.0[is].head, cycle as i32);
    assert_eq!(deps[cycle], "root", "deps: {deps:?}");
}

#[test]
fn modified_predicate_nominal_emits_category_tie() {
    // Track B (positive, second frame): "This is a Rust project" is the
    // modified-predicate-nominal reading of the same ambiguous shape (a +
    // Rust between be and project). Same dynamics: near-tie, provisional
    // key, confidence-axis refine — with the parse itself untouched.
    let (conf, analysis, keys) = ambiguity_of("This is a Rust project.");
    assert!(
        conf.oracle_tie_count >= 1,
        "modified predicate nominal must tie: {conf:?}"
    );
    assert!(
        analysis
            .ambiguities
            .iter()
            .any(|a| a.kind == spacy_rs::AmbiguityKind::AttachmentNearTie),
        "AttachmentNearTie expected: {analysis:?}"
    );
    assert!(
        keys.iter().any(|k| k.provisional),
        "ambiguous frame mints a provisional key"
    );
    let (reason, should) = refine_of("This is a Rust project.");
    assert!(
        matches!(reason, spacy_rs::RefineReason::Confidence(_)),
        "confidence-axis refine must fire, got {reason:?}"
    );
    assert!(should, "should_refine must be true for {reason:?}");
    let (_doc, set) = parse("This is a Rust project.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("is")], "cop", "deps: {deps:?}");
    assert_eq!(deps[at("project")], "root", "deps: {deps:?}");
}

#[test]
fn direct_copular_complement_emits_no_tie() {
    // Track B (must-NOT-fire control): a direct complement (`Your fee is
    // low` — nothing between be and predicate) is fully determined — no
    // tie, no ambiguity entry, permanent keys, no confidence-axis refine.
    let (conf, analysis, keys) = ambiguity_of("Your fee is low.");
    assert_eq!(conf.oracle_tie_count, 0, "direct copular must not tie: {conf:?}");
    assert!(
        analysis.ambiguities.is_empty(),
        "no ambiguity on a direct complement: {analysis:?}"
    );
    assert!(
        keys.iter().all(|k| !k.provisional),
        "clean frames mint permanent keys"
    );
    let (reason, should) = refine_of("Your fee is low.");
    assert_eq!(
        reason,
        spacy_rs::RefineReason::NoTrigger,
        "direct copular must not refine, got {reason:?}"
    );
    assert!(!should, "should_refine must be false");
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
fn because_clause_verb_attaches_ccomp() {
    // Refs (UD subordinate-02/07): snowed → ccomp → stayed. The clause verb
    // strands in repair-dep: the matrix verb reduces before the subordinate
    // arrives, so the (Verb, Verb) pair never meets.
    let (_doc, set) = parse("We stayed because it snowed.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (stayed, snowed) = (at("stayed"), at("snowed"));
    assert_eq!(deps[snowed], "ccomp", "deps: {deps:?}");
    assert_eq!(snowed as i32 + set.0[snowed].head, stayed as i32);
}

#[test]
fn when_clause_verb_attaches_advcl() {
    // Refs (UD subordinate-03/08): works → advcl → sings. Same stranding as
    // the because-frame, with marker-based label discrimination (when → advcl,
    // because → ccomp).
    let (_doc, set) = parse("She sings when she works.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (sings, works) = (at("sings"), at("works"));
    assert_eq!(deps[works], "advcl", "deps: {deps:?}");
    assert_eq!(works as i32 + set.0[works].head, sings as i32);
}

#[test]
fn complement_clause_without_subordinator_gets_no_ccomp() {
    // Must-NOT-fire: verb-headed "that" is DET (complementizer), so no SCONJ
    // stands between know and play — neither ccomp nor advcl may fire, and
    // the parataxis arm (no boundary) stays out too.
    let (_doc, set) = parse("I know that they play soccer.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(deps[at("play")], "ccomp", "deps: {deps:?}");
    assert_ne!(deps[at("play")], "advcl", "deps: {deps:?}");
    assert_ne!(deps[at("play")], "parataxis", "deps: {deps:?}");
}

#[test]
fn coordinated_nominals_attach_conj() {
    // Refs (UD coordination-01/06/07/12): the first conjunct heads the
    // phrase (Cats → nsubj → play), the second depends on the first
    // (dogs → conj → Cats), the marker on the second (and → cc → dogs).
    // The parser has no conj arm: the first conjunct strands in repair-dep
    // and the second steals its role.
    let (_doc, set) = parse("Cats and dogs play.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (cats, and, dogs, play) = (at("Cats"), at("and"), at("dogs"), at("play"));
    assert_eq!(deps[cats], "nsubj", "deps: {deps:?}");
    assert_eq!(cats as i32 + set.0[cats].head, play as i32);
    assert_eq!(deps[dogs], "conj", "deps: {deps:?}");
    assert_eq!(dogs as i32 + set.0[dogs].head, cats as i32);
    assert_eq!(deps[and], "cc", "deps: {deps:?}");
    assert_eq!(and as i32 + set.0[and].head, dogs as i32);
}

#[test]
fn bare_nominal_pair_without_conjunction_stays_compound() {
    // Must-NOT-fire: a bare nominal pair with no CCONJ between (chase/red
    // cars) stays on the compound path — the conj arm needs an intervening
    // conjunction, and the compound gate must keep firing without one.
    let (_doc, set) = parse("Dogs chase red cars.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (red, cars) = (at("red"), at("cars"));
    assert_eq!(deps[red], "compound", "deps: {deps:?}");
    assert_eq!(red as i32 + set.0[red].head, cars as i32);
}

#[test]
fn determiner_before_closed_verb_is_nominal() {
    // Refs (UD question-06): report → noun/nsubj → arrive, arrive → verb
    // root. The closed verb list fires before any nominal guard, so a
    // determiner-led noun that collides with the list ("the report") tags
    // VERB, steals root, and strands the true predicate. A determiner never
    // governs a finite verb, so DET + closed-verb-form reads nominal —
    // which also unblocks the initial-noun rule (DET+NOUN+NOUN → the third
    // position upgrades to VERB).
    let (_doc, set) = parse("Did the report arrive?");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (report, arrive) = (at("report"), at("arrive"));
    assert_eq!(pos[report], "noun", "pos: {pos:?}");
    assert_eq!(deps[at("the")], "det", "deps: {deps:?}");
    assert_eq!(at("the") as i32 + set.0[at("the")].head, report as i32);
    // Question inversion (Did + subject NP + bare verb): the finite verb
    // after a do-modal-hosted subject upgrades to VERB and takes root,
    // the subject as nsubj, the host as aux.
    assert_eq!(pos[arrive], "verb", "pos: {pos:?}");
    assert_eq!(deps[arrive], "root");
    assert_eq!(deps[report], "nsubj", "deps: {deps:?}");
    assert_eq!(report as i32 + set.0[report].head, arrive as i32);
    assert_eq!(deps[at("Did")], "aux", "deps: {deps:?}");
    assert_eq!(at("Did") as i32 + set.0[at("Did")].head, arrive as i32);
}

#[test]
fn copular_inversion_predicate_stays_adjective() {
    // Must-NOT-fire: the inversion-verb upgrade is keyed on do-modal
    // hosts — a be-hosted predicate ("Is the sky blue") keeps its
    // copular reading (blue → adj/root, sky → nsubj → blue).
    let (_doc, set) = parse("Is the sky blue?");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (sky, blue) = (at("sky"), at("blue"));
    assert_eq!(pos[blue], "adj", "pos: {pos:?}");
    assert_eq!(deps[blue], "root");
    assert_eq!(deps[sky], "nsubj", "deps: {deps:?}");
    assert_eq!(sky as i32 + set.0[sky].head, blue as i32);
}

#[test]
fn bare_closed_verb_without_determiner_stays_verb() {
    // Must-NOT-fire: the nominal reading needs an overt determiner —
    // "bark" after a subject noun stays VERB, and "sales" (never
    // VERB-tagged) stays NOUN with or without one.
    let (_doc, set) = parse("Dogs bark loudly.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("bark")], "verb", "pos: {pos:?}");
    assert_eq!(deps[at("bark")], "root");
    let (_doc, set) = parse("Show me the sales report.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("sales")], "noun", "pos: {pos:?}");
}

#[test]
fn parenthetical_anchor_survives_apposition() {
    // Refs (UD parenthetical-01/03/04/05/07/09/10/11): the anchor is the
    // matrix subject (brother → nsubj → lives), the appositive depends on
    // the anchor (doctor → appos → brother). The comma pops the anchor
    // (Reduce outbids Shift) before the matrix verb arrives, so the anchor
    // strands in repair-dep while the appositive — attaching earlier —
    // lands correctly.
    let (_doc, set) = parse("My brother, a doctor, lives here.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (brother, doctor, lives) = (at("brother"), at("doctor"), at("lives"));
    assert_eq!(deps[brother], "nsubj", "deps: {deps:?}");
    assert_eq!(brother as i32 + set.0[brother].head, lives as i32);
    assert_eq!(deps[doctor], "appos", "deps: {deps:?}");
    assert_eq!(doctor as i32 + set.0[doctor].head, brother as i32);
}

#[test]
fn verbless_concessive_before_comma_keeps_tie_dynamics() {
    // Must-NOT-fire: "tired" before a comma inside a subordinate-marker
    // span is the Track B verbless case — the anchor wait must not reroute
    // it. Both the near-tie emission and the (unchanged) heads are pinned.
    let (conf, _analysis, _keys) = ambiguity_of("Although tired, she kept pace.");
    assert!(
        conf.oracle_tie_count >= 1,
        "verbless concessive must still tie: {conf:?}"
    );
    let (_doc, set) = parse("Although tired, she kept pace.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("Although")], "dep", "deps: {deps:?}");
    assert_eq!(deps[at("tired")], "nsubj", "deps: {deps:?}");
}

#[test]
fn clausal_coordination_first_predicate_is_verb() {
    // Refs (UD multiclause-06/08/11): the first predicate of a clausal
    // coordination is verbal (rose → verb/root, Prices → nsubj → rose).
    // Coordination joins likes: a CCONJ-headed second clause with an overt
    // VERB predicate proves the first predicate verbal too, so a NOUN with
    // a nominal subject upgrades. The existing nsubj arm and root
    // selection do the rest — no new arc.
    let (_doc, set) = parse("Prices rose yet wages stalled.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (prices, rose) = (at("Prices"), at("rose"));
    assert_eq!(pos[rose], "verb", "pos: {pos:?}");
    assert_eq!(deps[rose], "root");
    assert_eq!(deps[prices], "nsubj", "deps: {deps:?}");
    assert_eq!(prices as i32 + set.0[prices].head, rose as i32);
}

#[test]
fn nominal_pair_without_clausal_frame_stays_noun() {
    // Must-NOT-fire: "chase" has a nominal subject but no CCONJ-headed
    // second clause ahead (verb-capability needs lexicon knowledge —
    // Track B territory, explicitly out of scope).
    let (_doc, set) = parse("Dogs chase red cars.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("chase")], "noun", "pos: {pos:?}");
}

#[test]
fn clausal_conjunction_second_verb_attaches_conj() {
    // Refs (UD multiclause-03/06/08/11): the second clause predicate
    // depends on the first (stayed → conj → passed), the marker on the
    // second predicate (but → cc → stayed), the subject on its own
    // predicate (floods → nsubj → stayed). The matrix verb reduces on the
    // conjunction before the clause verb arrives, so the pair never meets
    // and everything strands.
    let (_doc, set) = parse("The storm passed but floods stayed.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (passed, but_, floods, stayed) = (at("passed"), at("but"), at("floods"), at("stayed"));
    assert_eq!(deps[stayed], "conj", "deps: {deps:?}");
    assert_eq!(stayed as i32 + set.0[stayed].head, passed as i32);
    assert_eq!(deps[but_], "cc", "deps: {deps:?}");
    assert_eq!(but_ as i32 + set.0[but_].head, stayed as i32);
    assert_eq!(deps[floods], "nsubj", "deps: {deps:?}");
    assert_eq!(floods as i32 + set.0[floods].head, stayed as i32);
}

#[test]
fn elliptical_but_resolves_by_first_conjunct() {
    // Refs (UD coordination-02, corrected): fell → verb/conj → ran. Same
    // elliptical mechanism as the studied/failed pair on the but marker:
    // clause-final with a VERB two back, and no finite verb ahead (an
    // overt subject like floods always has its predicate after it).
    let (_doc, set) = parse("She ran but fell.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (ran, but_, fell) = (at("ran"), at("but"), at("fell"));
    assert_eq!(pos[fell], "verb", "pos: {pos:?}");
    assert_eq!(deps[fell], "conj", "deps: {deps:?}");
    assert_eq!(fell as i32 + set.0[fell].head, ran as i32);
    assert_eq!(deps[but_], "cc", "deps: {deps:?}");
    assert_eq!(but_ as i32 + set.0[but_].head, fell as i32);
}

#[test]
fn verbless_fallback_root_emits_attachment_tie() {
    // Track B (positive): "Define photosynthesis." has no verb or aux, so
    // the nominal fallback root (Define) is genuinely ambiguous between an
    // imperative verb–object reading and an NP-fragment reading — the two
    // are POS-identical without lexicon knowledge (see
    // refine_pos_directive_initial). The oracle must record a near-tie (→
    // RefineReason::Confidence(Ties)) and the frame stage an
    // AttachmentNearTie with a provisional key — never a confident silent
    // misparse. Heads/labels are unchanged (Shift wins ties by stable
    // order); only the margin drops.
    let (conf, analysis, keys) = ambiguity_of("Define photosynthesis.");
    assert!(
        conf.oracle_tie_count >= 1,
        "verbless fallback root must tie: {conf:?}"
    );
    assert!(
        conf.oracle_margins
            .iter()
            .any(|m| m.abs() <= spacy_rs::TIE_MARGIN_EPSILON),
        "near-zero margin expected: {conf:?}"
    );
    assert!(
        analysis
            .ambiguities
            .iter()
            .any(|a| a.kind == spacy_rs::AmbiguityKind::AttachmentNearTie),
        "AttachmentNearTie expected: {analysis:?}"
    );
    assert!(
        keys.iter().any(|k| k.provisional),
        "ambiguous frame mints a provisional key"
    );
    // Parse-stability pin: the tie must not re-head anything.
    let (_doc, set) = parse("Define photosynthesis.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("Define")], "noun", "pos: {pos:?}");
    assert_eq!(deps[at("Define")], "root");
}

#[test]
fn verb_anchored_sentence_emits_no_root_tie() {
    // Track B (must-NOT-fire control): a sentence with a verbal anchor
    // ("Dogs bark loudly.") is fully determined — no root tie, no
    // ambiguity entry, permanent keys.
    let (conf, analysis, keys) = ambiguity_of("Dogs bark loudly.");
    assert_eq!(conf.oracle_tie_count, 0, "anchored clause must not tie: {conf:?}");
    assert!(
        analysis.ambiguities.is_empty(),
        "no ambiguity on a clean clause: {analysis:?}"
    );
    assert!(
        keys.iter().all(|k| !k.provisional),
        "clean frames mint permanent keys"
    );
}

#[test]
fn asyndetic_second_predicate_attaches_conj() {
    // Refs (UD multiclause-04/09): asyndetic coordination — two imperatives
    // juxtaposed by comma with no overt subject (play → conj → Work). The
    // comma+nominal branch owns subject-ful juxtaposition (parataxis: "He
    // cooks, she cleans"); the subject-less comma shape is coordination.
    let (_doc, set) = parse("Work hard, play fair.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (work, play) = (at("Work"), at("play"));
    assert_eq!(deps[play], "conj", "deps: {deps:?}");
    assert_eq!(play as i32 + set.0[play].head, work as i32);
}

#[test]
fn comma_framed_participle_modifies_anchor() {
    // Refs (UD parenthetical-05/11): comma-framed -ing forms are
    // participial modifiers (smiling → verb/amod → CEO), not appositive
    // nominals. The -ing morphology (same allocation-free suffix check as
    // the copular-predicate rule) plus the comma frame identifies them; the
    // existing Right arms plus a guarded Left-nsubj do the rest, and the
    // matrix subject still meets its verb.
    let (_doc, set) = parse("The CEO, smiling, took questions.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (ceo, smiling, took) = (at("CEO"), at("smiling"), at("took"));
    assert_eq!(pos[smiling], "verb", "pos: {pos:?}");
    assert_eq!(deps[smiling], "amod", "deps: {deps:?}");
    assert_eq!(smiling as i32 + set.0[smiling].head, ceo as i32);
    assert_eq!(deps[ceo], "nsubj", "deps: {deps:?}");
    assert_eq!(ceo as i32 + set.0[ceo].head, took as i32);
}

#[test]
fn unframed_participle_after_aux_stays_noun() {
    // Must-NOT-fire: the participial upgrade needs the comma frame —
    // "coming" after an AUX with no commas stays NOUN (its progressive
    // reading belongs to copular handling, per the existing aux-prev
    // pin).
    let (_doc, set) = parse("They aren't coming today.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("coming")], "noun", "pos: {pos:?}");
}

#[test]
fn temporal_yet_after_predicate_is_adv() {
    // Refs (UD contraction-09): yet → adv/advmod → ready. The closed map
    // lexes every "yet" as CCONJ, but sentence-final "yet" after a
    // predicate adjective is the temporal adverb (still/yet aspectual
    // family), not a coordinator — coordinators always have a second
    // clause (finite verb) ahead.
    let (_doc, set) = parse("She isn't ready yet.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (ready, yet) = (at("ready"), at("yet"));
    assert_eq!(pos[yet], "adv", "pos: {pos:?}");
    assert_eq!(deps[yet], "advmod", "deps: {deps:?}");
    assert_eq!(yet as i32 + set.0[yet].head, ready as i32);
}

#[test]
fn predicate_adjective_takes_trailing_advmod() {
    // Refs (UD contraction-06): again → adv/advmod → late. Copular
    // predicates take trailing adverbials rightward; no arm offers
    // (Adj, Adv), so they strand in repair-dep with the right head.
    let (_doc, set) = parse("You're late again.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (late, again) = (at("late"), at("again"));
    assert_eq!(deps[again], "advmod", "deps: {deps:?}");
    assert_eq!(again as i32 + set.0[again].head, late as i32);
}

#[test]
fn clausal_yet_stays_conjunction() {
    // Must-NOT-fire: "yet" with a second finite clause ahead ("Prices
    // rose yet wages stalled") is the adversative coordinator — stays
    // CCONJ even though no nominal follows it directly.
    let (_doc, set) = parse("Prices rose yet wages stalled.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("yet")], "cconj", "pos: {pos:?}");
    assert_eq!(deps[at("yet")], "cc", "deps: {deps:?}");
}

#[test]
fn bare_object_noun_after_verb_is_nominal() {
    // Refs (UD contraction-03): calls → noun/dobj → answer. A closed-verb
    // s-form directly after a VERB at sentence end is a plural object
    // noun, never a finite verb — English morphosyntax forbids a finite
    // s-form after a bare verb (modals/causatives govern bare forms), so
    // the s-form reads nominal. The existing dobj arm lands it.
    let (_doc, set) = parse("She won't answer calls.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (answer, calls) = (at("answer"), at("calls"));
    assert_eq!(pos[calls], "noun", "pos: {pos:?}");
    assert_eq!(deps[calls], "dobj", "deps: {deps:?}");
    assert_eq!(calls as i32 + set.0[calls].head, answer as i32);
}

#[test]
fn verb_form_after_pronoun_stays_verbal() {
    // Must-NOT-fire: the nominal reading needs a VERB host — "calls"
    // after a pronoun ("she calls") keeps its finite reading, as does a
    // non-s-form after a verb ("called left": ends in -t, matrix root).
    let (_doc, set) = parse("She calls Kelly.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("calls")], "verb", "pos: {pos:?}");
}

#[test]
fn negator_before_predicate_adjective_is_neg() {
    // Refs (UD contraction-09): n't → part/neg → ready. The neg arm is
    // verb-locked, but negators equally negate predicate adjectives;
    // the head is already right via repair, only the label strands.
    let (_doc, set) = parse("She isn't ready yet.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (nt, ready) = (at("n't"), at("ready"));
    assert_eq!(deps[nt], "neg", "deps: {deps:?}");
    assert_eq!(nt as i32 + set.0[nt].head, ready as i32);
}

#[test]
fn possessive_object_reads_through_determiner() {
    // Refs (UD question-09): snack → noun/dobj → ate. A verb reduces on a
    // determiner buffer before its object arrives, stranding the object in
    // repair-dep. Possessive determiners (my/your/her/…) obligatorily head
    // a nominal on their right, so the matrix verb can wait out the
    // determiner: it shifts, heads its noun, and vacates the stack for the
    // dobj arm. Articles route through the existing det dynamics
    // (unchanged); complementizer "that" is excluded by word, pronouns by
    // POS.
    let (_doc, set) = parse("Who ate my snack?");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (ate, snack) = (at("ate"), at("snack"));
    assert_eq!(deps[snack], "dobj", "deps: {deps:?}");
    assert_eq!(snack as i32 + set.0[snack].head, ate as i32);
}

#[test]
fn complementizer_that_does_not_wait() {
    // Must-NOT-fire: verb-headed "that" is a complementizer, not a
    // determiner — the matrix verb reduces as before so the embedded
    // subject meets its own verb ("they" stays nsubj → play, never
    // dobj → know).
    let (_doc, set) = parse("I know that they play soccer.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (they, play) = (at("they"), at("play"));
    assert_eq!(deps[they], "nsubj", "deps: {deps:?}");
    assert_eq!(they as i32 + set.0[they].head, play as i32);
}

#[test]
fn bare_ed_predicate_with_transitive_frame_is_verb() {
    // Refs (UD subverbobj-06/12): bare -ed predicates with a nominal subject and
    // a determiner-led object (John opened the door) are past-tense
    // transitives, not noun compounds. Past morphology plus the
    // transitive frame identifies them: -ed adjectives are attributive
    // (DET-led: the tired man) or predicative after linking/be (AUX
    // prev) — never bare-initial-subject position. The pre-existing
    // Verb–Det wait (which only excludes closed-list verbs) holds the
    // object slot, and the dobj arm lands it.
    let (_doc, set) = parse("John opened the door.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (john, opened, door) = (at("John"), at("opened"), at("door"));
    assert_eq!(pos[opened], "verb", "pos: {pos:?}");
    assert_eq!(deps[opened], "root");
    assert_eq!(deps[john], "nsubj", "deps: {deps:?}");
    assert_eq!(john as i32 + set.0[john].head, opened as i32);
    assert_eq!(deps[door], "dobj", "deps: {deps:?}");
    assert_eq!(door as i32 + set.0[door].head, opened as i32);
}

#[test]
fn nontransitive_ed_frames_keep_tags() {
    // Must-NOT-fire (double guard): "dropped" has a conjunction (not a
    // determiner-led object) after it, and "launched" a proper-noun
    // object — neither is the transitive frame, so both keep their
    // current tags (verb via agreement, verb via the closed list).
    let (_doc, set) = parse("Grades dropped yet spirits rose.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("dropped")], "verb", "pos: {pos:?}");
    let (_doc, set) = parse("NASA launched HTML5.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("launched")], "verb", "pos: {pos:?}");
}

#[test]
fn as_comment_matrix_is_root() {
    // Refs (UD subordinate-01): a comment clause depends on its matrix
    // (know → parataxis → low), and the matrix predicate heads (low →
    // adj/root). The leftmost verb is the subordinate predicate, so
    // position alone crowns the wrong head — the SCONJ-marked frame plus
    // a later matrix predicate proves it subordinate. The fee subject
    // path (nsubj via the aux guard and the clause-boundary idiom) is
    // pinned unchanged.
    let (_doc, set) = parse("As you know, your fee is low.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (know, fee, low) = (at("know"), at("fee"), at("low"));
    assert_eq!(deps[low], "root");
    assert_eq!(deps[know], "parataxis", "deps: {deps:?}");
    assert_eq!(know as i32 + set.0[know].head, low as i32);
    assert_eq!(deps[fee], "nsubj", "deps: {deps:?}");
    assert_eq!(fee as i32 + set.0[fee].head, low as i32);
}

#[test]
fn matrixless_fronted_clause_keeps_subordinate_root() {
    // Must-NOT-fire: with no well-formed matrix predicate ahead (stay
    // tags NOUN — the verb-detection gap), skipping the subordinate verb
    // would orphan the only predicate — rains stays root exactly as
    // before.
    let (_doc, set) = parse("If it rains, stay home.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("rains")], "root");
}

#[test]
fn verbless_matrix_fronted_clause_keeps_subordinate_root() {
    // Must-NOT-fire: "ends" tags NOUN, so no matrix predicate exists
    // ahead — knows stays root exactly as before.
    let (_doc, set) = parse("As everybody knows, lunch ends early.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("knows")], "root");
}

#[test]
fn root_governed_bare_complement_attaches_ccomp() {
    // Refs (UD imperative-03): permissive "let" governs its bare
    // infinitive (go → ccomp → Let). Gated on the matrix verb holding
    // the crown with a bare (marker/comma-free) verb after it — relcl
    // matrices ("called left": called never roots) and marked clauses
    // never match. The object nominals (dead/bury) keep their current
    // tags — lexicon gaps, deliberately unasserted beyond stability.
    let (_doc, set) = parse("Let the dead go bury their dead.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (let_, go, bury) = (at("Let"), at("go"), at("bury"));
    assert_eq!(deps[go], "ccomp", "deps: {deps:?}");
    assert_eq!(go as i32 + set.0[go].head, let_ as i32);
    assert_eq!(deps[bury], "dobj", "deps: {deps:?}");
}

#[test]
fn relcl_matrix_verb_keeps_crown() {
    // Must-NOT-fire: "called" never holds the crown (relcl skip), so the
    // bare-complement gate — which needs a root matrix — cannot fire;
    // "left" stays root exactly as before.
    let (_doc, set) = parse("The man who called left.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("left")], "root");
}

#[test]
fn temporal_today_after_verb_is_advmod() {
    // Refs (UD imperative-05, contraction-02): temporal "today" (NOUN
    // per the frozen pin) heads an adverbial modifier of its verb
    // (today → advmod → Send/go), not a second direct object. Gated on
    // the word itself — the dobj arm outbids everything below 100, so
    // this ranks just above it.
    let (_doc, set) = parse("Send the invoice today.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (send, today) = (at("Send"), at("today"));
    assert_eq!(deps[today], "advmod", "deps: {deps:?}");
    assert_eq!(today as i32 + set.0[today].head, send as i32);
}

#[test]
fn temporal_today_after_modal_is_advmod() {
    // Same adverbial reading under a modal host (contraction-02).
    let (_doc, set) = parse("I can't go today.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (go, today) = (at("go"), at("today"));
    assert_eq!(deps[today], "advmod", "deps: {deps:?}");
    assert_eq!(today as i32 + set.0[today].head, go as i32);
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

/// End-to-end Track B probe: deterministic confidence + frame ambiguity for
/// one input text, through the public seams only (no synthetic margins).
fn ambiguity_of(
    text: &str,
) -> (
    spacy_rs::ParseConfidence,
    spacy_rs::FrameAnalysis,
    Vec<spacy_rs::FrameKey>,
) {
    let vocab = en_vocab();
    let tokenizer = spacy_rs::lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    let mut doc = tokenizer.tokenize(text).expect("tokenize");
    let annotator = ArcEagerAnnotator::en_default(vocab.clone());
    let (result, conf) = annotator.annotate_with_confidence(&doc).expect("parse");
    let margins = result.oracle_margins.clone().unwrap_or_default();
    spacy_rs::llm::attach(&mut doc, &result.records).expect("attach");
    spacy_rs::Sentencizer::new().process(&mut doc);
    let store: std::sync::Arc<spacy_rs::InMemoryConceptStore> =
        std::sync::Arc::new(spacy_rs::InMemoryConceptStore::new());
    let resolver = std::sync::Arc::new(spacy_rs::InterlinguaResolver::new(
        store.clone() as std::sync::Arc<dyn spacy_rs::ConceptStore>,
        std::sync::Arc::clone(vocab.strings()),
    ));
    let ex = spacy_rs::FrameExtractor::new(
        resolver,
        store.clone() as std::sync::Arc<dyn spacy_rs::ConceptStore>,
    );
    let analysis = ex.extract(&doc, Some(&margins));
    let keys = ex.keys(&doc, &analysis);
    (conf, analysis, keys)
}

#[test]
fn verbless_concessive_emits_attachment_tie() {
    // Track B (positive): "Although tired" has no finite verb in the child
    // span, so the marker's attachment is underdetermined. The oracle must
    // record a near-tie (→ RefineReason::Confidence(Ties)) and the frame
    // stage must emit AttachmentNearTie with a provisional key — never a
    // confident silent misparse. Heads/labels are unchanged (Shift wins
    // ties by stable order); only the margin drops.
    let (conf, analysis, keys) = ambiguity_of("Although tired, she kept pace.");
    assert!(
        conf.oracle_tie_count >= 1,
        "verbless concessive must tie: {conf:?}"
    );
    assert!(
        conf.oracle_margins
            .iter()
            .any(|m| m.abs() <= spacy_rs::TIE_MARGIN_EPSILON),
        "near-zero margin expected: {conf:?}"
    );
    assert!(
        analysis
            .ambiguities
            .iter()
            .any(|a| a.kind == spacy_rs::AmbiguityKind::AttachmentNearTie),
        "AttachmentNearTie expected: {analysis:?}"
    );
    assert!(
        keys.iter().any(|k| k.provisional),
        "ambiguous frame mints a provisional key"
    );
    let (reason, should) = refine_of("Although tired, she kept pace.");
    assert!(
        matches!(reason, spacy_rs::RefineReason::Confidence(_)),
        "confidence-axis refine must fire, got {reason:?}"
    );
    assert!(should, "should_refine must be true for {reason:?}");
    // Parse-stability pin: the tie must not re-head anything.
    let (_doc, set) = parse("Although tired, she kept pace.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("Although")], "dep", "deps: {deps:?}");
    assert_eq!(deps[at("tired")], "nsubj", "deps: {deps:?}");
}

#[test]
fn clean_subordinate_emits_no_tie() {
    // Track B (must-NOT-fire control): an overt-subject subordinate clause
    // ("because it snowed") is fully determined — no tie, no ambiguity
    // entry, permanent key.
    let (conf, analysis, keys) = ambiguity_of("We stayed because it snowed.");
    assert_eq!(conf.oracle_tie_count, 0, "clean clause must not tie: {conf:?}");
    assert!(
        analysis.ambiguities.is_empty(),
        "no ambiguity on a clean clause: {analysis:?}"
    );
    assert!(
        keys.iter().all(|k| !k.provisional),
        "clean frames mint permanent keys"
    );
}

/// End-to-end Track B probe of the refine seam: the confidence-axis
/// `RefineReason` for one input text, through the public seams only.
///
/// The policy is `OnUncertain` with the task-value flags OFF: the hermetic
/// helper wires an empty concept store, which would trip
/// `UnresolvedPropn` for every sentence (no registered senses) — that axis
/// needs a populated store and is covered by `refine_calibration`. Track B
/// isolates the confidence axis (`ParseConfidence` → `ConfidenceReason`),
/// which is fully determined by the deterministic parse.
fn refine_of(text: &str) -> (spacy_rs::RefineReason, bool) {
    let vocab = en_vocab();
    let tokenizer = spacy_rs::lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    let mut doc = tokenizer.tokenize(text).expect("tokenize");
    let annotator = ArcEagerAnnotator::en_default(vocab.clone());
    let (result, _conf) = annotator.annotate_with_confidence(&doc).expect("parse");
    spacy_rs::llm::attach(&mut doc, &result.records).expect("attach");
    spacy_rs::Sentencizer::new().process(&mut doc);
    let store: std::sync::Arc<spacy_rs::InMemoryConceptStore> =
        std::sync::Arc::new(spacy_rs::InMemoryConceptStore::new());
    let resolver = spacy_rs::InterlinguaResolver::new(
        store.clone() as std::sync::Arc<dyn spacy_rs::ConceptStore>,
        std::sync::Arc::clone(vocab.strings()),
    );
    resolver.resolve_doc(&mut doc, result.token_confidence());
    let (routing, signal) = spacy_rs::extract_routing_signals(&doc)
        .into_iter()
        .next()
        .map(|s| {
            let inter = s.interlingua.clone().unwrap_or(spacy_rs::InterlinguaSignal {
                predicate_id: None,
                subject_id: None,
                direct_object_id: None,
                indirect_object_id: None,
                concept_ids: Vec::new(),
                token_ids: Vec::new(),
                confidence: None,
            });
            (s, inter)
        })
        .expect("one routing signal");
    let policy = spacy_rs::RefinePolicy {
        mode: spacy_rs::RefineMode::OnUncertain,
        refine_on_unresolved_critical_role: false,
        refine_on_unresolved_propn: false,
        refine_on_collision_note: false,
        ..Default::default()
    };
    let reason = spacy_rs::refine_reason(&result, &signal, &routing, policy);
    let should = spacy_rs::should_refine(&result, &signal, &routing, policy);
    // The bool and the reason agree by construction (`should == (reason !=
    // NoTrigger)`); pin the agreement so the two seams cannot drift apart.
    assert_eq!(
        should,
        reason != spacy_rs::RefineReason::NoTrigger,
        "should_refine must agree with refine_reason for {text:?}"
    );
    (reason, should)
}

#[test]
fn verbless_adp_concessive_emits_attachment_tie() {
    // Track B (positive): "Although in pain" is a verbless concessive whose
    // marker faces an ADP — no finite verb in the child span, so the
    // marker's attachment is underdetermined. The oracle must record a
    // near-tie and the frame stage an AttachmentNearTie with a provisional
    // key — never a confident silent misparse. The confidence-axis refine
    // reason must fire (Ties, or RoleCoverage while the residual misparse
    // leaves the argument slots empty).
    let (conf, analysis, keys) = ambiguity_of("Although in pain, he smiled.");
    assert!(
        conf.oracle_tie_count >= 1,
        "verbless ADP concessive must tie: {conf:?}"
    );
    assert!(
        conf.oracle_margins
            .iter()
            .any(|m| m.abs() <= spacy_rs::TIE_MARGIN_EPSILON),
        "near-zero margin expected: {conf:?}"
    );
    assert!(
        analysis
            .ambiguities
            .iter()
            .any(|a| a.kind == spacy_rs::AmbiguityKind::AttachmentNearTie),
        "AttachmentNearTie expected: {analysis:?}"
    );
    assert!(
        keys.iter().any(|k| k.provisional),
        "ambiguous frame mints a provisional key"
    );
    let (reason, should) = refine_of("Although in pain, he smiled.");
    assert!(
        matches!(reason, spacy_rs::RefineReason::Confidence(_)),
        "confidence-axis refine must fire, got {reason:?}"
    );
    assert!(should, "should_refine must be true for {reason:?}");
    // Parse-stability pin: the tie must not re-head anything. (`he →
    // pobj` is the Track A residual the flag points at — flagged, not
    // silently confident.)
    let (_doc, set) = parse("Although in pain, he smiled.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (although, in_, pain, smiled) =
        (at("Although"), at("in"), at("pain"), at("smiled"));
    assert_eq!(deps[although], "dep", "deps: {deps:?}");
    assert_eq!(although as i32 + set.0[although].head, smiled as i32);
    assert_eq!(deps[in_], "dep", "deps: {deps:?}");
    assert_eq!(in_ as i32 + set.0[in_].head, smiled as i32);
    assert_eq!(deps[pain], "pobj", "deps: {deps:?}");
    assert_eq!(pain as i32 + set.0[pain].head, in_ as i32);
    assert_eq!(deps[smiled], "root");
}

#[test]
fn unanchored_double_embedding_emits_attachment_tie() {
    // Track B (positive): "I think she knows he left" doubly embeds — the
    // second-level verb (left) has no subordinator, no boundary
    // punctuation, and a non-root matrix (knows), so no licensed arm owns
    // the pair and it strands in repair-dep. The oracle must record a
    // near-tie (→ RefineReason::Confidence(Ties)) and the frame stage an
    // AttachmentNearTie with a provisional key — never a confident silent
    // strand. Heads/labels are unchanged (Shift wins ties by stable
    // order); only the margin drops.
    let (conf, analysis, keys) = ambiguity_of("I think she knows he left.");
    assert!(
        conf.oracle_tie_count >= 1,
        "unanchored verb–verb embedding must tie: {conf:?}"
    );
    assert!(
        conf.oracle_margins
            .iter()
            .any(|m| m.abs() <= spacy_rs::TIE_MARGIN_EPSILON),
        "near-zero margin expected: {conf:?}"
    );
    assert!(
        analysis
            .ambiguities
            .iter()
            .any(|a| a.kind == spacy_rs::AmbiguityKind::AttachmentNearTie),
        "AttachmentNearTie expected: {analysis:?}"
    );
    assert!(
        keys.iter().any(|k| k.provisional),
        "ambiguous frame mints a provisional key"
    );
    let (reason, should) = refine_of("I think she knows he left.");
    assert_eq!(
        reason,
        spacy_rs::RefineReason::Confidence(spacy_rs::ConfidenceReason::Ties),
        "double embedding must refine on Ties, got {reason:?}"
    );
    assert!(should, "should_refine must be true for {reason:?}");
    // Parse-stability pin: the tie must not re-head anything (knows keeps
    // its licensed ccomp; left keeps its repair-dep strand, flagged).
    let (_doc, set) = parse("I think she knows he left.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (think, knows, left) = (at("think"), at("knows"), at("left"));
    assert_eq!(deps[think], "root");
    assert_eq!(deps[knows], "ccomp", "deps: {deps:?}");
    assert_eq!(knows as i32 + set.0[knows].head, think as i32);
    assert_eq!(deps[left], "dep", "deps: {deps:?}");
}

#[test]
fn single_embedding_emits_no_tie() {
    // Track B (must-NOT-fire control): a singly-embedded bare complement
    // ("She thinks he left") is fully determined — the root-governed
    // ccomp arm owns the pair, so no tie, no ambiguity entry, permanent
    // keys, and no confidence-axis refine.
    let (conf, analysis, keys) = ambiguity_of("She thinks he left.");
    assert_eq!(conf.oracle_tie_count, 0, "licensed complement must not tie: {conf:?}");
    assert!(
        analysis.ambiguities.is_empty(),
        "no ambiguity on a licensed complement: {analysis:?}"
    );
    assert!(
        keys.iter().all(|k| !k.provisional),
        "clean frames mint permanent keys"
    );
    let (reason, should) = refine_of("She thinks he left.");
    assert_eq!(
        reason,
        spacy_rs::RefineReason::NoTrigger,
        "licensed complement must not refine, got {reason:?}"
    );
    assert!(!should, "should_refine must be false");
}

#[test]
fn clean_subverbobj_emits_no_refine() {
    // Track B (must-NOT-fire control): an unambiguous SVO sentence keeps
    // high ParseConfidence — no tie, no ambiguity, no refine.
    let (conf, analysis, _) = ambiguity_of("Dogs bark loudly.");
    assert_eq!(conf.oracle_tie_count, 0, "clean SVO must not tie: {conf:?}");
    assert!(
        analysis.ambiguities.is_empty(),
        "no ambiguity on clean SVO: {analysis:?}"
    );
    let (reason, should) = refine_of("Dogs bark loudly.");
    assert_eq!(
        reason,
        spacy_rs::RefineReason::NoTrigger,
        "clean SVO must not refine, got {reason:?}"
    );
    assert!(!should, "should_refine must be false");
}

#[test]
fn clean_copular_emits_no_refine() {
    // Track B (must-NOT-fire control): an unambiguous copular sentence
    // keeps high ParseConfidence — no tie, no ambiguity, no refine.
    let (conf, analysis, _) = ambiguity_of("Your fee is low.");
    assert_eq!(conf.oracle_tie_count, 0, "clean copular must not tie: {conf:?}");
    assert!(
        analysis.ambiguities.is_empty(),
        "no ambiguity on clean copular: {analysis:?}"
    );
    let (reason, should) = refine_of("Your fee is low.");
    assert_eq!(
        reason,
        spacy_rs::RefineReason::NoTrigger,
        "clean copular must not refine, got {reason:?}"
    );
    assert!(!should, "should_refine must be false");
}

#[test]
fn unsubordinated_single_verb_attaches_no_subordinate() {
    // Track A must-NOT-fire control for the advcl/ccomp Verb–Verb arm: an
    // un-subordinated matrix verb ("She sings loudly") has no second verb
    // at all, so neither ccomp nor advcl (nor parataxis) may fire — and
    // with no uncertainty the confidence axis stays quiet.
    let (_doc, set) = parse("She sings loudly.");
    let deps = deps(&set);
    assert!(
        deps.iter().all(|d| d != "ccomp" && d != "advcl" && d != "parataxis"),
        "no subordinate attachment without a subordinate clause: {deps:?}"
    );
    let (conf, _, _) = ambiguity_of("She sings loudly.");
    assert_eq!(conf.oracle_tie_count, 0, "single verb must not tie: {conf:?}");
    let (reason, should) = refine_of("She sings loudly.");
    assert_eq!(
        reason,
        spacy_rs::RefineReason::NoTrigger,
        "unsubordinated matrix must not refine, got {reason:?}"
    );
    assert!(!should, "should_refine must be false");
}
#[test]
fn imperative_pronoun_object_upgrades_initial_verb() {
    // Refs (UD command-04): Remind → verb/root, me → dobj → Remind. The
    // directive pass only covers DET-led objects, so pronoun objects strand
    // the initial verb as NOUN. The verbless-clause gate plus the pronoun
    // frame identifies them; the existing dobj arm lands the object, and
    // verb lemmatization lowercases the lemma. (UD pins iobj for me; the
    // oracle has no iobj arm, so dobj is the honest head with a label
    // residual.)
    let (_doc, set) = parse("Remind me at noon.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let lem = lemmas(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (remind, me) = (at("Remind"), at("me"));
    assert_eq!(pos[remind], "verb", "pos: {pos:?}");
    assert_eq!(deps[remind], "root");
    assert_eq!(lem[remind], "remind", "lemmas: {lem:?}");
    assert_eq!(deps[me], "dobj", "deps: {deps:?}");
    assert_eq!(me as i32 + set.0[me].head, remind as i32);
}

#[test]
fn imperative_pp_complement_upgrades_initial_verb() {
    // Refs (UD command-02): Translate → verb/root, hello → dobj →
    // Translate. A bare nominal object plus a prepositional adjunct later
    // in the clause is the transitive frame without a determiner. Nominal
    // pairs with no adjunct (`Dogs chase red cars`) keep the compound
    // dynamics — pinned by the control below.
    let (_doc, set) = parse("Translate hello to French.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let lem = lemmas(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (translate, hello) = (at("Translate"), at("hello"));
    assert_eq!(pos[translate], "verb", "pos: {pos:?}");
    assert_eq!(deps[translate], "root");
    assert_eq!(lem[translate], "translate", "lemmas: {lem:?}");
    assert_eq!(deps[hello], "dobj", "deps: {deps:?}");
    assert_eq!(hello as i32 + set.0[hello].head, translate as i32);
}

#[test]
fn imperative_possessive_object_upgrades_initial_verb() {
    // Refs (UD command-06): Explain → verb/root, theorem → dobj → Explain.
    // A nominal object plus a possessive 's is the proper-name object
    // frame. Bell/'s keep their current tags (the PROPN/case gaps are
    // their own iterations, deliberately unasserted beyond stability).
    let (_doc, set) = parse("Explain Bell's theorem.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let lem = lemmas(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (explain, theorem) = (at("Explain"), at("theorem"));
    assert_eq!(pos[explain], "verb", "pos: {pos:?}");
    assert_eq!(deps[explain], "root");
    assert_eq!(lem[explain], "explain", "lemmas: {lem:?}");
    assert_eq!(deps[theorem], "dobj", "deps: {deps:?}");
    assert_eq!(theorem as i32 + set.0[theorem].head, explain as i32);
}

#[test]
fn imperative_upgrade_needs_verbless_clause() {
    // Must-NOT-fire: the upgrade needs a verbless clause — "Anna" is a
    // subject (finished is VERB via the bare-ed pass, sequenced before),
    // and "Dogs" heads a nominal pair with no adjunct (the compound
    // dynamics own it). Both stay nominal.
    let (_doc, set) = parse("Anna finished her lunch.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(pos[at("Anna")], "verb", "pos: {pos:?}");
    assert_eq!(pos[at("finished")], "verb", "pos: {pos:?}");
    let (_doc, set) = parse("Dogs chase red cars.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("Dogs")], "noun", "pos: {pos:?}");
    assert_eq!(pos[at("chase")], "noun", "pos: {pos:?}");
}

#[test]
fn imperative_upgrade_skips_relative_pronouns() {
    // Must-NOT-fire: "who" heads a relative clause (`People who wait`),
    // so the initial noun is a subject, not an imperative verb — the
    // pronoun frame is object-forms only (the mirror of the
    // pronoun-subject pass, which upgrades after exactly the excluded
    // nominative forms).
    let (_doc, set) = parse("People who wait succeed.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_ne!(pos[at("People")], "verb", "pos: {pos:?}");
    assert_eq!(pos[at("wait")], "verb", "pos: {pos:?}");
    assert_eq!(deps[at("wait")], "relcl", "deps: {deps:?}");
    assert_eq!(deps[at("succeed")], "root");
}

#[test]
fn bare_negator_not_is_particle() {
    // Refs (UD negation-02): the bare negator `not` is categorically PART
    // (UD) — it has no nominal or verbal reading, so it tags in `infer_pos`
    // beside `n't` with no collision audit. Before, it fell to NOUN and the
    // bare-infinitive upgrade rooted it (`not` stole root, stranding `call`
    // in ccomp). Now the true verb crowns and the hosted-infinitive logic
    // is untouched.
    let (_doc, set) = parse("She did not call.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let lem = lemmas(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (she, did, not, call) = (at("She"), at("did"), at("not"), at("call"));
    assert_eq!(pos[not], "part", "pos: {pos:?}");
    assert_eq!(lem[not], "not", "lemmas: {lem:?}");
    assert_eq!(deps[not], "neg", "deps: {deps:?}");
    assert_eq!(not as i32 + set.0[not].head, call as i32);
    assert_eq!(pos[call], "verb", "pos: {pos:?}");
    assert_eq!(deps[call], "root");
    assert_eq!(deps[did], "aux", "deps: {deps:?}");
    assert_eq!(did as i32 + set.0[did].head, call as i32);
    assert_eq!(deps[she], "nsubj", "deps: {deps:?}");
    assert_eq!(she as i32 + set.0[she].head, call as i32);
}

#[test]
fn negator_never_steals_root() {
    // Must-NOT-fire: `not` never tags VERB, in imperatives (`Do not enter`
    // — enter crowns) or copular clauses (`This is not correct` — the
    // predicate adjective crowns with be-copula, the n't-shape it already
    // took in `isn't ready`).
    let (_doc, set) = parse("Do not enter.");
    let pos1 = pos_of(&set);
    let deps1 = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos1[at("not")], "part", "pos: {pos1:?}");
    assert_eq!(deps1[at("not")], "neg", "deps: {deps1:?}");
    assert_eq!(deps1[at("enter")], "root");
    let (_doc, set) = parse("This is not correct.");
    let pos2 = pos_of(&set);
    let deps2 = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos2[at("not")], "part", "pos: {pos2:?}");
    assert_eq!(deps2[at("correct")], "root");
    assert_eq!(deps2[at("is")], "cop", "deps: {deps2:?}");
    assert_eq!(deps2[at("not")], "neg", "deps: {deps2:?}");
}

#[test]
fn demonstrative_object_upgrades_pair() {
    // Refs (UD command-target-01): a demonstrative with no nominal head is
    // the object itself (UD: PRON), and the retag feeds the imperative
    // pronoun frame — Translate crowns via the same dynamics as Remind me.
    let (_doc, set) = parse("Translate this to French.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let lem = lemmas(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (translate, this) = (at("Translate"), at("this"));
    assert_eq!(pos[this], "pron", "pos: {pos:?}");
    assert_eq!(pos[translate], "verb", "pos: {pos:?}");
    assert_eq!(deps[translate], "root");
    assert_eq!(lem[translate], "translate", "lemmas: {lem:?}");
    assert_eq!(deps[this], "dobj", "deps: {deps:?}");
    assert_eq!(this as i32 + set.0[this].head, translate as i32);
}

#[test]
fn demonstrative_final_this_is_pronoun() {
    // Refs (UD punctuation-edge-05): sentence-final `this` has no nominal
    // to determine — PRON/dobj, and the imperative crowns (the punctuation
    // that broke the DET+NOUN frame no longer matters).
    let (_doc, set) = parse("Summarize this?");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (summarize, this) = (at("Summarize"), at("this"));
    assert_eq!(pos[this], "pron", "pos: {pos:?}");
    assert_eq!(pos[summarize], "verb", "pos: {pos:?}");
    assert_eq!(deps[summarize], "root");
    assert_eq!(deps[this], "dobj", "deps: {deps:?}");
    assert_eq!(this as i32 + set.0[this].head, summarize as i32);
}

#[test]
fn demonstrative_before_noun_stays_determiner() {
    // Must-NOT-fire: a demonstrative heading a nominal (`this equation`)
    // keeps the determiner reading — the upgrade needs an ADP, boundary,
    // or nothing after it. And `that` never retags: it relativizes (`Dogs
    // that bark bite`), owned by the that-relative pass.
    let (_doc, set) = parse("Solve this equation.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("this")], "det", "pos: {pos:?}");
    let (_doc, set) = parse("Dogs that bark bite.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    // `that` never retags: it relativizes, owned by the that-relative
    // pass. (`Dogs` crowns via the pre-existing directive DET+NOUN frame —
    // a known over-fire, out of scope here; this test owns only the
    // demonstrative line.)
    assert_eq!(pos[at("that")], "det", "pos: {pos:?}");
}

#[test]
fn epistemic_linking_frame_upgrades_pair() {
    // Refs (UD copular-edge-06): epistemic linkers (feel/seem/remain/
    // appear, base + -s) take the sensory two-step — the linker crowns as
    // VERB and the nominal complement lands as predicate ADJ/acomp.
    let (_doc, set) = parse("Something feels wrong.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let lem = lemmas(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (feels, wrong) = (at("feels"), at("wrong"));
    assert_eq!(pos[feels], "verb", "pos: {pos:?}");
    assert_eq!(deps[feels], "root");
    assert_eq!(lem[feels], "feel", "lemmas: {lem:?}");
    assert_eq!(pos[wrong], "adj", "pos: {pos:?}");
    assert_eq!(deps[wrong], "acomp", "deps: {deps:?}");
    assert_eq!(wrong as i32 + set.0[wrong].head, feels as i32);
}

#[test]
fn epistemic_linking_before_verbal_complement() {
    // Refs (UD copular-edge-05): `remains` fires before a VERB-tagged
    // complement (`uncertain`, crowned by the initial-noun rule) — and
    // because the linking pass runs first, the complement is still NOUN
    // here, so the ADJ step lands it as acomp instead of tying. Pass
    // ordering is load-bearing; pin it.
    let (_doc, set) = parse("This remains uncertain.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (remains, uncertain) = (at("remains"), at("uncertain"));
    assert_eq!(pos[remains], "verb", "pos: {pos:?}");
    assert_eq!(deps[remains], "root");
    assert_eq!(pos[uncertain], "adj", "pos: {pos:?}");
    assert_eq!(deps[uncertain], "acomp", "deps: {deps:?}");
}

#[test]
fn epistemic_linking_skips_nominal_readings() {
    // Must-NOT-fire: plural-noun `remains` (AUX-next) and noun `feel`
    // (ADP-next) keep their nominal readings — the verb step needs a
    // nominal, adjectival, or verbal complement, never a determiner-led,
    // auxiliary-led, or prepositional one.
    let (_doc, set) = parse("The remains were buried.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("remains")], "noun", "pos: {pos:?}");
    let (_doc, set) = parse("She has a feel for music.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("feel")], "noun", "pos: {pos:?}");
}

#[test]
fn ditransitive_recipient_is_iobj() {
    // Refs (UD dative-*): a bare pronoun between a verb and a DET-led
    // nominal is the indirect object — the nominal takes dobj below. The
    // 105 weight outbids dobj (100) inside the gated frame; routing and
    // yago already consume iobj downstream, so only the oracle had to
    // learn to emit it. (imperative-01's frozen ref pinned me→dobj,
    // contradicting the 8 dative refs and UD — corrected to iobj with
    // this arm.)
    let (_doc, set) = parse("Give me the report.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (give, me, report) = (at("Give"), at("me"), at("report"));
    assert_eq!(deps[me], "iobj", "deps: {deps:?}");
    assert_eq!(me as i32 + set.0[me].head, give as i32);
    assert_eq!(deps[report], "dobj", "deps: {deps:?}");
    assert_eq!(report as i32 + set.0[report].head, give as i32);
}

#[test]
fn single_object_pronoun_stays_dobj() {
    // Must-NOT-fire: without a DET-led nominal after it, the pronoun is
    // the direct object — before a preposition (Remind me at noon), a
    // bare nominal (Help them win), or an adverb (Call me later).
    for (text, verb, pron) in [
        ("Remind me at noon.", "Remind", "me"),
        ("Help them win.", "Help", "them"),
        ("Call me later.", "Call", "me"),
    ] {
        let (_doc, set) = parse(text);
        let deps = deps(&set);
        let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
        assert_eq!(deps[at(pron)], "dobj", "{text} deps: {deps:?}");
        assert_eq!(
            at(pron) as i32 + set.0[at(pron)].head,
            at(verb) as i32,
            "{text} deps: {deps:?}"
        );
    }
}

#[test]
fn discourse_initial_imperative_crowns_verb() {
    // Refs (UD imperative-do-04): a discourse marker before a bare verb
    // strands both — the marker roots as NOUN and the verb compounds into
    // it. The frame upgrades the verb (standard dobj dynamics land `date`)
    // and retags the marker (`please` → INTJ, inert: nothing else reads
    // INTJ, so it strands honestly via repair-dep instead of corrupting
    // the clause).
    let (_doc, set) = parse("Please confirm the date.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (please, confirm, date) = (at("Please"), at("confirm"), at("date"));
    assert_eq!(pos[please], "intj", "pos: {pos:?}");
    assert_eq!(pos[confirm], "verb", "pos: {pos:?}");
    assert_eq!(deps[confirm], "root");
    assert_eq!(deps[date], "dobj", "deps: {deps:?}");
    assert_eq!(date as i32 + set.0[date].head, confirm as i32);
}

#[test]
fn discourse_initial_never_takes_advmod() {
    // Refs (UD negation-04): `Never`/`Always`/`Just`/`Kindly` retag to ADV
    // and take the existing (Adv, Verb) advmod arm — `Never send that
    // email` goes fully correct (marker, crown, determiner, object).
    let (_doc, set) = parse("Never send that email.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (never, send, email) = (at("Never"), at("send"), at("email"));
    assert_eq!(pos[never], "adv", "pos: {pos:?}");
    assert_eq!(deps[never], "advmod", "deps: {deps:?}");
    assert_eq!(never as i32 + set.0[never].head, send as i32);
    assert_eq!(pos[send], "verb", "pos: {pos:?}");
    assert_eq!(deps[send], "root");
    assert_eq!(deps[email], "dobj", "deps: {deps:?}");
}

#[test]
fn discourse_frame_needs_verbal_complement() {
    // Must-NOT-fire: nominal thirds (`Just good friends`) are fragments,
    // not imperatives — the marker and the second noun both stay nominal.
    // Mid-sentence markers (`... please.`) never match either (initial
    // frame only).
    let (_doc, set) = parse("Just good friends.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("Just")], "noun", "pos: {pos:?}");
    assert_eq!(pos[at("good")], "noun", "pos: {pos:?}");
    let (_doc, set) = parse("Summarize this... please.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("please")], "noun", "pos: {pos:?}");
}

#[test]
fn copular_predicate_nominal_full_frame() {
    // Refs (UD copular-03): predicate nominals (`is a doctor`) strand
    // completely today — the subject crowns, be dangles, the predicate
    // strands. The package (pick_root crowns the last nominal of the
    // be+DET span; cop-Left attaches be; the gated nsubj lands the
    // subject) restores the full frame. Order matters throughout: det
    // fires before cop attaches, cop before the subject meets.
    let (_doc, set) = parse("She is a doctor.");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (she, is, doctor) = (at("She"), at("is"), at("doctor"));
    assert_eq!(pos[doctor], "noun", "pos: {pos:?}");
    assert_eq!(deps[doctor], "root");
    assert_eq!(deps[is], "cop", "deps: {deps:?}");
    assert_eq!(is as i32 + set.0[is].head, doctor as i32);
    assert_eq!(deps[she], "nsubj", "deps: {deps:?}");
    assert_eq!(she as i32 + set.0[she].head, doctor as i32);
}

#[test]
fn wh_copular_predicate_nominal_full_frame() {
    // Refs (UD wh-copula-02): same package through an interrogative
    // subject — Who crowns nothing, the predicate does.
    let (_doc, set) = parse("Who is the president?");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (who, is, president) = (at("Who"), at("is"), at("president"));
    assert_eq!(deps[president], "root");
    assert_eq!(deps[is], "cop", "deps: {deps:?}");
    assert_eq!(is as i32 + set.0[is].head, president as i32);
    assert_eq!(deps[who], "nsubj", "deps: {deps:?}");
    assert_eq!(who as i32 + set.0[who].head, president as i32);
    assert_eq!(pos[who], "pron", "pos: {pos:?}");
}

#[test]
fn existential_there_is_expl() {
    // Refs (UD copular-edge-01): existential `there` is an expletive, not
    // a subject — word-gated expl arm on the copular frame (the nsubj
    // gate excludes there/here explicitly).
    let (_doc, set) = parse("There is a problem.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (there, problem) = (at("There"), at("problem"));
    assert_eq!(deps[problem], "root");
    assert_eq!(deps[there], "expl", "deps: {deps:?}");
    assert_eq!(there as i32 + set.0[there].head, problem as i32);
}

#[test]
fn numeric_modifier_is_nummod() {
    // Refs (UD topic-bare-01): bare fragments strand numbers in
    // repair-dep; a numeral after a nominal head is its numeric modifier.
    let (_doc, set) = parse("invoice 1001");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (invoice, num) = (at("invoice"), at("1001"));
    assert_eq!(pos[num], "num", "pos: {pos:?}");
    assert_eq!(deps[invoice], "root");
    assert_eq!(deps[num], "nummod", "deps: {deps:?}");
    assert_eq!(num as i32 + set.0[num].head, invoice as i32);
}

#[test]
fn bare_nominal_chain_crowns_last() {
    // Refs (UD topic-bare-03/05/07): English compounds are head-final —
    // with no verb, aux, or adjective anywhere, a 3+ word bare-nominal
    // chain crowns its last nominal (checker, medication, logs) and the
    // earlier nouns chain as compound via the existing Left arm.
    // Determined frames: no oracle tie (Track B stays quiet).
    for (text, heads) in [
        ("Rust borrow checker", ("Rust", "borrow", "checker")),
        ("blood pressure medication", ("blood", "pressure", "medication")),
        ("server error logs", ("server", "error", "logs")),
    ] {
        let (_doc, set) = parse(text);
        let deps = deps(&set);
        let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
        let (first, mid, last) = (at(heads.0), at(heads.1), at(heads.2));
        assert_eq!(deps[last], "root", "{text}: deps: {deps:?}");
        assert_eq!(set.0[last].head, 0, "{text}: root head");
        assert_eq!(deps[mid], "compound", "{text}: deps: {deps:?}");
        assert_eq!(mid as i32 + set.0[mid].head, last as i32, "{text}");
        assert_eq!(deps[first], "compound", "{text}: deps: {deps:?}");
        let (conf, _, _) = ambiguity_of(text);
        assert_eq!(conf.oracle_tie_count, 0, "{text}: determined frame must not tie: {conf:?}");
    }
}

#[test]
fn prenominal_designator_heads_right() {
    // Refs (UD topic-bare-08): a designator number with a nominal after it
    // (flight 204 status) belongs to the FOLLOWING head — the Right-nummod
    // withhold lets it shift, the Left-nummod arm lands it on status, and
    // the head-final rung crowns status with flight compounding onto it.
    let (_doc, set) = parse("flight 204 status");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    let (flight, num, status) = (at("flight"), at("204"), at("status"));
    assert_eq!(pos[num], "num", "pos: {pos:?}");
    assert_eq!(deps[status], "root", "deps: {deps:?}");
    assert_eq!(deps[num], "nummod", "deps: {deps:?}");
    assert_eq!(num as i32 + set.0[num].head, status as i32);
    assert_eq!(deps[flight], "compound", "deps: {deps:?}");
    assert_eq!(flight as i32 + set.0[flight].head, status as i32);
}

#[test]
fn number_final_frame_still_heads_left() {
    // Must-NOT-fire: the pre-nominal withhold needs a nominal AFTER the
    // number — number-final `invoice 1001` attaches leftward exactly as
    // before (see numeric_modifier_is_nummod for the attachment pins).
    let (_doc, set) = parse("invoice 1001");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("invoice")], "root", "deps: {deps:?}");
    assert_eq!(deps[at("1001")], "nummod", "deps: {deps:?}");
}

#[test]
fn two_word_fragment_keeps_first_crown_and_tie() {
    // Must-NOT-fire: a two-word bare-nominal fragment (`Define
    // photosynthesis`) keeps the first-crown + Track B tie dynamics — the
    // imperative reading is still live there, so the head-final rung (3+
    // content tokens) and the designator arms never see the pair.
    let (_doc, set) = parse("Define photosynthesis.");
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(deps[at("Define")], "root", "deps: {deps:?}");
    let (conf, _, _) = ambiguity_of("Define photosynthesis.");
    assert!(
        conf.oracle_tie_count >= 1,
        "two-word fragment must still tie: {conf:?}"
    );
}

#[test]
fn nominal_chain_needs_pure_run() {
    // Must-NOT-fire: the head-final rung needs a pure nominal run —
    // determiner-led (`Send her the invoice`), adpositional (`Elaborate on
    // this`), and adverbial (`Study hard, rest well`) frames keep their
    // incumbent first crowns.
    for (text, crown) in [
        ("Send her the invoice.", "Send"),
        ("Elaborate on this.", "Elaborate"),
        ("Study hard, rest well.", "Study"),
    ] {
        let (_doc, set) = parse(text);
        let deps = deps(&set);
        let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
        assert_eq!(deps[at(crown)], "root", "{text}: deps: {deps:?}");
    }
}

#[test]
fn attributive_ly_is_adjective() {
    // Refs (UD topic-bare-04): `-ly` before a nominal head is an
    // attributive adjective (quarterly sales), not an adverbial — the
    // rule only sees NOUNs directly before nominals, so final (`daily`),
    // conjunct (`daily or`), comma-framed (`Sadly,`), and retagged
    // (`Kindly` is ADV by then) shapes never match.
    let (_doc, set) = parse("quarterly sales report");
    let pos = pos_of(&set);
    let deps = deps(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("quarterly")], "adj", "pos: {pos:?}");
    assert_eq!(deps[at("quarterly")], "amod", "deps: {deps:?}");
}

#[test]
fn copular_package_keeps_predicate_adjectives() {
    // Must-NOT-fire: a predicative adjective after the nominal span keeps
    // its crown (`Is the report ready`, `Is the sky blue`) — the
    // pick_root rule requires no ADJ after be, and prepositional spans
    // (`The book is on the table`, ADP after be) never match either.
    // (Where-initial interrogatives crown be instead — see
    // `where_be_question_crowns_aux`; they are deliberately not listed
    // here.)
    for (text, crown) in [
        ("Is the report ready?", "ready"),
        ("The book is on the table.", "book"),
    ] {
        let (_doc, set) = parse(text);
        let deps = deps(&set);
        let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
        assert_eq!(deps[at(crown)], "root", "{text} deps: {deps:?}");
    }
}

#[test]
fn imperative_upgrade_skips_aux_clauses() {
    // Must-NOT-fire: a clause with an AUX (`Big Data is a trend`) is
    // copular, not imperative — the initial nominal stays nominal even
    // though no VERB is present.
    let (_doc, set) = parse("Big Data is a trend.");
    let pos = pos_of(&set);
    let at = |w: &str| set.0.iter().position(|r| r.text == w).expect(w);
    assert_eq!(pos[at("Big")], "noun", "pos: {pos:?}");
}
