use super::*;
use crate::lang;
use crate::validate::AnnotationValidator;
use crate::vocab::Vocab;

fn en_vocab() -> Arc<Vocab> {
    Arc::new(Vocab::new(lang::en::lexicon_config()))
}

fn tokenize(text: &str) -> Doc {
    let vocab = en_vocab();
    lang::en::tokenizer(vocab)
        .expect("tokenizer")
        .tokenize(text)
        .expect("tokenize")
}

fn annotator() -> ArcEagerAnnotator {
    ArcEagerAnnotator::en_default(en_vocab())
}

fn label_hashes(vocab: &Arc<Vocab>) -> DepLabels {
    DepLabels::new(vocab.strings())
}

fn ortho() -> TaggerOrtho<'static> {
    TaggerOrtho::english()
}

// ── 9.1 infer_pos ──────────────────────────────────────────────────────

#[test]
fn infer_pos_categories() {
    let vocab = en_vocab();
    let flags = |s: &str| {
        let mut d = Doc::new(Arc::clone(&vocab));
        d.push_back(s, true).expect("push");
        d.token(0).lexeme.flags
    };
    assert_eq!(infer_pos(flags(".")), Upos::Punct);
    assert_eq!(infer_pos(flags("5")), Upos::Num);
    assert_eq!(infer_pos(flags("is")), Upos::Aux);
    assert_eq!(infer_pos(flags("of")), Upos::Adp);
    assert_eq!(infer_pos(flags("the")), Upos::Det);
    assert_eq!(infer_pos(flags("and")), Upos::Cconj);
    assert_eq!(infer_pos(flags("if")), Upos::Sconj);
    assert_eq!(infer_pos(flags("cat")), Upos::Noun);
    assert_eq!(infer_pos(flags("it")), Upos::Pron);
    // Adjectives are an open class with no lexeme signal — "big" falls to
    // NOUN (the honest heuristic limit; no adjective list, §8.1).
    assert_eq!(infer_pos(flags("big")), Upos::Noun);
}

#[test]
fn infer_pos_allcaps_is_propn_positive() {
    let vocab = en_vocab();
    let flags = |s: &str| {
        let mut d = Doc::new(Arc::clone(&vocab));
        d.push_back(s, true).expect("push");
        d.token(0).lexeme.flags
    };
    assert_eq!(infer_pos(flags("NASA")), Upos::Propn);
    assert_eq!(infer_pos(flags("HTML5")), Upos::Propn);
}

#[test]
fn infer_pos_title_case_is_never_propn_negative() {
    // Sentence-initial common noun / Title Case non-entity → NOT PROPN
    // (is_upper-only, never is_title). "Google"/"Paris" are the documented
    // false-negative class (§8.2) — lexeme-only POS cannot know them.
    for w in ["Dogs", "Big", "Google", "Paris", "The"] {
        let vocab = en_vocab();
        let mut d = Doc::new(Arc::clone(&vocab));
        d.push_back(w, true).expect("push");
        let pos = infer_pos(d.token(0).lexeme.flags);
        assert_ne!(pos, Upos::Propn, "{w} must not be PROPN");
    }
    let vocab = en_vocab();
    let mut d = Doc::new(Arc::clone(&vocab));
    d.push_back("Dogs", true).expect("push");
    assert_eq!(infer_pos(d.token(0).lexeme.flags), Upos::Noun);
}

// ── 9.2 state init / is_final ─────────────────────────────────────────

#[test]
fn state_init_has_unset_heads_and_is_final() {
    let st = ArcEagerState::new(4, 0);
    assert_eq!(st.heads, vec![-1; 4]);
    assert!(st.is_final());
    let labels = label_hashes(&en_vocab());
    let mut st = ArcEagerState::new(4, 0);
    st.reset_for_sentence(0, 4, 1, labels.root);
    assert!(!st.is_final());
    assert_eq!(
        st.buffer.iter().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(st.heads[1], -1, "root head stays unset (never re-attached)");
    assert_eq!(st.labels[1], labels.root);
}

// ── 9.3 candidate_actions ─────────────────────────────────────────────

#[test]
fn candidate_actions_noun_before_verb() {
    let vocab = en_vocab();
    let labels = label_hashes(&vocab);
    let mut st = ArcEagerState::new(2, 0);
    st.stack.push(0); // noun
    st.buffer.extend(1..2); // verb
    let pos = vec![Upos::Noun, Upos::Verb];
    let texts = vec!["dogs".to_string(), "bark".to_string()];
    let flags: Vec<LexemeFlags> = texts
        .iter()
        .map(|t| vocab.lexicon().get_or_create(t).flags)
        .collect();
    let actions = st.candidate_actions(&pos, &texts, &flags, &labels, &ortho());
    assert!(actions.iter().any(|a| a.move_type == ArcEagerMove::Shift));
    assert!(
        actions
            .iter()
            .any(|a| a.move_type == ArcEagerMove::Left && a.label == labels.nsubj)
    );
    assert!(actions.iter().any(|a| a.move_type == ArcEagerMove::Reduce));
}

#[test]
fn candidate_actions_verb_before_noun() {
    let labels = label_hashes(&en_vocab());
    let mut st = ArcEagerState::new(2, 0);
    st.stack.push(0); // verb
    st.buffer.extend(1..2); // noun
    let pos = vec![Upos::Verb, Upos::Noun];
    let texts = vec!["find".to_string(), "milk".to_string()];
    let vocab = en_vocab();
    let flags: Vec<LexemeFlags> = texts
        .iter()
        .map(|t| vocab.lexicon().get_or_create(t).flags)
        .collect();
    let actions = st.candidate_actions(&pos, &texts, &flags, &labels, &ortho());
    assert!(
        actions
            .iter()
            .any(|a| a.move_type == ArcEagerMove::Right && a.label == labels.dobj)
    );
}

#[test]
fn candidate_actions_punct_with_empty_stack() {
    let labels = label_hashes(&en_vocab());
    let mut st = ArcEagerState::new(1, 0);
    st.buffer.extend(0..1);
    let pos = vec![Upos::Punct];
    let texts = vec![".".to_string()];
    let vocab = en_vocab();
    let flags: Vec<LexemeFlags> = texts
        .iter()
        .map(|t| vocab.lexicon().get_or_create(t).flags)
        .collect();
    let actions = st.candidate_actions(&pos, &texts, &flags, &labels, &ortho());
    assert!(
        !actions.iter().any(|a| a.move_type == ArcEagerMove::Right),
        "nothing to attach punctuation to"
    );
    assert!(actions.iter().any(|a| a.move_type == ArcEagerMove::Break));
    assert!(actions.iter().any(|a| a.move_type == ArcEagerMove::Shift));
}

#[test]
fn candidate_actions_drain_on_empty_buffer() {
    let labels = label_hashes(&en_vocab());
    let mut st = ArcEagerState::new(2, 0);
    st.stack.push(1);
    st.buffer.clear();
    let pos = vec![Upos::Noun, Upos::Noun];
    let texts: Vec<String> = vec![];
    let flags: Vec<LexemeFlags> = vec![];
    let actions = st.candidate_actions(&pos, &texts, &flags, &labels, &ortho());
    assert_eq!(
        actions,
        vec![ArcEagerAction {
            move_type: ArcEagerMove::Reduce,
            label: 0,
        }]
    );
}

// ── 9.4 apply per move (absolute heads, F8) ───────────────────────────

#[test]
fn apply_shift_reduce_left_right_break() {
    let labels = label_hashes(&en_vocab());
    let mut st = ArcEagerState::new(4, 0);
    st.buffer.extend(0..4);

    // SHIFT 0.
    st.apply(&ArcEagerAction { move_type: ArcEagerMove::Shift, label: 0 });
    assert_eq!(st.stack, vec![0]);
    assert_eq!(st.buffer.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);

    // SHIFT 1 → stack [0, 1], buffer [2, 3].
    st.apply(&ArcEagerAction { move_type: ArcEagerMove::Shift, label: 0 });
    assert_eq!(st.stack, vec![0, 1]);

    // LEFT(nsubj): stack top 1 depends on buffer head 2 (absolute).
    st.apply(&ArcEagerAction { move_type: ArcEagerMove::Left, label: labels.nsubj });
    assert_eq!(st.heads[1], 2);
    assert_eq!(st.labels[1], labels.nsubj);
    assert!(st.left_children[2].contains(&1));
    assert_eq!(st.stack, vec![0], "LEFT pops the stack top");

    // RIGHT(prep): buffer head 2 depends on stack top 0.
    st.apply(&ArcEagerAction { move_type: ArcEagerMove::Right, label: labels.prep });
    assert_eq!(st.heads[2], 0);
    assert_eq!(st.labels[2], labels.prep);
    assert!(st.right_children[0].contains(&2));
    assert_eq!(st.stack, vec![0, 2], "RIGHT pushes the buffer head");

    // REDUCE pops the stack top.
    st.apply(&ArcEagerAction { move_type: ArcEagerMove::Reduce, label: 0 });
    assert_eq!(st.stack, vec![0]);

    // BREAK clears the stack and consumes the buffer head (always progresses).
    st.apply(&ArcEagerAction { move_type: ArcEagerMove::Break, label: 0 });
    assert!(st.stack.is_empty());
    assert!(st.buffer.is_empty(), "Break consumed the boundary token");
}

#[test]
fn apply_refuses_to_rehead_a_token() {
    // A token is attached at most once → acyclic output.
    let labels = label_hashes(&en_vocab());
    let mut st = ArcEagerState::new(3, 0);
    st.stack.push(0);
    st.buffer.extend(1..3);
    st.apply(&ArcEagerAction { move_type: ArcEagerMove::Left, label: labels.det });
    assert_eq!(st.heads[0], 1, "first attachment wins");
    st.stack.push(0);
    st.buffer.pop_front(); // 1 consumed conceptually
    st.apply(&ArcEagerAction { move_type: ArcEagerMove::Left, label: labels.nsubj });
    assert_eq!(st.heads[0], 1, "already-headed token is never re-headed");
}

// ── 9.5/9.7 output: relative heads + golden structure ──────────────────

#[test]
fn relative_heads_match_the_doc_contract() {
    // head is a signed offset: token.i + head == head_index; root head = 0.
    let doc = tokenize("The cat sat on the mat.");
    let (result, _pc) = annotator()
        .annotate_with_confidence(&doc)
        .expect("parse");
    let recs = result.records().records();
    assert_eq!(recs.len(), doc.len());
    let root_count = recs.iter().filter(|r| r.dep == "root").count();
    assert_eq!(root_count, 1, "exactly one ROOT per sentence");
    for (i, r) in recs.iter().enumerate() {
        let abs = i as i32 + r.head;
        assert!((0..doc.len() as i32).contains(&abs), "head in bounds");
        if r.dep == "root" {
            assert_eq!(r.head, 0, "root head is 0 (label-driven, F8)");
        } else {
            assert_ne!(r.head, 0, "non-root never has head == 0");
        }
    }
}

#[test]
fn verb_governs_argument_structure() {
    let doc = tokenize("The cat sat on the mat.");
    let (result, _pc) = annotator()
        .annotate_with_confidence(&doc)
        .expect("parse");
    let recs = result.records().records();
    let deps: Vec<&str> = recs.iter().map(|r| r.dep.as_str()).collect();
    assert!(deps.contains(&"nsubj"), "subject extracted: {deps:?}");
    assert!(deps.contains(&"prep"), "preposition frame: {deps:?}");
    assert!(deps.contains(&"pobj"), "prepositional object: {deps:?}");
    assert!(deps.contains(&"root"), "root present: {deps:?}");
    // "cat" is the nsubj of the root verb "sat".
    let sat = recs.iter().position(|r| r.text == "sat").expect("sat");
    let cat = recs.iter().position(|r| r.text == "cat").expect("cat");
    assert_eq!(recs[cat].dep, "nsubj");
    assert_eq!(cat as i32 + recs[cat].head, sat as i32);
}

#[test]
fn transitive_dobj_structure() {
    let doc = tokenize("NASA launched HTML5.");
    let (result, _pc) = annotator()
        .annotate_with_confidence(&doc)
        .expect("parse");
    let recs = result.records().records();
    let launched = recs.iter().position(|r| r.text == "launched").expect("verb");
    let nasa = recs.iter().position(|r| r.text == "NASA").expect("nasa");
    let html5 = recs.iter().position(|r| r.text == "HTML5").expect("html5");
    assert_eq!(recs[launched].dep, "root");
    assert_eq!(recs[nasa].dep, "nsubj");
    assert_eq!(recs[html5].dep, "dobj");
    assert_eq!(nasa as i32 + recs[nasa].head, launched as i32);
    assert_eq!(html5 as i32 + recs[html5].head, launched as i32);
}

// ── 9.6 oracle ────────────────────────────────────────────────────────

#[test]
fn oracle_best_with_margin_reports_ties() {
    let labels = label_hashes(&en_vocab());
    let oracle = DeterministicOracle;
    // Two actions that score identically → margin 0.
    let mut st = ArcEagerState::new(2, 0);
    st.stack.push(0);
    st.buffer.extend(1..2);
    let pos = vec![Upos::Noun, Upos::Verb];
    let texts = vec!["dogs".to_string(), "bark".to_string()];
    let vocab = en_vocab();
    let flags: Vec<LexemeFlags> = texts
        .iter()
        .map(|t| vocab.lexicon().get_or_create(t).flags)
        .collect();
    let actions = st.candidate_actions(&pos, &texts, &flags, &labels, &ortho());
    assert!(actions.len() >= 2);
    let (best, margin) = oracle
        .best_with_margin(&st, &actions, &pos, &labels)
        .expect("winner");
    // Left(nsubj) dominates noun-before-verb → positive margin, no tie.
    assert_eq!(best.move_type, ArcEagerMove::Left);
    assert_eq!(best.label, labels.nsubj);
    assert!(margin > 0.0, "dominant candidate has positive margin");
}

#[test]
fn oracle_empty_actions_returns_none() {
    let labels = label_hashes(&en_vocab());
    let oracle = DeterministicOracle;
    let st = ArcEagerState::new(2, 0);
    let pos = vec![Upos::Noun, Upos::Noun];
    assert!(oracle.best_with_margin(&st, &[], &pos, &labels).is_none());
}

// ── 9.8 ParseConfidence ───────────────────────────────────────────────

#[test]
fn parse_confidence_ties_reduce_overall() {
    let confident = ParseConfidence::compute(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], 1.0);
    let tied = ParseConfidence::compute(&[1.0, 1.0, 1.0], &[1.0, 0.0, 1.0], 1.0);
    assert!((confident.overall - 1.0).abs() < 1e-9);
    assert_eq!(tied.oracle_tie_count, 1);
    assert!((tied.overall - 0.95).abs() < 1e-9, "one tie → 5% penalty");
    assert!((tied.overall - (1.0 - 0.05)).abs() < 1e-9);
}

#[test]
fn parse_confidence_never_goes_below_zero() {
    let c = ParseConfidence::compute(&[0.1], &[0.0, 0.0, 0.0, 0.0], 0.0);
    assert_eq!(c.overall, 0.0);
}

#[test]
fn annotate_reports_sane_confidence() {
    let doc = tokenize("The cat sat on the mat.");
    let (_result, pc) = annotator()
        .annotate_with_confidence(&doc)
        .expect("parse");
    assert!((0.0..=1.0).contains(&pc.overall));
    assert!(pc.role_coverage > 0.0, "subject slot filled");
    assert_eq!(pc.token_scores.len(), doc.len());
}

// ── 9.9 ArcEagerRung always-return ────────────────────────────────────

#[test]
fn rung_returns_parse_with_source_and_confidence() {
    let vocab = en_vocab();
    let rung = ArcEagerRung::new(
        Arc::new(ArcEagerAnnotator::en_default(Arc::clone(&vocab))),
        Arc::new(AnnotationValidator::new()),
    );
    let doc = tokenize("The cat sat.");
    let out = crate::pipeline::tests::run_rung_sync(rung, &doc);
    let result = out.expect("always returns Some").expect("Some");
    assert_eq!(result.source(), AnnotationSource::ArcEager);
    assert!(result.parse_confidence.is_some());
    assert!(result.token_confidence().is_some());
}

#[test]
fn rung_returns_none_only_on_structural_failure() {
    let vocab = en_vocab();
    let rung = ArcEagerRung::new(
        Arc::new(ArcEagerAnnotator::en_default(Arc::clone(&vocab))),
        Arc::new(AnnotationValidator::new()),
    );
    let empty = Doc::new(vocab);
    let out = crate::pipeline::tests::run_rung_sync(rung, &empty);
    assert!(out.expect("no error").is_none(), "empty doc → None → RuleRung");
}

// ── 9.10 property: every output passes the 7-check validator ──────────

#[test]
fn random_pos_sequences_produce_valid_trees() {
    // A small LCG so the property test is deterministic (no rand dep).
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed >> 33
    };
    let pos_words: &[(Upos, &str)] = &[
        (Upos::Noun, "cat"),
        (Upos::Verb, "run"),
        (Upos::Det, "the"),
        (Upos::Adp, "of"),
        (Upos::Propn, "NASA"),
        (Upos::Punct, "."),
        (Upos::Aux, "is"),
        (Upos::Adj, "big"),
        (Upos::Adv, "quickly"),
        (Upos::Cconj, "and"),
        (Upos::Sconj, "if"),
        (Upos::Num, "5"),
        (Upos::Pron, "it"),
    ];
    let vocab = en_vocab();
    let annotator = ArcEagerAnnotator::en_default(Arc::clone(&vocab));
    let validator = AnnotationValidator::new();
    for _ in 0..120 {
        let len = 1 + (next() % 14) as usize;
        let mut doc = Doc::new(Arc::clone(&vocab));
        for _ in 0..len {
            let (_, w) = pos_words[(next() as usize) % pos_words.len()];
            doc.push_back(w, true).expect("push");
        }
        let (result, _pc) = annotator.annotate_with_confidence(&doc).expect("parse");
        validator.validate(&doc, result.records()).expect("7-check gate");
    }
}

// ── DRY: shared lowercase-initial primitive equivalence ────────────────
// `starts_lowercase` consolidates the 10 bespoke finite-verb guards
// (`word.chars().next().is_some_and(|c| c.is_lowercase())` over both the
// `word` and `texts[i]` receiver spellings, both polarities). This test
// proves the primitive agrees with the legacy spelling on every input
// (zero behavior change), including empty, non-alpha-first, digit-first,
// and unicode (ß/ẞ, É/é) edges where `chars().next()` — never byte
// indexing — is the correct operation.
#[test]
fn starts_lowercase_matches_legacy_spelling() {
    fn legacy(word: &str) -> bool {
        word.chars().next().is_some_and(|c| c.is_lowercase())
    }
    let words = [
        "opened", "failed", "left", "stands", "ended", "fell", "ready", "work", "calls",
        "wittily", "unlikely", "loudly", "quarterly", "smiling", "raining",
        "Anna", "July", "CEO", "NASA", "Big", "Dogs", "She", "It", "S", "A",
        "yet", "today", "hard", "go", "a", "x", "", "5", "2fast", ".", ",", "—",
        "éclair", "Éclair", "ßeta", "ẞeta", "über", "Über",
    ];
    for word in words {
        assert_eq!(
            starts_lowercase(word),
            legacy(word),
            "word={word:?}"
        );
        // Both receiver spellings in the file (`word` and `texts[i]`)
        // route through the same `&str` primitive.
        let owned = word.to_string();
        assert_eq!(starts_lowercase(&owned), legacy(word), "owned {word:?}");
    }
}

// ── DRY: shared copula-attachment primitive equivalence ────────────────
// `copula_is_attached` consolidates the 6 bespoke copula-frame guards
// (`left_children[b].iter().any(|&c| labels[c] == cop)` over both the
// `self` (candidate arms) and `state` (oracle weights) receiver
// spellings). This test proves the primitive agrees with the legacy
// spelling on controlled fixtures (zero behavior change): empty children,
// non-cop labels, cop NOT first among several children, and a cop label
// attached to a different head.
#[test]
fn copula_is_attached_matches_legacy_spelling() {
    let vocab = en_vocab();
    let labels = label_hashes(&vocab);
    fn legacy(state: &ArcEagerState, b: usize, cop: u64) -> bool {
        state.left_children[b]
            .iter()
            .any(|&c| state.labels[c] == cop)
    }
    // Fixture 1: cop attached (with a non-cop sibling before it).
    let mut attached = ArcEagerState::new(5, 0);
    attached.left_children[3].push(1);
    attached.labels[1] = labels.det;
    attached.left_children[3].push(2);
    attached.labels[2] = labels.cop;
    // Fixture 2: children present but none is a cop.
    let mut bare = ArcEagerState::new(5, 0);
    bare.left_children[3].push(1);
    bare.labels[1] = labels.det;
    // Fixture 3: a cop attached to a DIFFERENT head (b=1), not b=3.
    let mut elsewhere = ArcEagerState::new(5, 0);
    elsewhere.left_children[1].push(0);
    elsewhere.labels[0] = labels.cop;
    // Fixture 4: no children at all.
    let empty = ArcEagerState::new(5, 0);
    for (state, b, expected) in [
        (&attached, 3, true),
        (&attached, 1, false),
        (&bare, 3, false),
        (&elsewhere, 3, false),
        (&elsewhere, 1, true),
        (&empty, 3, false),
        (&empty, 0, false),
    ] {
        assert_eq!(legacy(state, b, labels.cop), expected, "legacy b={b}");
        assert_eq!(
            copula_is_attached(state, b, labels.cop),
            legacy(state, b, labels.cop),
            "b={b}"
        );
    }
}

// ── STREAMLINE: refine early-out contract (no candidate → no write) ────
// Every `refine_pos_*` loop body provably no-ops when its candidate POS is
// absent (all writes sit after the POS guard), so a
// `pos.iter().any(...)` pre-scan that skips the loop is behavior-neutral.
// This test pins that contract per rule on REAL flag streams (trigger bits
// set — relativizers, yet/clitic/wh/after/adverb words — with the POS
// scrubbed target-absent), proving flags alone can never trigger a write.
// It passes before the early-outs land (loop no-ops naturally) and must
// keep passing after (early-out skips legally).
#[test]
fn refine_early_out_no_candidate_no_write() {
    type Plain = fn(&[String], &mut [Upos], &[LexemeFlags]);
    type WithOrtho = for<'a, 'b> fn(
        &'a [String],
        &'a mut [Upos],
        &'a [LexemeFlags],
        &'a TaggerOrtho<'b>,
    );
    // Rules whose frames evaluate blob strings take `ortho`; the rest keep
    // the plain shape (interface segregation — no unused params).
    // (rule, refine fn, sentence with realistic trigger flags, scrub POS
    //  that leaves the rule's candidate set empty)
    let plain: &[(&str, Plain, &str, Upos)] = &[
    ];
    let ortho_table: &[(&str, WithOrtho, &str, Upos)] = &[
        ("bare_infinitive", refine_pos_bare_infinitive, "She won't answer calls.", Upos::Verb),
        ("bare_object_noun", refine_pos_bare_object_noun, "She won't answer calls.", Upos::Noun),
        ("directive_initial", refine_pos_directive_initial, "Call the office now.", Upos::Verb),
        ("demonstrative_object", refine_pos_demonstrative_object, "Translate this to French.", Upos::Noun),
        ("attributive_ly", refine_pos_attributive_ly, "quarterly sales report", Upos::Verb),
        ("shifted_det_noun_verb", refine_pos_shifted_det_noun_verb, "Truthfully, the plan failed.", Upos::Verb),
        ("relative_matrix_verb", refine_pos_relative_matrix_verb, "The man who called left.", Upos::Verb),
        ("relcl_matrix_complement", refine_pos_relcl_matrix_complement, "The dog that barked stands empty.", Upos::Verb),
        ("comma_adverbial", refine_pos_comma_adverbial, "In July, we met.", Upos::Verb),
        ("predicative_ly", refine_pos_predicative_ly, "That seems unlikely.", Upos::Verb),
        ("final_adverbial", refine_pos_final_adverbial, "The dog barks loudly.", Upos::Verb),
        ("temporal_yet", refine_pos_temporal_yet, "She is not ready yet.", Upos::Verb),
        ("linking_predicate", refine_pos_linking_predicate, "That seems unlikely.", Upos::Verb),
        ("medial_manner_ly", refine_pos_medial_manner_ly, "She spoke wittily at dinner.", Upos::Verb),
        ("post_comma_verb", refine_pos_post_comma_verb, "The test, honestly, scared us.", Upos::Verb),
        ("comma_participle", refine_pos_comma_participle, "The CEO, smiling, took questions.", Upos::Verb),
        ("inverted_copular", refine_pos_inverted_copular, "Is lunch ready?", Upos::Verb),
        ("be_predicate", refine_pos_be_predicate, "It's raining.", Upos::Verb),
        ("progressive_ing", refine_pos_progressive_ing, "It's raining.", Upos::Verb),
        ("modal_question_verb", refine_pos_modal_question_verb, "Will this work?", Upos::Verb),
        ("contracted_be", refine_pos_contracted_be, "It's raining.", Upos::Verb),
        ("discourse_initial_verb", refine_pos_discourse_initial_verb, "Please confirm the date.", Upos::Verb),
        ("imperative_non_det_object", refine_pos_imperative_non_det_object, "Eat apples daily.", Upos::Verb),
        ("bare_ed_transitive", refine_pos_bare_ed_transitive, "John opened the door.", Upos::Verb),
    ];
    for (name, refine, text, scrub) in plain {
        let doc = tokenize(text);
        let texts: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
        let flags: Vec<LexemeFlags> =
            (0..doc.len()).map(|i| doc.token(i).lexeme.flags).collect();
        let mut pos = vec![*scrub; doc.len()];
        refine(&texts, &mut pos, &flags);
        assert_eq!(
            pos,
            vec![*scrub; doc.len()],
            "{name}: write without candidate POS"
        );
    }
    for (name, refine, text, scrub) in ortho_table {
        let doc = tokenize(text);
        let texts: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
        let flags: Vec<LexemeFlags> =
            (0..doc.len()).map(|i| doc.token(i).lexeme.flags).collect();
        let mut pos = vec![*scrub; doc.len()];
        refine(&texts, &mut pos, &flags, &ortho());
        assert_eq!(
            pos,
            vec![*scrub; doc.len()],
            "{name}: write without candidate POS"
        );
    }
}
