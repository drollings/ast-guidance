use super::*;
use crate::hash::hash_utf8;
use crate::labels::DepLabelSet;
use crate::lexeme::LexiconConfig;
use crate::vocab::Vocab;
use std::str::FromStr;
use std::sync::Arc;

fn vocab() -> Arc<Vocab> {
    Arc::new(Vocab::new(LexiconConfig::default()))
}

/// A doc + a valid annotation set for "The cat sat ." (det, nsubj, root,
/// punct), the canonical §11.4 tree.
fn valid_doc_and_set() -> (Doc, AnnotationSet) {
    let mut doc = Doc::new(vocab());
    for (t, s) in [("The", true), ("cat", true), ("sat", true), (".", false)] {
        doc.push_back(t, s).expect("push");
    }
    let set = AnnotationSet(vec![
        AnnotationRecord {
            text: "The".into(),
            pos: "det".into(),
            tag: "DT".into(),
            dep: "det".into(),
            head: 1,
            lemma: "the".into(),
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        },
        AnnotationRecord {
            text: "cat".into(),
            pos: "noun".into(),
            tag: "NN".into(),
            dep: "nsubj".into(),
            head: 1,
            lemma: "cat".into(),
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        },
        AnnotationRecord {
            text: "sat".into(),
            pos: "verb".into(),
            tag: "VBD".into(),
            dep: "root".into(),
            head: 0,
            lemma: "sit".into(),
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        },
        AnnotationRecord {
            text: ".".into(),
            pos: "punct".into(),
            tag: ".".into(),
            dep: "punct".into(),
            head: -1,
            lemma: ".".into(),
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        },
    ]);
    (doc, set)
}

fn recorder(text: &str, pos: &str, dep: &str, head: i32) -> AnnotationRecord {
    AnnotationRecord {
        text: text.into(),
        pos: pos.into(),
        tag: String::new(),
        dep: dep.into(),
        head,
        lemma: String::new(),
        morph: String::new(),
        ent_iob: String::new(),
        ent_type: String::new(),
    }
}

#[test]
fn accepts_canonical_parse() {
    let (doc, set) = valid_doc_and_set();
    let validator = AnnotationValidator::new();
    assert_eq!(validator.validate(&doc, &set), Ok(()));
}

#[test]
fn rejects_count_mismatch() {
    let (doc, set) = valid_doc_and_set();
    let short = AnnotationSet(set.0[..3].to_vec());
    let err = AnnotationValidator::new().validate(&doc, &short);
    assert_eq!(
        err,
        Err(AnnotationError::CountMismatch {
            expected: 4,
            got: 3
        })
    );
}

#[test]
fn rejects_text_mismatch() {
    let (doc, mut set) = valid_doc_and_set();
    set.0[2].text = "sat.".into();
    let err = AnnotationValidator::new().validate(&doc, &set);
    assert!(matches!(
        err,
        Err(AnnotationError::TextMismatch { index: 2, .. })
    ));
}

#[test]
fn rejects_unknown_pos_and_dep() {
    let (doc, set) = valid_doc_and_set();
    let mut bad_pos = set.clone();
    bad_pos.0[0].pos = "notapos".into();
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &bad_pos),
        Err(AnnotationError::UnknownPos(_))
    ));

    let mut bad_dep = set;
    bad_dep.0[1].dep = "bogus_relation".into();
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &bad_dep),
        Err(AnnotationError::UnknownDep(_))
    ));
}

#[test]
fn rejects_head_out_of_bounds() {
    let (doc, mut set) = valid_doc_and_set();
    set.0[0].head = 99;
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &set),
        Err(AnnotationError::HeadOutOfBounds { token: 0, .. })
    ));
    set = valid_doc_and_set().1;
    set.0[3].head = -5;
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &set),
        Err(AnnotationError::HeadOutOfBounds { token: 3, .. })
    ));
}

#[test]
fn rejects_self_loop_non_root() {
    let (doc, mut set) = valid_doc_and_set();
    set.0[1].head = 0; // cat → cat but dep is "nsubj"
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &set),
        Err(AnnotationError::SelfLoop { token: 1 })
    ));
}

#[test]
fn rejects_root_head_mismatch() {
    let (doc, mut set) = valid_doc_and_set();
    set.0[2].head = 1; // dep root but head != 0
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &set),
        Err(AnnotationError::RootHeadMismatch { token: 2, .. })
    ));
}

#[test]
fn rejects_zero_roots() {
    // dep == root is replaced by a non-root head → no ROOT remains.
    let (doc, set) = valid_doc_and_set();
    let mut none = set;
    none.0[2].dep = "xcomp".into();
    none.0[2].head = 1; // token 2 → token 1, but nothing claims root
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &none),
        Err(AnnotationError::RootCount { found: 0, .. })
    ));
}

#[test]
fn accepts_multiple_roots_for_multiple_sentences() {
    // Two sentences, one root each: "Apple looks at buying R.G. startup."
    // and a second "It is good." — two `dep == root` tokens is valid now
    // (per-sentence roots, walkthrough §9.4).
    let mut doc = Doc::new(vocab());
    for t in ["Apple", "looks", "at", "buying", ".", "It", "is", "good"] {
        doc.push_back(t, true).expect("push");
    }
    let nsubj = "nsubj".to_string();
    let root = "root".to_string();
    let prep = "prep".to_string();
    let punct = "punct".to_string();
    let cop = "cop".to_string();
    let set = AnnotationSet(vec![
        recorder("Apple", "noun", &nsubj, 1),
        recorder("looks", "verb", &root, 0),
        recorder("at", "adp", &prep, -1),
        recorder("buying", "verb", "pcomp", -1),
        recorder(".", "punct", &punct, -3),
        recorder("It", "pron", &nsubj, 1),
        recorder("is", "aux", &cop, 1),
        recorder("good", "adj", &root, 0),
    ]);
    assert_eq!(AnnotationValidator::new().validate(&doc, &set), Ok(()));
}

#[test]
fn rejects_disconnected_component_as_cycle_or_unreachable() {
    // a → b (b root); c → d, d → c (an independent 2-cycle with no root)
    let mut doc = Doc::new(vocab());
    for t in ["a", "b", "c", "d"] {
        doc.push_back(t, true).expect("push");
    }
    let set = AnnotationSet(vec![
        recorder("a", "noun", "nsubj", 1),
        recorder("b", "noun", "root", 0),
        recorder("c", "noun", "dep", 1),
        recorder("d", "noun", "dep", -1),
    ]);
    // c→d→c is a cycle in a separate component
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &set),
        Err(AnnotationError::Cycle(_))
    ));
}

#[test]
fn rejects_cycle() {
    // a → b, b → a: a pure cycle in a separate component, alongside the
    // required root c. RootCount passes (exactly one root), then the
    // three-color walk detects the cycle.
    let mut doc = Doc::new(vocab());
    for t in ["a", "b", "c"] {
        doc.push_back(t, true).expect("push");
    }
    let set = AnnotationSet(vec![
        recorder("a", "noun", "dep", 1),
        recorder("b", "noun", "dep", -1),
        recorder("c", "noun", "root", 0),
    ]);
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &set),
        Err(AnnotationError::Cycle(_))
    ));
}

#[test]
fn accepts_single_entity_span_biluo() {
    let (doc, set) = valid_doc_and_set();
    let mut ent = set;
    ent.0[0].ent_iob = "B".into();
    ent.0[0].ent_type = "ORG".into();
    ent.0[1].ent_iob = "L".into();
    ent.0[1].ent_type = "ORG".into();
    assert_eq!(AnnotationValidator::new().validate(&doc, &ent), Ok(()));

    let (doc, set) = valid_doc_and_set();
    let mut uni = set;
    uni.0[0].ent_iob = "U".into();
    uni.0[0].ent_type = "PERSON".into();
    assert_eq!(AnnotationValidator::new().validate(&doc, &uni), Ok(()));
}

#[test]
fn rejects_malformed_biluo() {
    // I without a preceding B
    let (doc, set) = valid_doc_and_set();
    let mut bad = set;
    bad.0[1].ent_iob = "I".into();
    bad.0[1].ent_type = "ORG".into();
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &bad),
        Err(AnnotationError::MalformedBiluo { index: 1, .. })
    ));

    // B without an L (unterminated)
    let (doc, set) = valid_doc_and_set();
    let mut bad = set;
    bad.0[1].ent_iob = "B".into();
    bad.0[1].ent_type = "ORG".into();
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &bad),
        Err(AnnotationError::MalformedBiluo { index: 2, .. })
    ));

    // type required where IOB != O
    let (doc, set) = valid_doc_and_set();
    let mut bad = set;
    bad.0[1].ent_iob = "U".into();
    bad.0[1].ent_type = String::new();
    assert!(matches!(
        AnnotationValidator::new().validate(&doc, &bad),
        Err(AnnotationError::MalformedBiluo { index: 1, .. })
    ));
}

#[test]
fn rejects_non_projective_when_required() {
    // Non-projective: a → c (a heads c) crossing b → d, b → e, e root.
    // Tokens: a b c d e; heads: a→c(2), b→root(e,4), c→root(4), d→root(4).
    // Edge a→c crosses b (b is not a descendant of c).
    let mut doc = Doc::new(vocab());
    for t in ["a", "b", "c", "d", "e"] {
        doc.push_back(t, true).expect("push");
    }
    let set = AnnotationSet(vec![
        recorder("a", "noun", "nsubj", 2),
        recorder("b", "noun", "nsubj", 3),
        recorder("c", "verb", "xcomp", 2),
        recorder("d", "verb", "dep", 1),
        recorder("e", "verb", "root", 0),
    ]);
    // Default: projective not required → the (odd) tree is structurally valid
    assert_eq!(AnnotationValidator::new().validate(&doc, &set), Ok(()));
    // Required: a heads c across b → non-projective
    assert!(matches!(
        AnnotationValidator::new()
            .require_projectivity(true)
            .validate(&doc, &set),
        Err(AnnotationError::NonProjective { .. })
    ));
}

#[test]
fn rejects_projectivity_failure_with_clean_tree() {
    // A projective tree must pass with the requirement on.
    let (doc, set) = valid_doc_and_set();
    assert_eq!(
        AnnotationValidator::new()
            .require_projectivity(true)
            .validate(&doc, &set),
        Ok(())
    );
}

#[test]
fn rejects_empty_document() {
    let doc = Doc::new(vocab());
    let set = AnnotationSet(vec![recorder("x", "noun", "root", 0)]);
    assert_eq!(
        AnnotationValidator::new().validate(&doc, &set),
        Err(AnnotationError::EmptyDocument)
    );
}

#[test]
fn custom_dep_label_set_accepts_and_rejects() {
    let (doc, set) = valid_doc_and_set();
    // A set that excludes "punct" rejects the canonical parse: build a
    // minimal accepted set (removal from ud_default is not exposed).
    let strict = DepLabelSet::from_str("root,nsubj,det").expect("parse");
    let validator = AnnotationValidator::with_dep_labels(strict);
    assert!(matches!(
        validator.validate(&doc, &set),
        Err(AnnotationError::UnknownDep(_))
    ));
}

#[test]
fn attached_doc_is_navigable() {
    // Apply the golden set, then confirm the rebuilt tree is navigable and
    // the sent-starts are marked from the root.
    let (mut doc, set) = valid_doc_and_set();
    crate::llm::apply(&mut doc, &set).expect("apply");
    assert_eq!(doc.head_index(2), 2); // root
    assert_eq!(doc.head_index(0), 1);
    assert_eq!(doc.lefts(2), vec![1]);
    assert_eq!(doc.rights(2), vec![3]);
    assert_eq!(doc.token(2).pos, Upos::Verb);
    assert_eq!(doc.token(0).dep, hash_utf8("det"));
    assert_eq!(doc.token(0).sent_start, crate::doc::SentStart::Start);
}
