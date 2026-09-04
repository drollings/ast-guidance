use super::*;
use crate::hash::hash_utf8;
use crate::lexeme::LexiconConfig;
use fluent_types::InterlinguaNamespace;

fn vocab() -> Arc<Vocab> {
    Arc::new(Vocab::new(LexiconConfig::default()))
}

fn doc_with(texts: &[(&str, bool)]) -> Doc {
    let mut doc = Doc::new(vocab());
    for (text, spacy) in texts {
        doc.push_back(text, *spacy).expect("push");
    }
    doc
}

#[test]
fn push_back_computes_idx_and_spacy() {
    let doc = doc_with(&[("Hello", true), ("world", true), ("!", false)]);
    assert_eq!(doc.len(), 3);
    assert_eq!(doc.token(0).idx, 0);
    assert_eq!(doc.token(1).idx, 6); // "Hello" (5) + spacy (1)
    assert_eq!(doc.token(2).idx, 12); // "world" (5) + spacy (1)
    assert!(doc.token(0).spacy);
    assert!(doc.token(1).spacy);
    assert!(!doc.token(2).spacy);
}

#[test]
fn push_back_rejects_empty() {
    let mut doc = Doc::new(vocab());
    assert!(doc.push_back("", false).is_err());
}

#[test]
fn first_token_is_sent_start() {
    let doc = doc_with(&[("Hello", false)]);
    assert_eq!(doc.token(0).sent_start, SentStart::Start);
}

#[test]
fn text_reconstruction() {
    let doc = doc_with(&[("Hello", true), ("world", true), ("!", false)]);
    assert_eq!(doc.text(), "Hello world !");
}

#[test]
fn text_reconstruction_preserves_trailing_space() {
    let doc = doc_with(&[("Hello", true), ("world", true)]);
    assert_eq!(doc.text(), "Hello world ");
}

#[test]
fn token_text_resolves_through_store() {
    let doc = doc_with(&[("Apple", false)]);
    assert_eq!(doc.token_text(0), "Apple");
}

#[test]
fn to_array_matches_spacy_attr_contract() {
    let doc = doc_with(&[("Hello", true), ("world", false)]);
    let arr = doc
        .to_array(&[
            Attribute::Orth,
            Attribute::Length,
            Attribute::IsAlpha,
            Attribute::Idx,
        ])
        .expect("to_array");
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0], vec![hash_utf8("Hello"), 5, 1, 0]);
    assert_eq!(arr[1], vec![hash_utf8("world"), 5, 1, 6]);
}

#[test]
fn from_array_roundtrips_context_attrs() {
    let mut doc = doc_with(&[("The", false), ("cat", false), ("sat", false)]);
    let root_dep = hash_utf8("ROOT");
    let nsubj_dep = hash_utf8("nsubj");
    let det_dep = hash_utf8("det");
    let arr = vec![
        vec![1, Upos::Det.id(), det_dep],
        vec![1, Upos::Noun.id(), nsubj_dep],
        vec![0, Upos::Verb.id(), root_dep],
    ];
    doc.from_array(&[Attribute::Head, Attribute::Pos, Attribute::Dep], &arr)
        .expect("from_array");
    assert_eq!(doc.token(0).head, 1);
    assert_eq!(doc.token(1).head, 1);
    assert_eq!(doc.token(2).head, 0);
    assert_eq!(doc.token(2).pos, Upos::Verb);
    assert_eq!(doc.token(0).pos, Upos::Det);
    assert_eq!(doc.token(0).dep, det_dep);
    assert_eq!(doc.token(2).dep, root_dep);
}

#[test]
fn interlingua_attr_ids_roundtrip_via_to_from_array() {
    let mut doc = doc_with(&[("The", false), ("cat", false)]);
    let lemma_id = InterlinguaId::new(InterlinguaNamespace::SpacyLemma, 0x1234_5678_9abc);
    doc.token_mut(0).interlingua_lemma_id = Some(lemma_id);
    doc.token_mut(0).confidence = Some(0.875);
    let attrs = [
        Attribute::InterlinguaLemmaId,
        Attribute::InterlinguaEntityId,
        Attribute::AnnotationConfidence,
    ];
    let arr = doc.to_array(&attrs).expect("to_array");
    assert_eq!(arr[0][0], lemma_id.as_u64());
    assert_eq!(arr[0][1], 0, "unset entity id → 0");
    assert_eq!(arr[0][2], 0.875f64.to_bits());

    // from_array restores them (and 0 → None).
    let mut doc2 = doc_with(&[("The", false), ("cat", false)]);
    doc2.from_array(&attrs, &arr).expect("from_array");
    assert_eq!(doc2.token(0).interlingua_lemma_id, Some(lemma_id));
    assert_eq!(doc2.token(0).confidence, Some(0.875));
    assert!(doc2.token(1).interlingua_lemma_id.is_none());
    assert!(doc2.token(1).confidence.is_none());
}

#[test]
fn from_array_rejects_out_of_bounds_heads() {
    let mut doc = doc_with(&[("a", false), ("b", false)]);
    let err = doc
        .from_array(&[Attribute::Head], &[vec![5], vec![0]])
        .expect_err("head out of bounds");
    assert!(matches!(err, SpacyError::HeadOutOfBounds { token: 0, .. }));
}

#[test]
fn from_array_rejects_invalid_iob() {
    let mut doc = doc_with(&[("a", false)]);
    let err = doc
        .from_array(&[Attribute::EntIob], &[vec![9]])
        .expect_err("iob out of range");
    assert_eq!(err, SpacyError::InvalidEntIob(9));
}

#[test]
fn from_array_rejects_row_count_mismatch() {
    let mut doc = doc_with(&[("a", false), ("b", false)]);
    let err = doc
        .from_array(&[Attribute::Orth], &[vec![0]])
        .expect_err("row mismatch");
    assert_eq!(err, SpacyError::ArrayLengthMismatch { array: 1, doc: 2 });
}

/// A 4-token tree: "The cat sat ." with `cat`→ROOT(`sat`), `The`→`cat`,
/// `.`→`sat`.
fn parsed_doc() -> Doc {
    let mut doc = doc_with(&[("The", true), ("cat", true), ("sat", true), (".", false)]);
    // heads: The→cat (i+1), cat→sat (i+1), sat=ROOT (0), .→sat (i-1)
    let root = hash_utf8("ROOT");
    let det = hash_utf8("det");
    let nsubj = hash_utf8("nsubj");
    let punct = hash_utf8("punct");
    doc.from_array(
        &[Attribute::Head, Attribute::Dep],
        &[
            vec![1, det],
            vec![1, nsubj],
            vec![0, root],
            vec![(-1i32) as u64, punct],
        ],
    )
    .expect("parse");
    doc
}

#[test]
fn set_children_from_heads_rebuilds_edges() {
    let doc = parsed_doc();
    // Tree: The → cat → sat ← .
    assert_eq!(doc.token(2).l_kids, 1); // cat
    assert_eq!(doc.token(2).r_kids, 1); // .
    assert_eq!(doc.token(1).l_kids, 1); // The
    assert_eq!(doc.token(0).l_kids, 0);
    // edges
    assert_eq!(doc.left_edge(2), 0);
    assert_eq!(doc.right_edge(2), 3);
    assert_eq!(doc.left_edge(1), 0);
    assert_eq!(doc.right_edge(1), 1);
}

#[test]
fn tree_navigation_lefts_rights_ancestors() {
    let doc = parsed_doc();
    assert_eq!(doc.lefts(2), vec![1]);
    assert_eq!(doc.lefts(1), vec![0]);
    assert_eq!(doc.rights(2), vec![3]);
    assert_eq!(doc.children(2), vec![1, 3]);
    assert_eq!(doc.ancestors(0), vec![1, 2]);
    assert_eq!(doc.ancestors(2), Vec::<usize>::new());
    assert_eq!(doc.head_index(0), 1);
    assert_eq!(doc.head_index(2), 2);
    assert!(doc.is_ancestor(2, 0));
    assert!(!doc.is_ancestor(0, 2));
    assert_eq!(doc.subtree(1), vec![0, 1]);
    assert_eq!(doc.subtree(2), vec![0, 1, 2, 3]);
}

#[test]
fn sent_starts_marked_from_roots() {
    let doc = parsed_doc();
    assert_eq!(doc.token(0).sent_start, SentStart::Start);
    assert_eq!(doc.token(1).sent_start, SentStart::NotStart);
    assert_eq!(doc.token(2).sent_start, SentStart::NotStart);
    assert_eq!(doc.token(3).sent_start, SentStart::NotStart);
}

#[test]
fn non_projective_tree_settles() {
    // A non-projective configuration (a→c crosses b→d); the multi-pass
    // loop must terminate and produce edges enclosing every token.
    let mut doc = doc_with(&[("a", false), ("b", false), ("c", false), ("d", false)]);
    let root = hash_utf8("ROOT");
    doc.from_array(
        &[Attribute::Head, Attribute::Dep],
        &[
            vec![2, 1],              // a → c (crosses the next edge)
            vec![2, 1],              // b → c
            vec![0, root],           // c = ROOT
            vec![(-1i32) as u64, 1], // d → c
        ],
    )
    .expect("parse");
    for (i, token) in doc.tokens().iter().enumerate() {
        let l = token.l_edge as usize;
        let r = token.r_edge as usize;
        assert!(l <= i && i <= r, "edges enclose token {i}: {l}..={r}");
    }
}

#[test]
fn set_children_from_heads_rejects_bad_heads() {
    let mut doc = doc_with(&[("a", false)]);
    doc.token_mut(0).head = 3;
    assert!(matches!(
        doc.set_children_from_heads(),
        Err(SpacyError::HeadOutOfBounds { .. })
    ));
}

#[test]
fn read_only_attributes_rejected() {
    let mut doc = doc_with(&[("a", false)]);
    let err = doc
        .from_array(&[Attribute::Orth], &[vec![7]])
        .expect_err("orth is lexeme-derived");
    assert_eq!(err, SpacyError::ReadOnlyAttribute(Attribute::Orth.id()));
}
