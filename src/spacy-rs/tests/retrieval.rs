use super::*;
use crate::doc::Doc;
use crate::lang::en::lexicon_config;
use crate::vocab::Vocab;
use std::sync::Arc;

fn doc_with(texts: &[(&str, f64, Option<InterlinguaId>)]) -> Doc {
    // Build a small parsed doc: each entry is (token_text, per-token
    // confidence, interlingua_lemma_id). Lemma == lowercase token text.
    let vocab = Arc::new(Vocab::new(lexicon_config()));
    let mut doc = Doc::new(vocab.clone());
    for (i, (text, conf, il_id)) in texts.iter().enumerate() {
        let n = doc.push_back(text, i + 1 < texts.len()).expect("push");
        let _ = n;
        let last = doc.len() - 1;
        {
            let tokens = doc.tokens_mut();
            let tok = &mut tokens[last];
            tok.lemma = vocab.strings().add(&text.to_lowercase());
            tok.confidence = Some(*conf);
            tok.interlingua_lemma_id = *il_id;
        }
    }
    doc
}

#[test]
fn lemma_grep_returns_hits_with_confidence_and_lemma_id() {
    let doc = doc_with(&[
        ("Show", 0.9, Some(InterlinguaId::from_u64(7))),
        ("me", 0.8, None),
        ("the", 0.7, None),
        ("report", 0.6, Some(InterlinguaId::from_u64(9))),
    ]);
    let hits = lemma_grep(&doc, "show");
    assert_eq!(hits.len(), 1);
    let hit = &hits[0];
    assert_eq!(hit.lemma, "show");
    assert_eq!(hit.lemma_id, Some(InterlinguaId::from_u64(7)));
    assert_eq!(hit.parse_confidence, 0.9);
    // Byte span of the first token in "Show me the report".
    assert_eq!(hit.span, Span { start: 0, end: 4 });
}

#[test]
fn lemma_grep_is_case_insensitive_and_skips_unmatched() {
    let doc = doc_with(&[("show", 0.9, None), ("display", 0.5, None)]);
    assert_eq!(lemma_grep(&doc, "SHOW").len(), 1);
    assert_eq!(lemma_grep(&doc, "list").len(), 0);
}

#[test]
fn lemma_grep_skips_tokens_without_a_resolved_lemma() {
    let vocab = Arc::new(Vocab::new(lexicon_config()));
    let mut doc = Doc::new(vocab.clone());
    doc.push_back("show", false).expect("push");
    // lemma hash left 0 → strings.get(0) resolves to "", which cannot equal
    // the query, so the token is skipped rather than falsely matched.
    assert!(lemma_grep(&doc, "show").is_empty());
}
