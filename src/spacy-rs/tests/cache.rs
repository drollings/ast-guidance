use super::*;
use crate::review::CorrectionField;
use crate::vocab::Vocab;
use crate::lang;

fn doc_from(text: &str) -> Doc {
    let vocab = std::sync::Arc::new(Vocab::new(lang::en::lexicon_config()));
    let tok = lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    tok.tokenize(text).expect("tokenize")
}

#[test]
fn cache_is_content_addressed() {
    let a = doc_from("hello world");
    let b = doc_from("hello world");
    let focus_a = vec![0];
    assert_eq!(span_key(&a, &focus_a), span_key(&b, &focus_a));
    // Different focused orth → different key.
    let c = doc_from("goodbye world");
    assert_ne!(span_key(&a, &focus_a), span_key(&c, &focus_a));
    // Same focused word but different doc tail — same span key (span-level).
    let d = doc_from("hello there");
    assert_eq!(span_key(&a, &focus_a), span_key(&d, &focus_a));
}

#[test]
fn cache_empty_focus_never_cached() {
    let doc = doc_from("hello");
    assert_eq!(span_key(&doc, &[]), 0);
    let cache = InMemorySpanCache::new();
    cache.put(0, vec![]);
    assert_eq!(cache.len(), 0);
}

#[test]
fn put_get_invalidate() {
    let cache = InMemorySpanCache::new();
    let key = 42u64;
    let corr = vec![Correction {
        token_index: 0,
        field: CorrectionField::Pos,
        old_value: String::new(),
        new_value: "verb".into(),
    }];
    cache.put(key, corr.clone());
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(key).unwrap(), corr);
    cache.invalidate(key);
    assert!(cache.get(key).is_none());
    assert_eq!(cache.len(), 0);
}

#[test]
fn lowercasing_is_case_insensitive() {
    let a = doc_from("Hello");
    let b = doc_from("hello");
    let focus = vec![0];
    assert_eq!(span_key(&a, &focus), span_key(&b, &focus));
}
