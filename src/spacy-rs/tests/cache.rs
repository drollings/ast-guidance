use super::*;
use crate::review::CorrectionField;
use crate::vocab::Vocab;
use crate::lang;

fn doc_from(text: &str) -> Doc {
    let vocab = std::sync::Arc::new(Vocab::new(lang::en::lexicon_config()));
    let tok = lang::en::tokenizer(vocab.clone()).expect("tokenizer");
    tok.tokenize(text).expect("tokenize")
}

/// A hermetic [`SpanCacheSeam`] over a shared map (test-only double).
/// The ledger owns the production `SpanCache` trait + `InMemorySpanCache`;
/// the ladder only needs these three legs, so tests build the seam directly.
fn test_seam() -> (SpanCacheSeam, std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u64, Vec<Correction>>>>) {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    let map: Arc<Mutex<HashMap<u64, Vec<Correction>>>> = Arc::new(Mutex::new(HashMap::new()));
    let get_h = Arc::clone(&map);
    let put_h = Arc::clone(&map);
    let inv_h = Arc::clone(&map);
    let seam = SpanCacheSeam::new(
        Arc::new(move |key| get_h.lock().expect("lock").get(&key).cloned()),
        Arc::new(move |key, corrections| {
            if key != 0 {
                put_h.lock().expect("lock").insert(key, corrections);
            }
        }),
        Arc::new(move |key| {
            inv_h.lock().expect("lock").remove(&key);
        }),
    );
    (seam, map)
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
    let (seam, map) = test_seam();
    seam.put(0, vec![]);
    assert!(map.lock().expect("lock").is_empty());
}

#[test]
fn put_get_invalidate() {
    let (seam, _) = test_seam();
    let key = 42u64;
    let corr = vec![Correction {
        token_index: 0,
        field: CorrectionField::Pos,
        old_value: String::new(),
        new_value: "verb".into(),
    }];
    assert!(seam.get(key).is_none());
    seam.put(key, corr.clone());
    assert_eq!(seam.get(key).unwrap(), corr);
    seam.invalidate(key);
    assert!(seam.get(key).is_none());
}

#[test]
fn lowercasing_is_case_insensitive() {
    let a = doc_from("Hello");
    let b = doc_from("hello");
    let focus = vec![0];
    assert_eq!(span_key(&a, &focus), span_key(&b, &focus));
}
