use super::*;

fn blob() -> LemmaBlob {
    LemmaBlob::from_bytes(crate::lang::en::LEMMAS_BLOB)
        .expect("embedded English lemma blob parses")
}

#[test]
fn parses_all_pos_keys() {
    let b = blob();
    let keys: Vec<&str> = b.pos_keys().collect();
    assert_eq!(keys, vec!["adj", "adv", "noun", "punct", "verb"]);
}

#[test]
fn rules_are_present_per_pos() {
    let b = blob();
    assert_eq!(b.rules("noun").len(), 9);
    assert_eq!(b.rules("verb").len(), 8);
    assert_eq!(b.rules("punct").len(), 4);
    assert!(b.rules("noun").contains(&("ies", "y")));
    assert!(b.rules("missing").is_empty());
}

#[test]
fn index_binary_search_hits_edges_and_misses() {
    let b = blob();
    assert!(b.index_contains("noun", "'hood"));
    assert!(b.index_contains("noun", "zyrian"));
    assert!(b.index_contains("verb", "aah"));
    assert!(b.index_contains("adj", "zymotic"));
    assert!(!b.index_contains("noun", "zzzznope"));
    assert!(!b.index_contains("missing", "cat"));
}

#[test]
fn exc_for_returns_lemma_list() {
    let b = blob();
    let lemmas = b.exc_for("verb", "went").expect("went is an exception");
    let forms: Vec<&str> = lemmas
        .split(|&x| x == 0)
        .filter(|s| !s.is_empty())
        .filter_map(|s| std::str::from_utf8(s).ok())
        .collect();
    assert_eq!(forms, vec!["go"]);
    let children = b.exc_for("noun", "children").expect("children exception");
    let forms: Vec<&str> = children
        .split(|&x| x == 0)
        .filter(|s| !s.is_empty())
        .filter_map(|s| std::str::from_utf8(s).ok())
        .collect();
    assert_eq!(forms, vec!["child"]);
    assert!(b.exc_for("noun", "notanexception").is_none());
}

#[test]
fn rejects_bad_magic_and_version() {
    let data: &'static [u8] = &[0, 0, 0, 0];
    assert!(LemmaBlob::from_bytes(data).is_err());
    let data: &'static [u8] = &[
        0x31, 0x4D, 0x4C, 0x53, // "SLM1"
        2, 0, // version 2 (unsupported)
        0, 0, // zero pos
    ];
    assert!(LemmaBlob::from_bytes(data).is_err());
}
