use super::*;

#[test]
fn make_trigram_value() {
    let tri = make_trigram(b'a', b'b', b'c');
    assert_eq!(tri, 0x636261);
}

#[test]
fn build_and_search() {
    let mut idx = TrigramIndex::new();
    idx.build_from_content("hello.txt", "hello world");
    let hits = idx.search_bytes([b'h', b'e', b'l']);
    assert!(!hits.is_empty());
}

#[test]
fn candidates() {
    let mut idx = TrigramIndex::new();
    idx.build_from_content("a.txt", "hello world");
    let docs = idx.candidates("hello");
    assert_eq!(docs.len(), 1);
}

#[test]
fn trigram_roundtrip() {
    let mut idx = TrigramIndex::new();
    idx.build_from_content("a.txt", "hello world");
    idx.build_from_content("b.txt", "goodbye world");
    let data = idx.serialize();
    let deser = TrigramIndex::deserialize(&data).unwrap();
    assert_eq!(deser.doc_count, 2);
    assert!(!deser.search_bytes([b'h', b'e', b'l']).is_empty());
}

#[test]
fn trigram_empty_index_roundtrip() {
    let idx = TrigramIndex::new();
    let data = idx.serialize();
    let deser = TrigramIndex::deserialize(&data).unwrap();
    assert_eq!(deser.doc_count, 0);
}

#[test]
fn trigram_deserialize_wrong_magic() {
    let data = &[0u8; 16];
    let result = TrigramIndex::deserialize(data);
    assert!(result.is_err());
}

#[test]
fn search_trigram_by_value() {
    let mut idx = TrigramIndex::new();
    idx.build_from_content("a.txt", "hello world");
    let tri = make_trigram(b'h', b'e', b'l');
    let hits = idx.search_trigram(tri);
    assert!(!hits.is_empty());
}

#[test]
fn search_delegates_to_candidates() {
    let mut idx = TrigramIndex::new();
    idx.build_from_content("a.txt", "hello world");
    let docs = idx.search("hello");
    assert_eq!(docs.len(), 1);
}

#[test]
fn candidates_short_query_returns_empty() {
    let idx = TrigramIndex::new();
    let docs = idx.candidates("ab");
    assert!(docs.is_empty());
}

#[test]
fn trigram_index_default_is_empty() {
    let idx = TrigramIndex::default();
    assert_eq!(idx.doc_count, 0);
}
