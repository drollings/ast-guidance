use super::*;

#[test]
fn basic_index_and_search() {
    let mut wi = WordIndex::new();
    wi.index_file("test.txt", "hello world");
    let hits = wi.search("hello");
    assert!(!hits.is_empty());
    assert_eq!(wi.hit_path(hits[0]), "test.txt");
}

#[test]
fn doc_registry_roundtrip() {
    let mut reg = DocRegistry::new();
    let id = reg.get_or_create("/path/to/file.rs");
    assert_eq!(reg.path_for_id(id), "/path/to/file.rs");
}

#[test]
fn search_prefix() {
    let mut wi = WordIndex::new();
    wi.index_file("a.txt", "hello world");
    let hits = wi.search_prefix("hel");
    assert!(!hits.is_empty());
}

#[test]
fn remove_file() {
    let mut wi = WordIndex::new();
    wi.index_file("a.txt", "hello world");
    wi.remove_file("a.txt");
    assert!(wi.search("hello").is_empty());
}
