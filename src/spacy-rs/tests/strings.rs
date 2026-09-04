use super::*;

#[test]
fn add_returns_spacy_hash() {
    let store = StringStore::new();
    assert_eq!(store.add("hello"), hash_utf8("hello"));
    assert_eq!(store.add(""), 0, "empty string maps to hash 0");
}

#[test]
fn empty_string_is_always_present() {
    let store = StringStore::new();
    assert!(store.contains(""));
    assert_eq!(store.get(0).map(|s| s.to_string()), Some(String::new()));
}

#[test]
fn add_then_get_roundtrips() {
    let store = StringStore::new();
    let h = store.add("Apple");
    let back = store.get(h).expect("hash present");
    assert_eq!(&*back, "Apple");
}

#[test]
fn get_missing_returns_none() {
    let store = StringStore::new();
    assert!(store.get(hash_utf8("not-added")).is_none());
}

#[test]
fn add_is_idempotent_and_deduplicates() {
    let store = StringStore::new();
    let h1 = store.add("the");
    let h2 = store.add("the");
    assert_eq!(h1, h2);
    assert_eq!(store.len(), 1);
}

#[test]
fn lookup_does_not_intern() {
    let store = StringStore::new();
    assert_eq!(store.lookup("seen-once"), hash_utf8("seen-once"));
    assert!(!store.contains("seen-once"));
}

#[test]
fn contains_checks_hashes() {
    let store = StringStore::new();
    store.add("dog");
    assert!(store.contains("dog"));
    assert!(!store.contains("cat"));
}

#[test]
fn len_counts_distinct_nonempty() {
    let store = StringStore::new();
    store.add("a");
    store.add("a");
    store.add("b");
    assert_eq!(store.len(), 2);
}

#[test]
fn to_bytes_from_bytes_roundtrips_first_wins() {
    let store = StringStore::new();
    store.add("apple");
    store.add("banana");
    store.add("cherry");
    let bytes = store.to_bytes().expect("serialize");
    let reloaded = StringStore::from_bytes(&bytes).expect("deserialize");
    assert_eq!(reloaded.len(), 3);
    assert_eq!(reloaded.get(hash_utf8("apple")).map(|s| s.to_string()), Some("apple".into()));
    assert_eq!(reloaded.get(hash_utf8("banana")).map(|s| s.to_string()), Some("banana".into()));
    assert_eq!(reloaded.get(hash_utf8("cherry")).map(|s| s.to_string()), Some("cherry".into()));
}

#[test]
fn from_bytes_preserves_first_wins_on_duplicate_hash() {
    // A hand-edited blob with two canonicals claiming the same hash: the
    // first entry wins, exactly as in-memory interning does.
    let h = hash_utf8("first");
    let blob = serde_json::to_vec(&vec![(h, "first".to_string()), (h, "second".to_string())])
        .expect("blob");
    let store = StringStore::from_bytes(&blob).expect("deserialize");
    assert_eq!(store.len(), 1);
    assert_eq!(store.get(h).map(|s| s.to_string()), Some("first".into()));
}

#[test]
fn save_and_load_or_empty_roundtrip() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("strings.json");
    let store = StringStore::new();
    store.add("persisted");
    store.save(&path).expect("save");
    let reloaded = StringStore::load_or_empty(&path);
    assert_eq!(reloaded.len(), 1);
    assert_eq!(reloaded.get(hash_utf8("persisted")).map(|s| s.to_string()), Some("persisted".into()));
}

#[test]
fn load_or_empty_missing_file_is_empty() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let store = StringStore::load_or_empty(&dir.path().join("absent.json"));
    assert!(store.is_empty());
}
