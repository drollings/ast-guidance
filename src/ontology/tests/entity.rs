use super::*;

#[test]
fn extracts_capitalized_words() {
    let words = candidate_entity_words("Alice met Bob at Google");
    assert!(words.contains(&"Alice".to_string()));
    assert!(words.contains(&"Bob".to_string()));
    assert!(words.contains(&"Google".to_string()));
}

#[test]
fn filters_stoplist() {
    let words = candidate_entity_words("The quick brown Fox");
    assert!(!words.contains(&"The".to_string()));
    assert!(words.contains(&"Fox".to_string()));
}

#[test]
fn extract_entities_with_min_frequency() {
    let text = "Alice and Alice and Bob and Alice";
    let entities = extract_entities(text, 2);
    let alice = entities.iter().find(|e| e.name == "Alice").unwrap();
    assert_eq!(alice.frequency, 3);
}
