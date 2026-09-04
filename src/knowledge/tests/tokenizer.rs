use super::*;

#[test]
fn tokenizer_basic() {
    let tokens: Vec<&str> = WordTokenizer::new("hello world").collect();
    assert_eq!(tokens, vec!["hello", "world"]);
}

#[test]
fn tokenizer_with_symbols() {
    let tokens: Vec<&str> = WordTokenizer::new("hello, world! test").collect();
    assert_eq!(tokens, vec!["hello", "world", "test"]);
}

#[test]
fn split_identifier_snake_case() {
    let parts = split_identifier("hello_world");
    assert!(parts.contains(&"hello".to_string()));
    assert!(parts.contains(&"world".to_string()));
}

#[test]
fn split_identifier_camel_case() {
    let parts = split_identifier("helloWorld");
    assert!(parts.contains(&"hello".to_string()));
    assert!(parts.contains(&"World".to_string()));
}

#[test]
fn split_identifier_pascal_case() {
    let parts = split_identifier("HelloWorld");
    assert!(parts.contains(&"Hello".to_string()));
    assert!(parts.contains(&"World".to_string()));
}

#[test]
fn split_identifier_short_returns_empty() {
    let parts = split_identifier("a");
    assert!(parts.is_empty());
}

#[test]
fn normalize_char_cases() {
    assert_eq!(normalize_char('A'), 'a');
    assert_eq!(normalize_char('z'), 'z');
}
