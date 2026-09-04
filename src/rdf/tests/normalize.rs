use super::*;

#[test]
fn test_hash_iri_deterministic() {
    let iri = "http://example.org/foo";
    assert_eq!(hash_iri(iri), hash_iri(iri));
}

#[test]
fn test_hash_iri_different() {
    assert_ne!(
        hash_iri("http://example.org/foo"),
        hash_iri("http://example.org/bar")
    );
}

#[test]
fn test_hash_blank_node_deterministic() {
    assert_eq!(
        hash_blank_node("scope1", "b1"),
        hash_blank_node("scope1", "b1")
    );
}

#[test]
fn test_hash_blank_node_different_scopes() {
    assert_ne!(
        hash_blank_node("scope1", "b1"),
        hash_blank_node("scope2", "b1")
    );
}

#[test]
fn test_normalize_integer() {
    let tv = normalize_literal("42", None, Some(&format!("{XSD_NS}integer")));
    assert_eq!(tv, TypedValue::Integer(42));
}

#[test]
fn test_normalize_decimal() {
    let tv = normalize_literal("3.14", None, Some(&format!("{XSD_NS}decimal")));
    assert!(matches!(tv, TypedValue::Double(_)));
}

#[test]
fn test_normalize_boolean_true() {
    let tv = normalize_literal("true", None, Some(&format!("{XSD_NS}boolean")));
    assert_eq!(tv, TypedValue::Boolean(true));
}

#[test]
fn test_normalize_boolean_false() {
    let tv = normalize_literal("false", None, Some(&format!("{XSD_NS}boolean")));
    assert_eq!(tv, TypedValue::Boolean(false));
}

#[test]
fn test_normalize_lang_string() {
    let tv = normalize_literal("bonjour", Some("fr"), None);
    assert_eq!(tv, TypedValue::LangString);
}

#[test]
fn test_normalize_plain_string() {
    let tv = normalize_literal("hello", None, None);
    assert_eq!(tv, TypedValue::String);
}
