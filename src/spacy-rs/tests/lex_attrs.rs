use super::*;

#[test]
fn word_shape_basic() {
    assert_eq!(word_shape("Apple"), "Xxxxx");
    assert_eq!(word_shape("hello"), "xxxx"); // 5-run collapses to 4
    assert_eq!(word_shape("123"), "ddd");
    assert_eq!(word_shape("dyn-o-mite"), "xxx-x-xxxx");
}

#[test]
fn word_shape_long() {
    assert_eq!(word_shape(&"a".repeat(100)), "LONG");
    assert_eq!(word_shape(&"a".repeat(99)), "xxxx");
}

#[test]
fn prefix_suffix() {
    assert_eq!(prefix("hello"), "h");
    assert_eq!(suffix("hello"), "llo");
    assert_eq!(prefix(""), "");
    assert_eq!(suffix(""), "");
    assert_eq!(suffix("ab"), "ab");
}

#[test]
fn category_flags() {
    assert!(is_alpha("hello"));
    assert!(is_alpha("éclair"));
    assert!(!is_alpha("123"));
    assert!(is_ascii("hello"));
    assert!(!is_ascii("éclair"));
    assert!(is_digit("123"));
    assert!(is_digit("²"));
    assert!(!is_digit("½"));
    assert!(is_lower("hello"));
    assert!(!is_lower("Hello"));
    assert!(!is_lower("123"));
    assert!(is_upper("HELLO"));
    assert!(!is_upper("Hello"));
    assert!(is_title("Hello World"));
    assert!(!is_title("Hello world"));
    assert!(!is_title("HELLO"));
    assert!(is_space(" \t\n"));
    assert!(!is_space("x"));
    assert!(is_punct("?!,"));
    assert!(is_punct("，"));
    assert!(!is_punct("a"));
    assert!(is_bracket("("));
    assert!(is_quote("\""));
    assert!(is_left_punct("("));
    assert!(is_right_punct(")"));
    assert!(is_currency("$"));
    assert!(!is_currency("$a"));
}

#[test]
fn like_num_forms() {
    assert!(like_num("123"));
    assert!(like_num("-42"));
    assert!(like_num("+42"));
    assert!(like_num("1,000"));
    assert!(like_num("3.5"));
    assert!(like_num("1/2"));
    assert!(!like_num("abc"));
    assert!(!like_num("1/2/3"));
}

#[test]
fn like_url_forms() {
    assert!(like_url("http://example.com"));
    assert!(like_url("https://example.org/x"));
    assert!(like_url("www.example.com"));
    assert!(like_url("example.com"));
    assert!(like_url("sub.example.co.uk"));
    assert!(!like_url("example"));
    assert!(!like_url("foo@bar.com"));
    assert!(!like_url(".example.com"));
}

#[test]
fn like_email_forms() {
    assert!(like_email("a@b.co"));
    assert!(like_email("first.last+tag@example-domain.com"));
    assert!(!like_email("nope"));
    assert!(!like_email("a@b"));
}
