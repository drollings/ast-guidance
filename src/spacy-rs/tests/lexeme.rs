use super::*;
use crate::hash::hash_utf8;

fn lexicon() -> Lexicon {
    let strings = Arc::new(StringStore::new());
    let mut stop_words = HashSet::new();
    stop_words.insert("the".to_string());
    let mut norm_exceptions = HashMap::new();
    norm_exceptions.insert("n't".to_string(), "not".to_string());
    let lang = strings.lookup("en");
    Lexicon::new(
        strings,
        LexiconConfig {
            lang,
            stop_words,
            norm_exceptions,
            num_words: HashSet::new(),
        },
    )
}

#[test]
fn get_or_create_interning_is_deduplicated() {
    let lex = lexicon();
    let a = lex.get_or_create("apple");
    let b = lex.get_or_create("apple");
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(lex.len(), 1);
}

#[test]
fn empty_string_returns_empty_lexeme() {
    let lex = lexicon();
    let empty = lex.get_or_create("");
    assert_eq!(empty.orth, 0);
    assert_eq!(empty.id, OOV_RANK);
    assert_eq!(lex.len(), 0);
}

#[test]
fn lexeme_attr_hashes() {
    let lex = lexicon();
    let apple = lex.get_or_create("Apple");
    assert_eq!(apple.orth, hash_utf8("Apple"));
    assert_eq!(apple.lower, hash_utf8("apple"));
    assert_eq!(apple.shape, hash_utf8("Xxxxx"));
    assert_eq!(apple.prefix, hash_utf8("A"));
    assert_eq!(apple.suffix, hash_utf8("ple"));
    assert_eq!(apple.length, 5);
    assert_eq!(apple.lang, hash_utf8("en"));
}

#[test]
fn flags_computed_deterministically() {
    let lex = lexicon();
    let apple = lex.get_or_create("Apple");
    assert!(apple.flags.is_alpha());
    assert!(!apple.flags.is_digit());
    assert!(apple.flags.is_title());
    assert!(!apple.flags.is_lower());
    assert!(!apple.flags.is_upper());
    assert!(lex.get_or_create("hello").flags.is_lower());
    assert!(lex.get_or_create("123").flags.is_digit());
    assert!(lex.get_or_create("!").flags.is_punct());
}

#[test]
fn stop_word_flag_matches_lowercased_orth() {
    let lex = lexicon();
    assert!(lex.get_or_create("the").flags.is_stop());
    assert!(lex.get_or_create("The").flags.is_stop());
    assert!(!lex.get_or_create("cat").flags.is_stop());
}

#[test]
fn norm_exceptions_override_lower() {
    let lex = lexicon();
    let nt = lex.get_or_create("n't");
    assert_eq!(nt.norm, hash_utf8("not"));
}
