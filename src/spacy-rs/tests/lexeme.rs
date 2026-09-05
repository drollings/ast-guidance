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
            function_words: HashMap::new(),
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
fn function_word_bits_match_categories_case_insensitively() {
    // The English category map behind `LexiconConfig::function_words`:
    // spot-check every bit, multi-category overlap (`after` is ADP +
    // SCONJ-word + adverbial-subordinator), and case-insensitivity.
    let strings = Arc::new(StringStore::new());
    let lex = Lexicon::new(
        strings,
        crate::lang::en::lexicon_config(),
    );
    let flags = |w: &str| lex.get_or_create(w).flags;
    assert!(flags("the").is_det_word());
    assert!(flags("of").is_adp_word());
    assert!(flags("is").is_aux_word() && flags("is").is_be_verb() && flags("is").is_verb_word());
    assert!(flags("and").is_cconj_word());
    assert!(flags("because").is_sconj_word() && flags("because").is_subord_complement());
    assert!(flags("when").is_subord_adverbial() && !flags("when").is_subord_complement());
    assert!(flags("it").is_pron_word() && flags("it").is_nominative());
    assert!(flags("me").is_pron_word() && !flags("me").is_nominative());
    assert!(flags("sat").is_verb_word() && !flags("sat").is_aux_word());
    assert!(flags("do").is_bare_inf_host() && flags("wo").is_bare_inf_host());
    assert!(flags("n't").is_negator() && flags("NOT").is_negator());
    assert!(flags("my").is_det_word() && flags("my").is_possessive() && flags("my").is_pron_word());
    assert!(flags("who").is_relativizer() && flags("who").is_nominative());
    assert!(flags("that").is_that_word() && flags("that").is_relativizer());
    assert!(flags("that").is_det_word() && !flags("that").is_pron_word());
    assert!(!flags("that").is_demonstrative());
    assert!(flags("these").is_demonstrative());
    assert!(flags("where").is_where_word() && flags("where").is_relativizer());
    assert!(flags("there").is_locative() && flags("there").is_there_word());
    assert!(flags("here").is_locative() && !flags("here").is_there_word());
    assert!(flags("smells").is_sensory_verb() && !flags("smells").is_epistemic_verb());
    assert!(flags("seems").is_epistemic_verb());
    assert!(flags("please").is_discourse_marker() && flags("please").is_please_word());
    assert!(flags("hard").is_adverb_word());
    assert!(flags("today").is_today_word() && !flags("today").is_adverb_word());
    assert!(flags("as").is_as_word() && flags("as").is_adp_word());
    assert!(flags("after").is_adp_word() && !flags("after").is_sconj_word() && flags("after").is_after_word() && flags("after").is_subord_adverbial());
    assert!(flags("twice").is_twice_word());
    assert!(flags("yet").is_yet_word() && flags("yet").is_cconj_word() && flags("yet").is_adverb_word());
    assert!(flags("'s").is_be_verb() && flags("'s").is_be_clitic() && flags("'s").is_be_clitic_s());
    assert!(flags("'re").is_aux_word() && flags("'re").is_be_clitic() && !flags("'re").is_be_clitic_s());
    assert!(flags("IS").is_be_verb() && flags("IS").is_aux_word());
    assert!(flags("After").is_after_word());
    // Open-class words gain no category bits.
    let cat = flags("cat");
    assert!(!cat.is_det_word() && !cat.is_verb_word() && !cat.is_be_verb() && !cat.is_adverb_word());
}

#[test]
fn norm_exceptions_override_lower() {
    let lex = lexicon();
    let nt = lex.get_or_create("n't");
    assert_eq!(nt.norm, hash_utf8("not"));
}
