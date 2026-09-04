use super::*;
use crate::hash_utf8;

#[test]
fn vocab_creates_shared_store_and_lexicon() {
    let vocab = Vocab::new(LexiconConfig::default());
    let lex = vocab.lexicon().get_or_create("hello");
    assert_eq!(lex.orth, vocab.strings().lookup("hello"));
}

#[test]
fn vocab_save_and_load_or_empty_roundtrip() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let path = dir.path().join("vocab.json");
    let vocab = Vocab::new(LexiconConfig::default());
    let _lex = vocab.lexicon().get_or_create("persisted_word");
    vocab.save(&path).expect("save");

    let reloaded = Vocab::load_or_empty(&path, LexiconConfig::default());
    // The reverse mapping survives restart.
    assert_eq!(
        reloaded.strings().get(hash_utf8("persisted_word")).map(|s| s.to_string()),
        Some("persisted_word".into())
    );
    // The fresh lexicon lazily rebuilds the lexeme from the loaded store.
    let lex = reloaded.lexicon().get_or_create("persisted_word");
    assert_eq!(lex.orth, reloaded.strings().lookup("persisted_word"));
}

#[test]
fn vocab_load_or_empty_missing_path_is_empty() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let vocab = Vocab::load_or_empty(&dir.path().join("absent.json"), LexiconConfig::default());
    assert!(vocab.strings().is_empty());
}
