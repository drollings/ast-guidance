use super::*;
use crate::lexeme::LexiconConfig;
use crate::tokenizer::{Tokenizer, TokenizerConfig};
use crate::vocab::Vocab;
use std::sync::Arc;

fn tokenizer() -> Tokenizer {
    let vocab = Arc::new(Vocab::new(LexiconConfig::default()));
    let config = TokenizerConfig {
        // ASCII plus the fullwidth sentence-enders used by the unicode test.
        prefix_pattern: Some("^[.,!?;:。？]".to_string()),
        suffix_pattern: Some("[.,!?;:。？]$".to_string()),
        infix_pattern: None,
        token_match: None,
        url_match: None,
        faster_heuristics: true,
        max_cache_size: 10,
    };
    Tokenizer::new(vocab, &config, &[]).expect("tokenizer")
}

fn doc(text: &str) -> Doc {
    tokenizer().tokenize(text).expect("tokenize")
}

#[test]
fn default_punct_chars_is_the_spacy_set() {
    let chars = Sentencizer::default_punct_chars();
    assert_eq!(chars.len(), 128);
    assert!(chars.contains(&'!'));
    assert!(chars.contains(&'.'));
    assert!(chars.contains(&'？'));
    assert!(chars.contains(&'。'));
}

#[test]
fn single_sentence_is_one_start() {
    let s = Sentencizer::new();
    let d = doc("The cat sat.");
    let g = s.predict(&d);
    assert_eq!(g, vec![true, false, false, false]);
}

#[test]
fn multi_sentence_boundaries() {
    let s = Sentencizer::new();
    let d = doc("Hello world. How are you? Fine!");
    let g = s.predict(&d);
    // Hello world. | How are you? | Fine!
    assert_eq!(g, vec![true, false, false, true, false, false, false, true, false]);
}

#[test]
fn ellipsis_and_abbreviation_runs() {
    let s = Sentencizer::new();
    let d = doc("Wait... really?");
    let g = s.predict(&d);
    // "Wait" "." "." "." "really" "?" → the token after the "..." run
    // starts a new sentence; the trailing "?" does not.
    assert_eq!(g, vec![true, false, false, false, true, false]);
}

#[test]
fn set_annotations_writes_tristate_and_honors_existing() {
    let s = Sentencizer::new();
    let mut d = doc("A b. C");
    s.set_annotations(&mut d, &[true, false, false, true]);
    assert_eq!(d.token(0).sent_start, SentStart::Start);
    assert_eq!(d.token(1).sent_start, SentStart::NotStart);
    assert_eq!(d.token(3).sent_start, SentStart::Start);

    // Existing annotations are preserved unless overwrite.
    let mut d2 = doc("A b. C");
    d2.token_mut(1).sent_start = SentStart::Start;
    s.set_annotations(&mut d2, &[true, false, false, true]);
    assert_eq!(d2.token(1).sent_start, SentStart::Start, "preserved");
    let mut s2 = Sentencizer::new();
    s2.set_overwrite(true);
    s2.set_annotations(&mut d2, &[true, false, false, true]);
    assert_eq!(d2.token(1).sent_start, SentStart::NotStart, "overwritten");
}

#[test]
fn process_is_predict_then_attach() {
    let s = Sentencizer::new();
    let mut d = doc("A b. C");
    s.process(&mut d);
    assert_eq!(d.token(0).sent_start, SentStart::Start);
    assert_eq!(d.token(1).sent_start, SentStart::NotStart);
    assert_eq!(d.token(2).sent_start, SentStart::NotStart);
    assert_eq!(d.token(3).sent_start, SentStart::Start);
}

#[test]
fn empty_doc_predicts_empty() {
    let s = Sentencizer::new();
    let d = Doc::new(Arc::new(Vocab::new(LexiconConfig::default())));
    assert!(s.predict(&d).is_empty());
}

#[test]
fn unicode_punct_from_default_set() {
    let s = Sentencizer::new();
    let d = doc("Bonjour le monde 。 Comment ça va ？");
    let g = s.predict(&d);
    assert_eq!(g[0], true);
    // the token after the fullwidth 。 starts a new sentence
    assert!(g.iter().filter(|&&b| b).count() == 2);
}
