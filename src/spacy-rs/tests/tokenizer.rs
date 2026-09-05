use super::*;
use crate::lang::en;

fn en_tokenizer() -> Tokenizer {
    let vocab = Arc::new(Vocab::new(en::lexicon_config()));
    en::tokenizer(vocab).expect("english tokenizer builds")
}

fn orths(doc: &Doc) -> Vec<String> {
    let strings = doc.vocab().strings();
    doc.tokens()
        .iter()
        .map(|t| t.lexeme.orth_text(strings))
        .collect()
}

#[test]
fn tokenizes_simple_sentence() {
    let t = en_tokenizer();
    let doc = t.tokenize("Hello world").expect("tokenizes");
    assert_eq!(orths(&doc), ["Hello", "world"]);
    assert_eq!(doc.tokens()[0].idx, 0);
    assert_eq!(doc.tokens()[1].idx, 6);
    assert!(doc.tokens()[0].spacy);
    assert!(!doc.tokens()[1].spacy);
    assert_eq!(doc.tokens()[0].sent_start, crate::doc::SentStart::Start);
}

#[test]
fn multiple_spaces_make_a_space_token() {
    let t = en_tokenizer();
    let doc = t.tokenize("a  b").expect("tokenizes");
    assert_eq!(orths(&doc), ["a", " ", "b"]);
    assert!(doc.tokens()[0].spacy);
}

#[test]
fn contraction_special_case() {
    let t = en_tokenizer();
    let doc = t.tokenize("I can't go").expect("tokenizes");
    assert_eq!(orths(&doc), ["I", "ca", "n't", "go"]);
    let nnt = &doc.tokens()[2];
    assert_eq!(nnt.norm, hash_utf8("not"), "special-case NORM override");
}

#[test]
fn hyphen_infix_splits() {
    let t = en_tokenizer();
    let doc = t.tokenize("dyn-o-mite").expect("tokenizes");
    assert_eq!(orths(&doc), ["dyn", "-", "o", "-", "mite"]);
}

#[test]
fn hyphen_dash_run_stays_one_token() {
    let t = en_tokenizer();
    let doc = t.tokenize("foo---bar").expect("tokenizes");
    assert_eq!(orths(&doc), ["foo", "---", "bar"]);
}

#[test]
fn leading_space_is_tokenized() {
    let t = en_tokenizer();
    let doc = t.tokenize(" hello").expect("tokenizes");
    assert_eq!(orths(&doc), [" ", "hello"]);
}

#[test]
fn trailing_single_space_is_absorbed() {
    let t = en_tokenizer();
    let doc = t.tokenize("hello ").expect("tokenizes");
    assert_eq!(orths(&doc), ["hello"]);
    assert!(doc.tokens()[0].spacy);
}

#[test]
fn newline_is_a_token() {
    let t = en_tokenizer();
    let doc = t.tokenize("hello\nworld").expect("tokenizes");
    assert_eq!(orths(&doc), ["hello", "\n", "world"]);
    assert!(!doc.tokens()[0].spacy, "newline is not absorbed as spacy");
    assert!(!doc.tokens()[1].spacy);
    assert!(!doc.tokens()[2].spacy);
    assert_eq!(doc.tokens()[1].idx, 5);
    assert_eq!(doc.tokens()[2].idx, 6);
}

#[test]
fn multi_token_matcher_special_case() {
    let t = en_tokenizer();
    let doc = t.tokenize("y'all are great").expect("tokenizes");
    assert_eq!(orths(&doc), ["y'", "all", "are", "great"]);
    assert_eq!(doc.tokens()[1].idx, 2, "in-span idx recomputed");
    assert_eq!(doc.tokens()[2].idx, 6, "post-span idx preserved");
}

#[test]
fn times_abbreviation_special_case() {
    let t = en_tokenizer();
    let doc = t.tokenize("at 5a.m. sharp").expect("tokenizes");
    assert_eq!(orths(&doc), ["at", "5", "a.m.", "sharp"]);
}

#[test]
fn empty_and_whitespace_text() {
    let t = en_tokenizer();
    assert_eq!(t.tokenize("").expect("empty").len(), 0);
    let doc = t.tokenize("   ").expect("spaces");
    assert_eq!(orths(&doc), ["   "]);
    let doc = t.tokenize("  a   ").expect("wrapped spaces");
    assert_eq!(orths(&doc), ["  ", "a", "  "]);
    assert!(doc.tokens()[1].spacy);
}

#[test]
fn filter_special_spans_prefers_longest() {
    // Overlapping spans where "cannot" (0,2) is longer than "can" (0,1):
    // longest-first resolution keeps the longer span and drops the covered
    // one. A same-span tie keeps the first-listed rule.
    let rule_long = Arc::new(SpecialRule {
        key: "cannot".into(),
        tokens: vec![],
        phrase: vec![hash_utf8("cannot")],
    });
    let rule_short = Arc::new(SpecialRule {
        key: "can".into(),
        tokens: vec![],
        phrase: vec![hash_utf8("can")],
    });
    let matches = vec![
        (Arc::clone(&rule_short), 0, 1),
        (Arc::clone(&rule_long), 0, 2),
    ];
    let filtered = filter_special_spans(matches);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].0.key, "cannot");
    assert_eq!((filtered[0].1, filtered[0].2), (0, 2));
}
