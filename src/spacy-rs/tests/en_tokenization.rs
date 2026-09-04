//! Golden tokenization test: replays the corpus fixture generated from the
//! pinned spaCy 3.8.15 (`tools/gen_golden_corpus.py`) through the native
//! English tokenizer and asserts byte-for-byte parity on token boundaries,
//! char idx, spacy flags, and the deterministic lexeme surface attributes.
//!
//! The fixture is committed and hermetic — no live inference.

use serde::Deserialize;
use std::sync::Arc;

use spacy_rs::lang::en;
use spacy_rs::vocab::Vocab;

#[derive(Debug, Deserialize)]
struct GoldenCase {
    text: String,
    tokens: Vec<GoldenToken>,
}

#[derive(Debug, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
struct GoldenToken {
    orth: String,
    idx: u32,
    spacy: bool,
    lower: String,
    shape: String,
    prefix: String,
    suffix: String,
    norm: String,
    is_alpha: bool,
    is_ascii: bool,
    is_digit: bool,
    is_lower: bool,
    is_punct: bool,
    is_space: bool,
    is_title: bool,
    is_upper: bool,
    like_url: bool,
    like_num: bool,
    like_email: bool,
    is_stop: bool,
    is_bracket: bool,
    is_quote: bool,
    is_left_punct: bool,
    is_right_punct: bool,
    is_currency: bool,
}

fn norm_of(strings: &spacy_rs::StringStore, lexeme_norm: u64, token_norm: u64) -> String {
    let hash = if token_norm != 0 {
        token_norm
    } else {
        lexeme_norm
    };
    strings.get(hash).map(|s| s.to_string()).unwrap_or_default()
}

#[test]
fn english_tokenizer_matches_spacy_golden_corpus() {
    let corpus: Vec<GoldenCase> =
        serde_json::from_str(include_str!("data/en_tokenization.json")).expect("fixture parses");
    assert!(corpus.len() > 100, "fixture has a meaningful corpus");
    let vocab = Arc::new(Vocab::new(en::lexicon_config()));
    let tokenizer = en::tokenizer(Arc::clone(&vocab)).expect("english tokenizer builds");
    let strings = vocab.strings();

    let mut checked_tokens = 0usize;
    for case in &corpus {
        let doc = tokenizer
            .tokenize(&case.text)
            .unwrap_or_else(|e| panic!("tokenize {case:?}: {e}"));
        assert_eq!(
            doc.len(),
            case.tokens.len(),
            "token count mismatch for text {:?}",
            case.text
        );
        for (i, (actual, expected)) in doc.tokens().iter().zip(&case.tokens).enumerate() {
            let ctx = format!("token[{i}] of text {:?}", case.text);
            assert_eq!(
                actual.lexeme.orth_text(strings),
                expected.orth,
                "orth {ctx}"
            );
            assert_eq!(actual.idx, expected.idx, "idx {ctx}");
            assert_eq!(actual.spacy, expected.spacy, "spacy {ctx}");
            assert_eq!(
                norm_of(strings, actual.lexeme.norm, actual.norm),
                expected.norm,
                "norm {ctx}"
            );
            let surface = |h: u64| strings.get(h).map(|s| s.to_string()).unwrap_or_default();
            assert_eq!(surface(actual.lexeme.lower), expected.lower, "lower {ctx}");
            assert_eq!(surface(actual.lexeme.shape), expected.shape, "shape {ctx}");
            assert_eq!(
                surface(actual.lexeme.prefix),
                expected.prefix,
                "prefix {ctx}"
            );
            assert_eq!(
                surface(actual.lexeme.suffix),
                expected.suffix,
                "suffix {ctx}"
            );
            let flags = actual.lexeme.flags;
            assert_eq!(flags.is_alpha(), expected.is_alpha, "is_alpha {ctx}");
            assert_eq!(flags.is_ascii(), expected.is_ascii, "is_ascii {ctx}");
            assert_eq!(flags.is_digit(), expected.is_digit, "is_digit {ctx}");
            assert_eq!(flags.is_lower(), expected.is_lower, "is_lower {ctx}");
            assert_eq!(flags.is_punct(), expected.is_punct, "is_punct {ctx}");
            assert_eq!(flags.is_space(), expected.is_space, "is_space {ctx}");
            assert_eq!(flags.is_title(), expected.is_title, "is_title {ctx}");
            assert_eq!(flags.is_upper(), expected.is_upper, "is_upper {ctx}");
            assert_eq!(flags.like_url(), expected.like_url, "like_url {ctx}");
            assert_eq!(flags.like_num(), expected.like_num, "like_num {ctx}");
            assert_eq!(flags.like_email(), expected.like_email, "like_email {ctx}");
            assert_eq!(flags.is_stop(), expected.is_stop, "is_stop {ctx}");
            assert_eq!(flags.is_bracket(), expected.is_bracket, "is_bracket {ctx}");
            assert_eq!(flags.is_quote(), expected.is_quote, "is_quote {ctx}");
            assert_eq!(
                flags.is_left_punct(),
                expected.is_left_punct,
                "is_left_punct {ctx}"
            );
            assert_eq!(
                flags.is_right_punct(),
                expected.is_right_punct,
                "is_right_punct {ctx}"
            );
            assert_eq!(
                flags.is_currency(),
                expected.is_currency,
                "is_currency {ctx}"
            );
            checked_tokens += 1;
        }
    }
    assert!(
        checked_tokens > 500,
        "checked a meaningful number of tokens"
    );
}

#[test]
fn cache_does_not_change_results() {
    // Tokenize the corpus twice; the second run must be byte-identical even
    // though the per-span cache is now warm (and the rules flushed it).
    let corpus: Vec<GoldenCase> =
        serde_json::from_str(include_str!("data/en_tokenization.json")).expect("fixture parses");
    let vocab = Arc::new(Vocab::new(en::lexicon_config()));
    let tokenizer = en::tokenizer(Arc::clone(&vocab)).expect("english tokenizer builds");
    let strings = vocab.strings();
    let run = |cases: &[GoldenCase]| -> Vec<(usize, Vec<String>)> {
        cases
            .iter()
            .map(|c| {
                let doc = tokenizer.tokenize(&c.text).expect("tokenizes");
                let orth = doc
                    .tokens()
                    .iter()
                    .map(|t| t.lexeme.orth_text(strings))
                    .collect();
                (doc.len(), orth)
            })
            .collect()
    };
    let first = run(&corpus);
    let second = run(&corpus);
    assert_eq!(first, second, "warm-cache tokenization must be identical");
}
