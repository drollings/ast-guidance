//! Language data: the deterministic inputs the tokenizer and the lexicon
//! consume per language — affix pattern strings, special-case rules, stop
//! words, and number words. Generated data lives in the `en` module (from the
//! pinned spaCy 3.8.15 checkout); the base `url` module holds the
//! language-independent `URL_MATCH` pattern.

pub mod en;
pub mod norm_exceptions;
pub mod url;

/// The base `NORM` exception for `text`, if any (`BASE_NORMS` — applied by
/// the Vocab to every language before the `lower` fallback,
/// `vocab.pyx:34-36`).
#[must_use]
pub fn base_norm(text: &str) -> Option<&'static str> {
    norm_exceptions::BASE_NORMS
        .iter()
        .find_map(|(k, v)| (*k == text).then_some(*v))
}
