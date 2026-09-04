//! Language data: the deterministic inputs the tokenizer and the lexicon
//! consume per language — affix pattern strings, special-case rules, stop
//! words, and number words. Generated data lives in the `en` module (from the
//! pinned spaCy 3.8.15 checkout); the base `url` module holds the
//! language-independent `URL_MATCH` pattern.
//!
//! # Version-pinned rule-data audit surface (M2c)
//!
//! All "deterministic data that grows by promotion" is audited from this
//! tree plus the lemma-blob pipeline:
//!
//! - Tokenizer exceptions (`en::exceptions`, `norm_exceptions`) —
//!   longest-first special-case rules compiled into the tokenizer.
//! - Lemma tables (`crate::lemma_blob` + `build.rs` over
//!   `env/en_lemmatizer.json`) — versioned binary blob (`SLM2`).
//! - POS/NER promotions ([`genesis`]) — the only surface that grows at
//!   runtime: recurring refiner corrections promote to permanent
//!   deterministic data at the POS 3 / NER 5 thresholds. The parser consults
//!   it read-only through the [`GenesisIndex`](genesis::GenesisIndex) seam.

pub mod en;
pub mod genesis;
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
