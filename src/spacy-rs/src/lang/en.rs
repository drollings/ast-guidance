//! English language data — the port of `spacy/lang/en/__init__.py` and its
//! data modules: punctuation patterns, tokenizer exceptions, stop words, and
//! number words. Everything here is deterministic, version-pinned data
//! (generated from spaCy 3.8.15 by `tools/gen_en_regexes.py` and
//! `tools/gen_en_exceptions.py`).

pub mod exceptions;
pub mod function_words;
pub mod num_words;
pub mod patterns;
pub mod stop_words;
pub mod tag_map;

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::SpacyError;
use crate::hash::hash_utf8;
use crate::lexeme::LexiconConfig;
use crate::tag_map::TagMap;
use crate::tokenizer::{Tokenizer, TokenizerConfig};
use crate::vocab::Vocab;

/// The English tag → UPOS map, generated from `en_core_web_sm` 3.8.0.
#[must_use]
pub fn tag_map() -> TagMap {
    TagMap::from_pairs(tag_map::TAG_MAP).expect("generated tag map parses")
}

/// The versioned English lemma tables (rules/exc/index), compiled from
/// `../../env/en_lemmatizer.json` by `build.rs` into a compact binary blob.
pub static LEMMAS_BLOB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/en_lemmas.bin"));

/// The versioned English tagger orthography (morpheme suffixes, clause
/// punctuation, sentence boundaries), compiled from
/// `../../env/en_orthography.json` by `build.rs` into the `SOR1` blob the
/// ArcEager parser evaluates through [`crate::ortho::TaggerOrtho`] — never
/// as literals in the parser.
pub static ORTHO_BLOB: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/en_ortho.bin"));

/// The English `LexiconConfig`: `lang = "en"`, the stop-word set, the merged
/// cardinal + ordinal number words (English `LIKE_NUM`), no norm overrides,
/// and the closed-class function-word categories (attribute ids 19–47) the
/// parser matches as bits instead of hard-coded spellings.
#[must_use]
pub fn lexicon_config() -> LexiconConfig {
    LexiconConfig {
        lang: hash_utf8("en"),
        stop_words: stop_words::STOP_WORDS
            .iter()
            .map(|w| (*w).to_string())
            .collect(),
        norm_exceptions: HashMap::new(),
        num_words: num_words::NUM_WORDS
            .iter()
            .chain(num_words::ORDINAL_WORDS.iter())
            .map(|w| (*w).to_string())
            .collect(),
        function_words: function_words::function_word_bits(),
    }
}

/// The English `Tokenizer`: compiled affix patterns (base prefixes/suffixes,
/// English infixes), `token_match` disabled, `url_match = URL_MATCH`
/// (inherited from `BaseDefaults`), `faster_heuristics` on, default 10 000-slot
/// span cache.
pub fn tokenizer(vocab: Arc<Vocab>) -> Result<Tokenizer, SpacyError> {
    let config = TokenizerConfig {
        prefix_pattern: Some(patterns::PREFIX_PATTERN.to_string()),
        suffix_pattern: Some(patterns::SUFFIX_PATTERN.to_string()),
        infix_pattern: Some(patterns::INFIX_PATTERN.to_string()),
        token_match: None,
        url_match: Some(crate::lang::url::URL_PATTERN.to_string()),
        faster_heuristics: true,
        max_cache_size: 10_000,
    };
    Tokenizer::new(vocab, &config, exceptions::RULES)
}
