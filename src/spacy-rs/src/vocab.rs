//! The vocabulary: the shared owner of the string store, the lexicon, and the
//! morphology table, mirroring spaCy's `Vocab` (`spacy/vocab.pyx`). Docs and
//! tokenizers hold an `Arc<Vocab>` so lexemes, interned strings, and
//! morphology analyses are shared.

use std::io;
use std::path::Path;
use std::sync::Arc;

use crate::lexeme::{Lexicon, LexiconConfig};
use crate::morph::Morphology;
use crate::strings::StringStore;

/// The vocabulary: string store + lexicon + morphology, shared across docs.
#[derive(Debug, Clone)]
pub struct Vocab {
    strings: Arc<StringStore>,
    lexicon: Arc<Lexicon>,
    morphology: Arc<Morphology>,
}

impl Vocab {
    /// A vocabulary with the given lexicon configuration.
    #[must_use]
    pub fn new(config: LexiconConfig) -> Self {
        let strings = Arc::new(StringStore::new());
        let morphology = Arc::new(Morphology::new(Arc::clone(&strings)));
        Self {
            strings: Arc::clone(&strings),
            lexicon: Arc::new(Lexicon::new(strings, config)),
            morphology,
        }
    }

    /// A vocabulary with an empty lexicon configuration.
    #[must_use]
    pub fn default_with_empty_config() -> Self {
        Self::new(LexiconConfig::default())
    }

    /// The shared string store.
    #[must_use]
    pub fn strings(&self) -> &Arc<StringStore> {
        &self.strings
    }

    /// The shared lexicon.
    #[must_use]
    pub fn lexicon(&self) -> &Arc<Lexicon> {
        &self.lexicon
    }

    /// The shared morphology table.
    #[must_use]
    pub fn morphology(&self) -> &Arc<Morphology> {
        &self.morphology
    }
}

impl Vocab {
    /// Persist the string store's `(hash, string)` reverse mapping to `path`.
    /// This is what survives restart: lexemes are recreated lazily on first
    /// `get_or_create`, but the hash→string (and thus hash→InterlinguaId)
    /// resolution is durable.
    pub fn save(&self, path: &Path) -> Result<(), io::Error> {
        self.strings.save(path)
    }

    /// A vocab whose string store is pre-loaded from `path` (or empty when
    /// absent). Built over `config` like [`Vocab::new`].
    #[must_use]
    pub fn load_or_empty(path: &Path, config: LexiconConfig) -> Self {
        let strings = Arc::new(StringStore::load_or_empty(path));
        let morphology = Arc::new(Morphology::new(Arc::clone(&strings)));
        Self {
            strings: Arc::clone(&strings),
            lexicon: Arc::new(Lexicon::new(strings, config)),
            morphology,
        }
    }
}

#[cfg(test)]
#[path = "../tests/vocab.rs"]
mod tests;
