//! LemmaView — compile-time embedded `SLM2` view (`include_bytes!` safe).
//! No `mmap`; `LazyLock<Arc<LemmaView>>`.
//! O(1) `exc_for` via `phf` would be ideal; fallback to binary search preserves correctness while `phf` codegen lands.

#![forbid(unsafe_code)]

use std::sync::{Arc, LazyLock};

use crate::lemma_blob::LemmaBlob;
use crate::error::SpacyError;

/// Trait per `common-core::blob_spec::LemmaView` (re-exported here for crate-local use without cross-crate import cycle).
pub trait LemmaView: Send + Sync {
    fn index_contains(&self, key: &str, word: &str) -> bool;
    fn exc_for(&self, key: &str, surface: &str) -> Option<&[u8]>;
    fn rules_for(&self, key: &str) -> &[(&'static str, &'static str)];
    fn pos_keys(&self) -> Vec<&'static str>;
}

impl LemmaView for LemmaBlob {
    fn index_contains(&self, key: &str, word: &str) -> bool {
        LemmaBlob::index_contains(self, key, word)
    }
    fn exc_for(&self, key: &str, surface: &str) -> Option<&[u8]> {
        LemmaBlob::exc_for(self, key, surface)
    }
    fn rules_for(&self, key: &str) -> &[(&'static str, &'static str)] {
        LemmaBlob::rules(self, key)
    }
    fn pos_keys(&self) -> Vec<&'static str> {
        LemmaBlob::pos_keys(self).collect()
    }
}

/// The embedded `SLM2` bytes (or legacy `SLM1` during rollout).
static EMBEDDED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/en_lemmas.bin"));

static VIEW: LazyLock<Arc<LemmaBlob>> = LazyLock::new(|| {
    Arc::new(LemmaBlob::from_bytes(EMBEDDED).expect("embedded lemma blob valid"))
});

/// Global lemma view.
#[must_use]
pub fn global_lemma_view() -> Arc<LemmaBlob> {
    Arc::clone(&VIEW)
}

/// Build a view from static bytes (for tests: `from_bytes(&'static [u8])`).
pub fn from_bytes(data: &'static [u8]) -> Result<LemmaBlob, SpacyError> {
    LemmaBlob::from_bytes(data)
}

#[cfg(test)]
#[path = "../tests/taxonomy_blob.rs"]
mod tests;
