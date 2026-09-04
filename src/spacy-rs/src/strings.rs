//! Bidirectional string ↔ hash store, mirroring spaCy's `StringStore`
//! (`spacy/strings.pyx`).
//!
//! Every string that enters a vocabulary is stored as its MurmurHash64A
//! (seed 1) hash (`hash_utf8`). The reverse mapping (hash → string) lives in
//! an interned store: values are `ArcIntern<str>` so that sharing a string
//! across N docs/lexemes costs a refcount bump rather than a copy, and
//! identical strings across the corpus deduplicate automatically. This is the
//! two-model interning the walkthrough recommends (§2.2): content-addressed
//! hashing for the *id*, `ArcIntern` for *storage*.
//!
//! Hash 0 is reserved for the empty string, exactly as in spaCy
//! (`strings.pyx:43`, `_intern_utf8` treats 0 as "missing").

use internment::ArcIntern;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::RwLock;

use crate::hash::hash_utf8;

/// Thread-safe `StringStore`: hash → interned string.
#[derive(Debug, Default)]
pub struct StringStore {
    by_hash: RwLock<HashMap<u64, ArcIntern<str>>>,
}

impl StringStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the hash of `s`, interning `s` (and materializing its
    /// reverse mapping) on first sight. The empty string maps to `0` and is
    /// never stored.
    pub fn add(&self, s: &str) -> u64 {
        if s.is_empty() {
            return 0;
        }
        let h = hash_utf8(s);
        self.intern(h, s);
        h
    }

    /// Pure content hash of `s` with no interning — spaCy's `__getitem__(str)`
    /// path (`strings.pyx:146-151`). Symbols are *not* consulted; closed
    /// vocabularies live in `crate::labels`, not here.
    pub fn lookup(&self, s: &str) -> u64 {
        hash_utf8(s)
    }

    /// The interned string for `hash`, or `None` if never added. Hash `0`
    /// resolves to the empty string, matching spaCy `__getitem__(0) → ""`.
    #[must_use]
    pub fn get(&self, hash: u64) -> Option<ArcIntern<str>> {
        if hash == 0 {
            return Some(ArcIntern::from(""));
        }
        self.by_hash
            .read()
            .expect("StringStore read lock poisoned")
            .get(&hash)
            .cloned()
    }

    /// Whether `s` has been added to the store (by hash).
    #[must_use]
    pub fn contains(&self, s: &str) -> bool {
        s.is_empty()
            || self
                .by_hash
                .read()
                .expect("StringStore read lock poisoned")
                .contains_key(&hash_utf8(s))
    }

    /// Number of distinct non-empty strings interned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_hash
            .read()
            .expect("StringStore read lock poisoned")
            .len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Intern `s` under its hash unless already present.
    fn intern(&self, hash: u64, s: &str) {
        let mut map = self
            .by_hash
            .write()
            .expect("StringStore write lock poisoned");
        map.entry(hash).or_insert_with(|| ArcIntern::from(s));
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Durable persistence (§5 — the interlingua bridge's reverse mapping must
// survive restart: lemma hashes have to resolve back to strings, and thus to
// InterlinguaIds, across process lifetimes).
// ─────────────────────────────────────────────────────────────────────────

impl StringStore {
    /// Serialize the interned `(hash, string)` pairs. The in-memory store is
    /// already first-wins (`entry().or_insert`), so hashes are unique here.
    ///
    /// Serialized as `String` (not `ArcIntern`) so no `internment` serde
    /// feature is needed for the blob.
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        let map = self.by_hash.read().expect("StringStore read lock poisoned");
        let entries: Vec<(u64, String)> = map
            .iter()
            .map(|(&h, s)| (h, s.to_string()))
            .collect();
        serde_json::to_vec(&entries)
    }

    /// Rebuild a store from a `to_bytes` blob, preserving **first-wins** on
    /// reload even if a hand-edited blob contains duplicate hashes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, serde_json::Error> {
        let entries: Vec<(u64, String)> = serde_json::from_slice(data)?;
        let store = Self::new();
        {
            let mut map = store.by_hash.write().expect("StringStore write lock poisoned");
            for (hash, s) in entries {
                map.entry(hash).or_insert_with(|| ArcIntern::from(s));
            }
        }
        Ok(store)
    }

    /// Atomically write the interned pairs to `path` (temp file + rename, so
    /// a crash mid-write never leaves a truncated blob).
    pub fn save(&self, path: &Path) -> Result<(), io::Error> {
        let bytes = self.to_bytes().map_err(io::Error::other)?;
        let tmp = path.with_extension("strings.tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load the store from `path`, or an empty store when the file is absent
    /// or unreadable. Missing files are the normal cold-start case.
    pub fn load_or_empty(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(data) => Self::from_bytes(&data).unwrap_or_default(),
            Err(_) => Self::new(),
        }
    }
}

#[cfg(test)]
#[path = "../tests/strings.rs"]
mod tests;
