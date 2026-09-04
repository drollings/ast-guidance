//! The versioned canonical-form table (ROADMAP_20260827_ORT §6.3).
//!
//! Predicate canonicalization maps surface verb lemmas to a canonical lemma so
//! the content-addressed predicate id is stable across inflection ("ran",
//! "running", "runs" → "run"). The mapping is **built offline** (clustering of
//! verb-lemma embeddings) and **shipped as versioned data**, loaded like the
//! lemmatizer blob and consulted with a plain lookup on the hot path. It never
//! makes an inference call — the "ids are pure functions of content" contract
//! stays intact.
//!
//! The table is immutable once loaded (data-time, never runtime registration).
//! Lookup is first-wins against the baked map; an absent surface lemma is its
//! own canonical form (identity — never a guess, never a model call).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The YaGO `Entity` reference class IRI — the entity-link overlay's
/// `is_subclass_of` parent (M6.2): a candidate must be a subclass of this to
/// count as a genuine entity.
pub const ENTITY_ROOT_IRI: &str = "http://yago-knowledge.org/resource/Entity";

/// Errors surfaced while loading/parsing a canonical-form table.
#[derive(Debug, Error)]
pub enum CanonicalFormError {
    #[error("invalid canonical-form JSON: {0}")]
    Json(String),
}

/// A versioned `surface lemma → canonical lemma` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalFormTable {
    /// The table version (the offline build that produced it).
    version: String,
    /// Surface lemma → canonical lemma. Built once; never mutated at runtime.
    map: HashMap<String, String>,
}

impl CanonicalFormTable {
    /// An empty table (no canonicalization; every lemma is its own canonical).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            version: "0".into(),
            map: HashMap::new(),
        }
    }

    /// Build a table from versioned `(surface, canonical)` entries. First-wins:
    /// a duplicated surface keeps its first canonical.
    #[must_use]
    pub fn new(
        version: impl Into<String>,
        entries: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        let mut map = HashMap::new();
        for (surface, canonical) in entries {
            map.entry(surface).or_insert(canonical);
        }
        Self {
            version: version.into(),
            map,
        }
    }

    /// Parse a versioned JSON document of the shape
    /// `{"version": "1.0", "map": {"runs": "run", "ran": "run"}}`.
    ///
    /// M2: intentionally a pristine `from_str`, NOT the tolerant LLM codec.
    /// This loads a shipped, versioned data file (our own offline-built
    /// format) — never LLM-produced text. Widening acceptance here (fences,
    /// prose, repair) would silently bless corrupt data files instead of
    /// failing loudly at load.
    pub fn from_json(json: &str) -> Result<Self, CanonicalFormError> {
        serde_json::from_str(json).map_err(|e| CanonicalFormError::Json(e.to_string()))
    }

    /// The table version.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Pure lookup: the canonical form for a surface lemma, or the surface
    /// lemma itself when absent (identity). Never a model call, never a write.
    #[must_use]
    pub fn canonical<'a>(&'a self, surface: &'a str) -> &'a str {
        self.map.get(surface).map_or(surface, String::as_str)
    }

    /// Number of baked surface→canonical entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The baked entries (diagnostics/tests).
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}
#[cfg(test)]
#[path = "../../tests/overlay_canonical.rs"]
mod tests;
