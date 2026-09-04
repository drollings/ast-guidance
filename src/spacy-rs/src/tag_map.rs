//! The fine-grained tag → UPOS derivation table (walkthrough §8.1; the
//! model-metadata tag map of §9.1).
//!
//! spaCy derives `token.pos` from `token.tag` through a per-language tag map
//! (in `en_core_web_sm` this ships as the attribute-ruler patterns that set
//! `POS` from `TAG`). A `TagMap` is the serde-able, parseable form of that
//! table: `TAG=POS` pairs, comma-joined. The English default is generated from
//! `en_core_web_sm` 3.8.0 (`lang/en/tag_map.rs`).
//!
//! `resolve` returns the coarse `Upos` for a tag — a pure, data-driven
//! derivation with no model in the loop, usable by any tag-bearing annotation
//! source (LLM `tag` fields, a future deterministic tagger, the router's
//! routing-signal extraction).

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::SpacyError;
use crate::labels::Upos;

/// A tag → UPOS map. `"$"` → `SYM`, `"NN"` → `NOUN`, `"NNP"` → `PROPN`, ...
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TagMap {
    entries: HashMap<String, Upos>,
}

impl TagMap {
    /// A map from `(tag, pos)` pairs, resolving each POS name through
    /// [`Upos::from_str`] (case-insensitive). Unknown POS names are an error —
    /// the tag map is closed data, never silently accepted.
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Result<Self, SpacyError> {
        let mut entries = HashMap::with_capacity(pairs.len());
        for (tag, pos) in pairs {
            let pos = pos.parse::<Upos>()?;
            entries.insert((*tag).to_string(), pos);
        }
        Ok(Self { entries })
    }

    /// The coarse UPOS for `tag`, or `None` when the tag is not in the map.
    #[must_use]
    pub fn get(&self, tag: &str) -> Option<Upos> {
        self.entries.get(tag).copied()
    }

    /// Insert (or overwrite) the mapping for `tag`.
    pub fn insert(&mut self, tag: impl Into<String>, pos: Upos) {
        self.entries.insert(tag.into(), pos);
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(tag, pos)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, Upos)> {
        self.entries.iter().map(|(t, p)| (t.as_str(), *p))
    }
}

impl FromStr for TagMap {
    type Err = SpacyError;

    /// Parse a comma-joined `TAG=POS` list, e.g. `"NN=NOUN,VBD=VERB,NNP=PROPN"`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let pairs: Vec<(&str, &str)> = s
            .split(',')
            .filter(|p| !p.is_empty())
            .map(|p| {
                p.split_once('=')
                    .ok_or_else(|| SpacyError::Annotation(format!("tag map entry missing '=': {p:?}")))
            })
            .collect::<Result<_, _>>()?;
        Self::from_pairs(&pairs)
    }
}

impl fmt::Display for TagMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tags: Vec<(&str, Upos)> = self.entries.iter().map(|(t, p)| (t.as_str(), *p)).collect();
        tags.sort_by(|a, b| a.0.cmp(b.0));
        let pairs: Vec<String> = tags
            .iter()
            .map(|(t, p)| format!("{t}={p}"))
            .collect();
        write!(f, "{}", pairs.join(","))
    }
}

#[cfg(test)]
#[path = "../tests/tag_map.rs"]
mod tests;
