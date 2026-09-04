//! The morphology table (walkthrough §8.4): interning of normalized UFEATS
//! morphological analyses, mirroring `spacy/morphology.pyx`.
//!
//! A token stores only a `u64` key — the hash of its normalized UFEATS string
//! (the same MurmurHash64A the `StringStore` uses) — and the table resolves
//! the key back to a canonical string or feature dict. The separators and the
//! empty-morph placeholder are exact (`morphology.pyx:22-26`): fields join on
//! `|`, field and value on `=`, multi-values on `,`, and the empty analysis is
//! the placeholder `"_"`.
//!
//! Normalization (`normalize_features`/`normalize_attrs`) is the parity
//! contract: sorted fields, sorted multi-values, `POS` values canonicalized to
//! the UPOS name. Two analyses that differ only in feature order intern to the
//! same key.

use std::collections::HashMap;
use std::sync::Arc;

use crate::hash::hash_utf8;
use crate::labels::Upos;
use crate::strings::StringStore;

/// Feature separator between `Field=Value` pairs (`FEATURE_SEP`, `"|"`).
pub const FEATURE_SEP: char = '|';
/// Separator between a field and its value (`FIELD_SEP`, `"="`).
pub const FIELD_SEP: char = '=';
/// Separator between multiple values of one field (`VALUE_SEP`, `","`).
pub const VALUE_SEP: char = ',';
/// The canonical empty-morph placeholder, distinct from an unset morph
/// (`EMPTY_MORPH`, `"_"`).
pub const EMPTY_MORPH: &str = "_";

/// The morphology table: a typed view over the shared `StringStore` whose
/// content-addressed keys are the hashes of normalized UFEATS strings. Add an
/// analysis with [`Morphology::add`]; resolve a key with
/// [`Morphology::get`] / [`Morphology::to_dict`] / [`Morphology::get_by_field`].
#[derive(Debug, Clone)]
pub struct Morphology {
    strings: Arc<StringStore>,
}

impl Morphology {
    /// A table over `strings` (shared with the vocab it belongs to).
    #[must_use]
    pub fn new(strings: Arc<StringStore>) -> Self {
        Self { strings }
    }

    /// Insert a morphological analysis from a UFEATS string and return its
    /// interned key (`morphology.pyx:38-75`). The empty string normalizes to
    /// the `"_"` placeholder. Idempotent: the same analysis always yields the
    /// same key.
    pub fn add(&self, features: &str) -> u64 {
        let normalized = self.normalize_features(features);
        self.strings.add(&normalized)
    }

    /// Insert an analysis from a `Field -> Value` dict (values may contain
    /// `,`-separated multi-values) and return its interned key.
    pub fn add_dict(&self, features: &HashMap<String, String>) -> u64 {
        let normalized = Self::dict_to_feats(features);
        self.strings.add(&normalized)
    }

    /// The interned key for the canonical empty analysis (`"_"`).
    #[must_use]
    pub fn empty_key(&self) -> u64 {
        self.strings.add(EMPTY_MORPH)
    }

    /// Normalize a UFEATS string (or the `"_"` placeholder) to the canonical
    /// form: fields sorted, multi-values sorted, `POS` canonicalized
    /// (`morphology.pyx:77-96`).
    #[must_use]
    pub fn normalize_features(&self, features: &str) -> String {
        if features.is_empty() || features == EMPTY_MORPH {
            return EMPTY_MORPH.to_string();
        }
        let dict = Self::feats_to_dict(features);
        let normalized = Self::normalize_attrs(&dict);
        Self::dict_to_feats(&normalized)
    }

    /// Parse a UFEATS string into a `Field -> Value` dict, sorting each
    /// field's `,`-separated values (`Morphology.feats_to_dict`,
    /// `morphology.pyx:156-161`).
    #[must_use]
    pub fn feats_to_dict(feats: &str) -> HashMap<String, String> {
        let mut out = HashMap::new();
        if feats.is_empty() || feats == EMPTY_MORPH {
            return out;
        }
        for feat in feats.split(FEATURE_SEP) {
            let Some((field, values)) = feat.split_once(FIELD_SEP) else {
                continue;
            };
            let mut vals: Vec<&str> = values.split(VALUE_SEP).collect();
            vals.sort_unstable();
            out.insert(field.to_string(), vals.join(","));
        }
        out
    }

    /// Render a `Field -> Value` dict as a UFEATS string with sorted fields
    /// and sorted multi-values (`Morphology.dict_to_feats`,
    /// `morphology.pyx:163-167`).
    #[must_use]
    pub fn dict_to_feats(feats: &HashMap<String, String>) -> String {
        if feats.is_empty() {
            return EMPTY_MORPH.to_string();
        }
        let mut fields: Vec<&String> = feats.keys().collect();
        fields.sort();
        fields
            .iter()
            .map(|field| {
                let mut vals: Vec<&str> = feats[*field].split(VALUE_SEP).collect();
                vals.sort_unstable();
                format!("{field}{FIELD_SEP}{}", vals.join(","))
            })
            .collect::<Vec<_>>()
            .join(&FEATURE_SEP.to_string())
    }

    /// Normalize a feature dict: `POS` values canonicalized to the UPOS name
    /// (uppercase), every other field's values kept as strings with
    /// multi-values sorted (`Morphology.normalize_attrs`,
    /// `morphology.pyx:98-123`).
    #[must_use]
    pub fn normalize_attrs(feats: &HashMap<String, String>) -> HashMap<String, String> {
        let mut out = HashMap::new();
        for (field, value) in feats {
            let upper_field = field.to_ascii_uppercase();
            if upper_field == "POS" {
                let canonical = value
                    .parse::<Upos>()
                    .map_or_else(|_| value.clone(), |u| u.to_string().to_ascii_uppercase());
                out.insert(field.clone(), canonical);
                continue;
            }
            let mut vals: Vec<&str> = value.split(VALUE_SEP).collect();
            vals.sort_unstable();
            out.insert(field.clone(), vals.join(","));
        }
        out
    }

    /// The canonical UFEATS string for an interned key, or `None` when the key
    /// was never interned here.
    #[must_use]
    pub fn get(&self, key: u64) -> Option<String> {
        self.strings.get(key).map(|s| s.to_string())
    }

    /// The feature dict for an interned key.
    #[must_use]
    pub fn to_dict(&self, key: u64) -> Option<HashMap<String, String>> {
        self.get(key).map(|s| Self::feats_to_dict(&s))
    }

    /// Whether the analysis at `key` has the feature `"Field=Value"` (e.g.
    /// `"Number=Sing"`); `false` for an uninterned key.
    #[must_use]
    pub fn has_feature(&self, key: u64, feature: &str) -> bool {
        let Some((field, value)) = feature.split_once(FIELD_SEP) else {
            return false;
        };
        self.to_dict(key).is_some_and(|dict| {
            dict.get(field)
                .is_some_and(|values| values.split(VALUE_SEP).any(|v| v == value))
        })
    }

    /// Every value of `field` across the analysis at `key` (e.g. `"Number"`
    /// -> `["Sing"]`); empty for an uninterned key or absent field.
    #[must_use]
    pub fn get_by_field(&self, key: u64, field: &str) -> Vec<String> {
        self.to_dict(key)
            .and_then(|dict| dict.get(field).cloned())
            .map(|values| {
                values
                    .split(VALUE_SEP)
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Convenience for tests and callers that hold a `Morphology` in an `Arc`.
pub type ArcMorphology = Arc<Morphology>;

/// The hash of the canonical empty morph (`"_"`).
#[must_use]
pub fn empty_morph_hash() -> u64 {
    hash_utf8(EMPTY_MORPH)
}

#[cfg(test)]
#[path = "../tests/morph.rs"]
mod tests;
