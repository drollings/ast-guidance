//! Rule genesis for POS/NER (ROADMAP_20260831_ARCEAGER M6.2).
//!
//! The tokenizer's "rule genesis" pattern (LLM proposes → golden corpus
//! accepts → data absorbs as a version-pinned deterministic rule) is extended
//! from lexical boundaries to POS/NER. A refiner's correction on a recurring
//! pattern becomes permanent, version-pinned deterministic data — a standing
//! model cost becomes a committed rule.
//!
//! # Tenancy (M2c)
//!
//! This module is the version-pinned rule-data home for promoted POS/NER,
//! alongside the tokenizer exceptions (`en::exceptions`, `norm_exceptions`)
//! and the lemma-blob pipeline (`crate::lemma_blob` + `build.rs`) — one
//! audit surface for all "deterministic data that grows by promotion"
//! (see [`crate::lang`] docs). The promotion store (trait + persistence)
//! lives here; the parser keeps only consultation
//! (`RuleAnnotator::annotate` reads promoted entries through the
//! [`GenesisIndex`] seam).
//!
//! # Design
//!
//! * Evidence is a [`Correction`](crate::review::Correction) whose `field` is
//!   `Pos` or `Ner`. The store counts evidence per normalized orth
//!   (`to_ascii_lowercase`). POS and NER have **separate thresholds and
//!   counters**: POS promotes at `threshold` (default 3), NER at
//!   `ner_threshold` (default 5). Entity type is context-variant (the same
//!   surface "Washington"/"Jordan"/"Paris" can be different entity types
//!   depending on context), so the NER bar is substantially higher than POS.
//! * Promotion is monotonic — a promoted entry never demotes — mirroring the
//!   tokenizer's first-wins special-case table.
//! * The in-memory store is the hermetic test double. A file-backed store
//!   (`GenesisStore::load_or_empty` / `save`) is the version-pinned
//!   persistence (a `HashMap<String, (Upos, u32)>` JSON blob, analogous to the
//!   `StringStore` durability in `pipeline.rs` M7.8). The file path is supplied
//!   by the caller (router), not hard-coded.
//! * The production ledger view is the same file (or a sqlite view over the
//!   `genesis_pos` table) — no parallel in-memory index beyond the loaded
//!   file. The trait hides the backend.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::labels::{NerType, Upos};
use crate::review::{Correction, CorrectionField};

pub const DEFAULT_POS_THRESHOLD: u32 = 3;
pub const DEFAULT_NER_THRESHOLD: u32 = 5;

/// A normalized orth → POS/NER genesis entry.
///
/// `count` is POS evidence count (kept for backward compat). `ner_count`
/// and `ner_promoted` track NER separately so POS evidence does not
/// accelerate NER promotion — entity type is context-variant, so the NER
/// bar is higher and isolated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenesisEntry {
    /// The promoted POS (when `promoted`).
    pub pos: Option<String>,
    /// The promoted NER type (when `ner_promoted`).
    pub ner: Option<String>,
    /// POS evidence count (legacy `count` — kept for compat).
    pub count: u32,
    /// NER evidence count.
    #[serde(default)]
    pub ner_count: u32,
    /// Whether POS threshold has been reached.
    pub promoted: bool,
    /// Whether NER threshold has been reached.
    #[serde(default)]
    pub ner_promoted: bool,
}

/// The genesis index trait — object-safe so the pipeline can hold
/// `Option<Arc<dyn GenesisIndex>>`.
///
/// `Send + Sync` so the async ladder (shared across `ResultPool` workers)
/// can share one instance.
pub trait GenesisIndex: Send + Sync {
    /// The promoted POS for `normalized` (`to_ascii_lowercase`), if any.
    fn get_pos(&self, normalized: &str) -> Option<Upos>;
    /// The promoted NER type for `normalized`, if any.
    fn get_ner(&self, normalized: &str) -> Option<NerType>;
    /// Record one correction as evidence. When `count >= threshold`, the entry
    /// is promoted and future `get_*` calls return the value.
    fn record(&self, correction: &Correction, normalized: &str);
    /// How many evidence writes have landed for `normalized`.
    fn count_for(&self, normalized: &str) -> u32;
    /// Whether `normalized` is promoted.
    fn is_promoted(&self, normalized: &str) -> bool;
    /// Number of promoted entries (for metrics/tests).
    fn promoted_len(&self) -> usize;
    /// Total evidence entries.
    fn len(&self) -> usize;
    /// Persist to `path` (JSON blob — version-pinned data).
    fn save(&self, path: &Path) -> Result<(), std::io::Error>;
}

/// Hermetic in-memory genesis index (tests + single-process callers).
///
/// `Mutex<HashMap>` — genesis writes are rare (refiner corrections), reads are
/// on the deterministic path (one per token), so a single lock is adequate.
pub struct InMemoryGenesisIndex {
    /// normalized orth → (pos, ner, count, promoted)
    map: Mutex<HashMap<String, GenesisEntry>>,
    threshold: u32,
    ner_threshold: u32,
}

impl std::fmt::Debug for InMemoryGenesisIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let map = self.map.lock().expect("genesis lock");
        f.debug_struct("InMemoryGenesisIndex")
            .field("len", &map.len())
            .field("threshold", &self.threshold)
            .field("ner_threshold", &self.ner_threshold)
            .finish()
    }
}

impl InMemoryGenesisIndex {
    /// A new index with `threshold` evidence required for promotion.
    /// For tests, both POS and NER thresholds are set to the same value so
    /// `with_threshold(3)` keeps the existing NER happy-path promotion at 3.
    #[must_use]
    pub fn with_threshold(threshold: u32) -> Self {
        Self::with_thresholds(threshold, threshold)
    }

    /// A new index with distinct POS / NER thresholds.
    #[must_use]
    pub fn with_thresholds(pos_threshold: u32, ner_threshold: u32) -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            threshold: pos_threshold.max(1),
            ner_threshold: ner_threshold.max(1),
        }
    }

    /// A new index with the default promotion thresholds (POS 3, NER 5).
    #[must_use]
    pub fn new() -> Self {
        Self::with_thresholds(DEFAULT_POS_THRESHOLD, DEFAULT_NER_THRESHOLD)
    }

    /// Load from `path` if it exists, otherwise empty (M7.8 durability
    /// pattern, mirroring `Vocab::load_or_empty`). Production defaults
    /// (POS 3, NER 5) are used.
    #[must_use]
    pub fn load_or_empty(path: &Path) -> Self {
        Self::load_or_empty_with_thresholds(path, DEFAULT_POS_THRESHOLD, DEFAULT_NER_THRESHOLD)
    }

    /// Load with a custom POS threshold (NER threshold mirrors it — test helper).
    #[must_use]
    pub fn load_or_empty_with_threshold(path: &Path, threshold: u32) -> Self {
        Self::load_or_empty_with_thresholds(path, threshold, threshold)
    }

    /// Load with distinct POS / NER thresholds.
    #[must_use]
    pub fn load_or_empty_with_thresholds(path: &Path, pos_threshold: u32, ner_threshold: u32) -> Self {
        let mut map: HashMap<String, GenesisEntry> = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, GenesisEntry>>(&s).ok())
            .unwrap_or_default();
        // Backward compat: old files stored NER evidence in `count`/`promoted`
        // with `ner_count`/`ner_promoted` missing. Migrate them so an old
        // NER-promoted entry remains promoted after upgrade.
        for e in map.values_mut() {
            if e.ner.is_some() && e.promoted && !e.ner_promoted {
                e.ner_promoted = true;
                if e.ner_count == 0 {
                    e.ner_count = e.count;
                }
            }
        }
        Self {
            map: Mutex::new(map),
            threshold: pos_threshold.max(1),
            ner_threshold: ner_threshold.max(1),
        }
    }
}

impl Default for InMemoryGenesisIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl GenesisIndex for InMemoryGenesisIndex {
    fn get_pos(&self, normalized: &str) -> Option<Upos> {
        let map = self.map.lock().expect("genesis lock");
        let e = map.get(normalized)?;
        if !e.promoted {
            return None;
        }
        let s = e.pos.as_deref()?;
        s.parse().ok()
    }

    fn get_ner(&self, normalized: &str) -> Option<NerType> {
        let map = self.map.lock().expect("genesis lock");
        let e = map.get(normalized)?;
        if !e.ner_promoted {
            return None;
        }
        let s = e.ner.as_deref()?;
        s.parse().ok()
    }

    fn record(&self, correction: &Correction, normalized: &str) {
        // Only POS/NER fields feed genesis; other fields (dep/head/lemma) are
        // not version-pinned as lexical POS data.
        let (pos_val, ner_val) = match correction.field {
            CorrectionField::Pos => (Some(correction.new_value.clone()), None),
            CorrectionField::Ner => (None, Some(correction.new_value.clone())),
            _ => return,
        };
        let mut map = self.map.lock().expect("genesis lock");
        let entry = map.entry(normalized.to_string()).or_insert(GenesisEntry {
            pos: None,
            ner: None,
            count: 0,
            ner_count: 0,
            promoted: false,
            ner_promoted: false,
        });
        // Separate counters so POS evidence does not accelerate NER promotion
        // (entity type is context-variant — Washington/Jordan/Paris).
        if let Some(v) = pos_val {
            if entry.pos.is_none() {
                entry.pos = Some(v);
            }
            entry.count += 1;
            if entry.count >= self.threshold {
                entry.promoted = true;
            }
        }
        if let Some(v) = ner_val {
            if entry.ner.is_none() {
                entry.ner = Some(v);
            }
            entry.ner_count += 1;
            if entry.ner_count >= self.ner_threshold {
                entry.ner_promoted = true;
            }
        }
    }

    fn count_for(&self, normalized: &str) -> u32 {
        self.map
            .lock()
            .expect("genesis lock")
            .get(normalized)
            .map_or(0, |e| e.count)
    }

    fn is_promoted(&self, normalized: &str) -> bool {
        self.map
            .lock()
            .expect("genesis lock")
            .get(normalized)
            .is_some_and(|e| e.promoted || e.ner_promoted)
    }

    fn promoted_len(&self) -> usize {
        self.map
            .lock()
            .expect("genesis lock")
            .values()
            .filter(|e| e.promoted || e.ner_promoted)
            .count()
    }

    fn len(&self) -> usize {
        self.map.lock().expect("genesis lock").len()
    }

    fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let map = self.map.lock().expect("genesis lock");
        let json = serde_json::to_string_pretty(&*map).map_err(std::io::Error::other)?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, json)
    }
}

impl InMemoryGenesisIndex {
    /// NER evidence count for `normalized` (separate from POS).
    pub fn ner_count_for(&self, normalized: &str) -> u32 {
        self.map
            .lock()
            .expect("genesis lock")
            .get(normalized)
            .map_or(0, |e| e.ner_count)
    }
}

#[cfg(test)]
#[path = "../../tests/genesis.rs"]
mod tests;
