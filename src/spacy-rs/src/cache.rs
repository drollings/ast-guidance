//! Span-level detail cache for refiner output (ROADMAP_20260831_ARCEAGER M6.1).
//!
//! A repeated span need not re-consult a model: the corrections produced for
//! a focus span are cached content-addressably and replayed on the next
//! occurrence. The cache is `Arc<dyn SpanCache>` so both the async and sync
//! ladders share one instance and the router can back it with its ledger
//! sqlite (a **read-through view**, not a parallel store).
//!
//! # Content addressing
//!
//! The key is a 64-bit content hash of the span's lowercased orths
//! (`hash_utf8` seed 1, the same hash the ledger uses for `InterlinguaId`).
//! Different texts, different keys; same text, same key — order- and
//! universe-independent. 64 bits is sufficient for the expected span
//! population (`n=1M` → `≈2e-7` collision).
//!
//! # Invalidation
//!
//! Cached entries are keyed by content. A human review that writes a new
//! [`Correction`](crate::review::Correction) for the same span through the
//! [`CorrectionIndex`](crate::review::CorrectionIndex) must invalidate the
//! cached entry for that span key. The spacy-rs crate does not know the
//! ledger's sqlite schema; it exposes `invalidate(key)` and the router calls
//! it when it records a correction (the "invalidated through the correction
//! index" contract). The cache never goes stale silently — the key is the
//! content hash, so a content mutation is a different key by construction.
//!
//! # Storage
//!
//! `spacy-rs` ships only the trait and the hermetic `InMemorySpanCache`.
//! The production `LedgerSpanCache` (router) is a view over the shared
//! `interlingua_index` table (same sqlite file, `role='span_cache'`
//! sentinel) — no parallel store. The trait is object-safe so the pipeline
//! can hold `Option<Arc<dyn SpanCache>>` without knowing the backend.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::doc::Doc;
use crate::hash::hash_utf8;
use crate::review::Correction;

/// Content-addressed span cache for refiner corrections.
///
/// `Send + Sync` so the async ladder (shared `Arc` across `ResultPool`
/// workers) and the sync ladder (single-threaded) can share one instance.
pub trait SpanCache: Send + Sync {
    /// Return the cached corrections for `key`, if present.
    fn get(&self, key: u64) -> Option<Vec<Correction>>;
    /// Store `corrections` under `key` (overwrite).
    fn put(&self, key: u64, corrections: Vec<Correction>);
    /// Invalidate the entry for `key` (called when a `CorrectionIndex` write
    /// for the same span lands).
    fn invalidate(&self, key: u64);
    /// Number of entries (for tests/metrics).
    fn len(&self) -> usize;
    /// Whether the cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Hash the focused span's lowercased orths into a content key.
///
/// The key is `hash_utf8(lowercased_focus_tokens joined by 0x1F)`. The 0x1F
/// separator prevents `"ab"+"c"` colliding with `"a"+"bc"`. An empty focus
/// yields `0` (never cached — the ladder skips refine when focus is empty).
#[must_use]
pub fn span_key(doc: &Doc, focus: &[usize]) -> u64 {
    if focus.is_empty() {
        return 0;
    }
    let mut buf = String::new();
    for (i, &idx) in focus.iter().enumerate() {
        if i > 0 {
            buf.push('\x1F');
        }
        // `token_text` is the orth from the lexeme — the detail baseline.
        buf.push_str(&doc.token_text(idx).to_ascii_lowercase());
    }
    hash_utf8(&buf)
}

/// Hermetic in-memory span cache (tests + non-ledger callers).
///
/// `Mutex<HashMap>` — the cache is not a hot loop; a single lock per
/// get/put is negligible compared to a model call.
#[derive(Debug, Default)]
pub struct InMemorySpanCache {
    map: Mutex<HashMap<u64, Vec<Correction>>>,
}

impl InMemorySpanCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }
}

impl SpanCache for InMemorySpanCache {
    fn get(&self, key: u64) -> Option<Vec<Correction>> {
        self.map.lock().expect("cache lock").get(&key).cloned()
    }

    fn put(&self, key: u64, corrections: Vec<Correction>) {
        if key == 0 {
            return;
        }
        self.map
            .lock()
            .expect("cache lock")
            .insert(key, corrections);
    }

    fn invalidate(&self, key: u64) {
        self.map.lock().expect("cache lock").remove(&key);
    }

    fn len(&self) -> usize {
        self.map.lock().expect("cache lock").len()
    }
}

#[cfg(test)]
#[path = "../tests/cache.rs"]
mod tests;
