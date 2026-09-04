//! Span-level detail cache for refiner output (ROADMAP_20260831_ARCEAGER M6.1).
//!
//! A repeated span need not re-consult a model: the corrections produced for
//! a focus span are cached content-addressably and replayed on the next
//! occurrence.
//!
//! # Content addressing
//!
//! The key is a 64-bit content hash of the span's lowercased orths
//! (`hash_utf8` seed 1, the same hash the ledger uses for `InterlinguaId`).
//! Different texts, different keys; same text, same key — order- and
//! universe-independent. 64 bits is sufficient for the expected span
//! population (`n=1M` → `≈2e-7` collision).
//!
//! # Tenancy (M2b)
//!
//! This module keeps only the pure hash discipline ([`span_key`]) and the
//! dependency-free [`SpanCacheSeam`] callback bundle the ladder consults.
//! The `SpanCache` trait + `InMemorySpanCache` hermetic double live with the
//! ledger view owner (`fluent-router` `ledger::span_cache`, beside
//! `SqliteSpanCache` — the read-through view over the shared
//! `interlingua_index` table, same sqlite file, `role='span_cache'`
//! sentinel; no parallel store). The router adapts its
//! `Arc<dyn SpanCache>` into a [`SpanCacheSeam`] at the wiring site, so no
//! router edge ever enters this crate.
//!
//! # Invalidation
//!
//! Cached entries are keyed by content. A human review that writes a new
//! [`Correction`](crate::review::Correction) for the same span through the
//! [`CorrectionIndex`](crate::review::CorrectionIndex) must invalidate the
//! cached entry for that span key: the router calls the seam's `invalidate`
//! when it records a correction (the "invalidated through the correction
//! index" contract; see `SqliteSpanCache::invalidate_for_corrections`). The
//! cache never goes stale silently — the key is the content hash, so a
//! content mutation is a different key by construction.

use std::sync::Arc;

use crate::doc::Doc;
use crate::hash::hash_utf8;
use crate::review::Correction;

/// Read leg of the span-cache seam: cached corrections for `key`, if present.
pub type SpanCacheGet = Arc<dyn Fn(u64) -> Option<Vec<Correction>> + Send + Sync>;
/// Write leg of the span-cache seam: store `corrections` under `key`
/// (overwrite; key `0` is a no-op — never cached).
pub type SpanCachePut = Arc<dyn Fn(u64, Vec<Correction>) + Send + Sync>;
/// Invalidation leg of the span-cache seam: drop the entry for `key`
/// (called when a `CorrectionIndex` write for the same span lands).
pub type SpanCacheInvalidate = Arc<dyn Fn(u64) + Send + Sync>;

/// The dependency-free span-cache seam the ladder consults (M2b).
///
/// A bundle of closures in the same idiom as the fetch seams
/// ([`LlmRefineFetchSync`](crate::pipeline::LlmRefineFetchSync),
/// [`EncoderResidualFetch`](crate::pipeline::EncoderResidualFetch)):
/// the ladder needs `get`/`put` per refine plus `invalidate` for the
/// correction-index contract, and knows nothing about the backend
/// (in-memory map, ledger sqlite, or test stub). `Clone` is an `Arc` bump
/// per leg, so sharing one seam across async workers and sync calls is
/// cheap.
#[derive(Clone)]
pub struct SpanCacheSeam {
    /// Return the cached corrections for `key`, if present.
    pub get: SpanCacheGet,
    /// Store `corrections` under `key` (overwrite).
    pub put: SpanCachePut,
    /// Invalidate the entry for `key`.
    pub invalidate: SpanCacheInvalidate,
}

impl SpanCacheSeam {
    /// A seam over explicit closure legs.
    pub fn new(get: SpanCacheGet, put: SpanCachePut, invalidate: SpanCacheInvalidate) -> Self {
        Self { get, put, invalidate }
    }

    /// Return the cached corrections for `key`, if present.
    #[must_use]
    pub fn get(&self, key: u64) -> Option<Vec<Correction>> {
        (self.get)(key)
    }

    /// Store `corrections` under `key` (overwrite).
    pub fn put(&self, key: u64, corrections: Vec<Correction>) {
        (self.put)(key, corrections);
    }

    /// Invalidate the entry for `key`.
    pub fn invalidate(&self, key: u64) {
        (self.invalidate)(key);
    }
}

impl std::fmt::Debug for SpanCacheSeam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanCacheSeam").finish_non_exhaustive()
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

#[cfg(test)]
#[path = "../tests/cache.rs"]
mod tests;
