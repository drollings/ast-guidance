//! LLM response cache — the single owner (ROADMAP_20260903_LLM M4).
//!
//! Moved verbatim from `common_core::cache` (`CachedResponse`,
//! `ResponseCache` with TTL lazy-eviction and the
//! `"{model}:{sha256(request_json)}"` keying). The only cross-crate
//! dependencies are the generic `common_core::hash::sha256_hex` and
//! `common_core::time::now_secs` (both stay in `common-core`).
//!
//! What stays behind: the generic cache *mechanism* — `LoadCache`,
//! `ArcLoadCache`, and the weighted-LRU eviction engine (`eviction_score`,
//! `eviction_order`, `evict_until_fit`, `Budget`) — which `llm::runtime`
//! composes and which is not LLM policy. This module owns only the LLM
//! *policy/keying*: the one `"{model}:…"` key format in the workspace.
//!
//! M11 deleted the `common-core::cache` byte-identical shim copies (kept
//! through M10 under `#[deprecated]`); the owner goldens in
//! `tests/cache.rs` are the lasting contract.
//!
//! Calibration (roadmap §1, M10): cache identity is task-value
//! freshness/identity, not endorsement — a cache hit is never a correctness
//! vote, and a different model or an expired TTL must miss even on
//! identical text. The key format and TTL semantics move unchanged here;
//! earning the right to trust cached outputs is M10.

use common_core::hash::sha256_hex;
use common_core::time::now_secs;
use serde_json::Value;

/// A cache entry storing an LLM response with a timestamp for TTL checks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedResponse {
    /// Unix epoch seconds when this entry was stored.
    pub stored_at_secs: u64,
    /// The serialized LLM response JSON.
    pub response_json: Value,
}

/// A backend-agnostic response cache for LLM completions.
///
/// Wraps user-supplied closures for `check`, `store`, `delete`, and `clear`
/// so it works with any storage backend (in-memory `HashMap`, SQLite, Redis,
/// etc.).  Entries are lazily evicted — `get` returns `None` if the entry
/// exists but is older than the configured TTL.
///
/// Cache key format: `"{model}:{sha256(request_json)}"`.
type CheckCacheFn = Box<dyn Fn(&str) -> Option<CachedResponse> + Send + Sync>;
type StoreCacheFn = Box<dyn Fn(&str, &CachedResponse) + Send + Sync>;
type DeleteCacheFn = Box<dyn Fn(&str) + Send + Sync>;
type ClearCacheFn = Box<dyn Fn() + Send + Sync>;

pub struct ResponseCache {
    check: CheckCacheFn,
    store: StoreCacheFn,
    delete: Option<DeleteCacheFn>,
    clear: Option<ClearCacheFn>,
    ttl: Option<std::time::Duration>,
}

impl ResponseCache {
    /// Create a new `ResponseCache`.
    ///
    /// * `ttl` — optional time-to-live; entries older than this are treated as
    ///   misses (lazy eviction).
    /// * `check` — looks up a key in the backend, returning `None` if absent.
    /// * `store` — persists a key/value pair in the backend.
    /// * `delete` — removes a single key (optional; used by `invalidate_key`).
    /// * `clear` — removes all entries (optional; used by `invalidate_all`).
    pub fn new(
        ttl: Option<std::time::Duration>,
        check: impl Fn(&str) -> Option<CachedResponse> + Send + Sync + 'static,
        store: impl Fn(&str, &CachedResponse) + Send + Sync + 'static,
    ) -> Self {
        Self {
            check: Box::new(check),
            store: Box::new(store),
            delete: None,
            clear: None,
            ttl,
        }
    }

    /// Configure single-key deletion.
    #[must_use]
    pub fn with_delete(mut self, delete: impl Fn(&str) + Send + Sync + 'static) -> Self {
        self.delete = Some(Box::new(delete));
        self
    }

    /// Configure full cache clearing.
    #[must_use]
    pub fn with_clear(mut self, clear: impl Fn() + Send + Sync + 'static) -> Self {
        self.clear = Some(Box::new(clear));
        self
    }

    fn cache_key(model: &str, request_json: &str) -> String {
        format!("{}:{}", model, sha256_hex(request_json.as_bytes()))
    }

    /// Look up a cached response.  Returns `None` on miss or TTL expiry.
    pub fn get(&self, model: &str, request_json: &str) -> Option<CachedResponse> {
        let key = Self::cache_key(model, request_json);
        let entry = (self.check)(&key)?;
        if let Some(ttl) = self.ttl {
            let age = now_secs().saturating_sub(entry.stored_at_secs);
            if age >= ttl.as_secs() {
                return None;
            }
        }
        Some(entry)
    }

    /// Store a response in the cache (best-effort; failure is non-fatal).
    pub fn set(&self, model: &str, request_json: &str, response_json: Value) {
        let key = Self::cache_key(model, request_json);
        let entry = CachedResponse {
            stored_at_secs: now_secs(),
            response_json,
        };
        (self.store)(&key, &entry);
    }

    /// Remove a cached entry by model and request JSON.
    pub fn invalidate_key(&self, model: &str, request_json: &str) {
        if let Some(ref delete) = self.delete {
            let key = Self::cache_key(model, request_json);
            delete(&key);
        }
    }

    /// Remove a cached entry by raw cache key.
    pub fn invalidate_key_raw(&self, cache_key: &str) {
        if let Some(ref delete) = self.delete {
            delete(cache_key);
        }
    }

    /// Remove all cached entries.
    pub fn invalidate_all(&self) {
        if let Some(ref clear) = self.clear {
            clear();
        }
    }
}
