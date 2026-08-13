//! A read-through cache abstraction: checks the cache first; on miss, calls
//! `load`; on success, writes back via `store`.

type CheckFn<K, V> = Box<dyn Fn(&K) -> Option<V> + Send + Sync>;
type LoadFn<K, V> = Box<dyn Fn(&K) -> Result<V, String> + Send + Sync>;
type StoreFn<K, V> = Box<dyn Fn(&K, &V) + Send + Sync>;

/// A read-through cache: checks the cache first; on miss, calls `load`;
/// on success, writes back via `store`.
///
/// Uses `Box<dyn Fn>` closures for maximum flexibility — the `check` and
/// `store` closures can wrap any cache backend (LRU, SQLite, HashMap, etc.).
///
/// # Example
///
/// ```ignore
/// use std::collections::HashMap;
/// use common_core::cache::ReadThroughCache;
///
/// let mut store: HashMap<String, String> = HashMap::new();
/// let cache = ReadThroughCache::new(
///     |key: &String| store.get(key).cloned(),
///     |key: &String| Ok::<_, String>(format!("loaded: {key}")),
///     |key: &String, value: &String| { store.insert(key.clone(), value.clone()); },
/// );
/// let v = cache.get(&"hello".to_string()).unwrap();
/// assert_eq!(v, "loaded: hello");
/// ```
pub struct ReadThroughCache<K, V> {
    check: CheckFn<K, V>,
    load: LoadFn<K, V>,
    store: StoreFn<K, V>,
}

impl<K, V> ReadThroughCache<K, V> {
    pub fn new(
        check: impl Fn(&K) -> Option<V> + Send + Sync + 'static,
        load: impl Fn(&K) -> Result<V, String> + Send + Sync + 'static,
        store: impl Fn(&K, &V) + Send + Sync + 'static,
    ) -> Self {
        Self {
            check: Box::new(check),
            load: Box::new(load),
            store: Box::new(store),
        }
    }

    /// Get a value for `key`, using the cache if available.
    pub fn get(&self, key: &K) -> Result<V, String> {
        if let Some(v) = (self.check)(key) {
            return Ok(v);
        }
        let v = (self.load)(key)?;
        (self.store)(key, &v);
        Ok(v)
    }
}

// ─── Load Cache ────────────────────────────────────────────────────────────

use lru::LruCache;
use std::borrow::Borrow;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::Mutex;

type LruLoadFn<K, V, E> = Box<dyn Fn(&K) -> Result<V, E> + Send + Sync>;

/// A bounded, thread-safe get-or-load LRU cache.
///
/// Wraps a `Mutex<LruCache<K, V>>` plus a `load` closure invoked on a cache
/// miss. `get_or_load` keeps the hot path to a single lock acquisition on a
/// hit; on a miss it drops the lock, runs `load`, and re-acquires only to
/// insert the freshly loaded value.
///
/// Write-through consumers (caches filled explicitly via `insert`, never via
/// load-on-miss) can use the plain `get`/`insert`/`remove`/`contains`
/// accessors; the `load` closure is only ever invoked by `get_or_load`.
pub struct LoadCache<K, V, E> {
    inner: Mutex<LruCache<K, V>>,
    load: LruLoadFn<K, V, E>,
}

impl<K, V, E> LoadCache<K, V, E>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    /// Create a bounded cache holding up to `capacity` entries.
    ///
    /// `load` produces a value for a missing key (and may fail with `E`).
    /// Returns an error when `capacity` is zero.
    pub fn new(
        capacity: usize,
        load: impl Fn(&K) -> Result<V, E> + Send + Sync + 'static,
    ) -> Result<Self, String> {
        let cap = NonZeroUsize::new(capacity)
            .ok_or_else(|| format!("cache capacity must be non-zero, got {capacity}"))?;
        Ok(Self {
            inner: Mutex::new(LruCache::new(cap)),
            load: Box::new(load),
        })
    }

    /// Look up `key` without invoking the load closure. Returns a clone of the
    /// cached value, if present.
    pub fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.lock().unwrap().get(key).cloned()
    }

    /// `true` when `key` is present in the cache (never loads on a miss).
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Insert `value` under `key`, replacing any existing entry.
    pub fn insert(&self, key: K, value: V) {
        self.inner.lock().unwrap().put(key, value);
    }

    /// Remove `key`, returning the evicted value if present.
    pub fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.lock().unwrap().pop(key)
    }

    /// Look up `key`, loading and caching it on a miss via the `load` closure.
    pub fn get_or_load(&self, key: K) -> Result<V, E> {
        if let Some(value) = self.get(&key) {
            return Ok(value);
        }
        let value = (self.load)(&key)?;
        self.insert(key, value.clone());
        Ok(value)
    }

    /// Number of entries currently held.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// `true` when the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The maximum number of entries before the least-recently-used entry is
    /// evicted.
    pub fn capacity(&self) -> usize {
        self.inner.lock().unwrap().cap().get()
    }
}

// ─── Response Cache ────────────────────────────────────────────────────────

use crate::hash::sha256_hex;
use crate::time::now_secs;
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

// ─── Weighted-LRU eviction engine ─────────────────────────────────────────
//
// The canonical "evict the largest × coldest until under budget" engine. The
// residency/admission loop (`InstancePool`) composes these three functions.
// `ColdSnapshotIndex::evict` (the router's TTL metadata sweep) is a *predicate*
// filter, not a byte-budget eviction, so it intentionally does not use this
// engine.

/// The maximum coldness used as an overflow guard, and the coldness assigned
/// to an entity that was never used (`last_used < 0`). ~35k years in seconds.
const COLD_CAP: i64 = 1 << 40;

/// Eviction priority score: `freed_bytes * coldness`, where coldness is
/// seconds since `last_used` (capped at `COLD_CAP`; an entity never used is
/// maximally cold). A "cost of keeping" heuristic: the unit whose resident
/// footprint times its idle time is largest is the most valuable to evict. It
/// makes big cold footprints (a model's weights) outrank small hot ones, so
/// memory pressure reclaims the largest chunks while a just-used entity scores
/// near zero and stays.
pub fn eviction_score(freed_bytes: u64, last_used: i64, now: i64) -> u64 {
    let coldness = if last_used < 0 {
        COLD_CAP
    } else {
        now.saturating_sub(last_used).clamp(1, COLD_CAP)
    };
    freed_bytes.saturating_mul(coldness as u64)
}

/// Order `candidates` best-eviction-first: score descending, then `last_used`
/// descending (the newer of two equal-scoring units is kept).
///
/// Returns the same candidates reordered (an owned `Vec<C>` — the caller
/// supplies the candidates it gathered and gets back an ordering it can feed
/// to [`evict_until_fit`]).
pub fn eviction_order<C>(
    candidates: Vec<C>,
    now: i64,
    freed_of: impl Fn(&C) -> u64,
    last_used_of: impl Fn(&C) -> i64,
) -> Vec<C> {
    let mut ordered = candidates;
    ordered.sort_by(|a, b| {
        eviction_score(freed_of(b), last_used_of(b), now)
            .cmp(&eviction_score(freed_of(a), last_used_of(a), now))
            .then_with(|| last_used_of(b).cmp(&last_used_of(a)))
    });
    ordered
}

/// Evict candidates (already in [`eviction_order`]) until `used <= budget` or
/// `batch` candidates have been evicted.
///
/// `evict(&candidate)` performs the actual eviction and returns the freed
/// bytes, or `None` when the eviction failed (not counted toward `batch`).
/// Returns the updated `used` total and the number of successful evictions.
pub async fn evict_until_fit<C, F, Fut>(
    mut used: u64,
    budget: u64,
    batch: usize,
    candidates: Vec<C>,
    evict: F,
) -> (u64, usize)
where
    F: Fn(&C) -> Fut,
    Fut: std::future::Future<Output = Option<u64>>,
{
    let mut evicted = 0usize;
    for candidate in candidates {
        if used <= budget || evicted >= batch {
            break;
        }
        if let Some(freed) = evict(&candidate).await {
            evicted += 1;
            used = used.saturating_sub(freed);
        }
    }
    (used, evicted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    // ─── ResponseCache tests ────────────────────────────────────────────

    fn make_cache(store: Arc<Mutex<HashMap<String, CachedResponse>>>) -> ResponseCache {
        let s_check = Arc::clone(&store);
        let s_store = Arc::clone(&store);
        let s_clear = Arc::clone(&store);
        let s_delete = Arc::clone(&store);
        ResponseCache::new(
            None,
            move |key: &str| s_check.lock().unwrap().get(key).cloned(),
            move |key: &str, value: &CachedResponse| {
                s_store
                    .lock()
                    .unwrap()
                    .insert(key.to_string(), value.clone());
            },
        )
        .with_delete(move |key: &str| {
            s_delete.lock().unwrap().remove(key);
        })
        .with_clear(move || {
            s_clear.lock().unwrap().clear();
        })
    }

    #[test]
    fn response_cache_identical_request_hits() {
        let store: Arc<Mutex<HashMap<String, CachedResponse>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let cache = make_cache(Arc::clone(&store));
        let request_json = r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#;
        // Cache miss on first call
        assert!(cache.get("gpt-4", request_json).is_none());
        // Set a response
        cache.set(
            "gpt-4",
            request_json,
            serde_json::json!({"choices": [{"message": {"content": "hello"}}]}),
        );
        // Cache hit on second call
        let hit = cache.get("gpt-4", request_json);
        assert!(hit.is_some(), "expected cache hit");
        assert_eq!(
            hit.unwrap().response_json["choices"][0]["message"]["content"],
            "hello"
        );
    }

    #[test]
    fn response_cache_different_temperature_misses() {
        let store: Arc<Mutex<HashMap<String, CachedResponse>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let cache = make_cache(Arc::clone(&store));
        let req1 = r#"{"model":"test","messages":[],"temperature":0.5}"#;
        let req2 = r#"{"model":"test","messages":[],"temperature":1.0}"#;
        cache.set("gpt-4", req1, serde_json::json!({"result": "a"}));
        // Different temperature → different cache key → miss
        assert!(cache.get("gpt-4", req2).is_none());
    }

    #[test]
    fn response_cache_ttl_expiry_misses() {
        let store: Arc<Mutex<HashMap<String, CachedResponse>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let s_check = Arc::clone(&store);
        let s_store = Arc::clone(&store);
        // Use a 0-second TTL so entries are immediately expired
        let cache = ResponseCache::new(
            Some(std::time::Duration::from_secs(0)),
            move |key: &str| s_check.lock().unwrap().get(key).cloned(),
            move |key: &str, value: &CachedResponse| {
                s_store
                    .lock()
                    .unwrap()
                    .insert(key.to_string(), value.clone());
            },
        );
        let request_json = r#"{"model":"test"}"#;
        cache.set(
            "gpt-4",
            request_json,
            serde_json::json!({"result": "stale"}),
        );
        // Should be expired because TTL is 0
        assert!(cache.get("gpt-4", request_json).is_none());
    }

    #[test]
    fn response_cache_corrupted_entry_miss() {
        // The in-memory store wouldn't have "corrupted" entries,
        // but this tests that a None return from the backend is treated as a miss.
        let store: Arc<Mutex<HashMap<String, CachedResponse>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let cache = make_cache(Arc::clone(&store));
        // No entry stored → should be None
        assert!(cache.get("gpt-4", "anything").is_none());
    }

    #[test]
    fn response_cache_invalidate_key() {
        let store: Arc<Mutex<HashMap<String, CachedResponse>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let cache = make_cache(Arc::clone(&store));
        let req = r#"{"x":1}"#;
        cache.set("gpt-4", req, serde_json::json!({"result": "data"}));
        assert!(cache.get("gpt-4", req).is_some());
        cache.invalidate_key("gpt-4", req);
        assert!(cache.get("gpt-4", req).is_none());
    }

    #[test]
    fn response_cache_invalidate_all() {
        let store: Arc<Mutex<HashMap<String, CachedResponse>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let cache = make_cache(Arc::clone(&store));
        cache.set("gpt-4", r#"{"a":1}"#, serde_json::json!({"result": "a"}));
        cache.set("gpt-4", r#"{"b":2}"#, serde_json::json!({"result": "b"}));
        cache.invalidate_all();
        assert!(cache.get("gpt-4", r#"{"a":1}"#).is_none());
        assert!(cache.get("gpt-4", r#"{"b":2}"#).is_none());
    }

    #[test]
    fn response_cache_different_model_different_key() {
        let store: Arc<Mutex<HashMap<String, CachedResponse>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let cache = make_cache(Arc::clone(&store));
        let req = r#"{"model":"test"}"#;
        cache.set("gpt-4", req, serde_json::json!({"result": "gpt4"}));
        // Same request JSON but different model → different key → miss
        assert!(cache.get("claude-3", req).is_none());
    }

    // ─── ReadThroughCache tests ─────────────────────────────────────────

    #[test]
    fn test_read_through_cache_hit() {
        let store: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
        store.lock().unwrap().insert("a".into(), "cached_a".into());
        let s = Arc::clone(&store);
        let cache = ReadThroughCache::new(
            move |key: &String| s.lock().unwrap().get(key).cloned(),
            |_: &String| -> Result<String, String> { unreachable!("should not be called") },
            |_: &String, _: &String| {},
        );
        let v = cache.get(&"a".to_string()).unwrap();
        assert_eq!(v, "cached_a");
    }

    #[test]
    fn test_read_through_cache_miss() {
        let store: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
        let s_check = Arc::clone(&store);
        let s_store = Arc::clone(&store);
        let cache = ReadThroughCache::new(
            move |key: &String| s_check.lock().unwrap().get(key).cloned(),
            |key: &String| Ok::<_, String>(format!("loaded:{key}")),
            move |key: &String, value: &String| {
                s_store.lock().unwrap().insert(key.clone(), value.clone());
            },
        );
        let v = cache.get(&"b".to_string()).unwrap();
        assert_eq!(v, "loaded:b");
        assert_eq!(store.lock().unwrap().get("b").unwrap(), "loaded:b");
    }

    #[test]
    fn test_read_through_cache_miss_then_hit() {
        let store: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
        let load_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let s_check = Arc::clone(&store);
        let s_store = Arc::clone(&store);
        let lc_load = Arc::clone(&load_count);
        let cache = ReadThroughCache::new(
            move |key: &String| s_check.lock().unwrap().get(key).cloned(),
            move |key: &String| {
                *lc_load.lock().unwrap() += 1;
                Ok::<_, String>(format!("loaded:{key}"))
            },
            move |key: &String, value: &String| {
                s_store.lock().unwrap().insert(key.clone(), value.clone());
            },
        );

        let v1 = cache.get(&"c".to_string()).unwrap();
        assert_eq!(v1, "loaded:c");
        assert_eq!(*load_count.lock().unwrap(), 1);

        let v2 = cache.get(&"c".to_string()).unwrap();
        assert_eq!(v2, "loaded:c");
        assert_eq!(*load_count.lock().unwrap(), 1);
    }

    #[test]
    fn test_read_through_cache_load_error() {
        let store: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));
        let s = Arc::clone(&store);
        let cache = ReadThroughCache::new(
            move |key: &String| s.lock().unwrap().get(key).cloned(),
            |_: &String| -> Result<String, String> { Err("load failed".into()) },
            |_: &String, _: &String| {},
        );
        let result = cache.get(&"d".to_string());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "load failed");
    }

    // ─── LoadCache tests ────────────────────────────────────────────────

    fn make_load_cache(load_count: Arc<Mutex<usize>>) -> LoadCache<String, String, String> {
        LoadCache::new(10, move |key: &String| {
            *load_count.lock().unwrap() += 1;
            Ok(format!("loaded:{key}"))
        })
        .expect("capacity non-zero")
    }

    #[test]
    fn load_cache_miss_loads_and_caches() {
        let load_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let cache = make_load_cache(Arc::clone(&load_count));

        let v1 = cache.get_or_load("a".to_string()).unwrap();
        assert_eq!(v1, "loaded:a");
        assert_eq!(*load_count.lock().unwrap(), 1);

        let v2 = cache.get_or_load("a".to_string()).unwrap();
        assert_eq!(v2, "loaded:a");
        assert_eq!(*load_count.lock().unwrap(), 1, "hit must not reload");
    }

    #[test]
    fn load_cache_get_does_not_load() {
        let load_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let cache = make_load_cache(Arc::clone(&load_count));

        assert!(cache.get::<str>("missing").is_none());
        assert!(!cache.contains::<str>("missing"));
        assert_eq!(*load_count.lock().unwrap(), 0, "plain get never loads");
    }

    #[test]
    fn load_cache_load_error_propagates() {
        let cache = LoadCache::new(2, |_: &String| -> Result<String, String> {
            Err("load failed".into())
        })
        .expect("capacity non-zero");
        assert_eq!(
            cache.get_or_load("a".to_string()).unwrap_err(),
            "load failed"
        );
    }

    #[test]
    fn load_cache_insert_overrides_and_get_returns_clone() {
        let cache = make_load_cache(Arc::new(Mutex::new(0)));
        cache.insert("a".to_string(), "manual".to_string());
        assert_eq!(cache.get::<str>("a"), Some("manual".to_string()));
        assert!(!cache.get_or_load("a".to_string()).unwrap().is_empty());
    }

    #[test]
    fn load_cache_remove() {
        let cache = make_load_cache(Arc::new(Mutex::new(0)));
        cache.insert("a".to_string(), "manual".to_string());
        assert_eq!(cache.remove::<str>("a"), Some("manual".to_string()));
        assert!(cache.get::<str>("a").is_none());
        assert!(cache.remove::<str>("a").is_none());
    }

    #[test]
    fn load_cache_evicts_lru() {
        let cache = LoadCache::new(2, |key: &String| Ok::<_, String>(format!("loaded:{key}")))
            .expect("capacity non-zero");
        cache.insert("a".to_string(), "1".to_string());
        cache.insert("b".to_string(), "2".to_string());
        // Touching "a" makes it most-recently-used; inserting "c" evicts "b".
        cache.get::<str>("a");
        cache.insert("c".to_string(), "3".to_string());
        assert!(cache.get::<str>("a").is_some());
        assert!(cache.get::<str>("b").is_none());
        assert!(cache.get::<str>("c").is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn load_cache_zero_capacity_rejected() {
        assert!(LoadCache::new(0, |_: &String| -> Result<String, String> {
            Ok(String::new())
        })
        .is_err());
    }

    #[test]
    fn load_cache_capacity_and_len() {
        let cache = make_load_cache(Arc::new(Mutex::new(0)));
        assert_eq!(cache.capacity(), 10);
        assert!(cache.is_empty());
        cache.insert("a".to_string(), "1".to_string());
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    // ─── Weighted-LRU eviction engine tests ─────────────────────────────

    #[test]
    fn eviction_score_favors_large_and_cold() {
        // freed * coldness; a 10-byte unit idle 2s scores 20.
        assert_eq!(eviction_score(10, 100, 102), 20);
        // A bigger footprint at the same coldness scores higher.
        assert_eq!(eviction_score(1000, 100, 102), 2000);
        // A just-used unit (coldness clamped to 1) scores its size.
        assert_eq!(eviction_score(100, 200, 200), 100);
    }

    #[test]
    fn eviction_score_never_used_is_maximally_cold() {
        // last_used < 0 → COLD_CAP (the overflow guard).
        let big = eviction_score(1, -1, 123456789);
        let capped = eviction_score(1, 123456789, 123456789);
        assert!(big >= capped, "never-used must be at least as cold as any real age");
        // COLD_CAP = 2^40.
        assert_eq!(big, 1 << 40);
    }

    #[test]
    fn eviction_score_caps_coldness() {
        // Huge idle time clamps to COLD_CAP.
        assert_eq!(eviction_score(2, 0, i64::MAX), 2 * (1 << 40));
    }

    #[test]
    fn eviction_score_overflow_saturates() {
        assert_eq!(
            eviction_score(u64::MAX, 0, 1 << 45),
            u64::MAX,
            "saturating_mul must not overflow"
        );
    }

    #[test]
    fn eviction_order_sorts_score_desc_then_last_used_desc() {
        // Three candidates: (freed, last_used). Highest score first; ties
        // broken by newer last_used. Scores with now=10: (10,5)->50, (100,5)->500, (10,9)->10.
        let cands = vec![(10, 5), (100, 5), (10, 9)];
        let ordered = eviction_order(cands, 10, |c| c.0, |c| c.1);
        assert_eq!(ordered, vec![(100, 5), (10, 5), (10, 9)]);
    }

    #[tokio::test]
    async fn evict_until_fit_evicts_until_budget() {
        // used=100, budget=50 → evict until <=50. Candidates are best-first.
        let cands = vec![30u64, 20, 10];
        let (used, n) = evict_until_fit(100, 50, usize::MAX, cands, |c| {
            let v = *c;
            async move { Some(v) }
        })
        .await;
        // 30 then 20 → used=50; 10 is kept.
        assert_eq!(used, 50);
        assert_eq!(n, 2);
    }

    #[tokio::test]
    async fn evict_until_fit_honors_batch_cap() {
        let cands = vec![1u64, 1, 1, 1];
        let (used, n) = evict_until_fit(100, 0, 2, cands, |c| {
            let v = *c;
            async move { Some(v) }
        })
        .await;
        assert_eq!(n, 2, "batch caps evictions");
        assert_eq!(used, 98);
    }

    #[tokio::test]
    async fn evict_until_fit_counts_failed_evictions() {
        // Candidate 1 fails to evict; candidate 2 succeeds; candidate 3 fails.
        let cands = vec![1u64, 2, 1];
        let (used, n) = evict_until_fit(100, 0, usize::MAX, cands, |c| {
            let v = *c;
            async move {
                if v == 1 {
                    None // failed eviction
                } else {
                    Some(1)
                }
            }
        })
        .await;
        assert_eq!(n, 1, "only successful evictions count toward batch");
        assert_eq!(used, 99);
    }

    #[tokio::test]
    async fn evict_until_fit_stops_when_already_under_budget() {
        let cands = vec![1u64, 1];
        let (used, n) = evict_until_fit(40, 50, usize::MAX, cands, |c| {
            let v = *c;
            async move { Some(v) }
        })
        .await;
        assert_eq!(used, 40);
        assert_eq!(n, 0, "no eviction needed once under budget");
    }
}
