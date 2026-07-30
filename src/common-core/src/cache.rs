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
}
