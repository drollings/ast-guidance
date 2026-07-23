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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

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
