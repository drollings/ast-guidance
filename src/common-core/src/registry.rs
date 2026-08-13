//! Generic keyed registry — the canonical "register by key, look up by key"
//! primitive for the workspace.
//!
//! Consumers that store values under a string-ish key (guidance's
//! `PluginRegistry`, `memory-plugin`'s `MemoryPluginRegistry`) compose
//! `KeyedRegistry<K, V>` instead of hand-wrapping a `HashMap`/`BTreeMap`.
//! Zero-domain: `K` and `V` are fully generic, so no domain type leaks into
//! `common-core`.

use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use crate::sync::{lock_read, lock_write};

/// A generic keyed registry mapping `K` → `V`.
///
/// # Deterministic iteration
///
/// Entries are kept in `BTreeMap` key order, so `keys()`, `values()`, and
/// `iter()` return results in sorted key order. Consumers that expose a
/// listing method get a stable, reproducible order rather than arbitrary
/// hashing order.
pub struct KeyedRegistry<K, V> {
    entries: BTreeMap<K, V>,
}

impl<K, V> KeyedRegistry<K, V>
where
    K: Ord,
{
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Register `value` under `key`, replacing any existing entry.
    ///
    /// Returns the previously stored value, if one was present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.entries.insert(key, value)
    }

    /// Look up `key` by borrowed form — `get("foo")` works when the stored
    /// keys are `String` and the caller only has `&str`.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.entries.get(key)
    }

    /// Mutable borrow-based lookup — use this to mutate a stored value in place.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.entries.get_mut(key)
    }

    /// `true` when `key` is registered.
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.entries.contains_key(key)
    }

    /// All registered keys, in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.entries.keys()
    }

    /// All registered values, in sorted key order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.values()
    }

    /// `(key, value)` pairs, in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter()
    }

    /// Unregister `key`, returning its value if present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.entries.remove(key)
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no entries are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<K, V> Default for KeyedRegistry<K, V>
where
    K: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A thread-safe, `RwLock`-backed registry mapping `K` → `Arc<V>`.
///
/// This is the concurrent counterpart to [`KeyedRegistry`] for consumers that
/// share the registry across threads and need cheap, immutable lookups plus a
/// get-or-create (resolve-or-create) miss path. `get` is a single `RwLock`
/// read — the same cost as the `HashMap::get` it replaces — so it is safe on
/// hot paths. Values are shared as `Arc<V>` so callers hold a cheap clone that
/// keeps the value alive independent of the registry.
///
/// Zero-domain: `K` and `V` are fully generic; no router/guidance/coral type
/// leaks into `common-core`.
pub struct ConcurrentRegistry<K, V> {
    inner: RwLock<HashMap<K, Arc<V>>>,
}

impl<K, V> ConcurrentRegistry<K, V>
where
    K: Eq + Hash + Clone,
{
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    /// Look up `key`, returning a shared handle to the value if present.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        lock_read(&self.inner).get(key).cloned()
    }

    /// All registered keys.
    pub fn keys(&self) -> Vec<K> {
        lock_read(&self.inner).keys().cloned().collect()
    }

    /// Unregister `key`, returning its shared handle if present.
    pub fn remove(&self, key: &K) -> Option<Arc<V>> {
        lock_write(&self.inner).remove(key)
    }

    /// Register or replace `value` under `key`, returning the previous shared
    /// handle if one was present. The upsert analogue to [`KeyedRegistry::insert`]
    /// for the shared/concurrent form.
    pub fn insert(&self, key: K, value: V) -> Option<Arc<V>> {
        lock_write(&self.inner).insert(key, Arc::new(value))
    }

    /// Look up `key`, or construct and register a new value on a miss.
    ///
    /// The constructor runs only under the registry's write lock, so a given
    /// key is materialized exactly once even under concurrent contention; the
    /// second observer returns the first construction. The constructor is
    /// infallible by design — fallible construction belongs at the call site
    /// (resolve first, then `remove`+insert on failure), matching the shared
    /// `Arc<V>` immutable-sharing contract.
    pub fn resolve_or_create(
        &self,
        key: K,
        ctor: impl FnOnce(&K) -> V,
    ) -> Arc<V> {
        if let Some(existing) = self.get(&key) {
            return existing;
        }
        let mut guard = lock_write(&self.inner);
        if let Some(existing) = guard.get(&key) {
            return Arc::clone(existing);
        }
        let created = Arc::new(ctor(&key));
        guard.insert(key, Arc::clone(&created));
        created
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        lock_read(&self.inner).len()
    }

    /// `true` when no entries are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, V> Default for ConcurrentRegistry<K, V>
where
    K: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Clone for ConcurrentRegistry<K, V>
where
    K: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(lock_read(&self.inner).clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_by_borrowed_key() {
        let mut reg = KeyedRegistry::new();
        reg.insert("alpha".to_string(), 1);
        reg.insert("beta".to_string(), 2);

        assert_eq!(reg.get("alpha"), Some(&1));
        assert_eq!(reg.get("beta"), Some(&2));
        assert_eq!(reg.get("gamma"), None);
    }

    #[test]
    fn insert_replaces_and_returns_old() {
        let mut reg = KeyedRegistry::new();
        reg.insert("k".to_string(), "first".to_string());
        assert_eq!(
            reg.insert("k".to_string(), "second".to_string()),
            Some("first".to_string())
        );
        assert_eq!(reg.get("k").map(String::as_str), Some("second"));
    }

    #[test]
    fn get_mut_updates_in_place() {
        let mut reg = KeyedRegistry::new();
        reg.insert("k".to_string(), 10);
        *reg.get_mut("k").unwrap() += 5;
        assert_eq!(reg.get("k"), Some(&15));
    }

    #[test]
    fn contains_reflects_registration() {
        let mut reg = KeyedRegistry::new();
        assert!(!reg.contains("a"));
        reg.insert("a".to_string(), ());
        assert!(reg.contains("a"));
    }

    #[test]
    fn keys_are_sorted() {
        let mut reg = KeyedRegistry::new();
        reg.insert("zebra".to_string(), 1);
        reg.insert("apple".to_string(), 2);
        reg.insert("mango".to_string(), 3);

        let keys: Vec<&String> = reg.keys().collect();
        assert_eq!(
            keys,
            vec![
                &"apple".to_string(),
                &"mango".to_string(),
                &"zebra".to_string()
            ]
        );
    }

    #[test]
    fn values_in_key_order() {
        let mut reg = KeyedRegistry::new();
        reg.insert("b".to_string(), 2);
        reg.insert("a".to_string(), 1);
        reg.insert("c".to_string(), 3);

        let values: Vec<&i32> = reg.values().collect();
        assert_eq!(values, vec![&1, &2, &3]);
    }

    #[test]
    fn iter_yields_pairs_in_key_order() {
        let mut reg = KeyedRegistry::new();
        reg.insert("b".to_string(), "two".to_string());
        reg.insert("a".to_string(), "one".to_string());

        let pairs: Vec<(&String, &String)> = reg.iter().collect();
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[0].1, "one");
        assert_eq!(pairs[1].0, "b");
        assert_eq!(pairs[1].1, "two");
    }

    #[test]
    fn remove_returns_value_and_unregisters() {
        let mut reg = KeyedRegistry::new();
        reg.insert("k".to_string(), 42);
        assert_eq!(reg.remove("k"), Some(42));
        assert!(reg.get("k").is_none());
        assert!(reg.remove("k").is_none());
        assert!(reg.is_empty());
    }

    #[test]
    fn len_and_is_empty() {
        let mut reg = KeyedRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        reg.insert("a".to_string(), 1);
        reg.insert("b".to_string(), 2);
        assert_eq!(reg.len(), 2);
        assert!(!reg.is_empty());
    }

    #[test]
    fn default_is_empty() {
        let reg: KeyedRegistry<&'static str, i32> = KeyedRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn static_str_keys_lookup_by_str() {
        let mut reg = KeyedRegistry::new();
        reg.insert("holographic", 1);
        reg.insert("honcho", 2);
        assert_eq!(reg.get("holographic"), Some(&1));
        assert_eq!(reg.get("honcho"), Some(&2));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn concurrent_get_hit_and_miss() {
        let reg = ConcurrentRegistry::new();
        let v = Arc::new(42i32);
        reg.resolve_or_create("k".to_string(), |_| *v);
        assert_eq!(*reg.get(&"k".to_string()).unwrap(), 42);
        assert!(reg.get(&"nope".to_string()).is_none());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn concurrent_resolve_or_create_single_construction() {
        let reg = Arc::new(ConcurrentRegistry::new());
        let constructions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..16 {
            let reg = Arc::clone(&reg);
            let constructions = Arc::clone(&constructions);
            handles.push(std::thread::spawn(move || {
                let v = reg.resolve_or_create("shared".to_string(), |_| {
                    constructions.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    vec![1, 2, 3]
                });
                assert_eq!(v.as_slice(), &[1, 2, 3]);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            constructions.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "constructor must run exactly once"
        );
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn concurrent_reads_on_get() {
        let reg = Arc::new(ConcurrentRegistry::new());
        for i in 0..8 {
            reg.resolve_or_create(i.to_string(), |_| i);
        }
        let mut handles = Vec::new();
        for _ in 0..16 {
            let reg = Arc::clone(&reg);
            handles.push(std::thread::spawn(move || {
                let sum: i64 = (0..8)
                    .filter_map(|i| reg.get(&i.to_string()).map(|v| *v))
                    .sum();
                assert_eq!(sum, 28);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn concurrent_keys_remove_len_is_empty() {
        let reg = ConcurrentRegistry::new();
        reg.resolve_or_create("a".to_string(), |_| 1);
        reg.resolve_or_create("b".to_string(), |_| 2);
        let mut keys = reg.keys();
        keys.sort();
        assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(reg.remove(&"a".to_string()).map(|v| *v), Some(1));
        assert!(reg.get(&"a".to_string()).is_none());
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        reg.remove(&"b".to_string());
        assert!(reg.is_empty());
    }

    #[test]
    fn concurrent_resolve_or_create_existing_is_reused() {
        let reg = ConcurrentRegistry::new();
        let first = reg.resolve_or_create("k".to_string(), |_| String::from("original"));
        let second = reg.resolve_or_create("k".to_string(), |_| String::from("replacement"));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.as_str(), "original");
    }

    #[test]
    fn concurrent_default_is_empty() {
        let reg: ConcurrentRegistry<&'static str, i32> = ConcurrentRegistry::default();
        assert!(reg.is_empty());
    }
}
