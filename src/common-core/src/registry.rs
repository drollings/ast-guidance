//! Generic keyed registry — the canonical "register by key, look up by key"
//! primitive for the workspace.
//!
//! Consumers that store values under a string-ish key (guidance's
//! `PluginRegistry`, `memory-plugin`'s `MemoryPluginRegistry`) compose
//! `KeyedRegistry<K, V>` instead of hand-wrapping a `HashMap`/`BTreeMap`.
//! Zero-domain: `K` and `V` are fully generic, so no domain type leaks into
//! `common-core`.

use std::borrow::Borrow;
use std::collections::BTreeMap;

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
}
