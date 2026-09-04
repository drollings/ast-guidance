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

