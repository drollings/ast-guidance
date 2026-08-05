//! Typed in-process handoff accumulator.
//!
//! `OutputStore` is the primary channel for passing data between `WorkUnit`s
//! and pipeline stages that live in the same process. Unlike
//! `WorkOutput.data` (a `serde_json::Value` that forces a serialize /
//! deserialize round-trip at every boundary) and `WorkContext.metadata`
//! (untyped stringly-typed annotations), the store holds values as
//! `Arc<dyn Any + Send + Sync>` and exposes typed accessors: values are
//! written with `set::<T>(key, value)` and read with `get::<T>(key) ->
//! Option<&T>`. The compiler checks the type at the call site; a
//! `get::<T>` for the wrong `T` returns `None` instead of silently
//! mis-deserializing.
//!
//! Because the stored values are `Arc`-shared, `OutputStore` (and therefore
//! `WorkContext`) remains `Clone` without requiring every value to be
//! `Clone` — clones share the same typed allocation, which is exactly the
//! zero-copy handoff semantics an orchestrator wants when it fans a context
//! out to several stages.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// A typed accumulator for in-process handoff between work units and stages.
///
/// This is the **primary** inter-unit data channel (see the decision-rule doc
/// on `WorkContext`): write with [`OutputStore::set`], read with
/// [`OutputStore::get`]. Use `WorkOutput.data` only for payloads that
/// genuinely cross a serialization boundary (WASM units, network dispatch),
/// and `WorkContext.metadata` only for genuinely untyped annotations.
#[derive(Default)]
pub struct OutputStore {
    inner: HashMap<String, Arc<dyn Any + Send + Sync>>,
}

impl Clone for OutputStore {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for OutputStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputStore")
            .field("keys", &self.inner.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl OutputStore {
    /// Store a typed value under `key`, replacing any prior value.
    ///
    /// The value's concrete type is captured by `Arc<dyn Any>` at the call
    /// site; a later [`OutputStore::get::<T>`](Self::get) must use the same
    /// `T` to retrieve it.
    pub fn set<T: Send + Sync + 'static>(&mut self, key: impl Into<String>, value: T) {
        self.inner.insert(key.into(), Arc::new(value));
    }

    /// Read the typed value stored under `key`, if one exists with exactly
    /// type `T`. A type mismatch (or a missing key) yields `None` — never a
    /// partial or mis-typed read.
    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        self.inner.get(key).and_then(|v| v.downcast_ref::<T>())
    }

    /// Whether a value exists under `key` (regardless of its type).
    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    /// Remove the value under `key`. Returns `true` if a value was present.
    pub fn remove(&mut self, key: &str) -> bool {
        self.inner.remove(key).is_some()
    }

    /// Number of stored handoff values.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the store holds no values.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Remove all values.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_roundtrips_typed_values_without_serde() {
        let mut store = OutputStore::default();
        store.set("count", 42_i64);
        store.set("label", "hello".to_string());
        store.set("flag", true);

        assert_eq!(store.get::<i64>("count"), Some(&42));
        assert_eq!(store.get::<String>("label"), Some(&"hello".to_string()));
        assert_eq!(store.get::<bool>("flag"), Some(&true));
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn store_type_mismatch_and_missing_key_yield_none() {
        let mut store = OutputStore::default();
        store.set("count", 42_i64);

        assert_eq!(store.get::<i64>("count"), Some(&42));
        assert_eq!(store.get::<u32>("count"), None, "wrong T must be None");
        assert_eq!(store.get::<i64>("missing"), None);
        assert!(store.contains_key("count"));
        assert!(!store.contains_key("missing"));
    }

    #[test]
    fn store_overwrite_and_remove() {
        let mut store = OutputStore::default();
        store.set("x", 1_i32);
        store.set("x", 2_i32);
        assert_eq!(store.get::<i32>("x"), Some(&2));

        assert!(store.remove("x"));
        assert!(!store.contains_key("x"));
        assert!(!store.remove("x"));
        assert!(store.is_empty());
    }

    #[test]
    fn store_clone_shares_typed_allocations() {
        let mut store = OutputStore::default();
        store.set("count", 42_i64);
        let cloned = store.clone();
        assert_eq!(cloned.get::<i64>("count"), Some(&42));
        // Mutating the clone does not affect the original.
        let mut cloned = cloned;
        cloned.remove("count");
        assert_eq!(store.get::<i64>("count"), Some(&42));
    }

    #[test]
    fn store_debug_lists_keys() {
        let mut store = OutputStore::default();
        store.set("k", 1_u8);
        let debug = format!("{store:?}");
        assert!(debug.contains("k"), "debug should list keys, got: {debug}");
    }
}
