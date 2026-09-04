#![allow(unused_imports)]
#[allow(unused_imports)]
use fluent_wvr::store::*;
use fluent_wvr::prelude::*;
use fluent_wvr::{FieldAccess, WorkUnit, Component, Describable, FieldError};


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
