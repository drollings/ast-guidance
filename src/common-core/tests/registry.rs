use common_core::registry::*;
use std::sync::Arc;


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
