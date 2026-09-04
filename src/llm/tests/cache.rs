//! ROADMAP_20260903_LLM M4.3 — LLM response-cache goldens (moved, not copied).
//!
//! Canonical home for every `ResponseCache` golden: the TTL hit/miss,
//! unknown-key miss, per-model isolation, and lazy-eviction assertions
//! moved from `src/common-core/tests/cache.rs`, plus the must-MISS control
//! group (expired entry, cross-model same-text, backend-returns-`None`)
//! and the TTL-boundary lock (`age >= ttl` misses, `age == ttl - 1` hits).
//! Behavior is byte-identical to the removed `common_core::cache` shims
//! (M11 deleted them with the `parity_new_eq_old` dual-path test).
//!
//! Calibration (roadmap §1, M10): cache identity is task-value
//! freshness/identity, not endorsement — a hit is never a correctness
//! vote, and a different model or an expired TTL must miss even on
//! identical text.

use fluent_llm::cache::*;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type Store = Arc<Mutex<HashMap<String, CachedResponse>>>;

fn make_cache(store: Store) -> ResponseCache {
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

// ── Moved from common-core/tests/cache.rs ─────────────────────────────────

#[test]
fn response_cache_identical_request_hits() {
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
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
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
    let cache = make_cache(Arc::clone(&store));
    let req1 = r#"{"model":"test","messages":[],"temperature":0.5}"#;
    let req2 = r#"{"model":"test","messages":[],"temperature":1.0}"#;
    cache.set("gpt-4", req1, serde_json::json!({"result": "a"}));
    // Different temperature → different cache key → miss
    assert!(cache.get("gpt-4", req2).is_none());
}

#[test]
fn response_cache_ttl_expiry_misses() {
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
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
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
    let cache = make_cache(Arc::clone(&store));
    // No entry stored → should be None
    assert!(cache.get("gpt-4", "anything").is_none());
}

#[test]
fn response_cache_invalidate_key() {
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
    let cache = make_cache(Arc::clone(&store));
    let req = r#"{"x":1}"#;
    cache.set("gpt-4", req, serde_json::json!({"result": "data"}));
    assert!(cache.get("gpt-4", req).is_some());
    cache.invalidate_key("gpt-4", req);
    assert!(cache.get("gpt-4", req).is_none());
}

#[test]
fn response_cache_invalidate_all() {
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
    let cache = make_cache(Arc::clone(&store));
    cache.set("gpt-4", r#"{"a":1}"#, serde_json::json!({"result": "a"}));
    cache.set("gpt-4", r#"{"b":2}"#, serde_json::json!({"result": "b"}));
    cache.invalidate_all();
    assert!(cache.get("gpt-4", r#"{"a":1}"#).is_none());
    assert!(cache.get("gpt-4", r#"{"b":2}"#).is_none());
}

#[test]
fn response_cache_different_model_different_key() {
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
    let cache = make_cache(Arc::clone(&store));
    let req = r#"{"model":"test"}"#;
    cache.set("gpt-4", req, serde_json::json!({"result": "gpt4"}));
    // Same request JSON but different model → different key → miss
    assert!(cache.get("claude-3", req).is_none());
}

// ── Controls: must MISS (freshness/identity, never endorsement) ───────────

#[test]
fn control_expired_cross_model_and_unknown_must_miss() {
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
    // Backdated entry under a real TTL: already expired at write time.
    let now = common_core::time::now_secs();
    let ttl = std::time::Duration::from_secs(60);
    let req = r#"{"model":"test"}"#;
    let stale_key = format!(
        "gpt-4:{}",
        common_core::hash::sha256_hex(req.as_bytes())
    );
    store.lock().unwrap().insert(
        stale_key.clone(),
        CachedResponse {
            stored_at_secs: now - 61,
            response_json: serde_json::json!({"result": "stale"}),
        },
    );
    let s_check = Arc::clone(&store);
    let s_store = Arc::clone(&store);
    let cache = ResponseCache::new(
        Some(ttl),
        move |key: &str| s_check.lock().unwrap().get(key).cloned(),
        move |key: &str, value: &CachedResponse| {
            s_store
                .lock()
                .unwrap()
                .insert(key.to_string(), value.clone());
        },
    );
    // Expired entry must miss …
    assert!(cache.get("gpt-4", req).is_none());
    // … but lazy eviction leaves the row in the backend (no eager delete).
    assert!(store.lock().unwrap().contains_key(&stale_key));
    // Cross-model same-text must miss even when fresh.
    cache.set("gpt-4", req, serde_json::json!({"result": "fresh"}));
    assert!(cache.get("claude-3", req).is_none());
    // Unknown key must miss.
    assert!(cache.get("gpt-4", r#"{"never":"stored"}"#).is_none());
}

#[test]
fn control_ttl_boundary_is_locked() {
    // `age >= ttl` misses; well under TTL hits. Locks the `>=` comparison:
    // the edge entry (age == ttl exactly) must miss like any expired entry.
    // (The hit side uses a 10s margin so a wall-clock second rollover
    // between insert and `get` cannot flip it — the miss side is
    // rollover-safe in both directions.)
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
    let now = common_core::time::now_secs();
    let ttl_secs = 60u64;
    for (req, age) in [("edge-req", ttl_secs), ("inside-req", ttl_secs - 10)] {
        let key = format!("m:{}", common_core::hash::sha256_hex(req.as_bytes()));
        store.lock().unwrap().insert(
            key,
            CachedResponse {
                stored_at_secs: now - age,
                response_json: serde_json::json!({"req": req}),
            },
        );
    }
    let s = Arc::clone(&store);
    let cache = ResponseCache::new(
        Some(std::time::Duration::from_secs(ttl_secs)),
        move |key: &str| s.lock().unwrap().get(key).cloned(),
        |_: &str, _: &CachedResponse| {},
    );
    assert!(
        cache.get("m", "edge-req").is_none(),
        "age == ttl must miss (>= comparison)"
    );
    assert!(
        cache.get("m", "inside-req").is_some(),
        "age == ttl - 10 must hit"
    );
}

#[test]
fn control_key_format_is_model_colon_sha256() {
    // The one `"{model}:…"` key format: `model:sha256_hex(request bytes)`.
    let store: Store = Arc::new(Mutex::new(HashMap::new()));
    let cache = make_cache(Arc::clone(&store));
    let req = r#"{"model":"test","messages":[]}"#;
    cache.set("gpt-4", req, serde_json::json!({"result": "a"}));
    let want = format!("gpt-4:{}", common_core::hash::sha256_hex(req.as_bytes()));
    let keys: Vec<String> = store.lock().unwrap().keys().cloned().collect();
    assert_eq!(keys, vec![want]);
}
