// NOTE (ROADMAP_20260903_LLM M11): the `ResponseCache` goldens moved to
// `fluent-llm --test cache` (canonical owner `fluent_llm::cache`) in M4,
// and M11 deleted the `common_core::cache` shims (with the two shim-lock
// tests that pinned them). This file keeps the generic suites
// (`LoadCache`, `ArcLoadCache`, weighted-LRU eviction engine, M10.1
// characterization).

use common_core::cache::*;

use std::sync::{Arc, Mutex};

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

#[test]
fn arc_load_cache_hit_is_arc_clone_not_vec_clone() {
        let cache: ArcLoadCache<String, Vec<f32>, String> =
            ArcLoadCache::new(10, |_: &String| Ok(std::sync::Arc::new(vec![1.0; 768]))).unwrap();
        let inserted = std::sync::Arc::new(vec![1.0_f32; 768]);
        cache.insert("k".to_string(), std::sync::Arc::clone(&inserted));
        let a = cache.get::<str>("k").expect("hit");
        let b = cache.get::<str>("k").expect("hit");
        // Both hits point to same allocation (Arc ptr equality)
        assert!(std::sync::Arc::ptr_eq(&a, &b), "ArcLoadCache must return Arc::clone, not Vec clone");
        assert!(std::sync::Arc::ptr_eq(&a, &inserted), "inserted Arc must be same allocation");
        // LoadCache clone would allocate new Vec; check that ArcLoadCache does not clone Vec
        assert_eq!(a.len(), 768);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.capacity(), 10);
        assert!(!cache.is_empty());
}

#[test]
fn arc_load_cache_get_or_load_caches_arc() {
        let loads = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let loads2 = std::sync::Arc::clone(&loads);
        let cache: ArcLoadCache<String, String, String> =
            ArcLoadCache::new(10, move |k: &String| {
                *loads2.lock().unwrap() += 1;
                Ok(std::sync::Arc::new(format!("loaded:{k}")))
            })
            .unwrap();
        let v1 = cache.get_or_load("x".to_string()).unwrap();
        assert_eq!(v1.as_str(), "loaded:x");
        let v2 = cache.get_or_load("x".to_string()).unwrap();
        assert!(std::sync::Arc::ptr_eq(&v1, &v2), "second load must be cached Arc");
        assert_eq!(*loads.lock().unwrap(), 1);
}

// ─── M10.1 characterization: eviction-order ambiguity + empty store ───
// Locks the documented "evict largest × coldest" semantics: footprint ×
// coldness descending, then last_used descending. In particular this
// disagrees with a pure-LRU ordering on the pair below, pinning the
// footprint-weighted choice.

#[test]
fn eviction_order_footprint_coldness_disagrees_with_pure_lru() {
        // now=10. big-recent: freed=100, last_used=9 → coldness 1 → score 100.
        // small-old: freed=10, last_used=0 → coldness 10 → score 100.
        // Scores tie → newer last_used first: big before small.
        // A pure-LRU-first ordering would evict small (older) first instead.
        let cands = vec![(100u64, 9i64), (10u64, 0i64)];
        let ordered = eviction_order(cands, 10, |c| c.0, |c| c.1);
        assert_eq!(ordered, vec![(100, 9), (10, 0)]);
}

#[tokio::test]
async fn evict_until_fit_empty_candidates_leaves_used_unchanged() {
        let cands: Vec<u64> = vec![];
        let (used, n) = evict_until_fit(100, 50, usize::MAX, cands, |c| {
                let v = *c;
                async move { Some(v) }
        })
        .await;
        assert_eq!(used, 100);
        assert_eq!(n, 0, "empty store evicts nothing");
}
