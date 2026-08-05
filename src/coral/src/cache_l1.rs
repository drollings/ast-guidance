use serde::{Deserialize, Serialize};
use std::sync::Arc;

use common_core::cache::LoadCache;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheTier {
    L1Memory,
    L2WasmWorkflow,
    L3Graph,
    L4Semantic,
    L4_5Decompose,
    L5Frontier,
}

impl std::fmt::Display for CacheTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheTier::L1Memory => write!(f, "L1"),
            CacheTier::L2WasmWorkflow => write!(f, "L2"),
            CacheTier::L3Graph => write!(f, "L3"),
            CacheTier::L4Semantic => write!(f, "L4"),
            CacheTier::L4_5Decompose => write!(f, "L4.5"),
            CacheTier::L5Frontier => write!(f, "L5"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingResult {
    pub query: String,
    pub result: String,
    pub tier: CacheTier,
}

pub struct L1Cache {
    inner: LoadCache<String, Arc<RoutingResult>, crate::error::CacheError>,
    max_entries: usize,
}

impl L1Cache {
    pub fn new() -> Self {
        Self::with_capacity(10_000).expect("default L1 capacity is non-zero")
    }

    pub fn with_capacity(max_entries: usize) -> Result<Self, crate::error::CacheError> {
        let load = |_: &String| -> Result<Arc<RoutingResult>, crate::error::CacheError> {
            Err(crate::error::CacheError::Miss)
        };
        let inner = LoadCache::new(max_entries, load)
            .map_err(|_| crate::error::CacheError::InvalidCapacity(max_entries))?;
        Ok(Self { inner, max_entries })
    }

    pub fn get(&self, query: &str) -> Option<Arc<RoutingResult>> {
        self.inner.get(query)
    }

    pub fn set(&self, query: String, result: Arc<RoutingResult>) {
        self.inner.insert(query, result);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
}

impl Default for L1Cache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(query: &str) -> Arc<RoutingResult> {
        Arc::new(RoutingResult {
            query: query.into(),
            result: format!("r_{query}"),
            tier: CacheTier::L1Memory,
        })
    }

    #[test]
    fn test_l1_cache_set_and_get() {
        let cache = L1Cache::new();
        let result = Arc::new(RoutingResult {
            query: "hello".into(),
            result: "world".into(),
            tier: CacheTier::L1Memory,
        });
        cache.set("hello".into(), Arc::clone(&result));
        let cached = cache.get("hello").expect("should exist");
        assert_eq!(cached.result, "world");
        assert_eq!(cached.tier, CacheTier::L1Memory);
    }

    #[test]
    fn test_l1_cache_miss() {
        let cache = L1Cache::new();
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_l1_cache_empty() {
        let cache = L1Cache::new();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_eviction() {
        let cache = L1Cache::with_capacity(2).unwrap();
        cache.set("a".into(), make_result("a"));
        cache.set("b".into(), make_result("b"));
        cache.set("c".into(), make_result("c"));
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_lru_renew_on_get() {
        let cache = L1Cache::with_capacity(2).unwrap();
        cache.set("a".into(), make_result("a"));
        cache.set("b".into(), make_result("b"));
        cache.get("a");
        cache.set("c".into(), make_result("c"));
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_max_entries() {
        let cache = L1Cache::with_capacity(5).unwrap();
        assert_eq!(cache.max_entries(), 5);
    }

    #[test]
    fn test_lru_capacity_bound() {
        let cache = L1Cache::with_capacity(3).unwrap();
        for i in 0..100 {
            cache.set(format!("key{i}"), make_result(&format!("key{i}")));
        }
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_zero_capacity_errors() {
        assert!(L1Cache::with_capacity(0).is_err());
    }
}
