//! Capability registry: thread-safe string ↔ BitVec index for capability bitsets.

use bitvec::vec::BitVec;
use internment::ArcIntern;
use std::collections::HashMap;
use std::sync::RwLock;

pub struct CapabilityRegistry {
    names: RwLock<HashMap<ArcIntern<str>, usize>>,
    indices: RwLock<Vec<ArcIntern<str>>>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        Self {
            names: RwLock::new(HashMap::new()),
            indices: RwLock::new(Vec::new()),
        }
    }

    pub fn intern(&self, name: &str) -> usize {
        let interned: ArcIntern<str> = ArcIntern::from(name);
        {
            let names = self.names.read().unwrap();
            if let Some(&idx) = names.get(&interned) {
                return idx;
            }
        }
        let mut names = self.names.write().unwrap();
        if let Some(&idx) = names.get(&interned) {
            return idx;
        }
        let idx = names.len();
        names.insert(interned.clone(), idx);
        self.indices.write().unwrap().push(interned);
        idx
    }

    pub fn get_index(&self, name: &str) -> Option<usize> {
        let interned: ArcIntern<str> = ArcIntern::from(name);
        self.names.read().unwrap().get(&interned).copied()
    }

    pub fn get_name(&self, idx: usize) -> Option<ArcIntern<str>> {
        self.indices.read().unwrap().get(idx).cloned()
    }

    pub fn intern_list(&self, names: &[&str]) {
        for name in names {
            self.intern(name);
        }
    }

    pub fn to_bitvec(&self, names: &[&str]) -> BitVec {
        let mut bits = BitVec::new();
        for name in names {
            let idx = self.intern(name);
            if idx >= bits.len() {
                bits.resize(idx + 1, false);
            }
            bits.set(idx, true);
        }
        bits
    }

    pub fn bitvec_to_names(&self, bits: &BitVec) -> Vec<ArcIntern<str>> {
        let indices = self.indices.read().unwrap();
        bits.iter_ones()
            .filter_map(|i| indices.get(i).cloned())
            .collect()
    }

    pub fn count(&self) -> usize {
        self.names.read().unwrap().len()
    }
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

