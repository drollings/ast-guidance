use bitvec::vec::BitVec;
use bon::Builder;
pub use fluent_types::{ExecutorKind, TargetType};
use internment::ArcIntern;
use std::collections::HashMap;

use crate::error::RegistryError;

#[derive(Debug, Clone, Builder)]
#[builder(start_fn = new)]
pub struct Target {
    pub id: i64,
    pub name: ArcIntern<str>,
    pub target_type: TargetType,
    pub executor: ExecutorKind,
    pub depends: BitVec,
    pub provides: BitVec,
    #[builder(default)]
    pub command: String,
    #[builder(default = false)]
    pub essential: bool,
}

#[derive(Debug, Clone)]
pub struct TargetRegistry {
    targets: Vec<Target>,
    by_name: HashMap<ArcIntern<str>, usize>,
    by_bit_index: HashMap<usize, usize>,
    providers: HashMap<usize, Vec<usize>>,
}

impl TargetRegistry {
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
            by_name: HashMap::new(),
            by_bit_index: HashMap::new(),
            providers: HashMap::new(),
        }
    }

    pub fn register(&mut self, target: Target) -> Result<(), RegistryError> {
        if self.by_name.contains_key(&target.name) {
            return Err(RegistryError::DuplicateTarget {
                name: target.name.to_string(),
            });
        }
        let idx = self.targets.len();
        let bit_idx = target.id as usize;
        self.by_name.insert(target.name.clone(), idx);
        self.by_bit_index.insert(bit_idx, idx);
        for cap_idx in target.provides.iter_ones() {
            self.providers.entry(cap_idx).or_default().push(bit_idx);
        }
        self.targets.push(target);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Target> {
        let interned: ArcIntern<str> = ArcIntern::from(name);
        self.by_name.get(&interned).map(|&idx| &self.targets[idx])
    }

    pub fn get_by_index(&self, idx: usize) -> Option<&Target> {
        self.targets.get(idx)
    }

    pub fn get_by_bit_index(&self, bit_idx: usize) -> Option<&Target> {
        self.by_bit_index
            .get(&bit_idx)
            .map(|&idx| &self.targets[idx])
    }

    pub fn get_providers(&self, capability_bit_index: usize) -> Vec<&Target> {
        self.providers
            .get(&capability_bit_index)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|bit_idx| {
                        self.by_bit_index
                            .get(bit_idx)
                            .map(|&idx| &self.targets[idx])
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn find_providers(&self, required: &BitVec) -> Vec<&Target> {
        self.targets
            .iter()
            .filter(|t| {
                let prov = &t.provides;
                let missing: BitVec = required.clone() & !prov.clone();
                missing.not_any()
            })
            .collect()
    }

    pub fn list_names(&self) -> Vec<ArcIntern<str>> {
        self.targets.iter().map(|t| t.name.clone()).collect()
    }

    pub fn essential_targets(&self) -> Vec<&Target> {
        self.targets.iter().filter(|t| t.essential).collect()
    }

    pub fn abstract_targets(&self) -> Vec<&Target> {
        self.targets
            .iter()
            .filter(|t| t.target_type == TargetType::Abstract)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.targets.len()
    }
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

impl Default for TargetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub use common_core::interner::CapabilityRegistry;

#[cfg(test)]
#[path = "../tests/target.rs"]
mod tests;
