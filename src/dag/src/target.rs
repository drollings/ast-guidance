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
mod tests {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn register_and_retrieve() {
        let mut reg = TargetRegistry::new();
        let target = Target::new()
            .id(1)
            .name("build".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(bitvec::bitvec![0, 1])
            .provides(bitvec::bitvec![1, 0])
            .command("cargo build".into())
            .essential(true)
            .build();
        reg.register(target).unwrap();
        assert_eq!(reg.len(), 1);
        let t = reg.get("build").unwrap();
        assert_eq!(t.id, 1);
        assert!(t.essential);
    }

    #[test]
    fn duplicate_target_errors() {
        let mut reg = TargetRegistry::new();
        let t1 = Target::new()
            .id(1)
            .name("dup".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(BitVec::new())
            .provides(BitVec::new())
            .build();
        let t2 = Target::new()
            .id(2)
            .name("dup".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(BitVec::new())
            .provides(BitVec::new())
            .build();
        reg.register(t1).unwrap();
        let err = reg.register(t2).unwrap_err();
        assert!(matches!(err, RegistryError::DuplicateTarget { .. }));
    }

    #[test]
    fn get_by_bit_index() {
        let mut reg = TargetRegistry::new();
        let t = Target::new()
            .id(42)
            .name("test".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Wasm)
            .depends(BitVec::new())
            .provides(BitVec::new())
            .build();
        reg.register(t).unwrap();
        let found = reg.get_by_bit_index(42).unwrap();
        assert_eq!(&*found.name, "test");
    }

    #[test]
    fn essential_and_abstract_targets() {
        let mut reg = TargetRegistry::new();
        for i in 0..3 {
            let ttype = if i == 0 {
                TargetType::Abstract
            } else {
                TargetType::File
            };
            let t = Target::new()
                .id(i)
                .name(ArcIntern::from(format!("t{i}")))
                .target_type(ttype)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(BitVec::new())
                .essential(i == 1)
                .build();
            reg.register(t).unwrap();
        }
        assert_eq!(reg.abstract_targets().len(), 1);
        assert_eq!(reg.essential_targets().len(), 1);
    }

    #[test]
    fn target_builder_with_defaults() {
        let t = Target::new()
            .id(1)
            .name("defaults".into())
            .target_type(TargetType::Phony)
            .executor(ExecutorKind::Native)
            .depends(BitVec::new())
            .provides(BitVec::new())
            .build();
        assert!(!t.essential);
        assert!(t.command.is_empty());
    }
}
