use std::collections::HashMap;

use crate::target::TargetRegistry;
use bitvec::vec::BitVec;
use common_core::error::ResolverError;
use guidance_types::TargetType;

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub order: Vec<usize>,
    pub target_names: Vec<String>,
}

impl ExecutionPlan {
    pub fn len(&self) -> usize {
        self.order.len()
    }
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

/// Resolves target dependencies into an execution order using Kahn's algorithm.
///
/// # Examples
///
/// ```no_run
/// use fluent_dag::resolver::DependencyResolver;
/// use fluent_dag::target::TargetRegistry;
///
/// let registry = TargetRegistry::new();
/// // ... register targets ...
/// let resolver = DependencyResolver::new(&registry);
/// let plan = resolver.resolve(&["build".into()]).unwrap();
/// assert!(!plan.is_empty());
/// ```
pub struct DependencyResolver<'a> {
    registry: &'a TargetRegistry,
    strict: bool,
}

impl<'a> DependencyResolver<'a> {
    pub fn new(registry: &'a TargetRegistry) -> Self {
        Self {
            registry,
            strict: true,
        }
    }
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub fn resolve(&self, target_names: &[&str]) -> Result<ExecutionPlan, ResolverError> {
        let mut needed: HashMap<usize, usize> = HashMap::new();
        let mut stack: Vec<usize> = Vec::new();
        for name in target_names {
            let target = self
                .registry
                .get(name)
                .ok_or_else(|| ResolverError::TargetNotFound(name.to_string()))?;
            stack.push(target.id as usize);
        }
        while let Some(bit_idx) = stack.pop() {
            if needed.contains_key(&bit_idx) {
                continue;
            }
            let target =
                self.registry
                    .get_by_bit_index(bit_idx)
                    .ok_or(ResolverError::TargetNotFound(format!(
                        "bit_index {bit_idx}"
                    )))?;
            needed.insert(bit_idx, bit_idx);
            for cap_idx in target.depends.iter_ones() {
                let providers = self.registry.get_providers(cap_idx);
                if providers.is_empty() && self.strict {
                    return Err(ResolverError::MissingDependency(format!(
                        "no provider for capability {cap_idx} required by '{}'",
                        target.name
                    )));
                }
                for provider in providers {
                    let provider_bit_idx = provider.id as usize;
                    if !needed.contains_key(&provider_bit_idx) {
                        stack.push(provider_bit_idx);
                    }
                }
            }
        }
        let mut in_degree: HashMap<usize, usize> = needed.keys().map(|&k| (k, 0)).collect();
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &bit_idx in needed.keys() {
            let target =
                self.registry
                    .get_by_bit_index(bit_idx)
                    .ok_or(ResolverError::TargetNotFound(format!(
                        "bit_index {bit_idx}"
                    )))?;
            for cap_idx in target.depends.iter_ones() {
                let providers = self.registry.get_providers(cap_idx);
                for provider in providers {
                    let provider_bit_idx = provider.id as usize;
                    if needed.contains_key(&provider_bit_idx) && provider_bit_idx != bit_idx {
                        adj.entry(provider_bit_idx).or_default().push(bit_idx);
                        *in_degree.get_mut(&bit_idx).unwrap() += 1;
                    }
                }
            }
        }
        let mut queue: Vec<usize> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&k, _)| k)
            .collect();
        queue.sort_unstable();
        let mut order = Vec::with_capacity(needed.len());
        let mut head = 0;
        while head < queue.len() {
            let current = queue[head];
            head += 1;
            order.push(current);
            if let Some(dependents) = adj.get(&current) {
                for &dep in dependents {
                    if let Some(deg) = in_degree.get_mut(&dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(dep);
                            queue[head..].sort_unstable();
                        }
                    }
                }
            }
        }
        if order.len() != needed.len() {
            return Err(ResolverError::CircularDependency);
        }
        let target_names = order
            .iter()
            .map(|&bit_idx| {
                self.registry
                    .get_by_bit_index(bit_idx)
                    .map_or_else(|| format!("bit_{bit_idx}"), |t| t.name.to_string())
            })
            .collect();
        Ok(ExecutionPlan {
            order,
            target_names,
        })
    }

    pub fn resolve_abstract_dependencies(
        &self,
        target_names: &[&str],
        provided: &BitVec,
    ) -> Result<ExecutionPlan, ResolverError> {
        let mut combined: Vec<String> = target_names.iter().map(ToString::to_string).collect();
        for name in target_names {
            let target = self
                .registry
                .get(name)
                .ok_or_else(|| ResolverError::TargetNotFound(name.to_string()))?;
            if target.target_type == TargetType::Abstract {
                let required = &target.depends;
                let missing: BitVec = required.clone() & !provided.clone();
                if missing.not_any() {
                    continue;
                }
                for cap_idx in missing.iter_ones() {
                    for provider in self.registry.get_providers(cap_idx) {
                        let pname = provider.name.to_string();
                        if !combined.contains(&pname) {
                            combined.push(pname);
                        }
                    }
                }
            }
        }
        let names: Vec<&str> = combined.iter().map(String::as_str).collect();
        self.resolve(&names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{Target, TargetRegistry};
    use guidance_types::{ExecutorKind, TargetType};

    fn make_bitset(bits: &[usize]) -> BitVec {
        let max = bits.iter().max().copied().unwrap_or(0) + 1;
        let mut bv = BitVec::with_capacity(max);
        bv.resize(max, false);
        for &bit in bits {
            if bit < bv.len() {
                bv.set(bit, true);
            }
        }
        bv
    }

    fn make_registry(targets: Vec<Target>) -> TargetRegistry {
        let mut reg = TargetRegistry::new();
        for t in targets {
            reg.register(t).unwrap();
        }
        reg
    }

    #[test]
    fn test_linear_chain() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("compile".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[0]))
                .build(),
            Target::new()
                .id(1)
                .name("link".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .build(),
            Target::new()
                .id(2)
                .name("build".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[1]))
                .provides(make_bitset(&[2]))
                .build(),
        ];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver.resolve(&["build"]).expect("resolve");
        assert_eq!(plan.order, vec![0, 1, 2]);
    }

    #[test]
    fn test_diamond_graph() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("base".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[0]))
                .build(),
            Target::new()
                .id(1)
                .name("left".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .build(),
            Target::new()
                .id(2)
                .name("right".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[2]))
                .build(),
            Target::new()
                .id(3)
                .name("top".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[1, 2]))
                .provides(make_bitset(&[3]))
                .build(),
        ];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver.resolve(&["top"]).expect("resolve");
        assert_eq!(plan.order.len(), 4);
        assert_eq!(plan.order[0], 0);
        assert_eq!(plan.order[3], 3);
    }

    #[test]
    fn test_missing_dependency_strict() {
        let targets = vec![Target::new()
            .id(0)
            .name("orphan".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(make_bitset(&[0, 1]))
            .provides(make_bitset(&[2]))
            .build()];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg).with_strict(true);
        assert!(resolver.resolve(&["orphan"]).is_err());
    }

    #[test]
    fn test_circular_dependency() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("a".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[1]))
                .provides(make_bitset(&[0]))
                .build(),
            Target::new()
                .id(1)
                .name("b".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .build(),
        ];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        assert!(matches!(
            resolver.resolve(&["a"]),
            Err(ResolverError::CircularDependency)
        ));
    }

    #[test]
    fn test_abstract_dependency_resolution() {
        let mut reg = TargetRegistry::new();
        reg.register(
            Target::new()
                .id(0)
                .name("build".into())
                .target_type(TargetType::Abstract)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0, 1]))
                .provides(make_bitset(&[2]))
                .build(),
        )
        .unwrap();
        reg.register(
            Target::new()
                .id(1)
                .name("zig_compile".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[0]))
                .build(),
        )
        .unwrap();
        reg.register(
            Target::new()
                .id(2)
                .name("zig_link".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .build(),
        )
        .unwrap();
        let mut provided = BitVec::new();
        provided.resize(3, false);
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver
            .resolve_abstract_dependencies(&["build"], &provided)
            .expect("resolve abstract");
        assert!(plan.len() >= 2);
    }
}
