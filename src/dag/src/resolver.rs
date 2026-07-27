use std::collections::HashMap;

use crate::target::TargetRegistry;
use bitvec::vec::BitVec;
use common_core::error::ResolverError;
use fluent_types::TargetType;

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
    pub(crate) strict: bool,
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
    use fluent_types::{ExecutorKind, TargetType};
    use internment::ArcIntern;
    use std::time::Instant;

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

    #[test]
    fn test_non_strict_allows_unresolved_deps() {
        let targets = vec![Target::new()
            .id(0)
            .name("orphan".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(make_bitset(&[0, 1]))
            .provides(make_bitset(&[2]))
            .build()];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg).with_strict(false);
        let plan = resolver.resolve(&["orphan"]).expect("non-strict resolve");
        assert_eq!(plan.order, vec![0]);
    }

    #[test]
    fn test_deterministic_order() {
        let mut targets: Vec<Target> = Vec::new();
        for i in 0..50 {
            targets.push(
                Target::new()
                    .id(i as i64)
                    .name(ArcIntern::from(format!("target_{i:03}")))
                    .target_type(TargetType::File)
                    .executor(ExecutorKind::Native)
                    .depends(BitVec::new())
                    .provides(make_bitset(&[i]))
                    .build(),
            );
        }
        targets.push(
            Target::new()
                .id(100)
                .name("final".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends({
                    let mut bv = BitVec::new();
                    bv.resize(50, false);
                    for i in 0..50 {
                        bv.set(i, true);
                    }
                    bv
                })
                .provides(make_bitset(&[100]))
                .build(),
        );
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let plan1 = resolver.resolve(&["final"]).expect("resolve 1");
        let plan2 = resolver.resolve(&["final"]).expect("resolve 2");
        assert_eq!(plan1.order, plan2.order);
        assert_eq!(plan1.target_names, plan2.target_names);
    }

    #[test]
    fn test_deep_chain_resolve() {
        let depth = 200;
        let mut targets: Vec<Target> = Vec::new();
        for i in 0..depth {
            let deps = if i == 0 {
                BitVec::new()
            } else {
                make_bitset(&[i - 1])
            };
            targets.push(
                Target::new()
                    .id(i as i64)
                    .name(ArcIntern::from(format!("step_{i:04}")))
                    .target_type(TargetType::File)
                    .executor(ExecutorKind::Native)
                    .depends(deps)
                    .provides(make_bitset(&[i]))
                    .build(),
            );
        }
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let last = format!("step_{:04}", depth - 1);
        let plan = resolver.resolve(&[&last]).expect("deep chain resolve");
        assert_eq!(plan.order.len(), depth);
        for (i, &bit_idx) in plan.order.iter().enumerate() {
            assert_eq!(bit_idx, i);
        }
    }

    #[test]
    fn test_multiple_providers_for_capability() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("consumer".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[2]))
                .provides(BitVec::new())
                .build(),
            Target::new()
                .id(1)
                .name("provider_a".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[2]))
                .build(),
            Target::new()
                .id(2)
                .name("provider_b".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[2]))
                .build(),
        ];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver.resolve(&["consumer"]).expect("multi-provider resolve");
        assert_eq!(plan.order.len(), 3);
        assert!(plan.order.contains(&0));
        assert!(plan.order.contains(&1));
        assert!(plan.order.contains(&2));
        assert!(plan.order.iter().position(|&i| i == 1).unwrap() < plan.order.iter().position(|&i| i == 0).unwrap());
        assert!(plan.order.iter().position(|&i| i == 2).unwrap() < plan.order.iter().position(|&i| i == 0).unwrap());
    }

    #[test]
    fn test_self_providing_chain() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("self_provider".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[0]))
                .build(),
        ];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver.resolve(&["self_provider"]).expect("self-provide resolve");
        assert_eq!(plan.order, vec![0]);
    }

    #[test]
    fn test_empty_resolve() {
        let reg = TargetRegistry::new();
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver.resolve(&[]).expect("empty resolve on empty registry");
        assert!(plan.is_empty());
    }

    #[test]
    fn test_nonexistent_target() {
        let reg = TargetRegistry::new();
        let resolver = DependencyResolver::new(&reg);
        let result = resolver.resolve(&["nonexistent"]);
        assert!(matches!(result, Err(ResolverError::TargetNotFound(_))));
    }

    #[test]
    fn test_breadth_first_fan_out() {
        let fan_out = 100;
        let mut targets: Vec<Target> = Vec::new();
        targets.push(
            Target::new()
                .id(0)
                .name("root".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[0]))
                .build(),
        );
        let mut final_deps = BitVec::new();
        final_deps.resize(fan_out + 1, false);
        for i in 0..fan_out {
            let cap_idx = i + 1;
            final_deps.set(cap_idx, true);
            targets.push(
                Target::new()
                    .id(cap_idx as i64)
                    .name(ArcIntern::from(format!("leaf_{i:04}")))
                    .target_type(TargetType::File)
                    .executor(ExecutorKind::Native)
                    .depends(make_bitset(&[0]))
                    .provides(make_bitset(&[cap_idx]))
                    .build(),
            );
        }
        targets.push(
            Target::new()
                .id((fan_out + 1) as i64)
                .name("final".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(final_deps)
                .provides(make_bitset(&[fan_out + 1]))
                .build(),
        );
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let start = Instant::now();
        let plan = resolver.resolve(&["final"]).expect("fan-out resolve");
        let elapsed = start.elapsed();
        assert_eq!(plan.order.len(), fan_out + 2);
        assert_eq!(plan.order[0], 0);
        assert_eq!(*plan.order.last().unwrap(), (fan_out + 1) as usize);
        let per_node_us = elapsed.as_micros() as f64 / (fan_out + 2) as f64;
        eprintln!(
            "[perf] breadth_first_fan_out: {fan_out} leaves, {}us total, {per_node_us:.1}us/node",
            elapsed.as_micros()
        );
        assert!(elapsed.as_millis() < 200, "fan-out perf degraded: {}ms", elapsed.as_millis());
    }

    #[test]
    fn test_resolve_perf_large_dag() {
        let n = 300;
        let mut targets: Vec<Target> = Vec::new();
        for i in 0..n {
            let deps: BitVec = if i == 0 {
                BitVec::new()
            } else {
                let parent = i - 1;
                let mut bv = BitVec::new();
                bv.resize(parent + 1, false);
                bv.set(parent, true);
                bv
            };
            targets.push(
                Target::new()
                    .id(i as i64)
                    .name(ArcIntern::from(format!("t_{i:04}")))
                    .target_type(TargetType::File)
                    .executor(ExecutorKind::Native)
                    .depends(deps)
                    .provides(make_bitset(&[i]))
                    .build(),
            );
        }
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let start = Instant::now();
        let plan = resolver.resolve(&["t_0299"]).expect("large dag resolve");
        let elapsed = start.elapsed();
        eprintln!(
            "[perf] large_dag ({n} nodes linear): {}us total, {:.1}us/node",
            elapsed.as_micros(),
            elapsed.as_micros() as f64 / n as f64,
        );
        assert_eq!(plan.order.len(), n);
        assert!(elapsed.as_millis() < 100, "large DAG perf degraded: {}ms", elapsed.as_millis());
    }

    #[test]
    fn test_resolve_perf_diamond_deep() {
        let height = 80;
        let mut targets: Vec<Target> = Vec::new();
        for level in 0..height {
            let cap = level;
            let deps: BitVec = if level == 0 {
                BitVec::new()
            } else {
                make_bitset(&[level - 1])
            };
            targets.push(
                Target::new()
                    .id((level * 3) as i64)
                    .name(ArcIntern::from(format!("l{level:03}_base")))
                    .target_type(TargetType::File)
                    .executor(ExecutorKind::Native)
                    .depends(deps)
                    .provides(make_bitset(&[cap]))
                    .build(),
            );
            targets.push(
                Target::new()
                    .id((level * 3 + 1) as i64)
                    .name(ArcIntern::from(format!("l{level:03}_a")))
                    .target_type(TargetType::File)
                    .executor(ExecutorKind::Native)
                    .depends(make_bitset(&[cap]))
                    .provides(make_bitset(&[cap + 1]))
                    .build(),
            );
            targets.push(
                Target::new()
                    .id((level * 3 + 2) as i64)
                    .name(ArcIntern::from(format!("l{level:03}_b")))
                    .target_type(TargetType::File)
                    .executor(ExecutorKind::Native)
                    .depends(make_bitset(&[cap, cap + 1]))
                    .provides(make_bitset(&[cap + 2]))
                    .build(),
            );
        }
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let start = Instant::now();
        let top = format!("l{:03}_b", height - 1);
        let plan = resolver.resolve(&[&top]).expect("deep diamond resolve");
        let elapsed = start.elapsed();
        eprintln!(
            "[perf] deep_diamond (height={height}): {}us total, {} nodes resolved",
            elapsed.as_micros(),
            plan.order.len(),
        );
        assert!(plan.order.len() >= height * 2);
        assert!(elapsed.as_millis() < 200, "deep diamond perf degraded: {}ms", elapsed.as_millis());
    }

    #[test]
    fn test_abstract_resolution_with_multiple_providers() {
        let mut reg = TargetRegistry::new();
        reg.register(
            Target::new()
                .id(0)
                .name("build".into())
                .target_type(TargetType::Abstract)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0, 1]))
                .provides(make_bitset(&[2]))
                .essential(true)
                .build(),
        )
        .unwrap();
        reg.register(
            Target::new()
                .id(1)
                .name("rust_compile".into())
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
                .id(3)
                .name("rust_link".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .build(),
        )
        .unwrap();
        reg.register(
            Target::new()
                .id(4)
                .name("zig_link".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .build(),
        )
        .unwrap();
        let mut provided = BitVec::new();
        provided.resize(5, false);
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver
            .resolve_abstract_dependencies(&["build"], &provided)
            .expect("multi-provider abstract resolve");
        assert!(plan.order.contains(&0));
        assert!(plan.order.contains(&1) || plan.order.contains(&2));
        assert!(plan.order.contains(&3) || plan.order.contains(&4));
    }

    #[test]
    fn test_resolve_strict_missing_dependency_errors() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("consumer".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[99]))
                .provides(BitVec::new())
                .build(),
        ];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg).with_strict(true);
        let err = resolver.resolve(&["consumer"]).unwrap_err();
        assert!(matches!(err, ResolverError::MissingDependency(_)));
    }

    #[test]
    fn test_resolve_non_strict_skips_missing() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("consumer".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[99]))
                .provides(BitVec::new())
                .build(),
        ];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg).with_strict(false);
        let plan = resolver.resolve(&["consumer"]).expect("non-strict missing dep");
        assert_eq!(plan.order, vec![0]);
    }

    #[test]
    fn test_resolve_shared_dependency() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("lib".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[0]))
                .build(),
            Target::new()
                .id(1)
                .name("app_a".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .build(),
            Target::new()
                .id(2)
                .name("app_b".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[2]))
                .build(),
            Target::new()
                .id(3)
                .name("suite".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[1, 2]))
                .provides(BitVec::new())
                .build(),
        ];
        let reg = make_registry(targets);
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver.resolve(&["suite"]).expect("shared dep resolve");
        assert_eq!(plan.order[0], 0);
        assert_eq!(plan.order.len(), 4);
    }

    #[test]
    fn test_resolve_abstract_excludes_non_essential() {
        let mut reg = TargetRegistry::new();
        reg.register(
            Target::new()
                .id(0)
                .name("essential_root".into())
                .target_type(TargetType::Abstract)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .essential(true)
                .build(),
        )
        .unwrap();
        reg.register(
            Target::new()
                .id(1)
                .name("non_essential_provider".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[0]))
                .essential(false)
                .build(),
        )
        .unwrap();
        reg.register(
            Target::new()
                .id(2)
                .name("essential_provider".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[0]))
                .essential(true)
                .build(),
        )
        .unwrap();
        let mut provided = BitVec::new();
        provided.resize(3, false);
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver
            .resolve_abstract_dependencies(&["essential_root"], &provided)
            .expect("essential abstract resolve");
        assert!(plan.order.contains(&0));
        assert!(plan.order.contains(&1) || plan.order.contains(&2));
    }
}
