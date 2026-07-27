/// Ambiguous dependency resolver: extends Kahn's algorithm with
/// yamake-old.py-style provider disambiguation for abstract targets.
///
/// ## Design
///
/// The resolver operates in three phases:
///
/// 1. **Transitive closure** — given seed targets, compute the full set of
///    targets transitively needed, tracking which capability bits are
///    required and which are satisfied.
///
/// 2. **Disambiguation** — for each unsatisfied capability with multiple
///    possible providers, apply narrowing rules (essential filtering,
///    locality preference, dependency satisfaction) to select exactly one
///    provider. Report ambiguity when narrowing fails.
///
/// 3. **Kahn's topological sort** — feed the narrowed set into Kahn's
///    algorithm for deterministic ordering + cycle detection.
///
/// Phase 2 is what distinguishes this resolver from `DependencyResolver`:
/// instead of including *all* providers for a multiply-satisfied capability,
/// it narrows to one using the same set-logic heuristics as yamake-old.py.
use std::collections::HashSet;

use bitvec::vec::BitVec;

use common_core::error::ResolverError;
use common_core::interner::CapabilityRegistry;
use fluent_types::TargetType;

use crate::target::{Target, TargetRegistry};

use super::resolver::{DependencyResolver, ExecutionPlan};

#[derive(Debug, Clone)]
pub struct AmbiguityReport {
    pub dependency: String,
    pub candidates: Vec<String>,
}

pub struct AmbiguousDependencyResolver<'a> {
    registry: &'a TargetRegistry,
    caps: &'a CapabilityRegistry,
    inner: DependencyResolver<'a>,
}

impl<'a> AmbiguousDependencyResolver<'a> {
    pub fn new(registry: &'a TargetRegistry, caps: &'a CapabilityRegistry) -> Self {
        Self {
            registry,
            caps,
            inner: DependencyResolver::new(registry),
        }
    }

    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.inner = self.inner.with_strict(strict);
        self
    }

    pub fn resolve(&self, target_names: &[&str]) -> Result<ExecutionPlan, ResolverError> {
        if target_names.is_empty() {
            return self.inner.resolve(target_names);
        }

        let has_abstract = target_names.iter().any(|name| {
            self.registry
                .get(name)
                .is_some_and(|t| t.target_type == TargetType::Abstract)
        });

        if !has_abstract {
            let mut needs_disambig = false;
            let (tc, _) = self.compute_transitive_closure(target_names)?;
            for &bit in &tc {
                if let Some(target) = self.registry.get_by_bit_index(bit) {
                    for cap in target.depends.iter_ones() {
                        let providers = self.registry.get_providers(cap);
                        if providers.len() > 1 {
                            needs_disambig = true;
                            break;
                        }
                    }
                }
                if needs_disambig {
                    break;
                }
            }
            if !needs_disambig {
                return self.inner.resolve(target_names);
            }
        }

        let mut combined: Vec<String> = target_names.iter().map(|s| s.to_string()).collect();
        let mut resolved_set: HashSet<usize> = HashSet::new();

        for name in target_names {
            let target = self.registry.get(name).ok_or_else(|| {
                ResolverError::TargetNotFound(name.to_string())
            })?;
            resolved_set.insert(target.id as usize);
        }

        let mut changed = true;
        while changed {
            changed = false;

            let full_provides = self.compute_full_provides(&resolved_set);

            for name in combined.clone() {
                let target = self.registry.get(&name).ok_or_else(|| {
                    ResolverError::TargetNotFound(name.clone())
                })?;

                let is_abstract = target.target_type == TargetType::Abstract;
                let deps_satisfied = target.depends.not_any()
                    || target
                        .depends
                        .iter_ones()
                        .all(|c| full_provides.contains(&c));

                if !deps_satisfied {
                    for cap_idx in target.depends.iter_ones() {
                        if full_provides.contains(&cap_idx) {
                            continue;
                        }
                        let providers = self.registry.get_providers(cap_idx);
                        if providers.is_empty() {
                            if self.inner.strict {
                                let cap_name = self
                                    .caps
                                    .get_name(cap_idx)
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| format!("cap_{cap_idx}"));
                                return Err(ResolverError::MissingDependency(format!(
                                    "no provider for '{cap_name}' (required by '{}')",
                                    target.name
                                )));
                            }
                            continue;
                        }
                        if providers.len() == 1 {
                            let pname = providers[0].name.to_string();
                            if !combined.contains(&pname) {
                                combined.push(pname);
                                resolved_set.insert(providers[0].id as usize);
                                changed = true;
                            }
                            continue;
                        }
                        let narrowed = self.narrow_providers(providers, &full_provides);
                        match narrowed.len() {
                            0 => {
                                let cap_name = self
                                    .caps
                                    .get_name(cap_idx)
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| format!("cap_{cap_idx}"));
                                return Err(ResolverError::MissingDependency(format!(
                                    "no provider for '{cap_name}' (required by '{}') after narrowing",
                                    target.name
                                )));
                            }
                            1 => {
                                let pname = narrowed[0].name.to_string();
                                if !combined.contains(&pname) {
                                    combined.push(pname);
                                    resolved_set.insert(narrowed[0].id as usize);
                                    changed = true;
                                }
                            }
                            _ => {
                                let cap_name = self
                                    .caps
                                    .get_name(cap_idx)
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| format!("cap_{cap_idx}"));
                                let candidates: Vec<String> = narrowed
                                    .iter()
                                    .map(|t| t.name.to_string())
                                    .collect();
                                return Err(ResolverError::AmbiguousDependency {
                                    name: cap_name,
                                    candidates,
                                });
                            }
                        }
                    }
                }

                if is_abstract || target.depends.not_any() {
                    let cap_name = &name;
                    let cap_idx = self.caps.get_index(cap_name);
                    if let Some(ci) = cap_idx {
                        let providers = self.registry.get_providers(ci);
                        if providers.len() > 1 {
                            let narrowed = self.narrow_providers(providers, &full_provides);
                            match narrowed.len() {
                                1 => {
                                    let pname = narrowed[0].name.to_string();
                                    if !combined.contains(&pname) {
                                        combined.push(pname);
                                        resolved_set.insert(narrowed[0].id as usize);
                                        changed = true;
                                    }
                                }
                                0 => {}
                                _ => {
                                    let candidates: Vec<String> = narrowed
                                        .iter()
                                        .map(|t| t.name.to_string())
                                        .collect();
                                    return Err(ResolverError::AmbiguousDependency {
                                        name: name.clone(),
                                        candidates,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        let tc = self.expand_transitive_closure(&resolved_set)?;
        let disambiguated: HashSet<usize> = resolved_set.union(&tc).copied().collect();

        let names: Vec<&str> = disambiguated
            .iter()
            .filter_map(|&bit_idx| {
                self.registry
                    .get_by_bit_index(bit_idx)
                    .map(|t| t.name.as_ref())
            })
            .collect();

        self.inner.resolve(&names)
    }

    fn expand_transitive_closure(
        &self,
        seed: &HashSet<usize>,
    ) -> Result<HashSet<usize>, ResolverError> {
        let mut needed: HashSet<usize> = HashSet::new();
        let mut stack: Vec<usize> = seed.iter().copied().collect();

        while let Some(bit_idx) = stack.pop() {
            if needed.contains(&bit_idx) || seed.contains(&bit_idx) {
                continue;
            }
            let target = self
                .registry
                .get_by_bit_index(bit_idx)
                .ok_or(ResolverError::TargetNotFound(format!(
                    "bit_index {bit_idx}"
                )))?;

            needed.insert(bit_idx);

            for cap_idx in target.depends.iter_ones() {
                let providers = self.registry.get_providers(cap_idx);
                if providers.is_empty() {
                    let cap_name = self.caps.get_name(cap_idx);
                    if let Some(ref name) = cap_name {
                        if let Some(implicit) = self.registry.get(name) {
                            let p_bit = implicit.id as usize;
                            if !needed.contains(&p_bit) && !seed.contains(&p_bit) {
                                stack.push(p_bit);
                            }
                            continue;
                        }
                    }
                    if self.inner.strict {
                        let cap_name = self
                            .caps
                            .get_name(cap_idx)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("cap_{cap_idx}"));
                        return Err(ResolverError::MissingDependency(format!(
                            "no provider for '{cap_name}' required by '{}'",
                            target.name
                        )));
                    }
                    continue;
                }
                for provider in providers {
                    let p_bit = provider.id as usize;
                    if !needed.contains(&p_bit) && !seed.contains(&p_bit) {
                        stack.push(p_bit);
                    }
                }
            }
        }

        Ok(needed)
    }

    fn compute_transitive_closure(
        &self,
        seed: &[&str],
    ) -> Result<(HashSet<usize>, Vec<usize>), ResolverError> {
        let mut needed: HashSet<usize> = HashSet::new();
        let mut abstract_targets: Vec<usize> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();

        for name in seed {
            let target = self
                .registry
                .get(name)
                .ok_or_else(|| ResolverError::TargetNotFound(name.to_string()))?;
            stack.push(target.id as usize);
        }

        while let Some(bit_idx) = stack.pop() {
            if needed.contains(&bit_idx) {
                continue;
            }
            let target = self
                .registry
                .get_by_bit_index(bit_idx)
                .ok_or(ResolverError::TargetNotFound(format!(
                    "bit_index {bit_idx}"
                )))?;

            needed.insert(bit_idx);

            if target.target_type == TargetType::Abstract {
                abstract_targets.push(bit_idx);
            }

            for cap_idx in target.depends.iter_ones() {
                let providers = self.registry.get_providers(cap_idx);
                if providers.is_empty() && self.inner.strict {
                    return Err(ResolverError::MissingDependency(format!(
                        "no provider for capability {cap_idx} required by '{}'",
                        target.name
                    )));
                }
                for provider in providers {
                    let p_bit = provider.id as usize;
                    if !needed.contains(&p_bit) {
                        stack.push(p_bit);
                    }
                }
            }
        }

        Ok((needed, abstract_targets))
    }

    fn narrow_providers<'t>(
        &self,
        candidates: Vec<&'t Target>,
        full_provides: &HashSet<usize>,
    ) -> Vec<&'t Target> {
        if candidates.len() <= 1 {
            return candidates;
        }

        let mut narrowed = candidates;

        narrowed.sort_by_key(|t| t.name.clone());

        let with_essential: Vec<&Target> = narrowed
            .iter()
            .copied()
            .filter(|t| t.essential)
            .collect();
        if !with_essential.is_empty() {
            narrowed = with_essential;
        }

        if narrowed.len() <= 1 {
            return narrowed;
        }

        let strict: Vec<&Target> = narrowed
            .iter()
            .copied()
            .filter(|t| {
                if t.depends.not_any() {
                    return true;
                }
                t.depends.iter_ones().all(|c| full_provides.contains(&c))
            })
            .collect();

        if !strict.is_empty() {
            narrowed = strict;
        }

        if narrowed.len() <= 1 {
            return narrowed;
        }

        let locality: Vec<&Target> = narrowed
            .iter()
            .copied()
            .filter(|t| {
                if t.depends.not_any() {
                    return false;
                }
                t.depends.iter_ones().any(|c| full_provides.contains(&c))
            })
            .collect();

        if !locality.is_empty() {
            narrowed = locality;
        }

        if narrowed.len() <= 1 {
            return narrowed;
        }

        let no_deps: Vec<&Target> = narrowed
            .iter()
            .copied()
            .filter(|t| t.depends.not_any())
            .collect();

        if !no_deps.is_empty() && no_deps.len() < narrowed.len() {
            narrowed = no_deps;
        }

        narrowed
    }

    fn compute_full_provides(&self, target_set: &HashSet<usize>) -> HashSet<usize> {
        let mut provides: HashSet<usize> = HashSet::new();
        for &bit_idx in target_set {
            if let Some(target) = self.registry.get_by_bit_index(bit_idx) {
                for cap in target.provides.iter_ones() {
                    provides.insert(cap);
                }
            }
        }
        provides
    }

    pub fn resolve_abstract_dependencies(
        &self,
        target_names: &[&str],
        _provided: &BitVec,
    ) -> Result<ExecutionPlan, ResolverError> {
        self.resolve(target_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{Target, TargetRegistry};
    use fluent_types::{ExecutorKind, TargetType};
    use internment::ArcIntern;

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

    fn make_registry(targets: Vec<Target>) -> (TargetRegistry, CapabilityRegistry) {
        let mut reg = TargetRegistry::new();
        let caps = CapabilityRegistry::new();
        for t in targets {
            reg.register(t).unwrap();
        }
        (reg, caps)
    }

    fn register_targets(
        reg: &mut TargetRegistry,
        caps: &CapabilityRegistry,
        entries: &[(i64, &str, TargetType, &[&str], &[&str], bool)],
    ) {
        for &(id, name, ttype, depends, provides, essential) in entries {
            let d: BitVec = if depends.is_empty() {
                BitVec::new()
            } else {
                caps.to_bitvec(depends)
            };
            let p: BitVec = if provides.is_empty() {
                BitVec::new()
            } else {
                caps.to_bitvec(provides)
            };
            reg.register(
                Target::new()
                    .id(id)
                    .name(name.into())
                    .target_type(ttype)
                    .executor(ExecutorKind::Native)
                    .depends(d)
                    .provides(p)
                    .essential(essential)
                    .build(),
            )
            .unwrap();
        }
    }

    #[test]
    fn test_linear_chain_no_ambiguity() {
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
        let (reg, caps) = make_registry(targets);
        let resolver = AmbiguousDependencyResolver::new(&reg, &caps);
        let plan = resolver.resolve(&["build"]).expect("resolve");
        assert_eq!(plan.order.len(), 3);
        assert_eq!(plan.order[0], 0);
        assert_eq!(plan.order[1], 1);
        assert_eq!(plan.order[2], 2);
    }

    #[test]
    fn test_disambiguate_single_animal_provider() {
        let mut reg = TargetRegistry::new();
        let caps = CapabilityRegistry::new();
        for (i, (name, provides, deps)) in [
            ("bee", vec!["insect", "color_vision"], vec![]),
            ("insect", vec!["animal"], vec![]),
            ("jellyfish", vec!["animal"], vec![]),
            ("consumer", vec![], vec!["animal"]),
        ]
        .iter()
        .enumerate()
        {
            let deps_bits: BitVec = if deps.is_empty() {
                BitVec::new()
            } else {
                caps.to_bitvec(deps)
            };
            let provides_bits: BitVec = if provides.is_empty() {
                BitVec::new()
            } else {
                caps.to_bitvec(provides)
            };
            reg.register(
                Target::new()
                    .id(i as i64)
                    .name((*name).into())
                    .target_type(if provides.is_empty() {
                        TargetType::File
                    } else {
                        TargetType::Abstract
                    })
                    .executor(ExecutorKind::Native)
                    .depends(deps_bits)
                    .provides(provides_bits)
                    .build(),
            )
            .unwrap();
        }
        let resolver = AmbiguousDependencyResolver::new(&reg, &caps).with_strict(false);
        let result = resolver.resolve(&["consumer"]);
        match result {
            Err(ResolverError::AmbiguousDependency { name, candidates }) => {
                assert_eq!(name, "animal");
                assert_eq!(candidates.len(), 2);
            }
            other => panic!("expected AmbiguousDependency, got: {other:?}"),
        }
    }

    #[test]
    fn test_disambiguate_with_locality_preference() {
        let mut reg = TargetRegistry::new();
        let caps = CapabilityRegistry::new();

        let entries: &[(i64, &str, TargetType, &[&str], &[&str], bool)] = &[
            (0, "bee",           TargetType::File,     &[],                         &["insect", "color_vision"], false),
            (1, "stoat",         TargetType::File,     &[],                         &["mammal"],                false),
            (2, "insect",        TargetType::Abstract, &[],                         &["animal", "cognitive"],   false),
            (3, "mammal",        TargetType::Abstract, &[],                         &["animal", "cognitive"],   false),
            (4, "color_vision",  TargetType::Abstract, &[],                         &["vision"],                false),
            (5, "confuse_bee",   TargetType::File,     &["insect", "color_vision"], &["confuse", "agency"],     false),
            (6, "stun_stoat",    TargetType::File,     &["mammal"],                 &["confuse", "agency"],     false),
            (7, "confuse",       TargetType::Abstract, &[],                         &[],                        true),
            (8, "animal",        TargetType::Abstract, &[],                         &[],                        true),
            (9, "cognitive",     TargetType::Abstract, &[],                         &[],                        false),
            (10, "vision",       TargetType::Abstract, &[],                         &[],                        false),
        ];

        register_targets(&mut reg, &caps, entries);

        let resolver = AmbiguousDependencyResolver::new(&reg, &caps).with_strict(false);

        let plan_bee = resolver.resolve(&["confuse", "bee"]).expect("confuse bee");
        assert!(plan_bee.target_names.contains(&"bee".to_string()));
        assert!(
            plan_bee.target_names.contains(&"confuse_bee".to_string()),
            "should select confuse_bee over stun_stoat for bee input: {:?}",
            plan_bee.target_names
        );
        assert!(
            !plan_bee.target_names.contains(&"stun_stoat".to_string()),
            "stun_stoat should not be selected for bee input"
        );

        let plan_stoat = resolver.resolve(&["confuse", "stoat"]).expect("confuse stoat");
        assert!(plan_stoat.target_names.contains(&"stoat".to_string()));
        assert!(
            plan_stoat.target_names.contains(&"stun_stoat".to_string()),
            "should select stun_stoat for stoat input: {:?}",
            plan_stoat.target_names
        );
        assert!(
            !plan_stoat.target_names.contains(&"confuse_bee".to_string()),
            "confuse_bee should not be selected for stoat input"
        );
    }

    #[test]
    fn test_ambiguity_reported_when_multiple_providers() {
        let mut reg = TargetRegistry::new();
        let caps = CapabilityRegistry::new();

        register_targets(
            &mut reg,
            &caps,
            &[
                (0, "consumer",   TargetType::File,     &["animal"],   &[],       false),
                (1, "provider_a", TargetType::Abstract, &[],           &["animal"], false),
                (2, "provider_b", TargetType::Abstract, &[],           &["animal"], false),
            ],
        );

        let resolver = AmbiguousDependencyResolver::new(&reg, &caps).with_strict(false);
        let result = resolver.resolve(&["consumer"]);
        match result {
            Err(ResolverError::AmbiguousDependency { name, candidates }) => {
                assert_eq!(name, "animal");
                assert_eq!(candidates.len(), 2);
                assert!(candidates.contains(&"provider_a".to_string()));
                assert!(candidates.contains(&"provider_b".to_string()));
            }
            Ok(plan) => {
                panic!("expected AmbiguousDependency, got Ok with: {:?}", plan.target_names);
            }
            Err(other) => panic!("expected AmbiguousDependency, got: {other:?}"),
        }
    }

    #[test]
    fn test_circular_dependency_detected() {
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
        let (reg, caps) = make_registry(targets);
        let resolver = AmbiguousDependencyResolver::new(&reg, &caps);
        assert!(matches!(
            resolver.resolve(&["a"]),
            Err(ResolverError::CircularDependency)
        ));
    }

    #[test]
    fn test_perf_deep_chain_no_abstraction() {
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
        let (reg, caps) = make_registry(targets);
        let resolver = AmbiguousDependencyResolver::new(&reg, &caps);
        let last = format!("step_{:04}", depth - 1);
        let start = std::time::Instant::now();
        let plan = resolver.resolve(&[&last]).expect("deep chain");
        let elapsed = start.elapsed();
        eprintln!(
            "[perf] ambiguous_resolver deep_chain ({depth} nodes): {}us total, {:.1}us/node",
            elapsed.as_micros(),
            elapsed.as_micros() as f64 / depth as f64,
        );
        assert_eq!(plan.order.len(), depth);
        assert!(elapsed.as_millis() < 100, "perf degraded: {}ms", elapsed.as_millis());
    }
}
