use std::collections::HashSet;

use common_core::error::ResolverError;
use common_core::interner::CapabilityRegistry;

use crate::narrowing;
use crate::target::TargetRegistry;

pub(crate) struct ClosureCtx<'a> {
    pub registry: &'a TargetRegistry,
    pub caps: Option<&'a CapabilityRegistry>,
    pub strict: bool,
}

/// DFS over `depends` edges pushing all non-rejected, non-visited providers.
///
/// For each required capability:
///   - skip if `satisfied` contains it (NarrowOne durability guard);
///   - skip providers in `rejected` (narrowing losers;
///   - 0 providers → implicit name fallback (NarrowOne only: a target whose
///     name matches the capability name), else strict-error or skip;
///   - N providers → push all non-rejected, non-visited providers.
pub(crate) fn transitive_closure(
    ctx: &ClosureCtx,
    seeds: impl IntoIterator<Item = usize>,
    satisfied: Option<&HashSet<usize>>,
    rejected: Option<&HashSet<usize>>,
) -> Result<HashSet<usize>, ResolverError> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = seeds.into_iter().collect();

    while let Some(bit_idx) = stack.pop() {
        if visited.contains(&bit_idx) {
            continue;
        }
        let target =
            ctx.registry
                .get_by_bit_index(bit_idx)
                .ok_or(ResolverError::TargetNotFound(format!(
                    "bit_index {bit_idx}"
                )))?;

        visited.insert(bit_idx);

        for cap_idx in target.depends.iter_ones() {
            if satisfied.is_some_and(|sat| sat.contains(&cap_idx)) {
                continue;
            }

            let providers = ctx.registry.get_providers(cap_idx);

            if providers.is_empty() {
                if let Some(caps) = ctx.caps {
                    if let Some(cap_name) = caps.get_name(cap_idx) {
                        if let Some(implicit) = ctx.registry.get(&cap_name) {
                            let p_bit = implicit.id as usize;
                            let skip = rejected.is_some_and(|r| r.contains(&p_bit));
                            if !visited.contains(&p_bit) && !skip {
                                stack.push(p_bit);
                            }
                            continue;
                        }
                    }
                }
                if ctx.strict {
                    return Err(narrowing::missing_provider_error(
                        ctx.caps,
                        cap_idx,
                        &target.name,
                    ));
                }
                continue;
            }

            for provider in &providers {
                let p_bit = provider.id as usize;
                let skip = rejected.is_some_and(|r| r.contains(&p_bit));
                if !visited.contains(&p_bit) && !skip {
                    stack.push(p_bit);
                }
            }
        }
    }

    Ok(visited)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{Target, TargetRegistry};
    use crate::tests::common::{make_bitset, make_registry};
    use bitvec::vec::BitVec;
    use common_core::interner::CapabilityRegistry;
    use fluent_types::{ExecutorKind, TargetType};

    #[test]
    fn test_linear_chain() {
        let targets = crate::tests::common::linear_chain();
        let reg = make_registry(targets);
        let ctx = ClosureCtx {
            registry: &reg,
            caps: None,
            strict: true,
        };
        let result = transitive_closure(&ctx, [0, 1, 2], None, None).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
        assert!(result.contains(&2));
    }

    #[test]
    fn test_diamond() {
        let targets = crate::tests::common::diamond();
        let reg = make_registry(targets);
        let ctx = ClosureCtx {
            registry: &reg,
            caps: None,
            strict: true,
        };
        let result = transitive_closure(&ctx, [3], None, None).unwrap();
        assert_eq!(result.len(), 4);
        assert!(result.contains(&0));
        assert!(result.contains(&3));
    }

    #[test]
    fn test_self_providing_target() {
        let targets = crate::tests::common::self_providing();
        let reg = make_registry(targets);
        let ctx = ClosureCtx {
            registry: &reg,
            caps: None,
            strict: true,
        };
        let result = transitive_closure(&ctx, [0], None, None).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0));
    }

    #[test]
    fn test_strict_missing_provider_errors() {
        let targets = vec![Target::new()
            .id(0)
            .name("orphan".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(make_bitset(&[99]))
            .provides(make_bitset(&[0]))
            .build()];
        let reg = make_registry(targets);
        let ctx = ClosureCtx {
            registry: &reg,
            caps: None,
            strict: true,
        };
        let err = transitive_closure(&ctx, [0], None, None).unwrap_err();
        assert!(matches!(err, ResolverError::MissingDependency(_)));
    }

    #[test]
    fn test_non_strict_skips_missing_provider() {
        let targets = vec![Target::new()
            .id(0)
            .name("orphan".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(make_bitset(&[99]))
            .provides(make_bitset(&[0]))
            .build()];
        let reg = make_registry(targets);
        let ctx = ClosureCtx {
            registry: &reg,
            caps: None,
            strict: false,
        };
        let result = transitive_closure(&ctx, [0], None, None).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_satisfied_guard_skips_cap() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("dep".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[1]))
                .build(),
            Target::new()
                .id(1)
                .name("consumer".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[1]))
                .provides(BitVec::new())
                .build(),
        ];
        let reg = make_registry(targets);
        let ctx = ClosureCtx {
            registry: &reg,
            caps: None,
            strict: true,
        };
        let mut satisfied = HashSet::new();
        satisfied.insert(1);
        let result = transitive_closure(&ctx, [1], Some(&satisfied), None).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(&1));
    }

    #[test]
    fn test_rejected_provider_skipped() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("provider_a".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[1]))
                .build(),
            Target::new()
                .id(1)
                .name("provider_b".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[1]))
                .build(),
            Target::new()
                .id(2)
                .name("consumer".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[1]))
                .provides(BitVec::new())
                .build(),
        ];
        let reg = make_registry(targets);
        let ctx = ClosureCtx {
            registry: &reg,
            caps: None,
            strict: true,
        };
        let mut rejected = HashSet::new();
        rejected.insert(0);
        let result = transitive_closure(&ctx, [2], None, Some(&rejected)).unwrap();
        assert!(result.contains(&1));
        assert!(!result.contains(&0));
    }

    #[test]
    fn test_implicit_name_fallback() {
        let mut reg = TargetRegistry::new();
        let caps = CapabilityRegistry::new();
        let cap_idx = caps.intern("helper_tool");
        reg.register(
            Target::new()
                .id(0)
                .name("helper_tool".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[cap_idx]))
                .build(),
        )
        .unwrap();
        reg.register(
            Target::new()
                .id(1)
                .name("consumer".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[cap_idx]))
                .provides(BitVec::new())
                .build(),
        )
        .unwrap();
        let ctx = ClosureCtx {
            registry: &reg,
            caps: Some(&caps),
            strict: true,
        };
        let result = transitive_closure(&ctx, [1], None, None).unwrap();
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }

    #[test]
    fn test_empty_seeds_returns_empty_set() {
        let reg = TargetRegistry::new();
        let ctx = ClosureCtx {
            registry: &reg,
            caps: None,
            strict: true,
        };
        let result = transitive_closure(&ctx, std::iter::empty(), None, None).unwrap();
        assert!(result.is_empty());
    }
}
