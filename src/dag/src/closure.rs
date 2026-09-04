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
#[path = "../tests/closure.rs"]
mod tests;
