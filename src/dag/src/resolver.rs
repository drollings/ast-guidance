//! Capability-aware dependency resolver.
//!
//! ## Two-phase design — read this before editing
//!
//! Resolution is deliberately split into **two phases**, and the boundary
//! between them is load-bearing. The two phases are NOT duplicates of each
//! other:
//!
//! 1. **Phase 1 — narrowing (semantic selection).** `resolve_narrow_one`
//!    decides *which* targets participate in the plan. Targets are
//!    duck-typed: an `Abstract` target provides a capability by **name**
//!    match rather than self-provision, and a capability can have multiple
//!    competing providers. So this phase *chooses* providers — it does not
//!    traverse a graph. It applies the narrowing pipeline (`narrowing.rs`),
//!    resolves implicit name self-provision (`closure.rs`), and records
//!    narrowing losers in a `rejected` set so they can never re-enter the
//!    plan through a different capability path.
//!
//! 2. **Phase 2 — pure Kahn's algorithm.** Once narrowing has reduced the
//!    ambiguous, multi-provider capability graph to a clean selection,
//!    `plan_from_set` topologically orders that *already-selected* set. It
//!    shares the Kahn loop with `DependencyGraph::topo_sort_inner` via
//!    `crate::dep_graph::kahn_sort` — only the edge derivation differs (this
//!    resolver resolves `depends` capability bits to their providers through
//!    the registry, then restricts to the selected set).
//!
//! `plan_from_set` cannot delegate wholesale to `DependencyGraph::topo_sort`
//! because (a) target ids and capability ids are two `usize` index spaces
//! that `DependencyGraph<K>`'s single key type would alias inside a single
//! registered graph, and (b) it must order exactly the selected set with no
//! re-expansion — re-expanding would re-introduce rejected (narrowing-loser)
//! targets through uncontested capabilities. The shared `kahn_sort` core
//! sidesteps both: it orders only the nodes handed to it, and each caller
//! builds its own `in_degree`/`adjacency` from its own source of truth.
//!
//! See also: `REVIEW_20260804_PROGRESS.md` §3.3 for the design rationale.

use std::collections::{HashMap, HashSet};

use crate::closure::{self, ClosureCtx};
use crate::narrowing;
use crate::target::TargetRegistry;
use common_core::error::ResolverError;
use common_core::interner::CapabilityRegistry;
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

/// Provider-selection policy for the resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderSelection {
    /// Classic include-all semantics: every provider produces a distinct
    /// artifact. Correct for build graphs where multiple providers for the
    /// same capability should all be included.
    All,
    /// Yamake capability-graph semantics: apply narrowing rules to pick
    /// exactly one provider per contested capability. Report structured
    /// ambiguity when narrowing fails.
    NarrowOne,
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
    caps: Option<&'a CapabilityRegistry>,
    selection: ProviderSelection,
}

impl<'a> DependencyResolver<'a> {
    pub fn new(registry: &'a TargetRegistry) -> Self {
        Self {
            registry,
            strict: true,
            caps: None,
            selection: ProviderSelection::All,
        }
    }

    pub fn with_narrowing(registry: &'a TargetRegistry, caps: &'a CapabilityRegistry) -> Self {
        Self {
            registry,
            strict: true,
            caps: Some(caps),
            selection: ProviderSelection::NarrowOne,
        }
    }

    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    #[must_use]
    pub fn with_selection(mut self, sel: ProviderSelection) -> Self {
        self.selection = sel;
        self
    }

    pub fn resolve(&self, target_names: &[&str]) -> Result<ExecutionPlan, ResolverError> {
        if target_names.is_empty() {
            return Ok(ExecutionPlan {
                order: Vec::new(),
                target_names: Vec::new(),
            });
        }

        let seed_bits: Result<Vec<usize>, ResolverError> = target_names
            .iter()
            .map(|name| {
                self.registry
                    .get(name)
                    .map(|t| t.id as usize)
                    .ok_or_else(|| ResolverError::TargetNotFound(name.to_string()))
            })
            .collect();
        let seed_bits = seed_bits?;

        match self.selection {
            ProviderSelection::All => self.resolve_all(seed_bits),
            ProviderSelection::NarrowOne => {
                let caps = self.caps.ok_or_else(|| {
                    ResolverError::MissingDependency(
                        "NarrowOne requires a CapabilityRegistry".into(),
                    )
                })?;
                self.resolve_narrow_one(&seed_bits, target_names, caps)
            }
        }
    }

    fn resolve_all(&self, seed_bits: Vec<usize>) -> Result<ExecutionPlan, ResolverError> {
        let ctx = ClosureCtx {
            registry: self.registry,
            caps: None,
            strict: self.strict,
        };
        let needed = closure::transitive_closure(&ctx, seed_bits, None, None)?;
        self.plan_from_set(&needed)
    }

    /// Phase 1 of the two-phase design (see the module doc comment): narrow
    /// the duck-typed, multi-provider capability graph down to the set of
    /// targets that will execute.
    ///
    /// This is a *selection* pass (a fixpoint over provider contests), not a
    /// graph walk. It records narrowing losers in the `rejected` set so the
    /// Phase 2 Kahn sort (`plan_from_set`) never re-introduces them. Keep it
    /// separate from the ordering phase — the two phases are deliberately
    /// distinct and must not be collapsed into one.
    fn resolve_narrow_one(
        &self,
        seed_bits: &[usize],
        target_names: &[&str],
        caps: &'a CapabilityRegistry,
    ) -> Result<ExecutionPlan, ResolverError> {
        let ctx = ClosureCtx {
            registry: self.registry,
            caps: Some(caps),
            strict: self.strict,
        };

        let has_abstract_seed = target_names.iter().any(|name| {
            self.registry
                .get(name)
                .is_some_and(|t| t.target_type == TargetType::Abstract)
        });

        // Step 2: compute full closure to check for multi-provider caps
        let closure_set = closure::transitive_closure(&ctx, seed_bits.iter().copied(), None, None)?;

        let mut multi_provider_cap = false;
        for &bit in &closure_set {
            if let Some(target) = self.registry.get_by_bit_index(bit) {
                for cap in target.depends.iter_ones() {
                    if self.registry.get_providers(cap).len() > 1 {
                        multi_provider_cap = true;
                        break;
                    }
                }
            }
            if multi_provider_cap {
                break;
            }
        }

        // Fast path: no ambiguity at all → delegate to All (zero narrowing overhead)
        if !has_abstract_seed && !multi_provider_cap {
            return self.resolve_all(seed_bits.to_vec());
        }

        let mut resolved_set: HashSet<usize> = seed_bits.iter().copied().collect();
        let mut rejected: HashSet<usize> = HashSet::new();
        let mut narrowed_caps: HashSet<usize> = HashSet::new();
        let mut changed = true;

        // Step 4: NarrowOne fixpoint loop
        while changed {
            changed = false;
            let full_provides = compute_full_provides(self.registry, &resolved_set);

            let snapshot: Vec<usize> = resolved_set.iter().copied().collect();

            for &bit_idx in &snapshot {
                let target = self
                    .registry
                    .get_by_bit_index(bit_idx)
                    .ok_or_else(|| ResolverError::TargetNotFound(format!("bit_index {bit_idx}")))?;

                let is_abstract = target.target_type == TargetType::Abstract;
                let deps_satisfied = target.depends.not_any()
                    || target
                        .depends
                        .iter_ones()
                        .all(|c| full_provides.contains(&c));

                if !deps_satisfied {
                    for cap_idx in target.depends.iter_ones() {
                        if narrowed_caps.contains(&cap_idx) || full_provides.contains(&cap_idx) {
                            continue;
                        }
                        let providers = self.registry.get_providers(cap_idx);
                        if providers.is_empty() {
                            if let Some(cap_name) = caps.get_name(cap_idx) {
                                if let Some(implicit) = self.registry.get(&cap_name) {
                                    if resolved_set.insert(implicit.id as usize) {
                                        changed = true;
                                    }
                                    continue;
                                }
                            }
                            if self.strict {
                                return Err(narrowing::missing_provider_error(
                                    Some(caps),
                                    cap_idx,
                                    &target.name,
                                ));
                            }
                            continue;
                        }
                        if providers.len() == 1 {
                            if resolved_set.insert(providers[0].id as usize) {
                                changed = true;
                            }
                            continue;
                        }
                        narrowed_caps.insert(cap_idx);
                        let narrowed =
                            narrowing::narrow_providers(providers.clone(), &full_provides);
                        match narrowed.len() {
                            0 => {
                                return Err(narrowing::missing_provider_error(
                                    Some(caps),
                                    cap_idx,
                                    &target.name,
                                ));
                            }
                            1 => {
                                if resolved_set.insert(narrowed[0].id as usize) {
                                    changed = true;
                                }
                                for candidate in providers {
                                    if candidate.id != narrowed[0].id {
                                        rejected.insert(candidate.id as usize);
                                    }
                                }
                            }
                            _ => {
                                return Err(narrowing::ambiguous_error(
                                    Some(caps),
                                    cap_idx,
                                    &narrowed,
                                ));
                            }
                        }
                    }
                }

                // Abstract self-provision branch
                if is_abstract || target.depends.not_any() {
                    let cap_idx = caps.get_index(&target.name);
                    if let Some(ci) = cap_idx {
                        if narrowed_caps.contains(&ci) {
                            continue;
                        }
                        let providers = self.registry.get_providers(ci);
                        if providers.len() > 1 {
                            narrowed_caps.insert(ci);
                            let narrowed =
                                narrowing::narrow_providers(providers.clone(), &full_provides);
                            match narrowed.len() {
                                1 => {
                                    if resolved_set.insert(narrowed[0].id as usize) {
                                        changed = true;
                                    }
                                    for candidate in providers {
                                        if candidate.id != narrowed[0].id {
                                            rejected.insert(candidate.id as usize);
                                        }
                                    }
                                }
                                0 => {}
                                _ => {
                                    return Err(narrowing::ambiguous_error(
                                        Some(caps),
                                        ci,
                                        &narrowed,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 5: Final expansion with satisfied + rejected guards
        let full_provides = compute_full_provides(self.registry, &resolved_set);
        let final_expansion = closure::transitive_closure(
            &ctx,
            resolved_set.iter().copied(),
            Some(&full_provides),
            Some(&rejected),
        )?;
        let mut combined: HashSet<usize> = resolved_set;
        combined.extend(final_expansion);

        // Step 6: Kahn's topological sort — use the combined set directly
        // (no re-expansion: narrowing decisions must be preserved).
        self.plan_from_set(&combined)
    }

    /// Phase 2 of the two-phase design (see the module doc comment): a pure
    /// Kahn's topological sort over the already-selected `needed` set.
    ///
    /// The Kahn loop itself is shared with `DependencyGraph::topo_sort_inner`
    /// via `crate::dep_graph::kahn_sort`; only the edge derivation differs
    /// (this method resolves a target's `depends` capability bits to their
    /// providers through the registry, then restricts to the already-selected
    /// `needed` set). It must not re-expand the set — narrowing losers in
    /// `rejected` stay out because `needed` is exactly the selection result,
    /// and `kahn_sort` only ever orders the nodes passed to it.
    fn plan_from_set(&self, needed: &HashSet<usize>) -> Result<ExecutionPlan, ResolverError> {
        let mut in_degree: HashMap<usize, usize> = needed.iter().map(|&k| (k, 0)).collect();
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &bit_idx in needed {
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
                    if needed.contains(&provider_bit_idx) && provider_bit_idx != bit_idx {
                        adj.entry(provider_bit_idx).or_default().push(bit_idx);
                        *in_degree.get_mut(&bit_idx).unwrap() += 1;
                    }
                }
            }
        }
        let Ok(order) = crate::dep_graph::kahn_sort(&mut in_degree, &adj, needed.len()) else {
            return Err(ResolverError::CircularDependency);
        };
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
}

fn compute_full_provides(registry: &TargetRegistry, target_set: &HashSet<usize>) -> HashSet<usize> {
    let mut provides: HashSet<usize> = HashSet::new();
    for &bit_idx in target_set {
        if let Some(target) = registry.get_by_bit_index(bit_idx) {
            for cap in target.provides.iter_ones() {
                provides.insert(cap);
            }
        }
    }
    provides
}

#[cfg(test)]
#[path = "../tests/resolver.rs"]
mod tests;
