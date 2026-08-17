use std::collections::HashSet;

use common_core::error::ResolverError;
use common_core::interner::CapabilityRegistry;

use crate::target::Target;

type Stage = (&'static str, fn(&Target, &HashSet<usize>) -> bool);

const STAGES: &[Stage] = &[
    ("essential", |t, _fp| t.essential),
    ("strict_sat", |t, fp| {
        t.depends.not_any() || t.depends.iter_ones().all(|c| fp.contains(&c))
    }),
    ("locality", |t, fp| {
        !t.depends.not_any() && t.depends.iter_ones().any(|c| fp.contains(&c))
    }),
];

/// Narrow providers for a capability using the yamake-old.py narrowing
/// pipeline. Filters are applied in priority order; when exactly one
/// candidate remains, the function returns immediately.
///
/// | Priority | Filter | Description |
/// |---|---|---|
/// | 1 | Essential | If any candidates have `essential: true`, drop non-essentials |
/// | 2 | Strict sat | Keep candidates whose **all** deps are in `full_provides` |
/// | 3 | Locality | Keep candidates whose **any** dep is in `full_provides` |
/// | 4 | No-dep | Prefer candidates with **no** dependencies (strict-reduction) |
pub(crate) fn narrow_providers<'t>(
    candidates: Vec<&'t Target>,
    full_provides: &HashSet<usize>,
) -> Vec<&'t Target> {
    if candidates.len() <= 1 {
        return candidates;
    }

    let mut narrowed = candidates;
    narrowed.sort_by(|a, b| a.name.cmp(&b.name));

    for (_name, keep) in STAGES {
        let kept: Vec<&Target> = narrowed
            .iter()
            .copied()
            .filter(|t| keep(t, full_provides))
            .collect();
        if !kept.is_empty() {
            narrowed = kept;
        }
        if narrowed.len() <= 1 {
            return narrowed;
        }
    }

    // Stage 4 (no-dep): only replace when it strictly reduces the candidate
    // set (yamake-old.py parity).
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

/// Get the human-readable name for a capability index, falling back to
/// `cap_{idx}` when no registry is available or the name is unknown.
pub(crate) fn cap_name(caps: Option<&CapabilityRegistry>, cap_idx: usize) -> String {
    caps.and_then(|c| c.get_name(cap_idx))
        .map_or_else(|| format!("cap_{cap_idx}"), |s| s.to_string())
}

/// Build a `MissingDependency` error for a capability with no providers.
/// Canonical message: `"no provider for '{cap}' (required by '{target}')"`
pub(crate) fn missing_provider_error(
    caps: Option<&CapabilityRegistry>,
    cap_idx: usize,
    target_name: &str,
) -> ResolverError {
    ResolverError::MissingDependency(format!(
        "no provider for '{}' (required by '{target_name}')",
        cap_name(caps, cap_idx),
    ))
}

/// Build an `AmbiguousDependency` error when narrowing produces >1 candidate.
pub(crate) fn ambiguous_error(
    caps: Option<&CapabilityRegistry>,
    cap_idx: usize,
    narrowed: &[&Target],
) -> ResolverError {
    ResolverError::AmbiguousDependency {
        name: cap_name(caps, cap_idx),
        candidates: narrowed.iter().map(|t| t.name.to_string()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::make_bitset;
    use common_core::interner::CapabilityRegistry;
    use fluent_types::{ExecutorKind, TargetType};

    use crate::target::Target;

    fn t(id: i64, name: &str, depends: &[usize], provides: &[usize], essential: bool) -> Target {
        Target::new()
            .id(id)
            .name(name.into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(make_bitset(depends))
            .provides(make_bitset(provides))
            .essential(essential)
            .build()
    }

    #[test]
    fn test_single_candidate_returns_as_is() {
        let only = t(0, "only", &[], &[0], false);
        let candidates = vec![&only];
        let result = narrow_providers(candidates.clone(), &HashSet::new());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name.to_string(), "only");
    }

    #[test]
    fn test_empty_candidates_returns_empty() {
        let result = narrow_providers(vec![], &HashSet::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_essential_stage_filters_non_essential() {
        let a = t(0, "a", &[], &[0], false);
        let b = t(1, "b", &[], &[0], true);
        let candidates = vec![&a, &b];
        let result = narrow_providers(candidates, &HashSet::new());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name.to_string(), "b");
    }

    #[test]
    fn test_essential_stage_keeps_all_essential() {
        let a = t(0, "a", &[], &[0], true);
        let b = t(1, "b", &[], &[0], true);
        let candidates = vec![&a, &b];
        let result = narrow_providers(candidates, &HashSet::new());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_strict_sat_keeps_only_satisfied() {
        let a = t(0, "a", &[1], &[0], false);
        let b = t(1, "b", &[2], &[0], false);
        let candidates = vec![&a, &b];
        let mut fp = HashSet::new();
        fp.insert(1);
        let result = narrow_providers(candidates, &fp);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name.to_string(), "a");
    }

    #[test]
    fn test_strict_sat_no_dep_always_kept() {
        let a = t(0, "a", &[1], &[0], false);
        let b = t(1, "b", &[], &[0], false);
        let candidates = vec![&a, &b];
        let fp = HashSet::new();
        let result = narrow_providers(candidates, &fp);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name.to_string(), "b");
    }

    #[test]
    fn test_locality_keeps_candidates_with_any_dep_in_fp() {
        let a = t(0, "a", &[1], &[0], false);
        let b = t(1, "b", &[2], &[0], false);
        let candidates = vec![&a, &b];
        let mut fp = HashSet::new();
        fp.insert(1);
        let result = narrow_providers(candidates, &fp);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name.to_string(), "a");
    }

    #[test]
    fn test_locality_excludes_no_dep_candidates() {
        let a = t(0, "a", &[], &[0], false);
        let b = t(1, "b", &[1], &[0], false);
        let candidates = vec![&a, &b];
        let mut fp = HashSet::new();
        fp.insert(1);
        let result = narrow_providers(candidates, &fp);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name.to_string(), "b");
    }

    #[test]
    fn test_no_dep_stage_reduces_only_when_strictly_smaller() {
        let a = t(0, "a", &[], &[0], false);
        let b = t(1, "b", &[], &[0], false);
        let candidates = vec![&a, &b];
        let result = narrow_providers(candidates, &HashSet::new());
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_stage_ordering_determinism() {
        let bee = t(0, "bee", &[], &[0], false);
        let stoat = t(1, "stoat", &[2], &[0], false);
        let candidates = vec![&bee, &stoat];
        let mut fp = HashSet::new();
        fp.insert(2);
        let r1 = narrow_providers(candidates.clone(), &fp);
        let r2 = narrow_providers(candidates, &fp);
        assert_eq!(r1.len(), r2.len());
    }

    #[test]
    fn test_cap_name_with_registry() {
        let caps = CapabilityRegistry::new();
        let idx = caps.intern("animal");
        let name = cap_name(Some(&caps), idx);
        assert_eq!(name, "animal");
    }

    #[test]
    fn test_cap_name_fallback() {
        let name = cap_name(None, 42);
        assert_eq!(name, "cap_42");
    }

    #[test]
    fn test_cap_name_with_registry_unknown_index() {
        let caps = CapabilityRegistry::new();
        let name = cap_name(Some(&caps), 99);
        assert_eq!(name, "cap_99");
    }

    #[test]
    fn test_missing_provider_error_format() {
        let err = missing_provider_error(None, 5, "test_target");
        assert!(matches!(err, ResolverError::MissingDependency(_)));
        let msg = err.to_string();
        assert!(msg.contains("cap_5"));
        assert!(msg.contains("test_target"));
    }

    #[test]
    fn test_ambiguous_error_format() {
        let a = t(0, "provider_a", &[], &[1], false);
        let b = t(1, "provider_b", &[], &[1], false);
        let candidates = vec![&a, &b];
        let caps = CapabilityRegistry::new();
        let idx = caps.intern("animal");
        let err = ambiguous_error(Some(&caps), idx, &candidates);
        assert!(matches!(err, ResolverError::AmbiguousDependency { .. }));
        let msg = err.to_string();
        assert!(msg.contains("animal"));
        assert!(msg.contains("provider_a"));
        assert!(msg.contains("provider_b"));
    }
}
