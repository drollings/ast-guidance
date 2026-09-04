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
#[path = "../tests/narrowing.rs"]
mod tests;
