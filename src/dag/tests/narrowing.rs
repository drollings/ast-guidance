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
