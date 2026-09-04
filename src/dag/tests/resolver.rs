use super::*;
use crate::target::{Target, TargetRegistry};
use crate::tests::common::{make_bitset, make_registry};
use bitvec::vec::BitVec;
use fluent_types::{ExecutorKind, TargetType};
use internment::ArcIntern;
use std::time::Instant;

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
fn test_linear_chain() {
    let targets = crate::tests::common::linear_chain();
    let reg = make_registry(targets);
    let resolver = DependencyResolver::new(&reg);
    let plan = resolver.resolve(&["build"]).expect("resolve");
    assert_eq!(plan.order, vec![0, 1, 2]);
}

#[test]
fn test_diamond_graph() {
    let targets = crate::tests::common::diamond();
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
fn test_narrow_one_empty_input() {
    let reg = TargetRegistry::new();
    let caps = CapabilityRegistry::new();
    let resolver = DependencyResolver::with_narrowing(&reg, &caps);
    let plan = resolver.resolve(&[]).expect("empty input");
    assert!(plan.is_empty());
}

#[test]
fn test_narrow_one_strict_missing_provider() {
    let mut reg = TargetRegistry::new();
    let caps = CapabilityRegistry::new();
    register_targets(
        &mut reg,
        &caps,
        &[(
            0,
            "consumer",
            TargetType::File,
            &["nonexistent"],
            &[],
            false,
        )],
    );
    let resolver = DependencyResolver::with_narrowing(&reg, &caps).with_strict(true);
    let err = resolver.resolve(&["consumer"]).unwrap_err();
    assert!(matches!(err, ResolverError::MissingDependency(_)));
}

#[test]
fn test_narrow_one_disambiguate_single_animal_provider() {
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
    let resolver = DependencyResolver::with_narrowing(&reg, &caps).with_strict(false);
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
fn test_narrow_one_disambiguate_with_locality_preference() {
    let mut reg = TargetRegistry::new();
    let caps = CapabilityRegistry::new();

    let entries: &[(i64, &str, TargetType, &[&str], &[&str], bool)] = &[
        (
            0,
            "bee",
            TargetType::File,
            &[],
            &["insect", "color_vision"],
            false,
        ),
        (1, "stoat", TargetType::File, &[], &["mammal"], false),
        (
            2,
            "insect",
            TargetType::Abstract,
            &[],
            &["animal", "cognitive"],
            false,
        ),
        (
            3,
            "mammal",
            TargetType::Abstract,
            &[],
            &["animal", "cognitive"],
            false,
        ),
        (
            4,
            "color_vision",
            TargetType::Abstract,
            &[],
            &["vision"],
            false,
        ),
        (
            5,
            "confuse_bee",
            TargetType::File,
            &["insect", "color_vision"],
            &["confuse", "agency"],
            false,
        ),
        (
            6,
            "stun_stoat",
            TargetType::File,
            &["mammal"],
            &["confuse", "agency"],
            false,
        ),
        (7, "confuse", TargetType::Abstract, &[], &[], true),
        (8, "animal", TargetType::Abstract, &[], &[], true),
        (9, "cognitive", TargetType::Abstract, &[], &[], false),
        (10, "vision", TargetType::Abstract, &[], &[], false),
    ];

    register_targets(&mut reg, &caps, entries);

    let resolver = DependencyResolver::with_narrowing(&reg, &caps).with_strict(false);

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

    let plan_stoat = resolver
        .resolve(&["confuse", "stoat"])
        .expect("confuse stoat");
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
fn test_narrow_one_ambiguity_reported_when_multiple_providers() {
    let mut reg = TargetRegistry::new();
    let caps = CapabilityRegistry::new();

    register_targets(
        &mut reg,
        &caps,
        &[
            (0, "consumer", TargetType::File, &["animal"], &[], false),
            (
                1,
                "provider_a",
                TargetType::Abstract,
                &[],
                &["animal"],
                false,
            ),
            (
                2,
                "provider_b",
                TargetType::Abstract,
                &[],
                &["animal"],
                false,
            ),
        ],
    );

    let resolver = DependencyResolver::with_narrowing(&reg, &caps).with_strict(false);
    let result = resolver.resolve(&["consumer"]);
    match result {
        Err(ResolverError::AmbiguousDependency { name, candidates }) => {
            assert_eq!(name, "animal");
            assert_eq!(candidates.len(), 2);
            assert!(candidates.contains(&"provider_a".to_string()));
            assert!(candidates.contains(&"provider_b".to_string()));
        }
        Ok(plan) => {
            panic!(
                "expected AmbiguousDependency, got Ok with: {:?}",
                plan.target_names
            );
        }
        Err(other) => panic!("expected AmbiguousDependency, got: {other:?}"),
    }
}

#[test]
fn test_narrow_one_loser_not_readded_by_final_expansion() {
    let mut reg = TargetRegistry::new();
    let caps = CapabilityRegistry::new();
    register_targets(
        &mut reg,
        &caps,
        &[
            (
                0,
                "bee",
                TargetType::File,
                &[],
                &["insect", "color_vision"],
                false,
            ),
            (1, "stoat", TargetType::File, &[], &["mammal"], false),
            (
                2,
                "confuse_bee",
                TargetType::File,
                &["insect", "color_vision"],
                &["confuse"],
                false,
            ),
            (
                3,
                "stun_stoat",
                TargetType::File,
                &["mammal"],
                &["confuse"],
                false,
            ),
            (4, "confuse", TargetType::Abstract, &[], &[], true),
            (
                5,
                "rate_confusion",
                TargetType::File,
                &["confuse"],
                &["rated"],
                false,
            ),
        ],
    );
    let resolver = DependencyResolver::with_narrowing(&reg, &caps).with_strict(false);
    let plan = resolver
        .resolve(&["confuse", "bee", "rate_confusion"])
        .expect("resolve");
    assert!(plan.target_names.contains(&"confuse_bee".to_string()));
    assert!(
        !plan.target_names.contains(&"stun_stoat".to_string()),
        "narrowing loser stun_stoat was re-added: {:?}",
        plan.target_names
    );
}

#[test]
fn test_narrow_one_loser_re_enters_via_different_cap() {
    let mut reg = TargetRegistry::new();
    let caps = CapabilityRegistry::new();
    register_targets(
        &mut reg,
        &caps,
        &[
            (
                0,
                "bee",
                TargetType::File,
                &[],
                &["insect", "color_vision"],
                false,
            ),
            (1, "stoat", TargetType::File, &[], &["mammal"], false),
            (
                2,
                "confuse_bee",
                TargetType::File,
                &["insect", "color_vision"],
                &["confuse"],
                false,
            ),
            (
                3,
                "stun_stoat",
                TargetType::File,
                &["mammal"],
                &["confuse", "stun_grenade"],
                false,
            ),
            (4, "confuse", TargetType::Abstract, &[], &[], true),
            (
                5,
                "rate_confusion",
                TargetType::File,
                &["confuse"],
                &["rated"],
                false,
            ),
            (
                6,
                "grenade_user",
                TargetType::File,
                &["stun_grenade"],
                &[],
                false,
            ),
        ],
    );
    let resolver = DependencyResolver::with_narrowing(&reg, &caps).with_strict(false);
    let plan = resolver
        .resolve(&["confuse", "bee", "rate_confusion", "grenade_user"])
        .expect("resolve");
    assert!(plan.target_names.contains(&"confuse_bee".to_string()));
    assert!(
        plan.target_names.contains(&"stun_stoat".to_string()),
        "stun_stoat should re-enter via its uncontested 'stun_grenade' cap: {:?}",
        plan.target_names
    );
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
    let plan = resolver
        .resolve(&["consumer"])
        .expect("multi-provider resolve");
    assert_eq!(plan.order.len(), 3);
    assert!(plan.order.contains(&0));
    assert!(plan.order.contains(&1));
    assert!(plan.order.contains(&2));
    assert!(
        plan.order.iter().position(|&i| i == 1).unwrap()
            < plan.order.iter().position(|&i| i == 0).unwrap()
    );
    assert!(
        plan.order.iter().position(|&i| i == 2).unwrap()
            < plan.order.iter().position(|&i| i == 0).unwrap()
    );
}

#[test]
fn test_self_providing_chain() {
    let targets = vec![Target::new()
        .id(0)
        .name("self_provider".into())
        .target_type(TargetType::File)
        .executor(ExecutorKind::Native)
        .depends(make_bitset(&[0]))
        .provides(make_bitset(&[0]))
        .build()];
    let reg = make_registry(targets);
    let resolver = DependencyResolver::new(&reg);
    let plan = resolver
        .resolve(&["self_provider"])
        .expect("self-provide resolve");
    assert_eq!(plan.order, vec![0]);
}

#[test]
fn test_empty_resolve() {
    let reg = TargetRegistry::new();
    let resolver = DependencyResolver::new(&reg);
    let plan = resolver
        .resolve(&[])
        .expect("empty resolve on empty registry");
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
    assert!(
        elapsed.as_millis() < 200,
        "fan-out perf degraded: {}ms",
        elapsed.as_millis()
    );
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
    assert!(
        elapsed.as_millis() < 100,
        "large DAG perf degraded: {}ms",
        elapsed.as_millis()
    );
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
    assert!(
        elapsed.as_millis() < 200,
        "deep diamond perf degraded: {}ms",
        elapsed.as_millis()
    );
}

#[test]
fn test_resolve_strict_missing_dependency_errors() {
    let targets = vec![Target::new()
        .id(0)
        .name("consumer".into())
        .target_type(TargetType::File)
        .executor(ExecutorKind::Native)
        .depends(make_bitset(&[99]))
        .provides(BitVec::new())
        .build()];
    let reg = make_registry(targets);
    let resolver = DependencyResolver::new(&reg).with_strict(true);
    let err = resolver.resolve(&["consumer"]).unwrap_err();
    assert!(matches!(err, ResolverError::MissingDependency(_)));
}

#[test]
fn test_resolve_non_strict_skips_missing() {
    let targets = vec![Target::new()
        .id(0)
        .name("consumer".into())
        .target_type(TargetType::File)
        .executor(ExecutorKind::Native)
        .depends(make_bitset(&[99]))
        .provides(BitVec::new())
        .build()];
    let reg = make_registry(targets);
    let resolver = DependencyResolver::new(&reg).with_strict(false);
    let plan = resolver
        .resolve(&["consumer"])
        .expect("non-strict missing dep");
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
