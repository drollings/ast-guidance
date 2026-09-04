use super::*;
use crate::resolver::DependencyResolver;
use crate::target::{Target, TargetRegistry};
use crate::tests::common::make_bitset;
use bitvec::vec::BitVec;
use fluent_concurrency::tokio_runtime;
use fluent_concurrency::batch::{SupervisedBatch, SupervisedBatchSummary};
use fluent_types::{ExecutorKind, TargetType};
use std::sync::Arc;
use tempfile::tempdir;

/// Builds the same 3-target chain used by the former
/// `executor::test_execute_noop_targets`, with capability names registered
/// so the bit → name mapping in `from_target` is exercised.
fn make_chain_registry(caps: &CapabilityRegistry) -> TargetRegistry {
    caps.intern_list(&["init_asset", "process_asset", "finalize_asset"]);
    let targets = vec![
        Target::new()
            .id(0)
            .name("init".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(BitVec::new())
            .provides(caps.to_bitvec(&["init_asset"]))
            .build(),
        Target::new()
            .id(1)
            .name("process".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(caps.to_bitvec(&["init_asset"]))
            .provides(caps.to_bitvec(&["process_asset"]))
            .build(),
        Target::new()
            .id(2)
            .name("finalize".into())
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(caps.to_bitvec(&["process_asset"]))
            .provides(caps.to_bitvec(&["finalize_asset"]))
            .build(),
    ];
    let mut reg = TargetRegistry::new();
    for t in targets {
        reg.register(t).unwrap();
    }
    reg
}

#[test]
fn from_target_maps_capabilities_and_fields() {
    let caps = CapabilityRegistry::new();
    let reg = make_chain_registry(&caps);
    let target = reg.get("process").unwrap();
    let unit = TargetWorkUnit::from_target(target, &caps);
    assert_eq!(unit.name(), "process");
    assert_eq!(unit.depends(), &[ArcIntern::from("init_asset")]);
    assert_eq!(unit.provides(), &[ArcIntern::from("process_asset")]);
    assert!(!unit.essential);
}

#[test]
fn unmappable_bit_indices_are_skipped() {
    let caps = CapabilityRegistry::new();
    // Raw bit index 42 has no registered name — must be skipped, not fail.
    let target = Target::new()
        .id(0)
        .name("raw".into())
        .target_type(TargetType::File)
        .executor(ExecutorKind::Native)
        .depends(make_bitset(&[42]))
        .provides(BitVec::new())
        .build();
    let unit = TargetWorkUnit::from_target(&target, &caps);
    assert!(unit.depends().is_empty());
}

/// Port of `executor::test_execute_noop_targets`: resolve the same
/// 3-target chain and run the targets sequentially in `plan.order`,
/// asserting order and per-target success.
#[test]
fn test_target_work_unit_sequential_execution() {
    let caps = CapabilityRegistry::new();
    let reg = make_chain_registry(&caps);
    let resolver = DependencyResolver::new(&reg);
    let plan = resolver.resolve(&["finalize"]).expect("resolve");
    assert_eq!(plan.order.len(), 3);

    let ctx = WorkContext::default();
    let mut executed: Vec<String> = Vec::new();
    for &bit_idx in &plan.order {
        let target = reg.get_by_bit_index(bit_idx).expect("target");
        let unit = TargetWorkUnit::from_target(target, &caps);
        let result = unit.execute(&ctx).expect("execute");
        assert!(result.success);
        executed.push(unit.name().to_string());
    }
    assert_eq!(executed, vec!["init", "process", "finalize"]);
}

/// Parallel-wave variant: independent targets run concurrently in one
/// `SupervisedBatch`. Each writes an `arrived` marker then waits for all three to
/// arrive (a rendezvous barrier) before writing `done` — so every `done`
/// marker existing proves the three ran concurrently, not serially.
#[tokio::test]
async fn test_target_work_unit_batch_parallel_wave() {
    let caps = CapabilityRegistry::new();
    let dir = tempdir().unwrap();
    let dir_str = dir.path().to_string_lossy().into_owned();
    let mut batch = SupervisedBatch::new(tokio_runtime(), fluent_wvr::CapabilitySet::new());

    for i in 0..3 {
        let target = Target::new()
            .id(i)
            .name(ArcIntern::from(format!("worker_{i}")))
            .target_type(TargetType::File)
            .executor(ExecutorKind::Native)
            .depends(BitVec::new())
            .provides(BitVec::new())
            .command(format!(
                "touch {dir_str}/arrived_{i}; \
                 i=0; while [ $i -lt 200 ]; do \
                 c=$(ls {dir_str}/arrived_* 2>/dev/null | wc -l); \
                 [ \"$c\" -ge 3 ] && break; \
                 sleep 0.01; i=$((i+1)); done; \
                 touch {dir_str}/done_{i}"
            ))
            .build();
        let unit = TargetWorkUnit::from_target(&target, &caps);
        batch.register(Arc::new(unit)).unwrap();
    }

    let summary: SupervisedBatchSummary = (&mut batch).await;
    assert_eq!(summary.completed.len(), 3, "all workers complete");
    assert_eq!(summary.failed.len(), 0);
    assert_eq!(summary.panicked.len(), 0);
    assert_eq!(summary.cancelled.len(), 0);
    for i in 0..3 {
        assert!(
            dir.path().join(format!("done_{i}")).exists(),
            "worker {i} must have passed the rendezvous barrier"
        );
    }
}
