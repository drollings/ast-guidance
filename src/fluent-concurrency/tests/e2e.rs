use super::*;
use crate::batch::{SupervisedBatch, SupervisedBatchEvent, SupervisedBatchSummary};
use fluent_wvr_testutil::StubComponent;
use std::sync::Arc;

/// End-to-end: SupervisedBatch handles mixed success/failure/cancellation
#[tokio::test(start_paused = true)]
async fn test_e2e_batch_mixed_outcomes() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut batch = SupervisedBatch::new(runtime, caps);
    // root fails permanently (providing `shared`), child1 waits on `shared`
    // (transient, stays pending), independent panics.
    batch.register(Arc::new(
        StubComponent::fail("root").with_provides("shared"),
    ))
    .unwrap()
    .register_with_context(
        Arc::new(StubComponent::dep_fail("child1").with_dep("shared")),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap()
    .register(Arc::new(StubComponent::panic("independent")))
    .unwrap();

    let summary: SupervisedBatchSummary = (&mut batch).await;
    assert_eq!(summary.completed.len(), 0);
    assert_eq!(summary.failed.len(), 1, "root fails with execution error");
    assert_eq!(summary.panicked.len(), 1, "independent panics");
    assert_eq!(summary.cancelled.len(), 1);
}

/// E2E Panic Cascade: verify that a panicking task aborts its (multiple)
/// transitive dependents while independent neighbors continue unhindered.
/// This is the single owner of the panic-cascade-with-neighbors assertion; it
/// folds in the former m2 `test_batch_panic_cancels_transitive_dependents`
/// check that one panic cancels several sibling dependents (see ROADMAP M2.5).
#[tokio::test(start_paused = true)]
async fn test_e2e_panic_cascade_with_independent_neighbors() {
    let mut batch = make_batch();

    batch.register(Arc::new(
        StubComponent::panic("parent").with_provides("shared"),
    ))
    .unwrap();
    batch.register_with_context(
        Arc::new(StubComponent::dep_fail("child").with_dep("shared")),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();
    batch.register_with_context(
        Arc::new(StubComponent::dep_fail("child2").with_dep("shared")),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();
    batch.register(Arc::new(
        StubComponent::ok("neighbor").with_provides("independent"),
    ))
    .unwrap();
    batch.register_with_context(
        Arc::new(StubComponent::ok("grandchild").with_dep("independent")),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();

    let summary: SupervisedBatchSummary = (&mut batch).await;
    assert_eq!(
        summary.completed.len(),
        2,
        "neighbor and grandchild should complete"
    );
    assert_eq!(summary.panicked.len(), 1, "parent should panic");
    assert_eq!(
        summary.cancelled.len(),
        2,
        "both child and child2 should be cancelled"
    );
    assert!(summary
        .panicked
        .iter()
        .any(|e| matches!(e, SupervisedBatchEvent::Panicked { name, .. } if &**name == "parent")));
    assert!(summary
        .cancelled
        .iter()
        .any(|e| matches!(e, SupervisedBatchEvent::Cancelled { name, .. } if &**name == "child")));
    assert!(summary
        .cancelled
        .iter()
        .any(|e| matches!(e, SupervisedBatchEvent::Cancelled { name, .. } if &**name == "child2")));
}

/// E2E Cycle Resiliency: verify that a circular dependency does not hang
/// the SupervisedBatch and that the cascade breaks the loop safely.
#[tokio::test(start_paused = true)]
async fn test_e2e_cycle_resiliency() {
    let mut batch = make_batch();

    // A depends on b_provides and fails immediately with Execution; B depends
    // on a_provides and stays pending (transient) until a dependent's failure
    // cancels it — forming a cycle that must not hang the batch.
    batch
        .register(Arc::new(
            StubComponent::fail("A")
                .with_dep("b_provides")
                .with_provides("a_provides"),
        ))
        .unwrap();
    batch
        .register_with_context(
            Arc::new(
                StubComponent::dep_fail("B")
                    .with_dep("a_provides")
                    .with_provides("b_provides"),
            ),
            WorkContext {
                max_retries: 10,
                ..WorkContext::default()
            },
        )
        .unwrap();

    let summary: SupervisedBatchSummary = (&mut batch).await;
    // A fails immediately with Execution error, B is a dependent in a cycle.
    // The cycle should be detected and B should be cancelled.
    assert_eq!(
        summary.failed.len(),
        1,
        "A should fail with Execution error"
    );
    assert_eq!(
        summary.cancelled.len(),
        1,
        "B should be cancelled due to cycle detection"
    );
    assert!(summary
        .failed
        .iter()
        .any(|e| matches!(e, SupervisedBatchEvent::Failed { name, .. } if &**name == "A")));
    assert!(summary
        .cancelled
        .iter()
        .any(|e| matches!(e, SupervisedBatchEvent::Cancelled { name, .. } if &**name == "B")));
}
