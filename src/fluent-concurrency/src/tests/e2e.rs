use crate::pool::WorkerPool;
use crate::batch::{SupervisedBatch, SupervisedBatchEvent, SupervisedBatchSummary};
use fluent_wvr::prelude::*;
use fluent_wvr_testutil::{impl_component_for_test, StubComponent};
use std::sync::Arc;
use std::time::Duration;

/// End-to-end: SupervisedBatch orchestrates WorkerPool-backed tasks
#[tokio::test(start_paused = true)]
async fn test_e2e_batch_with_worker_pool() {
    tokio::time::resume();
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let pool = Arc::new(WorkerPool::new(
        Arc::clone(&runtime),
        2,
        10,
        |job: i32| async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let _ = job * 2;
        },
    ));

    struct PoolWorkUnit {
        name: String,
        _pool: Arc<WorkerPool<i32>>,
        input: i32,
    }
    impl WorkUnit for PoolWorkUnit {
        fn name(&self) -> &str {
            &self.name
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            Ok(WorkOutput::ok_with_data(
                "done",
                serde_json::json!({ "result": self.input * 2 }),
            ))
        }
    }
    impl_component_for_test!(PoolWorkUnit);

    let mut batch = SupervisedBatch::new(runtime, caps);
    batch.register(Arc::new(PoolWorkUnit {
        name: "task1".into(),
        _pool: Arc::clone(&pool),
        input: 5,
    }))
    .unwrap()
    .register(Arc::new(PoolWorkUnit {
        name: "task2".into(),
        _pool: Arc::clone(&pool),
        input: 10,
    }))
    .unwrap();

    let summary: SupervisedBatchSummary = (&mut batch).await;
    assert_eq!(summary.completed.len(), 2);
    assert_eq!(summary.panicked.len(), 0);
    assert_eq!(summary.cancelled.len(), 0);
}

/// End-to-end: SupervisedBatch handles mixed success/failure/cancellation
#[tokio::test(start_paused = true)]
async fn test_e2e_batch_mixed_outcomes() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();

    struct OutcomeUnit {
        name: String,
        outcome: &'static str, // "ok", "fail", "panic"
        deps: Vec<ArcIntern<str>>,
        provides: Vec<ArcIntern<str>>,
    }
    impl WorkUnit for OutcomeUnit {
        fn name(&self) -> &str {
            &self.name
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &self.deps
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &self.provides
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            match self.outcome {
                "ok" => Ok(WorkOutput::ok("done")),
                // Permanent failure — non-retryable; the unit fails on the
                // first attempt.
                "fail" => Err(WorkError::Execution("failed".into())),
                // Transient failure — retryable; keeps the unit pending under
                // a retrying SupervisedBatch until a dependent provider's failure reaches it.
                "dep_fail" => Err(WorkError::Dependency("awaiting shared".into())),
                "panic" => panic!("intentional panic"),
                _ => unreachable!(),
            }
        }
    }
    impl_component_for_test!(OutcomeUnit);

    let shared = ArcIntern::<str>::from("shared");
    let mut batch = SupervisedBatch::new(runtime, caps);
    batch.register(Arc::new(OutcomeUnit {
        name: "root".into(),
        outcome: "fail",
        deps: vec![],
        provides: vec![shared.clone()],
    }))
    .unwrap()
    .register_with_context(
        Arc::new(OutcomeUnit {
            name: "child1".into(),
            outcome: "dep_fail",
            deps: vec![shared.clone()],
            provides: vec![],
        }),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap()
    .register(Arc::new(OutcomeUnit {
        name: "independent".into(),
        outcome: "panic",
        deps: vec![],
        provides: vec![],
    }))
    .unwrap();

    let summary: SupervisedBatchSummary = (&mut batch).await;
    assert_eq!(summary.completed.len(), 0);
    assert_eq!(summary.failed.len(), 1, "root fails with execution error");
    assert_eq!(summary.panicked.len(), 1, "independent panics");
    assert_eq!(summary.cancelled.len(), 1);
}

/// E2E Panic Cascade: verify that a panicking task aborts its transitive
/// dependents while independent neighbors continue unhindered.
#[tokio::test(start_paused = true)]
async fn test_e2e_panic_cascade_with_independent_neighbors() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut batch = SupervisedBatch::new(runtime, caps);

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
    assert_eq!(summary.cancelled.len(), 1, "child should be cancelled");
    assert!(summary
        .panicked
        .iter()
        .any(|e| matches!(e, SupervisedBatchEvent::Panicked { name, .. } if &**name == "parent")));
    assert!(summary
        .cancelled
        .iter()
        .any(|e| matches!(e, SupervisedBatchEvent::Cancelled { name, .. } if &**name == "child")));
}

/// E2E Cycle Resiliency: verify that a circular dependency does not hang
/// the SupervisedBatch and that the cascade breaks the loop safely.
#[tokio::test(start_paused = true)]
async fn test_e2e_cycle_resiliency() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut batch = SupervisedBatch::new(runtime, caps);

    struct CycleUnit {
        name: String,
        deps: Vec<ArcIntern<str>>,
        provides: Vec<ArcIntern<str>>,
        /// `false` → permanent `WorkError::Execution` (fails immediately);
        /// `true` → transient `WorkError::Dependency` (retries, stays pending
        /// until a dependent's failure cancels it).
        retryable: bool,
    }
    impl WorkUnit for CycleUnit {
        fn name(&self) -> &str {
            &self.name
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &self.deps
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &self.provides
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            if self.retryable {
                Err(WorkError::Dependency("cycle member".into()))
            } else {
                Err(WorkError::Execution("cycle member".into()))
            }
        }
    }
    impl_component_for_test!(CycleUnit);

    let a_provides = ArcIntern::<str>::from("a_provides");
    let b_provides = ArcIntern::<str>::from("b_provides");

    batch.register(Arc::new(CycleUnit {
        name: "A".into(),
        deps: vec![b_provides.clone()],
        provides: vec![a_provides.clone()],
        retryable: false,
    }))
    .unwrap();
    batch.register_with_context(
        Arc::new(CycleUnit {
            name: "B".into(),
            deps: vec![a_provides.clone()],
            provides: vec![b_provides.clone()],
            retryable: true,
        }),
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
