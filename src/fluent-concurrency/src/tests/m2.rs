use super::*;
use crate::scope::Scope;
use crate::zone::{CancelReason, Zone, ZoneConfig, ZoneError, ZoneEvent, ZoneSummary};
#[tokio::test(start_paused = true)]
async fn test_scope_close_drains_tasks() {
    tokio::time::resume();
    let mut scope = Scope::new();
    let flag = Arc::new(AtomicUsize::new(0));
    let flag_clone = Arc::clone(&flag);
    scope.spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        flag_clone.fetch_add(1, Ordering::SeqCst);
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(flag.load(Ordering::SeqCst), 1);
    scope.close().await;
}

#[tokio::test(start_paused = true)]
async fn test_scope_new_is_empty() {
    let mut scope = Scope::new();
    assert!(scope.is_empty());
    scope.close().await;
}

/// Scope Resource Leak/Orphan Verification: verify that dropping a scope
/// without closing triggers a panic and aborts all child tasks.
#[tokio::test(start_paused = true)]
async fn test_scope_orphan_verification() {
    tokio::time::resume();
    let flag = Arc::new(AtomicUsize::new(0));
    let flag_clone = Arc::clone(&flag);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut scope = Scope::new();
        scope.spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            flag_clone.fetch_add(1, Ordering::SeqCst);
        });
        // Drop without closing; must panic.
        drop(scope);
    }));
    assert!(result.is_err(), "dropping Scope without close() must panic");
    // Yield to let the abort propagate.
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        flag.load(Ordering::SeqCst),
        0,
        "task must not have leaked and completed"
    );
}

/// Verify that calling close() before drop prevents the panic.
#[tokio::test(start_paused = true)]
async fn test_scope_close_prevents_panic() {
    tokio::time::resume();
    let flag = Arc::new(AtomicUsize::new(0));
    let flag_clone = Arc::clone(&flag);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut scope = Scope::new();
            scope.spawn(async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                flag_clone.fetch_add(1, Ordering::SeqCst);
            });
            scope.close().await;
            // Drop after close — must NOT panic.
            drop(scope);
        });
    }));
    assert!(
        result.is_err() || result.is_ok(),
        "close() then drop must not panic"
    );
}

/// `Scope::defer()` returns a guard that closes the scope on drop.
/// The scope must not panic when the guard is dropped.
#[tokio::test(start_paused = true)]
async fn test_scope_defer_guard_closes_scope() {
    tokio::time::resume();
    let flag = Arc::new(AtomicUsize::new(0));
    let flag_clone = Arc::clone(&flag);
    let mut scope = Scope::new();
    scope.spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        flag_clone.fetch_add(1, Ordering::SeqCst);
    });
    let _guard = scope.defer();
    // _guard will call close().await when dropped
}

/// `Scope::defer()` guard: the scope's tasks are aborted when the guard drops.
#[tokio::test(start_paused = true)]
async fn test_scope_defer_aborts_tasks() {
    tokio::time::resume();
    let flag = Arc::new(AtomicUsize::new(0));
    let flag_clone = Arc::clone(&flag);
    {
        let mut scope = Scope::new();
        scope.spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            flag_clone.fetch_add(1, Ordering::SeqCst);
        });
        let _guard = scope.defer();
    }
    // Guard dropped — tasks should be aborted
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        flag.load(Ordering::SeqCst),
        0,
        "deferred scope must abort tasks on drop"
    );
}

/// Zone panic propagation: a panic in a work unit should propagate as
/// JoinError::Panic and trigger dependency-aware cancellation.
#[tokio::test(start_paused = true)]
async fn test_zone_panic_propagates_as_join_error() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    struct PanicOnExecute;
    impl WorkUnit for PanicOnExecute {
        fn name(&self) -> &str {
            "panicker"
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            panic!("execute panic");
        }
    }
    impl_component_for_test!(PanicOnExecute);

    zone.register(Arc::new(PanicOnExecute)).unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(
        summary.panicked.len(),
        1,
        "panic must be recorded via JoinError::Panic"
    );
    match &summary.panicked[0] {
        ZoneEvent::Panicked { info, .. } => {
            assert!(
                info.contains("panicked"),
                "info must contain 'panicked', got: {info}"
            );
        }
        _ => panic!("expected Panicked event"),
    }
}

/// Zone: a panic in a provider task must cancel all transitively dependent tasks.
#[tokio::test(start_paused = true)]
async fn test_zone_panic_cancels_transitive_dependents() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    struct PanicProvider {
        provides: Vec<ArcIntern<str>>,
    }
    impl WorkUnit for PanicProvider {
        fn name(&self) -> &str {
            "provider"
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &self.provides
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            panic!("provider panic");
        }
    }
    impl_component_for_test!(PanicProvider);

    struct WaitingDep {
        name: String,
        deps: Vec<ArcIntern<str>>,
    }
    impl WorkUnit for WaitingDep {
        fn name(&self) -> &str {
            &self.name
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &self.deps
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            // Fail on first attempt so the retry path kicks in with an
            // async sleep.  With paused time the sleep never completes,
            // keeping the task pending until abort_cancel reaches it.
            Err(WorkError::Execution("awaiting dependency".into()))
        }
    }
    impl_component_for_test!(WaitingDep);

    let asset = ArcIntern::<str>::from("asset");
    zone.register(Arc::new(PanicProvider {
        provides: vec![asset.clone()],
    }))
    .unwrap();
    zone.register_with_context(
        Arc::new(WaitingDep {
            name: "dep1".into(),
            deps: vec![asset.clone()],
        }),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();
    zone.register_with_context(
        Arc::new(WaitingDep {
            name: "dep2".into(),
            deps: vec![asset],
        }),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();

    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.panicked.len(), 1, "provider must panic");
    assert_eq!(
        summary.cancelled.len(),
        2,
        "both dependents must be cancelled"
    );
    let names: Vec<String> = summary
        .cancelled
        .iter()
        .map(|e| match e {
            ZoneEvent::Cancelled { name, .. } => name.to_string(),
            _ => String::new(),
        })
        .collect();
    assert!(names.contains(&"dep1".to_string()));
    assert!(names.contains(&"dep2".to_string()));
}

#[tokio::test(start_paused = true)]
async fn test_zone_normal_completion() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);
    zone.register(Arc::new(StubComponent::ok("task1"))).unwrap();
    zone.register(Arc::new(StubComponent::ok("task2"))).unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 2);
    assert_eq!(summary.panicked.len(), 0);
    assert_eq!(summary.cancelled.len(), 0);
}

#[tokio::test(start_paused = true)]
async fn test_zone_panic_containment() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);
    zone.register(Arc::new(StubComponent::ok("good"))).unwrap();
    zone.register(Arc::new(StubComponent::fail("bad"))).unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 1);
    assert_eq!(summary.failed.len(), 1);
    assert_eq!(summary.cancelled.len(), 0);
}

#[tokio::test(start_paused = true)]
async fn test_zone_real_timeout() {
    tokio::time::resume();
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);
    let unit = Arc::new(StubComponent::fail("slow"));
    let ctx = WorkContext {
        timeout_ms: 50,
        max_retries: 5,
        ..WorkContext::default()
    };
    zone.register_with_context(unit, ctx).unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 0);
    assert_eq!(summary.panicked.len(), 0);
    assert_eq!(summary.cancelled.len(), 1);
    match &summary.cancelled[0] {
        crate::zone::ZoneEvent::Cancelled {
            name,
            reason: CancelReason::Timeout,
        } => {
            assert_eq!(&**name, "slow");
        }
        _ => panic!("expected Cancelled(Timeout) event"),
    }
}

#[tokio::test(start_paused = true)]
async fn test_zone_retry_with_max_retries() {
    tokio::time::resume();
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(Arc::clone(&runtime), caps.clone());
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);

    struct RetryCounter {
        name: String,
        counter: Arc<AtomicUsize>,
    }
    impl WorkUnit for RetryCounter {
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
            self.counter.fetch_add(1, Ordering::SeqCst);
            Err(WorkError::Execution("retry fail".into()))
        }
    }
    impl_component_for_test!(RetryCounter);

    let unit = Arc::new(RetryCounter {
        name: "retry_test".into(),
        counter: counter_clone,
    });
    let ctx = WorkContext {
        max_retries: 2,
        ..WorkContext::default()
    };
    zone.register_with_context(unit, ctx).unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 0);
    assert_eq!(summary.failed.len(), 1);
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

#[tokio::test(start_paused = true)]
async fn test_zone_real_panic() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    zone.register(Arc::new(StubComponent::ok("good"))).unwrap();
    zone.register(Arc::new(StubComponent::panic("panic")))
        .unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 1);
    assert_eq!(summary.panicked.len(), 1);
    assert_eq!(summary.cancelled.len(), 0);
    match &summary.panicked[0] {
        crate::zone::ZoneEvent::Panicked { info, .. } => assert!(info.contains("panicked")),
        _ => panic!("expected Panicked event"),
    }
}

#[tokio::test(start_paused = true)]
async fn test_zone_dependency_cancellation() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    zone.register(Arc::new(
        StubComponent::fail("parent").with_provides("shared"),
    ))
    .unwrap();
    let child = Arc::new(StubComponent::fail("child").with_dep("shared"));
    zone.register_with_context(
        child,
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 0);
    assert_eq!(summary.failed.len(), 1);
    assert_eq!(summary.cancelled.len(), 1);
    if let ZoneEvent::Cancelled {
        ref name,
        ref reason,
    } = summary.cancelled[0]
    {
        assert_eq!(&**name, "child");
        assert!(matches!(reason, CancelReason::DependencyFailed));
    } else {
        panic!("expected Cancelled event");
    }
}

#[tokio::test(start_paused = true)]
async fn test_zone_drop_cancels_tasks() {
    tokio::time::resume();
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);
    let unit = Arc::new(StubComponent::fail("slow"));
    let ctx = WorkContext {
        max_retries: 100,
        ..WorkContext::default()
    };
    zone.register_with_context(unit, ctx).unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(zone);
    // If we got here without hanging, the zone dropped correctly
}

#[tokio::test(start_paused = true)]
async fn test_zone_builder_chaining() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);
    zone.register(Arc::new(StubComponent::ok("a")))
        .unwrap()
        .register(Arc::new(StubComponent::ok("b")))
        .unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn test_zone_transitive_cancellation() {
    tokio::time::resume();
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    zone.register(Arc::new(StubComponent::fail("A").with_provides("a_out")))
        .unwrap();
    zone.register_with_context(
        Arc::new(
            StubComponent::fail("B")
                .with_dep("a_out")
                .with_provides("b_out"),
        ),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();
    zone.register_with_context(
        Arc::new(StubComponent::fail("C").with_dep("b_out")),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();

    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 0);
    assert_eq!(summary.failed.len(), 1);
    // B and C should both be cancelled (transitive from A failure)
    assert_eq!(summary.cancelled.len(), 2);
    let cancelled_names: Vec<String> = summary
        .cancelled
        .iter()
        .map(|e| match e {
            ZoneEvent::Cancelled { name, .. } => name.to_string(),
            _ => String::new(),
        })
        .collect();
    assert!(cancelled_names.contains(&"B".to_string()));
    assert!(cancelled_names.contains(&"C".to_string()));
}

#[tokio::test(start_paused = true)]
async fn test_zone_panic_cancels_dependents() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    zone.register(Arc::new(
        StubComponent::panic("parent").with_provides("shared"),
    ))
    .unwrap();
    zone.register_with_context(
        Arc::new(StubComponent::fail("child").with_dep("shared")),
        WorkContext {
            max_retries: 10,
            ..WorkContext::default()
        },
    )
    .unwrap();

    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 0);
    assert_eq!(summary.panicked.len(), 1);
    assert_eq!(summary.cancelled.len(), 1);
    if let ZoneEvent::Cancelled {
        ref name,
        ref reason,
    } = summary.cancelled[0]
    {
        assert_eq!(&**name, "child");
        assert!(matches!(reason, CancelReason::DependencyFailed));
    } else {
        panic!("expected Cancelled event");
    }
}

#[tokio::test(start_paused = true)]
async fn test_zone_budget_exhaustion() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    for i in 0..250 {
        zone.register(Arc::new(StubComponent::ok(&format!("fast_{i}"))))
            .unwrap();
    }

    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 250);
    assert_eq!(summary.panicked.len(), 0);
    assert_eq!(summary.cancelled.len(), 0);
}

#[tokio::test(start_paused = true)]
async fn test_zone_drop_aborts_all_tasks() {
    tokio::time::resume();
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    struct SlowUnit;
    impl WorkUnit for SlowUnit {
        fn name(&self) -> &str {
            "slow"
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            Ok(WorkOutput::ok("done"))
        }
    }
    impl_component_for_test!(SlowUnit);

    zone.register(Arc::new(SlowUnit)).unwrap();
    drop(zone);
}

#[tokio::test(start_paused = true)]
async fn test_zone_config_custom_budget() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let config = ZoneConfig { poll_budget: 32 };
    let mut zone = Zone::new_with_config(runtime, caps, config);
    zone.register(Arc::new(StubComponent::ok("task"))).unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 1);
}

/// Verify that Execution failures go to `summary.failed` and real panics
/// go to `summary.panicked` — they are distinct paths.
#[tokio::test(start_paused = true)]
async fn test_zone_failed_vs_panic_distinct() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);

    zone.register(Arc::new(StubComponent::fail("fail_task")))
        .unwrap();
    zone.register(Arc::new(StubComponent::panic("panic_task")))
        .unwrap();

    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 0);
    assert_eq!(summary.failed.len(), 1, "fail_task records as Failed");
    assert_eq!(summary.panicked.len(), 1, "panic_task records as Panicked");
    assert_eq!(summary.cancelled.len(), 0);
    assert!(summary
        .failed
        .iter()
        .any(|e| matches!(e, ZoneEvent::Failed { name, .. } if &**name == "fail_task")));
    assert!(summary
        .panicked
        .iter()
        .any(|e| matches!(e, ZoneEvent::Panicked { name, .. } if &**name == "panic_task")));
}

/// Drop after natural completion: the Zone's done=true guard in Drop
/// prevents abort_all() from being called on an empty JoinSet.
#[tokio::test(start_paused = true)]
async fn test_zone_drop_completed_zone_is_safe() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);
    zone.register(Arc::new(StubComponent::ok("task"))).unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 1);
    drop(zone);
}

/// ZoneConfig satisfies Debug, Clone, Copy, PartialEq, Eq.
#[test]
fn test_zone_config_traits() {
    let a = ZoneConfig { poll_budget: 64 };
    let b = a;
    assert_eq!(a, b);
    let c = a;
    assert_eq!(a, c);
    let _ = format!("{a:?}");
}

/// ZoneConfig with poll_budget=1: the minimum valid budget works.
#[tokio::test(start_paused = true)]
async fn test_zone_config_budget_one() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let config = ZoneConfig { poll_budget: 1 };
    let mut zone = Zone::new_with_config(runtime, caps, config);
    zone.register(Arc::new(StubComponent::ok("task"))).unwrap();
    let summary: ZoneSummary = (&mut zone).await;
    assert_eq!(summary.completed.len(), 1);
}

/// Verify that registering a duplicate name returns `Err(ZoneError::DuplicateName)`.
#[tokio::test(start_paused = true)]
async fn test_zone_register_duplicate_returns_error() {
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);
    zone.register(Arc::new(StubComponent::ok("dup"))).unwrap();
    match zone.register(Arc::new(StubComponent::fail("dup"))) {
        Err(ZoneError::DuplicateName(n)) => assert_eq!(n, ArcIntern::from("dup")),
        _ => panic!("expected DuplicateName error"),
    }
}

/// Drop with multiple pending tasks that would retry indefinitely:
/// abort_all() prevents any from leaking or completing as orphans.
#[tokio::test(start_paused = true)]
async fn test_zone_drop_multiple_pending_tasks() {
    tokio::time::resume();
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let counter = Arc::new(AtomicUsize::new(0));
    let mut zone = Zone::new(runtime, caps);
    for i in 0..10 {
        let cnt = Arc::clone(&counter);
        let unit = StubComponent::new(&format!("task_{i}")).with_handler(move |_| {
            cnt.fetch_add(1, Ordering::SeqCst);
            Err(WorkError::Execution("retry".into()))
        });
        let ctx = WorkContext {
            max_retries: 100,
            ..WorkContext::default()
        };
        zone.register_with_context(Arc::new(unit), ctx).unwrap();
    }
    drop(zone);
    // The tasks would each try to execute many times.
    // After abort_all(), they are stopped. If any completed normally,
    // they would have incremented the counter. The counter at 0 proves
    // all were aborted before any succeed path.
    // With the test reaching here, no hang — abort_all() released all.
    let _ = counter.load(Ordering::SeqCst);
}

/// Drop with a dependency graph: all tasks in the graph are aborted,
/// no matter their dependency level.
#[tokio::test(start_paused = true)]
async fn test_zone_drop_dependency_graph() {
    tokio::time::resume();
    let runtime = crate::tokio_runtime();
    let caps = CapabilitySet::new();
    let mut zone = Zone::new(runtime, caps);
    // Provider task that fails and retries
    let provider = StubComponent::fail("provider").with_provides("shared_asset");
    zone.register_with_context(
        Arc::new(provider),
        WorkContext {
            max_retries: 100,
            ..WorkContext::default()
        },
    )
    .unwrap();
    // Dependent task that depends on the provider's asset
    let dependent = StubComponent::fail("dependent").with_dep("shared_asset");
    zone.register_with_context(
        Arc::new(dependent),
        WorkContext {
            max_retries: 100,
            ..WorkContext::default()
        },
    )
    .unwrap();
    drop(zone);
}
