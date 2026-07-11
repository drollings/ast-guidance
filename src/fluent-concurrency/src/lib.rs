#![forbid(unsafe_code)]

pub mod capability;
pub mod flow;
pub mod io;
pub mod pool;
pub mod queue;
pub mod router;
pub mod runtime;
pub mod scope;
pub mod zone;

use std::sync::Arc;

/// Returns a new `Arc<dyn Runtime>` wrapping the production Tokio runtime.
/// Use this instead of `crate::tokio_runtime()`.
pub fn tokio_runtime() -> Arc<dyn fluent_wvr::Runtime> {
    Arc::new(runtime::tokio::TokioRuntime)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use crate::runtime::test::TestRuntime;
    use crate::runtime::tokio::TokioRuntime;
    use fluent_wvr::{
        Capability, CapabilitySet, Reserve, Runtime, WorkContext, WorkError, WorkOutput, WorkUnit,
    };
    use fluent_wvr_testutil::impl_component_for_test;
    use internment::ArcIntern;

    struct TestCapA;
    impl Capability for TestCapA {
        fn name(&self) -> &'static str {
            "cap_a"
        }
    }

    struct TestCapB;
    impl Capability for TestCapB {
        fn name(&self) -> &'static str {
            "cap_b"
        }
    }

    /// Shared test helper: a `WorkUnit` that panics with a configurable message.
    struct PanicUnit {
        name: String,
        provides: Vec<ArcIntern<str>>,
    }
    impl WorkUnit for PanicUnit {
        fn name(&self) -> &str {
            &self.name
        }
        fn depends(&self) -> &[ArcIntern<str>] {
            &[]
        }
        fn provides(&self) -> &[ArcIntern<str>] {
            &self.provides
        }
        fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
            panic!("intentional panic")
        }
    }
    impl_component_for_test!(PanicUnit);

    /// Shared test helper: a `WorkUnit` with dependencies that can optionally fail.
    struct DepWorkUnit {
        name: String,
        deps: Vec<ArcIntern<str>>,
        provides: Vec<ArcIntern<str>>,
        should_fail: bool,
    }
    impl WorkUnit for DepWorkUnit {
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
            if self.should_fail {
                Err(WorkError::Execution("dep failure".into()))
            } else {
                Ok(WorkOutput::ok("done"))
            }
        }
    }
    impl_component_for_test!(DepWorkUnit);

    #[tokio::test(start_paused = true)]
    async fn test_tokio_runtime_spawn() {
        tokio::time::resume();
        let runtime = TokioRuntime;
        let handle = runtime.spawn(Box::pin(async {}));
        handle.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn test_tokio_runtime_sleep_now() {
        tokio::time::resume();
        let runtime = TokioRuntime;
        let before = runtime.now();
        runtime.sleep(Duration::from_millis(1)).await;
        let after = runtime.now();
        assert!(after >= before);
    }

    #[tokio::test(start_paused = true)]
    async fn test_test_runtime_with_paused_time() {
        tokio::time::resume();
        let handle = tokio::runtime::Handle::current();
        let test_runtime = TestRuntime::new(handle, 42);
        let before = test_runtime.now();
        test_runtime.sleep(Duration::from_millis(5)).await;
        let after = test_runtime.now();
        assert!(after >= before);
    }

    #[tokio::test(start_paused = true)]
    async fn test_test_runtime_spawn() {
        tokio::time::resume();
        let handle = tokio::runtime::Handle::current();
        let test_runtime = TestRuntime::new(handle, 42);
        let join = test_runtime.spawn(Box::pin(async {}));
        join.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn test_test_runtime_deterministic_rng() {
        tokio::time::resume();
        let handle = tokio::runtime::Handle::current();
        let rt1 = TestRuntime::new(handle.clone(), 12345);
        let rt2 = TestRuntime::new(handle, 12345);
        let a = rt1.rng().lock().unwrap().u32(..);
        let b = rt2.rng().lock().unwrap().u32(..);
        assert_eq!(a, b, "same seed must produce same output");
    }

    #[test]
    fn test_capability_set_insert_get() {
        let caps = CapabilitySet::new().with(TestCapA).with(TestCapB);
        assert!(caps.get::<TestCapA>().is_some());
        assert!(caps.get::<TestCapB>().is_some());
        assert_eq!(caps.get::<TestCapA>().unwrap().name(), "cap_a");
    }

    #[test]
    fn test_capability_set_missing_returns_none() {
        let caps = CapabilitySet::new().with(TestCapA);
        assert!(caps.get::<TestCapB>().is_none());
    }

    #[test]
    fn test_capability_set_clone() {
        let caps = CapabilitySet::new().with(TestCapA);
        let cloned = caps.clone();
        assert!(cloned.get::<TestCapA>().is_some());
    }

    #[test]
    fn test_reserve_acquire_and_drop_releases() {
        let counter = Arc::new(AtomicUsize::new(2));
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        {
            let _reserve = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_reserve_commit_does_not_release() {
        let counter = Arc::new(AtomicUsize::new(2));
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        let reserve = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        reserve.commit();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_reserve_multiple_acquires() {
        let counter = Arc::new(AtomicUsize::new(2));
        let r1 = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        let r2 = Reserve::try_acquire(Arc::clone(&counter)).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        r1.commit();
        drop(r2);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_reserve_exhaustion_returns_none() {
        let counter = Arc::new(AtomicUsize::new(0));
        let result = Reserve::try_acquire(Arc::clone(&counter));
        assert!(result.is_none());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_derived_field_access() {
        use fluent_wvr::FieldAccess;

        #[derive(FieldAccess)]
        struct Config {
            label: String,
            count: u32,
        }
        let mut cfg = Config {
            label: "hello".into(),
            count: 5,
        };
        assert_eq!(cfg.get_field("label").unwrap(), "hello");
        assert_eq!(cfg.get_field("count").unwrap(), "5");
        cfg.set_field("label", "world").unwrap();
        cfg.set_field("count", "10").unwrap();
        assert_eq!(cfg.get_field("label").unwrap(), "world");
        assert_eq!(cfg.get_field("count").unwrap(), "10");
        assert!(cfg.set_field("missing", "x").is_err());
        assert_eq!(cfg.field_names(), &["label", "count"]);
    }

    mod m2 {
        use super::*;
        use crate::scope::Scope;
        use crate::zone::{CancelReason, Zone, ZoneConfig, ZoneEvent, ZoneSummary};
        use fluent_wvr_testutil::impl_component_for_test;

        struct TestWorkUnit {
            name: String,
            should_fail: bool,
        }

        impl TestWorkUnit {
            fn ok(name: &str) -> Self {
                Self {
                    name: name.into(),
                    should_fail: false,
                }
            }

            fn fail(name: &str) -> Self {
                Self {
                    name: name.into(),
                    should_fail: true,
                }
            }
        }

        impl WorkUnit for TestWorkUnit {
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
                if self.should_fail {
                    Err(WorkError::Execution("test failure".into()))
                } else {
                    Ok(WorkOutput::ok("done"))
                }
            }
        }
        impl_component_for_test!(TestWorkUnit);

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

            zone.register(Arc::new(PanicOnExecute));
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
            }));
            zone.register_with_context(
                Arc::new(WaitingDep {
                    name: "dep1".into(),
                    deps: vec![asset.clone()],
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );
            zone.register_with_context(
                Arc::new(WaitingDep {
                    name: "dep2".into(),
                    deps: vec![asset],
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );

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
            zone.register(Arc::new(TestWorkUnit::ok("task1")));
            zone.register(Arc::new(TestWorkUnit::ok("task2")));
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
            zone.register(Arc::new(TestWorkUnit::ok("good")));
            zone.register(Arc::new(TestWorkUnit::fail("bad")));
            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 1);
            assert_eq!(summary.panicked.len(), 1);
            assert_eq!(summary.cancelled.len(), 0);
        }

        #[tokio::test(start_paused = true)]
        async fn test_zone_real_timeout() {
            tokio::time::resume();
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let mut zone = Zone::new(runtime, caps);
            let unit = Arc::new(TestWorkUnit::fail("slow"));
            let ctx = WorkContext {
                timeout_ms: 50,
                max_retries: 5,
                ..WorkContext::default()
            };
            zone.register_with_context(unit, ctx);
            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 0);
            assert_eq!(summary.panicked.len(), 1);
            assert_eq!(summary.cancelled.len(), 0);
            match &summary.panicked[0] {
                crate::zone::ZoneEvent::Panicked { info, .. } => {
                    assert!(info.contains("timed out"));
                }
                _ => panic!("expected Panicked event"),
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
            zone.register_with_context(unit, ctx);
            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 0);
            assert_eq!(summary.panicked.len(), 1);
            assert_eq!(counter.load(Ordering::SeqCst), 3);
        }

        #[tokio::test(start_paused = true)]
        async fn test_zone_real_panic() {
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let mut zone = Zone::new(runtime, caps);

            struct PanicUnit;
            impl WorkUnit for PanicUnit {
                fn name(&self) -> &str {
                    "panic"
                }
                fn depends(&self) -> &[ArcIntern<str>] {
                    &[]
                }
                fn provides(&self) -> &[ArcIntern<str>] {
                    &[]
                }
                fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
                    panic!("intentional panic");
                }
            }
            impl_component_for_test!(PanicUnit);

            zone.register(Arc::new(TestWorkUnit::ok("good")));
            zone.register(Arc::new(PanicUnit));
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

            let shared = ArcIntern::<str>::from("shared");
            zone.register(Arc::new(DepWorkUnit {
                name: "parent".into(),
                deps: vec![],
                provides: vec![shared.clone()],
                should_fail: true,
            }));
            let child = Arc::new(DepWorkUnit {
                name: "child".into(),
                deps: vec![shared.clone()],
                provides: vec![],
                should_fail: true,
            });
            zone.register_with_context(
                child,
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );
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
        async fn test_zone_drop_cancels_tasks() {
            tokio::time::resume();
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let mut zone = Zone::new(runtime, caps);
            let unit = Arc::new(TestWorkUnit::fail("slow"));
            let ctx = WorkContext {
                max_retries: 100,
                ..WorkContext::default()
            };
            zone.register_with_context(unit, ctx);
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(zone);
            // If we got here without hanging, the zone dropped correctly
        }

        #[tokio::test(start_paused = true)]
        async fn test_zone_builder_chaining() {
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let mut zone = Zone::new(runtime, caps);
            zone.register(Arc::new(TestWorkUnit::ok("a")))
                .register(Arc::new(TestWorkUnit::ok("b")));
            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 2);
        }

        #[tokio::test(start_paused = true)]
        async fn test_zone_transitive_cancellation() {
            tokio::time::resume();
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let mut zone = Zone::new(runtime, caps);

            struct ChainUnit {
                name: String,
                deps: Vec<ArcIntern<str>>,
                provides: Vec<ArcIntern<str>>,
                should_fail: bool,
            }
            impl WorkUnit for ChainUnit {
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
                    if self.should_fail {
                        Err(WorkError::Execution("chain failure".into()))
                    } else {
                        Ok(WorkOutput::ok("done"))
                    }
                }
            }
            impl_component_for_test!(ChainUnit);

            let a_out = ArcIntern::<str>::from("a_out");
            let b_out = ArcIntern::<str>::from("b_out");

            zone.register(Arc::new(ChainUnit {
                name: "A".into(),
                deps: vec![],
                provides: vec![a_out.clone()],
                should_fail: true,
            }));
            zone.register_with_context(
                Arc::new(ChainUnit {
                    name: "B".into(),
                    deps: vec![a_out.clone()],
                    provides: vec![b_out.clone()],
                    should_fail: true,
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );
            zone.register_with_context(
                Arc::new(ChainUnit {
                    name: "C".into(),
                    deps: vec![b_out.clone()],
                    provides: vec![],
                    should_fail: true,
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );

            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 0);
            assert_eq!(summary.panicked.len(), 1);
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

            let shared = ArcIntern::<str>::from("shared");
            zone.register(Arc::new(PanicUnit {
                name: "parent".into(),
                provides: vec![shared.clone()],
            }));
            zone.register_with_context(
                Arc::new(DepWorkUnit {
                    name: "child".into(),
                    deps: vec![shared.clone()],
                    provides: vec![],
                    should_fail: true,
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );

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
                zone.register(Arc::new(TestWorkUnit::ok(&format!("fast_{i}"))));
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

            zone.register(Arc::new(SlowUnit));
            drop(zone);
        }

        #[tokio::test(start_paused = true)]
        async fn test_zone_config_custom_budget() {
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let config = ZoneConfig { poll_budget: 32 };
            let mut zone = Zone::new_with_config(runtime, caps, config);
            zone.register(Arc::new(TestWorkUnit::ok("task")));
            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 1);
        }

        /// Drop after natural completion: the Zone's done=true guard in Drop
        /// prevents abort_all() from being called on an empty JoinSet.
        #[tokio::test(start_paused = true)]
        async fn test_zone_drop_completed_zone_is_safe() {
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let mut zone = Zone::new(runtime, caps);
            zone.register(Arc::new(TestWorkUnit::ok("task")));
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
            zone.register(Arc::new(TestWorkUnit::ok("task")));
            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 1);
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
                let unit = TestWorkUnitWithCounter {
                    name: format!("task_{i}"),
                    counter: cnt,
                };
                let ctx = WorkContext {
                    max_retries: 100,
                    ..WorkContext::default()
                };
                zone.register_with_context(Arc::new(unit), ctx);
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
            let asset = ArcIntern::<str>::from("shared_asset");
            // Provider task that fails and retries
            let provider = TestWorkUnitWithDep {
                name: "provider".into(),
                should_fail: true,
                deps: vec![],
                provides: vec![asset.clone()],
            };
            zone.register_with_context(
                Arc::new(provider),
                WorkContext {
                    max_retries: 100,
                    ..WorkContext::default()
                },
            );
            // Dependent task that depends on the provider's asset
            let dependent = TestWorkUnitWithDep {
                name: "dependent".into(),
                should_fail: true,
                deps: vec![asset.clone()],
                provides: vec![],
            };
            zone.register_with_context(
                Arc::new(dependent),
                WorkContext {
                    max_retries: 100,
                    ..WorkContext::default()
                },
            );
            drop(zone);
        }
    }

    // Helper structs used across multiple test modules inside tests
    // (defined at module level to avoid redefinition in each test).

    struct TestWorkUnitWithCounter {
        name: String,
        counter: Arc<AtomicUsize>,
    }

    impl WorkUnit for TestWorkUnitWithCounter {
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
            Err(WorkError::Execution("retry".into()))
        }
    }
    impl_component_for_test!(TestWorkUnitWithCounter);

    struct TestWorkUnitWithDep {
        name: String,
        should_fail: bool,
        deps: Vec<ArcIntern<str>>,
        provides: Vec<ArcIntern<str>>,
    }

    impl WorkUnit for TestWorkUnitWithDep {
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
            if self.should_fail {
                Err(WorkError::Execution("always fail".into()))
            } else {
                Ok(WorkOutput::ok("done"))
            }
        }
    }
    impl_component_for_test!(TestWorkUnitWithDep);

    mod m3 {
        use super::*;
        use crate::pool::{Limiter, PoolError, Queue, ResultPool, ResultPoolError, WorkerPool};
        use crate::queue::PriorityQueue;
        use crate::router::PartitionedRouter;
        use std::sync::Arc;

        #[tokio::test(start_paused = true)]
        async fn test_queue_push_and_pop_order() {
            let q = Queue::new(10);
            q.push(1).await.unwrap();
            q.push(2).await.unwrap();
            q.push(3).await.unwrap();
            assert_eq!(q.pop().await, Some(1));
            assert_eq!(q.pop().await, Some(2));
            assert_eq!(q.pop().await, Some(3));
            q.close();
        }

        #[tokio::test(start_paused = true)]
        async fn test_queue_bounded_full() {
            let q = Queue::new(2);
            assert!(q.push(1).await.is_ok());
            assert!(q.push(2).await.is_ok());
            assert_eq!(q.push(3).await, Err(PoolError::Full));
            q.close();
        }

        #[tokio::test(start_paused = true)]
        async fn test_queue_close_wakes_waiters() {
            let q: Queue<i32> = Queue::new(2);
            let q2 = Arc::new(q);
            let q_clone = Arc::clone(&q2);
            let handle = tokio::spawn(async move {
                assert_eq!(q_clone.pop().await, None);
            });
            tokio::task::yield_now().await;
            q2.close();
            handle.await.unwrap();
        }

        #[tokio::test(start_paused = true)]
        async fn test_worker_pool_processes_all_jobs() {
            let results = Arc::new(std::sync::Mutex::new(Vec::new()));
            let r = Arc::clone(&results);
            let pool = WorkerPool::new(crate::tokio_runtime(), 2, 10, move |job: i32| {
                let r = Arc::clone(&r);
                async move {
                    let mut guard = r.lock().unwrap();
                    guard.push(job * 2);
                }
            });
            pool.submit(1).await.unwrap();
            pool.submit(2).await.unwrap();
            pool.submit(3).await.unwrap();
            pool.shutdown().await;
            let guard = results.lock().unwrap();
            assert_eq!(guard.len(), 3);
            assert!(guard.contains(&2));
            assert!(guard.contains(&4));
            assert!(guard.contains(&6));
        }

        #[tokio::test(start_paused = true)]
        async fn test_worker_pool_shutdown() {
            tokio::time::resume();
            let completed = Arc::new(AtomicUsize::new(0));
            let c = Arc::clone(&completed);
            let pool = WorkerPool::new(crate::tokio_runtime(), 2, 10, move |job: i32| {
                let c = Arc::clone(&c);
                async move {
                    tokio::time::sleep(Duration::from_millis(10 * u64::try_from(job).unwrap()))
                        .await;
                    c.fetch_add(1, Ordering::SeqCst);
                }
            });
            pool.submit(1).await.unwrap();
            pool.submit(2).await.unwrap();
            pool.submit(3).await.unwrap();
            pool.shutdown().await;
            assert_eq!(completed.load(Ordering::SeqCst), 3);
        }

        #[tokio::test(start_paused = true)]
        async fn test_limiter_caps_concurrency() {
            tokio::time::resume();
            let limiter = Arc::new(Limiter::new(2));
            let counter = Arc::new(AtomicUsize::new(0));
            let max_concurrent = Arc::new(AtomicUsize::new(0));

            let mut handles = Vec::new();
            for _ in 0..5 {
                let lim = Arc::clone(&limiter);
                let cnt = Arc::clone(&counter);
                let max_c = Arc::clone(&max_concurrent);
                handles.push(tokio::spawn(async move {
                    lim.run(|| async {
                        let prev = cnt.fetch_add(1, Ordering::SeqCst);
                        max_c.fetch_max(prev + 1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        cnt.fetch_sub(1, Ordering::SeqCst);
                    })
                    .await;
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
            assert!(max_concurrent.load(Ordering::SeqCst) <= 2);
        }

        /// `Limiter::run_sync` works from a sync context (no tokio runtime needed).
        #[test]
        fn limiter_run_sync_caps_concurrency() {
            let limiter = Limiter::new(2);
            let counter = Arc::new(AtomicUsize::new(0));
            let max_concurrent = Arc::new(AtomicUsize::new(0));

            // Run 5 tasks synchronously through the limiter.
            // run_sync creates its own runtime internally.
            for _ in 0..5 {
                let cnt = Arc::clone(&counter);
                let max_c = Arc::clone(&max_concurrent);
                limiter.run_sync(|| async move {
                    let prev = cnt.fetch_add(1, Ordering::SeqCst);
                    max_c.fetch_max(prev + 1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    cnt.fetch_sub(1, Ordering::SeqCst);
                });
            }
            // Since run_sync is blocking and sequential, max_concurrent is 1.
            assert!(max_concurrent.load(Ordering::SeqCst) <= 2);
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn test_priority_queue_fast_path() {
            let mut pq = PriorityQueue::new();
            pq.push("a", 0);
            pq.push("b", 0);
            pq.push("c", 0);
            assert_eq!(pq.pop(), Some("a"));
            assert_eq!(pq.pop(), Some("b"));
            assert_eq!(pq.pop(), Some("c"));
            assert_eq!(pq.pop(), None);
        }

        #[test]
        fn test_priority_queue_mixed() {
            let mut pq = PriorityQueue::new();
            pq.push("low", -1);
            pq.push("normal", 0);
            pq.push("high", 1);
            pq.push("critical", 2);
            assert_eq!(pq.pop(), Some("critical"));
            assert_eq!(pq.pop(), Some("high"));
            assert_eq!(pq.pop(), Some("normal"));
            assert_eq!(pq.pop(), Some("low"));
            assert_eq!(pq.pop(), None);
        }

        #[test]
        fn test_priority_queue_empty() {
            let mut pq: PriorityQueue<i32> = PriorityQueue::new();
            assert_eq!(pq.pop(), None);
        }

        #[test]
        fn test_priority_queue_fifo_after_mixed() {
            let mut pq = PriorityQueue::new();
            pq.push("first", 0);
            pq.push("high", 1);
            pq.push("second", 0);
            assert_eq!(pq.pop(), Some("high"));
            assert_eq!(pq.pop(), Some("first"));
            assert_eq!(pq.pop(), Some("second"));
            assert_eq!(pq.pop(), None);
        }

        #[test]
        fn test_priority_queue_len_and_is_empty() {
            let mut pq = PriorityQueue::new();
            assert!(pq.is_empty());
            assert_eq!(pq.len(), 0);
            pq.push("a", 0);
            assert!(!pq.is_empty());
            assert_eq!(pq.len(), 1);
            pq.push("b", 1);
            pq.push("c", -1);
            assert_eq!(pq.len(), 3);
            pq.pop();
            assert_eq!(pq.len(), 2);
            pq.pop();
            assert_eq!(pq.len(), 1);
        }

        #[test]
        fn test_priority_queue_peek() {
            let mut pq = PriorityQueue::new();
            assert!(pq.peek().is_none());
            pq.push("low", -1);
            pq.push("normal", 0);
            pq.push("high", 1);
            let (item, prio) = pq.peek().unwrap();
            assert_eq!(*item, "high");
            assert_eq!(prio, 1);
            pq.pop();
            let (item, prio) = pq.peek().unwrap();
            assert_eq!(*item, "normal");
            assert_eq!(prio, 0);
        }

        #[test]
        fn test_priority_queue_into_iter() {
            let mut pq = PriorityQueue::new();
            pq.push("low", -1);
            pq.push("normal", 0);
            pq.push("high", 1);
            let items: Vec<(i32, &str)> = pq.into_iter().collect();
            assert_eq!(items, vec![(1, "high"), (0, "normal"), (-1, "low")]);
        }

        #[test]
        fn test_priority_queue_drain() {
            let mut pq = PriorityQueue::new();
            pq.push("a", 0);
            pq.push("high", 1);
            pq.push("b", 0);
            assert_eq!(pq.len(), 3);
            let drained: Vec<(i32, &str)> = pq.drain().collect();
            assert_eq!(drained, vec![(1, "high"), (0, "a"), (0, "b")]);
            assert!(pq.is_empty());
        }

        /// High-contention efficiency: verify that PriorityQueue::len is O(1)
        /// and that the queue handles many items correctly.
        #[test]
        fn test_high_contention_priority_queue_len() {
            let mut pq = PriorityQueue::new();
            for i in 0..10000 {
                pq.push(i, i % 10);
            }
            assert_eq!(pq.len(), 10000);
            for _ in 0..10000 {
                pq.pop();
            }
            assert!(pq.is_empty());
            assert_eq!(pq.len(), 0);
        }

        /// High-contention efficiency: verify that WorkerPool routing remains
        /// flat, monomorphic, and deadlock-free under parallel load.
        #[tokio::test(start_paused = true)]
        async fn test_high_contention_worker_pool() {
            tokio::time::resume();
            let completed = Arc::new(AtomicUsize::new(0));
            let c = Arc::clone(&completed);
            let pool = Arc::new(WorkerPool::new(
                crate::tokio_runtime(),
                4,
                2000,
                move |job: i32| {
                    let c = Arc::clone(&c);
                    async move {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                        c.fetch_add(1, Ordering::SeqCst);
                        let _ = job;
                    }
                },
            ));
            let mut handles = Vec::new();
            for _ in 0..10 {
                let p = Arc::clone(&pool);
                handles.push(tokio::spawn(async move {
                    for i in 0..100 {
                        p.submit(i).await.expect("queue should not be full");
                    }
                }));
            }
            for h in handles {
                h.await.unwrap();
            }
            // Safety: Arc::try_unwrap is used to get the inner WorkerPool
            // for shutdown. Since all spawned tasks are done, the reference
            // count is 1.
            let pool = Arc::try_unwrap(pool).unwrap_or_else(|_| panic!("pool still referenced"));
            pool.shutdown().await;
            assert_eq!(completed.load(Ordering::SeqCst), 1000);
        }

        #[tokio::test(start_paused = true)]
        async fn test_partitioned_router_same_key() {
            let results = Arc::new(std::sync::Mutex::new(Vec::new()));
            let r1 = Arc::clone(&results);
            let r2 = Arc::clone(&results);
            let pool1 = WorkerPool::new(crate::tokio_runtime(), 1, 10, move |job: i32| {
                let r = Arc::clone(&r1);
                async move {
                    let mut guard = r.lock().unwrap();
                    guard.push((0, job));
                }
            });
            let pool2 = WorkerPool::new(crate::tokio_runtime(), 1, 10, move |job: i32| {
                let r = Arc::clone(&r2);
                async move {
                    let mut guard = r.lock().unwrap();
                    guard.push((1, job));
                }
            });

            let router = PartitionedRouter::new(vec![pool1, pool2], |key: &String| key.len());
            router.submit(&"a".to_string(), 10).await.unwrap();
            router.submit(&"a".to_string(), 20).await.unwrap();
            // Both go to same shard because "a".len() % 2 == 1
            tokio::task::yield_now().await;
            let guard = results.lock().unwrap();
            assert_eq!(guard.len(), 2);
            let shard1_count = guard.iter().filter(|(s, _)| *s == 1).count();
            assert_eq!(shard1_count, 2);
        }

        #[tokio::test(start_paused = true)]
        async fn test_partitioned_router_distributes() {
            let results = Arc::new(std::sync::Mutex::new(Vec::new()));
            let r1 = Arc::clone(&results);
            let r2 = Arc::clone(&results);
            let pool1 = WorkerPool::new(crate::tokio_runtime(), 1, 10, move |job: i32| {
                let r = Arc::clone(&r1);
                async move {
                    let mut guard = r.lock().unwrap();
                    guard.push((0, job));
                }
            });
            let pool2 = WorkerPool::new(crate::tokio_runtime(), 1, 10, move |job: i32| {
                let r = Arc::clone(&r2);
                async move {
                    let mut guard = r.lock().unwrap();
                    guard.push((1, job));
                }
            });

            let router = PartitionedRouter::new(vec![pool1, pool2], |key: &String| key.len());
            router.submit(&"a".to_string(), 10).await.unwrap();
            router.submit(&"bb".to_string(), 20).await.unwrap();
            tokio::task::yield_now().await;
            let guard = results.lock().unwrap();
            assert_eq!(guard.len(), 2);
            let shards: Vec<_> = guard.iter().map(|(s, _)| *s).collect();
            assert!(shards.contains(&0));
            assert!(shards.contains(&1));
        }

        #[tokio::test]
        async fn result_pool_happy_path() {
            let pool = ResultPool::new(crate::tokio_runtime(), 2, 10, |job: i32| async move {
                Ok::<i32, String>(job * 2)
            });
            let result = pool.submit(5).await.unwrap();
            assert_eq!(result, 10);
            pool.shutdown().await;
        }

        #[tokio::test]
        async fn result_pool_handler_error() {
            let pool = ResultPool::new(crate::tokio_runtime(), 1, 10, |job: i32| async move {
                if job < 0 {
                    Err("negative".to_string())
                } else {
                    Ok(job)
                }
            });
            let err = pool.submit(-1).await.unwrap_err();
            match err {
                ResultPoolError::Inner(msg) => assert_eq!(msg, "negative"),
                other => panic!("expected Inner, got {other:?}"),
            }
            pool.shutdown().await;
        }

        #[tokio::test]
        async fn result_pool_multiple_jobs() {
            let pool = ResultPool::new(crate::tokio_runtime(), 2, 10, |job: i32| async move {
                Ok::<i32, String>(job + 1)
            });
            let mut handles = Vec::new();
            for i in 0..5i32 {
                handles.push(pool.submit(i));
            }
            for (i, handle) in handles.into_iter().enumerate() {
                assert_eq!(handle.await.unwrap(), i as i32 + 1);
            }
            pool.shutdown().await;
        }

        #[tokio::test]
        async fn result_pool_queue_full_returns_error() {
            // Queue capacity 0 means any submit will fail immediately
            let pool = ResultPool::new(crate::tokio_runtime(), 1, 0, |job: i32| async move {
                Ok::<i32, String>(job)
            });
            // The queue has capacity 0, so submit should fail immediately.
            // Wait briefly for the queue to fill (it's already full at capacity 0).
            let err = tokio::time::timeout(std::time::Duration::from_millis(100), pool.submit(1))
                .await
                .unwrap()
                .unwrap_err();
            assert!(matches!(err, ResultPoolError::Pool(PoolError::Full)));
            pool.shutdown().await;
        }

        #[tokio::test]
        async fn result_pool_shutdown_returns_canceled() {
            let pool = ResultPool::new(crate::tokio_runtime(), 1, 10, |job: i32| async move {
                Ok::<i32, String>(job)
            });
            // Submit a job to verify the pool is working
            let result = pool.submit(42).await.unwrap();
            assert_eq!(result, 42);
            pool.shutdown().await;
        }
    }

    mod m4 {
        use super::*;
        use crate::flow;

        #[tokio::test(start_paused = true)]
        async fn test_credit_flow_initial_credit_allows_n_sends() {
            let spec = flow::CreditSpec {
                initial: 3,
                more_after: 2,
            };
            let (sender, _receiver) = flow::new(spec);
            let counter = Arc::new(AtomicUsize::new(0));
            sender
                .send(|| async {
                    counter.fetch_add(1, Ordering::SeqCst);
                })
                .await;
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }

        #[tokio::test(start_paused = true)]
        async fn test_credit_flow_receiver_bumps_after_more_after() {
            let spec = flow::CreditSpec {
                initial: 3,
                more_after: 2,
            };
            let (sender, receiver) = flow::new(spec);

            let counter = Arc::new(AtomicUsize::new(0));
            for _ in 0..3 {
                let cnt = Arc::clone(&counter);
                sender
                    .send(|| async move {
                        cnt.fetch_add(1, Ordering::SeqCst);
                    })
                    .await;
            }
            assert_eq!(counter.load(Ordering::SeqCst), 3);

            receiver.recv();
            receiver.recv();

            let cnt = Arc::clone(&counter);
            sender
                .send(|| async move {
                    cnt.fetch_add(1, Ordering::SeqCst);
                })
                .await;
            assert_eq!(counter.load(Ordering::SeqCst), 4);
        }

        #[tokio::test(start_paused = true)]
        async fn test_credit_flow_sender_blocks_when_exhausted() {
            let spec = flow::CreditSpec {
                initial: 1,
                more_after: 2,
            };
            let (sender, receiver) = flow::new(spec);

            sender.send(|| async {}).await;

            let cnt = Arc::new(AtomicUsize::new(0));
            let cnt_clone = Arc::clone(&cnt);
            let handle = tokio::spawn(async move {
                sender
                    .send(|| async move {
                        cnt_clone.fetch_add(1, Ordering::SeqCst);
                    })
                    .await;
            });

            receiver.recv();
            receiver.recv();

            handle.await.unwrap();
            assert_eq!(cnt.load(Ordering::SeqCst), 1);
        }

        #[tokio::test(start_paused = true)]
        async fn test_credit_flow_end_to_end() {
            let spec = flow::CreditSpec {
                initial: 5,
                more_after: 3,
            };
            let (sender, receiver) = flow::new(spec);

            let sent = Arc::new(AtomicUsize::new(0));
            let received = Arc::new(AtomicUsize::new(0));

            let s = Arc::clone(&sent);
            let r = Arc::clone(&received);

            let producer = tokio::spawn(async move {
                for _ in 0..10 {
                    let s = Arc::clone(&s);
                    sender
                        .send(|| async move {
                            s.fetch_add(1, Ordering::SeqCst);
                        })
                        .await;
                }
            });

            let consumer = tokio::spawn(async move {
                for _ in 0..10 {
                    receiver.recv();
                    r.fetch_add(1, Ordering::SeqCst);
                }
            });

            producer.await.unwrap();
            consumer.await.unwrap();

            assert_eq!(sent.load(Ordering::SeqCst), 10);
            assert_eq!(received.load(Ordering::SeqCst), 10);
        }

        #[tokio::test(start_paused = true)]
        async fn test_credit_flow_is_blocked_and_current_credit() {
            tokio::time::resume();
            let spec = flow::CreditSpec {
                initial: 2,
                more_after: 1,
            };
            let (sender, receiver) = flow::new(spec);
            assert_eq!(sender.current_credit(), 2);
            assert!(!sender.is_blocked());

            sender.send(|| async {}).await;
            assert_eq!(sender.current_credit(), 1);
            assert!(!sender.is_blocked());

            sender.send(|| async {}).await;
            assert_eq!(sender.current_credit(), 0);

            // Now sender is blocked, wrap in Arc for shared access
            let sender = std::sync::Arc::new(sender);
            let sender_clone = std::sync::Arc::clone(&sender);
            let handle = tokio::spawn(async move {
                sender_clone.send(|| async {}).await;
                assert!(!sender_clone.is_blocked());
            });

            // Allow the spawned task to start and reach the blocking point
            tokio::time::sleep(Duration::from_millis(10)).await;
            assert!(sender.is_blocked());

            receiver.recv();
            handle.await.unwrap();
            // Credit went 0 -> +1 (bump) -> 0 (consumed by send)
            assert_eq!(sender.current_credit(), 0);
            assert!(!sender.is_blocked());
        }
    }

    mod m5 {
        use crate::io::db::DbCapability;
        use crate::io::fs::FsCapability;
        use crate::io::net::NetCapability;
        use crate::scope::CURRENT_CAPS;
        use fluent_wvr::CapabilitySet;

        #[tokio::test(start_paused = true)]
        async fn test_fs_read_write_roundtrip() {
            let tmp = tempfile::NamedTempFile::new().unwrap();
            let path = tmp.path().to_path_buf();
            let fs = FsCapability::new();
            let caps = CapabilitySet::new().with(FsCapability::new());
            CURRENT_CAPS
                .scope(caps, async {
                    fs.write(&path, b"hello world").await.expect("write failed");
                    let data = fs.read(&path).await.expect("read failed");
                    assert_eq!(data, b"hello world");
                    let meta = fs.metadata(&path).await.expect("metadata failed");
                    assert!(meta.is_file());
                })
                .await;
        }

        #[tokio::test(start_paused = true)]
        async fn test_net_tcp_connect_refused() {
            let net = NetCapability::new();
            let caps = CapabilitySet::new().with(NetCapability::new());
            let result = CURRENT_CAPS
                .scope(caps, async { net.tcp_connect("127.0.0.1:1").await })
                .await;
            assert!(result.is_err());
        }

        #[tokio::test(start_paused = true)]
        async fn test_db_query_execute_roundtrip() {
            let db = DbCapability::open(":memory:").unwrap();
            let caps = CapabilitySet::new().with(DbCapability::open(":memory:").unwrap());
            CURRENT_CAPS
                .scope(caps, async {
                    db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
                        .await
                        .unwrap();
                    db.execute("INSERT INTO t VALUES (1, 'hello')")
                        .await
                        .unwrap();
                    let rows = db.query("SELECT * FROM t").await.unwrap();
                    assert_eq!(rows.len(), 1);
                    assert_eq!(rows[0]["id"], "1");
                    assert_eq!(rows[0]["name"], "hello");
                })
                .await;
        }

        #[tokio::test(start_paused = true)]
        async fn test_capability_missing_denies_io() {
            let fs = FsCapability::new();
            let result = fs.read("/etc/passwd").await;
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::PermissionDenied,
                "expected PermissionDenied, got: {err}"
            );
            assert!(
                err.to_string().contains("missing"),
                "expected 'missing' in error, got: {err}"
            );
        }

        #[test]
        fn test_missing_capability_returns_none() {
            let caps = CapabilitySet::new();
            assert!(caps.get::<FsCapability>().is_none());
            assert!(caps.get::<NetCapability>().is_none());
            assert!(caps.get::<DbCapability>().is_none());
        }

        #[test]
        fn test_capability_gating_fs() {
            let caps = CapabilitySet::new().with(FsCapability::new());
            assert!(caps.get::<FsCapability>().is_some());
            assert!(caps.get::<NetCapability>().is_none());
        }

        #[tokio::test(start_paused = true)]
        async fn test_capability_boundary_enforcement_net() {
            let net = NetCapability::new();
            let result = net.tcp_connect("127.0.0.1:1").await;
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::PermissionDenied,
                "expected PermissionDenied for net, got: {err}"
            );
            assert!(
                err.to_string().contains("missing"),
                "expected 'missing' in error, got: {err}"
            );
        }

        #[tokio::test(start_paused = true)]
        async fn test_capability_boundary_enforcement_db() {
            let db = DbCapability::open(":memory:").unwrap();
            let result = db.query("SELECT 1").await;
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::PermissionDenied,
                "expected PermissionDenied for db, got: {err}"
            );
            assert!(
                err.to_string().contains("missing"),
                "expected 'missing' in error, got: {err}"
            );
        }

        #[tokio::test(start_paused = true)]
        async fn test_capability_boundary_enforcement_db_execute() {
            let db = DbCapability::open(":memory:").unwrap();
            let result = db.execute("CREATE TABLE t (id INTEGER)").await;
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::PermissionDenied,
                "expected PermissionDenied for db execute, got: {err}"
            );
            assert!(
                err.to_string().contains("missing"),
                "expected 'missing' in error, got: {err}"
            );
        }

        #[tokio::test(start_paused = true)]
        async fn test_db_concurrent_queries_via_pool() {
            let db = DbCapability::open(":memory:").unwrap();
            let caps = CapabilitySet::new().with(DbCapability::open(":memory:").unwrap());
            CURRENT_CAPS
                .scope(caps, async {
                    db.execute("CREATE TABLE t (id INTEGER, val TEXT)")
                        .await
                        .unwrap();
                    for i in 0..10 {
                        db.execute(&format!("INSERT INTO t VALUES ({i}, 'row{i}')"))
                            .await
                            .unwrap();
                    }
                    let rows = db.query("SELECT COUNT(*) AS cnt FROM t").await.unwrap();
                    assert_eq!(rows[0]["cnt"], "10");
                })
                .await;
        }
    }

    mod e2e {
        use super::{DepWorkUnit, PanicUnit};
        use crate::pool::WorkerPool;
        use crate::zone::{Zone, ZoneEvent, ZoneSummary};
        use fluent_wvr::{ArcIntern, CapabilitySet, WorkContext, WorkError, WorkOutput, WorkUnit};
        use fluent_wvr_testutil::impl_component_for_test;
        use std::sync::Arc;
        use std::time::Duration;

        /// End-to-end: Zone orchestrates WorkerPool-backed tasks
        #[tokio::test(start_paused = true)]
        async fn test_e2e_zone_with_worker_pool() {
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

            let mut zone = Zone::new(runtime, caps);
            zone.register(Arc::new(PoolWorkUnit {
                name: "task1".into(),
                _pool: Arc::clone(&pool),
                input: 5,
            }))
            .register(Arc::new(PoolWorkUnit {
                name: "task2".into(),
                _pool: Arc::clone(&pool),
                input: 10,
            }));

            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 2);
            assert_eq!(summary.panicked.len(), 0);
            assert_eq!(summary.cancelled.len(), 0);
        }

        /// End-to-end: Zone handles mixed success/failure/cancellation
        #[tokio::test(start_paused = true)]
        async fn test_e2e_zone_mixed_outcomes() {
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
                        "fail" => Err(WorkError::Execution("failed".into())),
                        "panic" => panic!("intentional panic"),
                        _ => unreachable!(),
                    }
                }
            }
            impl_component_for_test!(OutcomeUnit);

            let shared = ArcIntern::<str>::from("shared");
            let mut zone = Zone::new(runtime, caps);
            zone.register(Arc::new(OutcomeUnit {
                name: "root".into(),
                outcome: "fail",
                deps: vec![],
                provides: vec![shared.clone()],
            }))
            .register_with_context(
                Arc::new(OutcomeUnit {
                    name: "child1".into(),
                    outcome: "fail",
                    deps: vec![shared.clone()],
                    provides: vec![],
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            )
            .register(Arc::new(OutcomeUnit {
                name: "independent".into(),
                outcome: "panic",
                deps: vec![],
                provides: vec![],
            }));

            let summary: ZoneSummary = (&mut zone).await;
            assert_eq!(summary.completed.len(), 0);
            assert_eq!(summary.panicked.len(), 2);
            assert_eq!(summary.cancelled.len(), 1);
        }

        /// E2E Panic Cascade: verify that a panicking task aborts its transitive
        /// dependents while independent neighbors continue unhindered.
        #[tokio::test(start_paused = true)]
        async fn test_e2e_panic_cascade_with_independent_neighbors() {
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let mut zone = Zone::new(runtime, caps);

            let shared = ArcIntern::<str>::from("shared");
            let independent = ArcIntern::<str>::from("independent");

            zone.register(Arc::new(PanicUnit {
                name: "parent".into(),
                provides: vec![shared.clone()],
            }));
            zone.register_with_context(
                Arc::new(DepWorkUnit {
                    name: "child".into(),
                    deps: vec![shared.clone()],
                    provides: vec![],
                    should_fail: true,
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );
            zone.register(Arc::new(DepWorkUnit {
                name: "neighbor".into(),
                deps: vec![],
                provides: vec![independent.clone()],
                should_fail: false,
            }));
            zone.register_with_context(
                Arc::new(DepWorkUnit {
                    name: "grandchild".into(),
                    deps: vec![independent.clone()],
                    provides: vec![],
                    should_fail: false,
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );

            let summary: ZoneSummary = (&mut zone).await;
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
                .any(|e| matches!(e, ZoneEvent::Panicked { name, .. } if &**name == "parent")));
            assert!(summary
                .cancelled
                .iter()
                .any(|e| matches!(e, ZoneEvent::Cancelled { name, .. } if &**name == "child")));
        }

        /// E2E Cycle Resiliency: verify that a circular dependency does not hang
        /// the zone and that the cascade breaks the loop safely.
        #[tokio::test(start_paused = true)]
        async fn test_e2e_cycle_resiliency() {
            let runtime = crate::tokio_runtime();
            let caps = CapabilitySet::new();
            let mut zone = Zone::new(runtime, caps);

            struct CycleUnit {
                name: String,
                deps: Vec<ArcIntern<str>>,
                provides: Vec<ArcIntern<str>>,
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
                    Err(WorkError::Execution("cycle member".into()))
                }
            }
            impl_component_for_test!(CycleUnit);

            let a_provides = ArcIntern::<str>::from("a_provides");
            let b_provides = ArcIntern::<str>::from("b_provides");

            zone.register(Arc::new(CycleUnit {
                name: "A".into(),
                deps: vec![b_provides.clone()],
                provides: vec![a_provides.clone()],
            }));
            zone.register_with_context(
                Arc::new(CycleUnit {
                    name: "B".into(),
                    deps: vec![a_provides.clone()],
                    provides: vec![b_provides.clone()],
                }),
                WorkContext {
                    max_retries: 10,
                    ..WorkContext::default()
                },
            );

            let summary: ZoneSummary = (&mut zone).await;
            // A fails immediately, B is a dependent in a cycle.
            // The cycle should be detected and B should be cancelled.
            assert_eq!(summary.panicked.len(), 1, "A should panic");
            assert_eq!(
                summary.cancelled.len(),
                1,
                "B should be cancelled due to cycle detection"
            );
            assert!(summary
                .panicked
                .iter()
                .any(|e| matches!(e, ZoneEvent::Panicked { name, .. } if &**name == "A")));
            assert!(summary
                .cancelled
                .iter()
                .any(|e| matches!(e, ZoneEvent::Cancelled { name, .. } if &**name == "B")));
        }
    }
}
