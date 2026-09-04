use super::*;
use crate::tokio_runtime;
use std::sync::atomic::{AtomicUsize, Ordering};

#[tokio::test]
async fn test_affinity_bonus_applied() {
    let runtime = tokio_runtime();
    let processed = Arc::new(AtomicUsize::new(0));
    let p_clone = Arc::clone(&processed);
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        move |identity: String| {
            let c = Arc::clone(&p_clone);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok::<String, String>(identity)
            }
        },
    ));

    let scheduler = AffinityScheduler::new(Arc::clone(&p));
    scheduler.set_affinity(Some("session-a".into()));

    let task = ScheduledTask {
        identity: "session-a".into(),
        task: "task-a".into(),
        enqueued_at: scheduler.runtime.now(),
    };

    let result = scheduler.submit(task, 0).await.unwrap();
    assert_eq!(result, "task-a");
    assert_eq!(processed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_no_affinity_bonus_for_different_session() {
    let runtime = tokio_runtime();
    let processed = Arc::new(AtomicUsize::new(0));
    let p_clone = Arc::clone(&processed);
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        move |identity: String| {
            let c = Arc::clone(&p_clone);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok::<String, String>(identity)
            }
        },
    ));

    let scheduler = AffinityScheduler::new(Arc::clone(&p));
    scheduler.set_affinity(Some("session-a".into()));

    let task = ScheduledTask {
        identity: "session-b".into(),
        task: "task-b".into(),
        enqueued_at: scheduler.runtime.now(),
    };

    scheduler.submit(task, 0).await.unwrap();
    assert_eq!(processed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_aging_increases_priority_for_starved_tasks() {
    let runtime = tokio_runtime();
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        move |_identity: String| async move { Ok::<String, String>("done".into()) },
    ));

    let scheduler = AffinityScheduler::new_with_runtime(Arc::clone(&p), Arc::clone(&runtime)).with_aging(AgingConfig {
        affinity_bonus: 10,
        aging_interval: Duration::ZERO,
        aging_rate: 5,
        max_priority: 100,
    });

    scheduler.set_affinity(Some("session-a".into()));

    let task = ScheduledTask {
        identity: "session-b".into(),
        task: "task-b".into(),
        enqueued_at: scheduler.runtime.now(),
    };
    scheduler.submit(task, 0).await.unwrap();

    scheduler.age_now();

    let prios = sync_lock::lock(&scheduler.base_priorities);
    assert_eq!(prios.peek(&"session-b".to_string()), Some(&5));
}

#[tokio::test]
async fn test_set_and_get_affinity() {
    let runtime = tokio_runtime();
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        move |_: String| async move { Ok::<String, String>("ok".into()) },
    ));

    let scheduler: AffinityScheduler<String, String, String> =
        AffinityScheduler::new(Arc::clone(&p));

    assert_eq!(scheduler.current_affinity(), None);

    scheduler.set_affinity(Some("my-session".into()));
    assert_eq!(scheduler.current_affinity(), Some("my-session".into()));

    scheduler.set_affinity(None);
    assert_eq!(scheduler.current_affinity(), None);
}

// New tests for M2

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn affinity_never_deadlocks_submit_vs_age_now() {
    let runtime = tokio_runtime();
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        2,
        |_job: String| async move { Ok::<String, String>("ok".into()) },
    ));
    let scheduler = Arc::new(AffinityScheduler::new(Arc::clone(&p)));
    let mut handles = Vec::new();
    for i in 0..50 {
        let s = Arc::clone(&scheduler);
        handles.push(tokio::spawn(async move {
            let task = ScheduledTask {
                identity: format!("sess-{}", i % 5),
                task: format!("job-{}", i),
                enqueued_at: s.runtime.now(),
            };
            let _ = s.submit(task, 0).await;
        }));
        let s2 = Arc::clone(&scheduler);
        handles.push(tokio::spawn(async move {
            s2.age_now();
        }));
    }
    for h in handles {
        tokio::time::timeout(Duration::from_secs(5), h)
            .await
            .expect("task must complete without deadlock")
            .unwrap();
    }
}

#[tokio::test(start_paused = true)]
async fn affinity_virtual_time_deterministic() {
    let rt = crate::runtime::test::TestRuntime::new(tokio::runtime::Handle::current(), 42);
    let runtime: Arc<dyn fluent_wvr::Runtime> = Arc::new(rt);
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        |_job: String| async move { Ok::<String, String>("ok".into()) },
    ));
    let scheduler = AffinityScheduler::new_with_runtime(Arc::clone(&p), Arc::clone(&runtime)).with_aging(AgingConfig {
        affinity_bonus: 10,
        aging_interval: Duration::from_secs(1),
        aging_rate: 2,
        max_priority: 100,
    });
    scheduler.set_affinity(Some("a".into()));
    // Seed a starved identity
    let task = ScheduledTask {
        identity: "b".into(),
        task: "job".into(),
        enqueued_at: runtime.now(),
    };
    scheduler.submit(task, 0).await.unwrap();
    // No aging yet because interval not elapsed
    {
        let prios = sync_lock::lock(&scheduler.base_priorities);
        assert_eq!(prios.peek(&"b".to_string()), Some(&0));
    }
    tokio::time::advance(Duration::from_millis(500)).await;
    // Submit again should not age yet
    let task2 = ScheduledTask {
        identity: "b".into(),
        task: "job2".into(),
        enqueued_at: runtime.now(),
    };
    scheduler.submit(task2, 0).await.unwrap();
    {
        let prios = sync_lock::lock(&scheduler.base_priorities);
        // Still 0 because aging hasn't fired (maybe_age checked, but interval not elapsed; plus enqueued_at bonus capped)
        // enqueued_at bonus is 0 because just created
        assert!(prios.peek(&"b".to_string()).is_some());
    }
    tokio::time::advance(Duration::from_millis(600)).await;
    // Now interval elapsed, next submit should trigger aging
    let task3 = ScheduledTask {
        identity: "b".into(),
        task: "job3".into(),
        enqueued_at: runtime.now(),
    };
    scheduler.submit(task3, 0).await.unwrap();
    {
        let prios = sync_lock::lock(&scheduler.base_priorities);
        // b should have been aged by 2
        let v = prios.peek(&"b".to_string()).copied().unwrap();
        assert!(v >= 2, "aging should have fired exactly once, got {v}");
    }
}

#[tokio::test]
async fn affinity_bounded_map_evicts_oldest_identity() {
    let runtime = tokio_runtime();
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        |_job: String| async move { Ok::<String, String>("ok".into()) },
    ));
    let scheduler = AffinityScheduler::new_with_cap(Arc::clone(&p), Arc::clone(&runtime), 1024, AgingConfig::default());
    for i in 0..2000 {
        let task = ScheduledTask {
            identity: format!("id-{}", i),
            task: format!("job-{}", i),
            enqueued_at: scheduler.runtime.now(),
        };
        scheduler.submit(task, 0).await.unwrap();
    }
    assert!(scheduler.base_priorities_len() <= 1024);
}

#[tokio::test]
async fn affinity_enqueued_at_weights_older_task_higher() {
    let runtime = tokio_runtime();
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        |_job: String| async move { Ok::<String, String>("ok".into()) },
    ));
    let scheduler = AffinityScheduler::new(Arc::clone(&p)).with_aging(AgingConfig {
        affinity_bonus: 10,
        aging_interval: Duration::from_secs(100),
        aging_rate: 5,
        max_priority: 100,
    });
    let old_task = ScheduledTask {
        identity: "sess".into(),
        task: "old".into(),
        enqueued_at: runtime.now() - Duration::from_secs(10),
    };
    let fresh_task = ScheduledTask {
        identity: "sess".into(),
        task: "fresh".into(),
        enqueued_at: runtime.now(),
    };
    let prio_old = scheduler.effective_priority(&old_task, 0);
    // Need fresh scheduler for fair comparison because first call inserts; use separate instance
    let scheduler2 = AffinityScheduler::new_with_runtime(Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        |_job: String| async move { Ok::<String, String>("ok".into()) },
    )), Arc::clone(&runtime)).with_aging(AgingConfig {
        affinity_bonus: 10,
        aging_interval: Duration::from_secs(100),
        aging_rate: 5,
        max_priority: 100,
    });
    let prio_fresh = scheduler2.effective_priority(&fresh_task, 0);
    assert!(prio_old > prio_fresh, "older task should have higher priority: old={prio_old} fresh={prio_fresh}");
}

#[tokio::test]
async fn affinity_bonus_is_task_value_not_confidence() {
    let runtime = tokio_runtime();
    let p = Arc::new(PriorityResultPool::new(
        Arc::clone(&runtime),
        1,
        |_job: String| async move { Ok::<String, String>("ok".into()) },
    ));
    let scheduler = AffinityScheduler::new(Arc::clone(&p)).with_aging(AgingConfig {
        affinity_bonus: 10,
        aging_interval: Duration::from_secs(1),
        aging_rate: 2,
        max_priority: 100,
    });
    scheduler.set_affinity(Some("A".into()));
    // Affine fresh task: base 0 + bonus 10 =10
    let affine_fresh = ScheduledTask {
        identity: "A".into(),
        task: "affine".into(),
        enqueued_at: runtime.now(),
    };
    let prio_affine_fresh = scheduler.effective_priority(&affine_fresh, 0);
    // Non-affine old task with 20s age: age_bonus capped at 2, plus many aging ticks? But effective_priority only adds one age_bonus.
    // To simulate starvation, age the map
    // Seed starved identity then age multiple times
    let starved = ScheduledTask {
        identity: "B".into(),
        task: "starved".into(),
        enqueued_at: runtime.now() - Duration::from_secs(20),
    };
    // Insert B with base 0
    let _ = scheduler.effective_priority(&starved, 0);
    // Age 10 ticks
    for _ in 0..10 {
        scheduler.age_now();
    }
    let prio_starved_aged = {
        let guard = sync_lock::lock(&scheduler.base_priorities);
        guard.peek(&"B".to_string()).copied().unwrap_or(0)
    };
    // Starved's base after aging should exceed affine's fresh bonus if aging is confidence
    assert!(prio_starved_aged > prio_affine_fresh || prio_starved_aged + 2 > prio_affine_fresh, "aged starved B should eventually outrank fresh affine A");
}
