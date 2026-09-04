use common_core::watchdog::*;


#[test]
fn test_budget_watchdog_fires() {
        let dog = BudgetWatchdog::new(3);
        assert!(dog.check().is_none());
        assert!(dog.check().is_none());
        assert!(dog.check().is_none());
        let event = dog.check().expect("should fire on 4th item");
        match event {
            WatchdogEvent::BudgetExceeded { limit, actual } => {
                assert_eq!(limit, 3);
                assert_eq!(actual, 4);
            }
            _ => panic!("expected BudgetExceeded"),
        }
}

#[test]
fn test_budget_watchdog_never_fires_within_limit() {
        let dog = BudgetWatchdog::new(100);
        for _ in 0..100 {
            assert!(dog.check().is_none());
        }
}

#[test]
fn test_budget_watchdog_reset() {
        let dog = BudgetWatchdog::new(2);
        assert!(dog.check().is_none());
        assert!(dog.check().is_none());
        dog.reset();
        assert!(dog.check().is_none());
}

#[test]
fn test_wall_clock_watchdog_fires_after_deadline() {
        let dog = WallClockWatchdog::new(0);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let event = dog.check().expect("should fire");
        match event {
            WatchdogEvent::WallClock { .. } => {} // ok
            _ => panic!("expected WallClock"),
        }
}

#[test]
fn test_wall_clock_watchdog_not_fired_within_deadline() {
        let dog = WallClockWatchdog::new(3600);
        assert!(dog.check().is_none());
}

#[test]
fn test_repetition_watchdog_fires() {
        let dog = RepetitionWatchdog::new(4, 10);
        for _ in 0..3 {
            assert!(dog.check("hello").is_none());
        }
        let event = dog.check("hello").expect("should fire on 4th repeat");
        match event {
            WatchdogEvent::Repetition {
                threshold,
                consecutive,
            } => {
                assert_eq!(threshold, 4);
                assert_eq!(consecutive, 4);
            }
            _ => panic!("expected Repetition"),
        }
}

#[test]
fn test_repetition_watchdog_not_fires_with_different_items() {
        let dog = RepetitionWatchdog::new(4, 10);
        let items = ["a", "b", "c", "d", "e"];
        for t in &items {
            assert!(dog.check(t).is_none());
        }
}

#[test]
fn test_repetition_watchdog_reset() {
        let dog = RepetitionWatchdog::new(3, 10);
        assert!(dog.check("x").is_none());
        assert!(dog.check("x").is_none());
        dog.reset();
        assert!(dog.check("x").is_none());
}

#[test]
fn test_watchdog_set_composition() {
        let set = WatchdogSet::new(100, 3600, 3, 10);
        assert!(set.check(Some("hello")).is_none());
        assert!(set.check(Some("hello")).is_none());
        let event = set.check(Some("hello")).expect("repetition should fire");
        match event {
            WatchdogEvent::Repetition { .. } => {}
            _ => panic!("expected Repetition"),
        }
}

#[test]
fn test_watchdog_set_budget_triggers() {
        let set = WatchdogSet::new(1, 3600, 10, 10);
        assert!(set.check(None).is_none());
        let event = set.check(Some("hello")).expect("budget should fire");
        match event {
            WatchdogEvent::BudgetExceeded { .. } => {}
            _ => panic!("expected BudgetExceeded"),
        }
}

#[test]
fn test_log_event_does_not_panic() {
        let event = WatchdogEvent::BudgetExceeded {
            limit: 100,
            actual: 101,
        };
        WatchdogSet::log_event(&event);
        let event = WatchdogEvent::WallClock {
            deadline_secs: 300,
            elapsed_secs: 301,
        };
        WatchdogSet::log_event(&event);
        let event = WatchdogEvent::Repetition {
            threshold: 10,
            consecutive: 10,
        };
        WatchdogSet::log_event(&event);
}

#[test]
fn test_watchdog_event_type_from_event() {
        assert_eq!(
            WatchdogEventType::from(&WatchdogEvent::BudgetExceeded {
                limit: 1,
                actual: 2
            }),
            WatchdogEventType::BudgetExceeded
        );
        assert_eq!(
            WatchdogEventType::from(&WatchdogEvent::WallClock {
                deadline_secs: 1,
                elapsed_secs: 2
            }),
            WatchdogEventType::WallClock
        );
        assert_eq!(
            WatchdogEventType::from(&WatchdogEvent::Repetition {
                threshold: 1,
                consecutive: 2
            }),
            WatchdogEventType::Repetition
        );
}
