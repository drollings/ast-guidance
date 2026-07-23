use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Three distinct, independently configurable watchdogs.
///
/// # Design
/// Each watchdog is its own struct with its own state. `WatchdogSet` composes
/// them and provides a unified `check()` method. This is explicitly NOT a
/// single generic timeout — each watchdog tracks a different signal and has
/// independent configuration.
pub struct WatchdogSet {
    pub max_tokens: MaxTokenWatchdog,
    pub wall_clock: WallClockWatchdog,
    pub repetition: RepetitionWatchdog,
}

impl WatchdogSet {
    /// Create a new watchdog set with the given configurations.
    pub fn new(
        max_token_limit: u32,
        wall_clock_secs: u64,
        repeat_threshold: usize,
        repeat_window: usize,
    ) -> Self {
        Self {
            max_tokens: MaxTokenWatchdog::new(max_token_limit),
            wall_clock: WallClockWatchdog::new(wall_clock_secs),
            repetition: RepetitionWatchdog::new(repeat_threshold, repeat_window),
        }
    }

    /// Check all watchdogs. Returns the first watchdog that fired.
    pub fn check(&self, token: Option<&str>) -> Option<WatchdogEvent> {
        if let Some(event) = self.max_tokens.check() {
            return Some(event);
        }

        if let Some(event) = self.wall_clock.check() {
            return Some(event);
        }

        if let Some(token) = token {
            if let Some(event) = self.repetition.check(token) {
                return Some(event);
            }
        }

        None
    }

    /// Log which watchdog fired.
    pub fn log_event(event: &WatchdogEvent) {
        match event {
            WatchdogEvent::MaxTokens { limit, actual } => {
                tracing::warn!("watchdog: max tokens ({limit}) reached with {actual} tokens");
            }
            WatchdogEvent::WallClock {
                deadline_secs,
                elapsed_secs,
            } => {
                tracing::warn!(
                    "watchdog: wall-clock deadline ({deadline_secs}s) exceeded at {elapsed_secs}s"
                );
            }
            WatchdogEvent::Repetition {
                threshold,
                consecutive,
            } => {
                tracing::warn!(
                    "watchdog: repetition threshold ({threshold}) exceeded with {consecutive} consecutive identical tokens"
                );
            }
        }
    }
}

/// Token budget watchdog. Fires when the total token count exceeds the limit.
pub struct MaxTokenWatchdog {
    pub limit: u32,
    pub count: AtomicU32,
}

impl MaxTokenWatchdog {
    pub fn new(limit: u32) -> Self {
        Self {
            limit,
            count: AtomicU32::new(0),
        }
    }

    pub fn check(&self) -> Option<WatchdogEvent> {
        let current = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        if current > self.limit {
            Some(WatchdogEvent::MaxTokens {
                limit: self.limit,
                actual: current,
            })
        } else {
            None
        }
    }

    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
    }
}

/// Wall-clock deadline watchdog. Fires when the deadline is exceeded.
pub struct WallClockWatchdog {
    pub deadline_secs: u64,
    pub started_at: Instant,
}

impl WallClockWatchdog {
    pub fn new(deadline_secs: u64) -> Self {
        Self {
            deadline_secs,
            started_at: Instant::now(),
        }
    }

    pub fn check(&self) -> Option<WatchdogEvent> {
        let elapsed = self.started_at.elapsed();
        if elapsed >= Duration::from_secs(self.deadline_secs) {
            Some(WatchdogEvent::WallClock {
                deadline_secs: self.deadline_secs,
                elapsed_secs: elapsed.as_secs(),
            })
        } else {
            None
        }
    }

    pub fn reset(&mut self, deadline_secs: u64) {
        self.deadline_secs = deadline_secs;
        self.started_at = Instant::now();
    }
}

/// Repetition/loop detection watchdog. Fires when N consecutive identical
/// tokens are observed within a sliding window.
pub struct RepetitionWatchdog {
    pub threshold: usize,
    pub window: usize,
    pub buffer: Mutex<VecDeque<String>>,
}

impl RepetitionWatchdog {
    pub fn new(threshold: usize, window: usize) -> Self {
        Self {
            threshold,
            window,
            buffer: Mutex::new(VecDeque::with_capacity(window)),
        }
    }

    pub fn check(&self, token: &str) -> Option<WatchdogEvent> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.push_back(token.to_string());
        if buffer.len() > self.window {
            buffer.pop_front();
        }

        if buffer.len() >= self.threshold {
            let all_same = buffer.iter().rev().take(self.threshold).all(|t| t == token);
            if all_same {
                return Some(WatchdogEvent::Repetition {
                    threshold: self.threshold,
                    consecutive: self.threshold,
                });
            }
        }

        None
    }

    pub fn reset(&self) {
        self.buffer.lock().unwrap().clear();
    }
}

/// Events that can fire from any of the three watchdogs.
#[derive(Debug, Clone)]
pub enum WatchdogEvent {
    MaxTokens {
        limit: u32,
        actual: u32,
    },
    WallClock {
        deadline_secs: u64,
        elapsed_secs: u64,
    },
    Repetition {
        threshold: usize,
        consecutive: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_token_watchdog_fires() {
        let dog = MaxTokenWatchdog::new(3);
        assert!(dog.check().is_none());
        assert!(dog.check().is_none());
        assert!(dog.check().is_none());
        let event = dog.check().expect("should fire on 4th token");
        match event {
            WatchdogEvent::MaxTokens { limit, actual } => {
                assert_eq!(limit, 3);
                assert_eq!(actual, 4);
            }
            _ => panic!("expected MaxTokens"),
        }
    }

    #[test]
    fn test_max_token_watchdog_never_fires_within_limit() {
        let dog = MaxTokenWatchdog::new(100);
        for _ in 0..100 {
            assert!(dog.check().is_none());
        }
    }

    #[test]
    fn test_max_token_watchdog_reset() {
        let dog = MaxTokenWatchdog::new(2);
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
    fn test_repetition_watchdog_not_fires_with_different_tokens() {
        let dog = RepetitionWatchdog::new(4, 10);
        let tokens = ["a", "b", "c", "d", "e"];
        for t in &tokens {
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
        // Use a high max_tokens limit so repetition fires first
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
    fn test_watchdog_set_max_tokens_triggers() {
        let set = WatchdogSet::new(1, 3600, 10, 10);
        assert!(set.check(None).is_none());
        let event = set.check(Some("hello")).expect("max tokens should fire");
        match event {
            WatchdogEvent::MaxTokens { .. } => {}
            _ => panic!("expected MaxTokens"),
        }
    }

    #[test]
    fn test_log_event_does_not_panic() {
        let event = WatchdogEvent::MaxTokens {
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
}
