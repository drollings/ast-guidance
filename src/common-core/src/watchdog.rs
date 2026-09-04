use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Three independent, composable guardrails for long-running operations.
///
/// Each watchdog tracks a distinct signal — item budget, wall-clock
/// deadline, or repetition — and fires when its threshold is exceeded.
/// `WatchdogSet` composes them with a unified `check()`.
pub struct WatchdogSet {
    pub budget: BudgetWatchdog,
    pub wall_clock: WallClockWatchdog,
    pub repetition: RepetitionWatchdog,
}

impl WatchdogSet {
    pub fn new(
        budget_limit: u32,
        wall_clock_secs: u64,
        repeat_threshold: usize,
        repeat_window: usize,
    ) -> Self {
        Self {
            budget: BudgetWatchdog::new(budget_limit),
            wall_clock: WallClockWatchdog::new(wall_clock_secs),
            repetition: RepetitionWatchdog::new(repeat_threshold, repeat_window),
        }
    }

    /// Check all watchdogs. Returns the first event that fired.
    pub fn check(&self, item: Option<&str>) -> Option<WatchdogEvent> {
        if let Some(event) = self.budget.check() {
            return Some(event);
        }

        if let Some(event) = self.wall_clock.check() {
            return Some(event);
        }

        if let Some(item) = item {
            if let Some(event) = self.repetition.check(item) {
                return Some(event);
            }
        }

        None
    }

    /// Log which watchdog fired via `tracing::warn!`.
    pub fn log_event(event: &WatchdogEvent) {
        match event {
            WatchdogEvent::BudgetExceeded { limit, actual } => {
                tracing::warn!("watchdog: budget ({limit}) exceeded with {actual} items");
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
                    "watchdog: repetition threshold ({threshold}) exceeded with {consecutive} consecutive identical items"
                );
            }
        }
    }
}

/// Item budget guardrail. Fires when the total item count exceeds the limit.
pub struct BudgetWatchdog {
    pub limit: u32,
    pub count: AtomicU32,
}

impl BudgetWatchdog {
    pub fn new(limit: u32) -> Self {
        Self {
            limit,
            count: AtomicU32::new(0),
        }
    }

    pub fn check(&self) -> Option<WatchdogEvent> {
        let current = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        if current > self.limit {
            Some(WatchdogEvent::BudgetExceeded {
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

/// Wall-clock deadline guardrail. Fires when the elapsed time exceeds the deadline.
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

/// Repetition / loop-detection guardrail. Fires when N consecutive identical
/// items are observed within a sliding window.
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

    pub fn check(&self, item: &str) -> Option<WatchdogEvent> {
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        buffer.push_back(item.to_string());
        if buffer.len() > self.window {
            buffer.pop_front();
        }

        if buffer.len() >= self.threshold {
            let all_same = buffer.iter().rev().take(self.threshold).all(|t| t == item);
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
        self.buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

/// Events emitted by the three watchdogs.
#[derive(Debug, Clone)]
pub enum WatchdogEvent {
    BudgetExceeded {
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

/// Discriminant keys for watchdog event counters in metrics aggregation.
///
/// Distinct from `WatchdogEvent` (which carries payload data); this enum
/// provides label-safe variant keys suitable for `HashMap` keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WatchdogEventType {
    BudgetExceeded,
    WallClock,
    Repetition,
}

impl From<&WatchdogEvent> for WatchdogEventType {
    fn from(event: &WatchdogEvent) -> Self {
        match event {
            WatchdogEvent::BudgetExceeded { .. } => WatchdogEventType::BudgetExceeded,
            WatchdogEvent::WallClock { .. } => WatchdogEventType::WallClock,
            WatchdogEvent::Repetition { .. } => WatchdogEventType::Repetition,
        }
    }
}

