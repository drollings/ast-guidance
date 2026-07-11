//! Lock-free latency histogram with 12 fixed buckets.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub const BUCKET_MS: [u64; 11] = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000];
pub const BUCKET_COUNT: usize = 12;

/// Lock-free latency histogram with 12 fixed buckets.
///
/// Buckets are: `[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, +∞]` ms.
/// Thread-safe via `AtomicU64` — safe to share across tasks.
///
/// # Examples
///
/// ```
/// use std::time::Instant;
/// use common_core::metrics::LatencyHistogram;
///
/// let hist = LatencyHistogram::new();
/// let start = Instant::now();
/// // ... do work ...
/// hist.observe_duration(start);
///
/// assert_eq!(hist.count(), 1);
/// assert!(hist.sum_ms() < 1000);
/// ```
pub struct LatencyHistogram {
    buckets: [AtomicU64; BUCKET_COUNT],
    count: AtomicU64,
    sum: AtomicU64,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            buckets: Default::default(),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
        }
    }

    fn bucket_index(duration_ms: u64) -> usize {
        for (i, &bound) in BUCKET_MS.iter().enumerate() {
            if duration_ms <= bound {
                return i;
            }
        }
        BUCKET_COUNT - 1
    }

    pub fn observe(&self, duration_ms: u64) {
        let idx = Self::bucket_index(duration_ms);
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(duration_ms, Ordering::Relaxed);
    }

    pub fn observe_duration(&self, start: Instant) {
        let elapsed = start.elapsed();
        self.observe(elapsed.as_millis() as u64);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn sum_ms(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
    }

    pub fn bucket(&self, idx: usize) -> u64 {
        if idx < BUCKET_COUNT {
            self.buckets[idx].load(Ordering::Relaxed)
        } else {
            0
        }
    }

    pub fn estimate_percentile(&self, pct: f64) -> u64 {
        let total = self.count();
        if total == 0 {
            return 0;
        }
        let target = (total as f64 * pct / 100.0) as u64;
        let mut cumulative = 0u64;
        for (i, &bound) in BUCKET_MS.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            if cumulative >= target {
                return bound;
            }
        }
        *BUCKET_MS.last().unwrap_or(&5000)
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn observe_increments_count_and_sum() {
        let h = LatencyHistogram::new();
        h.observe(10);
        assert_eq!(h.count(), 1);
        assert_eq!(h.sum_ms(), 10);
    }

    #[test]
    fn observe_routes_to_correct_bucket() {
        let h = LatencyHistogram::new();
        h.observe(1);
        h.observe(10);
        h.observe(100);
        assert_eq!(h.bucket(0), 1); // 1ms ≤ 1 → bucket 0
        assert_eq!(h.bucket(2), 1); // 10ms ≤ 10 → bucket 2
        assert_eq!(h.bucket(5), 1); // 100ms ≤ 100 → bucket 5
    }

    #[test]
    fn large_value_goes_to_inf_bucket() {
        let h = LatencyHistogram::new();
        h.observe(99999);
        assert_eq!(h.bucket(BUCKET_COUNT - 1), 1);
    }

    #[test]
    fn estimate_percentile_returns_zero_when_empty() {
        let h = LatencyHistogram::new();
        assert_eq!(h.estimate_percentile(50.0), 0);
    }

    #[test]
    fn estimate_percentile_p50_with_known() {
        let h = LatencyHistogram::new();
        h.observe(1);
        h.observe(10);
        h.observe(100);
        let p50 = h.estimate_percentile(50.0);
        assert!(p50 <= 100);
    }

    #[test]
    fn cumulative_bucket_includes_earlier() {
        let h = LatencyHistogram::new();
        h.observe(5);
        h.observe(50);
        assert_eq!(h.bucket(0), 0); // No value ≤ 1ms
        assert_eq!(h.bucket(1), 1); // 5ms ≤ 5 → bucket 1
        assert_eq!(h.bucket(4), 1); // 50ms ≤ 50 → bucket 4
    }

    #[test]
    fn thread_safe_concurrent_observe() {
        let h = Arc::new(LatencyHistogram::new());
        let mut handles = Vec::new();
        for _ in 0..10 {
            let h_clone = Arc::clone(&h);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    h_clone.observe(5);
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(h.count(), 1000);
    }

    #[test]
    fn observe_duration_records_millis() {
        let h = LatencyHistogram::new();
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(1));
        h.observe_duration(start);
        assert!(h.count() >= 1);
        assert!(h.sum_ms() >= 1);
    }

    #[test]
    fn bucket_out_of_range_returns_zero() {
        let h = LatencyHistogram::new();
        assert_eq!(h.bucket(99), 0);
    }

    #[test]
    fn estimate_percentile_returns_max_when_target_in_last_bucket() {
        let h = LatencyHistogram::new();
        h.observe(99999);
        let pct = h.estimate_percentile(100.0);
        assert!(pct >= 5000);
    }

    #[test]
    fn default_creates_empty_histogram() {
        let h = LatencyHistogram::default();
        assert_eq!(h.count(), 0);
    }
}
