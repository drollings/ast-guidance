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

    /// Snapshot of the per-bucket counts, indexed by `BUCKET_MS`.
    pub fn bucket_counts(&self) -> [u64; BUCKET_COUNT] {
        let mut out = [0u64; BUCKET_COUNT];
        for (i, b) in out.iter_mut().enumerate() {
            *b = self.buckets[i].load(Ordering::Relaxed);
        }
        out
    }

    /// Weighted percentile across a set of histograms.
    ///
    /// Bucket arrays are summed across all histograms first, then the same
    /// cumulative walk as [`Self::estimate_percentile`] maps the target rank
    /// to a bucket bound. This is the canonical aggregate for p50/p99 across
    /// a multi-tier cascade (previously duplicated in coral's reactor).
    pub fn aggregate(histograms: &[&LatencyHistogram], pct: f64) -> u64 {
        let mut buckets = [0u64; BUCKET_COUNT];
        for h in histograms {
            let counts = h.bucket_counts();
            for (i, b) in buckets.iter_mut().enumerate() {
                *b += counts[i];
            }
        }
        Self::percentile_from_counts(&buckets, pct)
    }

    fn percentile_from_counts(buckets: &[u64; BUCKET_COUNT], pct: f64) -> u64 {
        let total: u64 = buckets.iter().sum();
        if total == 0 {
            return 0;
        }
        let target = (total as f64 * pct / 100.0) as u64;
        let mut cumulative = 0u64;
        for (i, &bound) in BUCKET_MS.iter().enumerate() {
            cumulative += buckets[i];
            if cumulative >= target {
                return bound;
            }
        }
        *BUCKET_MS.last().unwrap_or(&5000)
    }

    pub fn estimate_percentile(&self, pct: f64) -> u64 {
        Self::percentile_from_counts(&self.bucket_counts(), pct)
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

    #[test]
    fn bucket_counts_snapshot_matches_individual_bucket() {
        let h = LatencyHistogram::new();
        h.observe(5);
        h.observe(50);
        let counts = h.bucket_counts();
        assert_eq!(counts[1], 1); // 5ms ≤ 5 → bucket 1
        assert_eq!(counts[4], 1); // 50ms ≤ 50 → bucket 4
        assert_eq!(counts.iter().sum::<u64>(), h.count());
    }

    #[test]
    fn aggregate_sums_buckets_across_histograms() {
        let a = LatencyHistogram::new();
        let b = LatencyHistogram::new();
        a.observe(1);
        a.observe(100);
        b.observe(10);
        b.observe(100);
        // values: 1, 10, 100, 100 → p50 target rank 2 → 10ms bucket
        let p50 = LatencyHistogram::aggregate(&[&a, &b], 50.0);
        assert_eq!(p50, 10);
        // p99 target rank 3 → crosses the 100ms bucket
        let p99 = LatencyHistogram::aggregate(&[&a, &b], 99.0);
        assert_eq!(p99, 100);
    }

    #[test]
    fn aggregate_empty_returns_zero() {
        let a = LatencyHistogram::new();
        assert_eq!(LatencyHistogram::aggregate(&[&a], 50.0), 0);
        assert_eq!(LatencyHistogram::aggregate(&[], 50.0), 0);
    }

    #[test]
    fn aggregate_matches_estimate_percentile_for_single_histogram() {
        let h = LatencyHistogram::new();
        for ms in [1u64, 10, 100, 1000] {
            h.observe(ms);
        }
        assert_eq!(
            LatencyHistogram::aggregate(&[&h], 50.0),
            h.estimate_percentile(50.0)
        );
        assert_eq!(
            LatencyHistogram::aggregate(&[&h], 99.0),
            h.estimate_percentile(99.0)
        );
    }
}
