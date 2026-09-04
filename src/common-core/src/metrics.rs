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

