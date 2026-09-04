use common_core::metrics::*;
use std::time::Instant;


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
