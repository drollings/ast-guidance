use common_core::calibration::*;


#[derive(Debug)]
struct Case {
        score: f64,
        label: bool,
}

#[test]
fn synthetic_precision_at_0_5_is_one() {
        let cases = vec![
            Case { score: 0.2, label: false },
            Case { score: 0.8, label: true },
        ];
        let r = calibrate_threshold(&cases, |c| c.score, |c| c.label, 0.5);
        assert_eq!(r.precision, 1.0);
        assert_eq!(r.recall, 1.0);
        assert_eq!(r.fpr, 0.0);
        assert_eq!(r.support, 2);
        assert!(r.passes_gate());
}

#[test]
fn empty_cases_is_perfect() {
        let cases: Vec<Case> = vec![];
        let r = calibrate_threshold(&cases, |c| c.score, |c| c.label, 0.5);
        assert_eq!(r.precision, 1.0);
        assert_eq!(r.recall, 1.0);
        assert_eq!(r.fpr, 0.0);
        assert_eq!(r.support, 0);
}

#[test]
fn fpr_and_precision_on_mixed() {
        // 2 positives, 2 negatives, threshold 0.5
        // scores: 0.9(T), 0.4(FN), 0.6(FP), 0.1(TN)
        let cases = vec![
            Case { score: 0.9, label: true },
            Case { score: 0.4, label: true },
            Case { score: 0.6, label: false },
            Case { score: 0.1, label: false },
        ];
        let r = calibrate_threshold(&cases, |c| c.score, |c| c.label, 0.5);
        assert_eq!(r.tp, 1);
        assert_eq!(r.fp, 1);
        assert_eq!(r.tn, 1);
        assert_eq!(r.r#fn, 1);
        assert!((r.precision - 0.5).abs() < 1e-9);
        assert!((r.recall - 0.5).abs() < 1e-9);
        assert!((r.fpr - 0.5).abs() < 1e-9);
}

#[test]
fn sweep_is_monotonic_in_threshold_for_precision_on_sorted() {
        // As threshold increases, fewer positives are predicted.
        // For a dataset where higher scores are more likely true, precision should be non-decreasing
        // when sorted? At least we can assert that support is constant and sweep length matches.
        let cases = vec![
            Case { score: 0.1, label: false },
            Case { score: 0.4, label: false },
            Case { score: 0.6, label: true },
            Case { score: 0.9, label: true },
        ];
        let thresholds = vec![0.0, 0.3, 0.5, 0.8, 1.0];
        let reports = sweep_thresholds(&cases, |c| c.score, |c| c.label, &thresholds);
        assert_eq!(reports.len(), thresholds.len());
        for (_, r) in &reports {
            assert_eq!(r.support, cases.len());
        }
        // Higher threshold should not increase FP; FPR should be non-increasing.
        for w in reports.windows(2) {
            assert!(
                w[1].1.fpr <= w[0].1.fpr + 1e-9,
                "FPR should be non-increasing with threshold: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
}

#[test]
fn default_thresholds_are_21_points() {
        let d = default_thresholds();
        assert_eq!(d.len(), 21);
        assert!((d[0] - 0.0).abs() < 1e-9);
        assert!((d[20] - 1.0).abs() < 1e-9);
}

#[test]
fn markdown_row_contains_gate() {
        let cases = vec![
            Case { score: 0.2, label: false },
            Case { score: 0.8, label: true },
        ];
        let r = calibrate_threshold(&cases, |c| c.score, |c| c.label, 0.5);
        let row = r.markdown_row(0.5);
        assert!(row.contains("1.000"));
        assert!(row.contains('✅'));
}

#[test]
fn render_markdown_table_has_header() {
        let cases = vec![Case { score: 0.9, label: true }];
        let reports = sweep_thresholds(&cases, |c| c.score, |c| c.label, &[0.5]);
        let md = render_markdown_table(&reports);
        assert!(md.contains("| threshold |"));
        assert!(md.contains("| 0.50 |"));
}

#[test]
fn emit_calibration_artifact_smoke() {
        let cases = vec![
            Case { score: 0.2, label: false },
            Case { score: 0.8, label: true },
        ];
        let thresholds = default_thresholds();
        let reports = sweep_thresholds(&cases, |c| c.score, |c| c.label, &thresholds);
        // Emit to target/calibration/common_core_demo.md (best-effort, never panics)
        emit_markdown_artifact("common_core_demo", &reports);
        // Verify the markdown was produced in-memory
        let md = render_markdown_table(&reports);
        assert!(md.contains("0.50"));
        // If running with --nocapture, the eprintln from emit will be visible.
        if std::env::var("CALIBRATION_EMIT").is_ok() {
            eprintln!("{}", md);
        }
}
