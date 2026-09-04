//! Calibration harness for confidence vs task-value thresholds.
//!
//! Pure, zero-domain: no I/O, no domain types. Composed everywhere thresholds
//! are trusted for caching/persisting decisions. The gate is precision ≥0.90
//! and FPR ≤0.05 on control groups (see ROADMAP_20260901_FIXES_4.md M1).

/// Precision/recall/FPR at a single operating point.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationReport {
    /// TP / (TP+FP) — 1.0 when no positive predictions (no false positives).
    pub precision: f64,
    /// TP / (TP+FN) — 1.0 when no actual positives.
    pub recall: f64,
    /// FP / (FP+TN) — 0.0 when no actual negatives.
    pub fpr: f64,
    /// Number of cases evaluated (TP+FP+TN+FN).
    pub support: usize,
    /// Raw counts for audit.
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub r#fn: usize,
}

impl CalibrationReport {
    /// Whether this operating point passes the roadmap M1 gate.
    #[must_use]
    pub fn passes_gate(&self) -> bool {
        self.precision >= 0.90 && self.fpr <= 0.05
    }

    /// Markdown row for CI artifact tables.
    #[must_use]
    pub fn markdown_row(&self, threshold: f64) -> String {
        format!(
            "| {:.2} | {:.3} | {:.3} | {:.3} | {} | {} |",
            threshold,
            self.precision,
            self.recall,
            self.fpr,
            self.support,
            if self.passes_gate() { "✅" } else { "❌" }
        )
    }
}

/// Compute precision/recall/FPR at `threshold`.
///
/// `score_of` is the detector confidence (e.g. cosine similarity, overall
/// confidence), `label_of` is the ground-truth correctness (true = should fire).
/// A case fires when `score >= threshold`.
pub fn calibrate_threshold<S>(
    cases: &[S],
    score_of: impl Fn(&S) -> f64,
    label_of: impl Fn(&S) -> bool,
    threshold: f64,
) -> CalibrationReport {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut r#fn = 0usize;
    for c in cases {
        let score = score_of(c);
        let label = label_of(c);
        let predicted = score >= threshold;
        match (predicted, label) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => r#fn += 1,
        }
    }
    let precision = if tp + fp == 0 {
        1.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + r#fn == 0 {
        1.0
    } else {
        tp as f64 / (tp + r#fn) as f64
    };
    let fpr = if fp + tn == 0 {
        0.0
    } else {
        fp as f64 / (fp + tn) as f64
    };
    CalibrationReport {
        precision,
        recall,
        fpr,
        support: cases.len(),
        tp,
        fp,
        tn,
        r#fn,
    }
}

/// Sweep `thresholds` over the same case set, returning one report per threshold
/// in the order supplied.
pub fn sweep_thresholds<S>(
    cases: &[S],
    score_of: impl Fn(&S) -> f64,
    label_of: impl Fn(&S) -> bool,
    thresholds: &[f64],
) -> Vec<(f64, CalibrationReport)> {
    thresholds
        .iter()
        .map(|&t| {
            let r = calibrate_threshold(cases, &score_of, &label_of, t);
            (t, r)
        })
        .collect()
}

/// Default sweep grid 0.0..1.0 step 0.05 (21 points) plus caller-supplied extras.
pub fn default_thresholds() -> Vec<f64> {
    (0..=20).map(|i| i as f64 * 0.05).collect()
}

/// Render a markdown table for `reports` (from `sweep_thresholds`) to `path`.
///
/// Creates parent directories as needed. Returns the markdown string.
pub fn render_markdown_table(reports: &[(f64, CalibrationReport)]) -> String {
    let mut out = String::new();
    out.push_str("| threshold | precision | recall | FPR | support | gate |\n");
    out.push_str("|-----------|-----------|--------|-----|---------|------|\n");
    for (t, r) in reports {
        out.push_str(&r.markdown_row(*t));
        out.push('\n');
    }
    out
}

/// Write `reports` as markdown to `target/calibration/<name>.md` (best-effort).
/// Creates `target/calibration` if missing. Never panics — logs on error.
/// Tries workspace root and current dir to be robust under `cargo test -p`.
pub fn emit_markdown_artifact(name: &str, reports: &[(f64, CalibrationReport)]) {
    let md = render_markdown_table(reports);
    let candidates = [
        std::path::PathBuf::from("target/calibration"),
        std::path::PathBuf::from("../target/calibration"),
        std::path::PathBuf::from("../../target/calibration"),
        std::path::PathBuf::from("../../../target/calibration"),
    ];
    // Also try to locate workspace root by walking up from current_dir for Cargo.toml with [workspace]
    let mut extra = Vec::new();
    if let Ok(cur) = std::env::current_dir() {
        let mut p = cur.as_path();
        for _ in 0..6 {
            let candidate = p.join("target/calibration");
            if p.join("Cargo.toml").exists() {
                extra.push(candidate);
            }
            if let Some(parent) = p.parent() {
                p = parent;
            } else {
                break;
            }
        }
        // Also try CARGO_MANIFEST_DIR parent chain
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            let mut pm = std::path::Path::new(&manifest);
            for _ in 0..6 {
                extra.push(pm.join("../../target/calibration"));
                extra.push(pm.join("../target/calibration"));
                if let Some(parent) = pm.parent() {
                    pm = parent;
                } else {
                    break;
                }
            }
        }
    }
    let mut dirs = candidates.to_vec();
    dirs.extend(extra);
    // Deduplicate
    dirs.sort();
    dirs.dedup();
    let mut wrote = false;
    for dir in dirs {
        if std::fs::create_dir_all(&dir).is_ok() {
            let path = dir.join(format!("{name}.md"));
            if std::fs::write(&path, &md).is_ok() {
                eprintln!("calibration artifact: {}", path.display());
                wrote = true;
            }
        }
    }
    if !wrote {
        // Fallback: original path
        let dir = std::path::Path::new("target/calibration");
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(format!("{name}.md")), &md);
    }
}

