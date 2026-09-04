//! Live-AI tests for the Policy-Linter (`PolicyLinter`, ROADMAP §3.1).
//!
//! Performs REAL ONNX inference. Compiled only under the `live-ai` feature,
//! `#[ignore]`d, run via `make ort-test-live` / `make test-live`.
//!
//! Env contract (see `tests/live/README.md`):
//! - `ORT_LIVE_POLICY_MODEL` — path to the Policy-Linter directory (contains
//!   `onnx/` artifacts, `config.json` `head` block, `tokenizer.json`).
//! - `ORT_LIVE_POLICY_LABELS` — path to a policy-labels JSON file (a JSON
//!   array of rule strings).
//! - `ORT_LIVE_POLICY_REFERENCE` — optional; a second directory for the
//!   quant-vs-reference threshold-flip measurement (falls back to the on-disk
//!   q4 artifact — the export ships no fp32, README-documented).
//!
//! When the model var is absent the tests skip cleanly.

use std::path::{Path, PathBuf};

use fluent_onnx::{load_policy_labels, OnnxConfig, OnnxTask, PolicyLinter, Quant, SessionLoader};

fn live_policy_dir() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_POLICY_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn live_policy_labels() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_POLICY_LABELS")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn live_reference_dir() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_POLICY_REFERENCE")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Build a `PolicyLinter` session for `quant` over the model in `dir`.
fn build_linter(
    dir: &Path,
    quant: Quant,
    labels: &[String],
) -> Result<PolicyLinter, fluent_onnx::OrtError> {
    let config = OnnxConfig::new()
        .model_path(dir.join("onnx"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .task(OnnxTask::ZeroShotTokenMatching)
        .quantization(quant)
        .build();
    config.validate()?;
    let handle = fluent_onnx::OrtSessionLoader.load(&config, "live-policy")?;
    let worker = fluent_onnx::TwoTowerWorker::from_handle(&handle, &config, "live-policy")?;
    Ok(PolicyLinter::new(std::sync::Arc::new(worker), labels.to_vec(), 0.5))
}

/// The primary quantization for the on-disk export: q8 (the shipped artifact —
/// no fp32 is exported; the README accuracy table is vs the PyTorch reference).
fn primary_quant() -> Quant {
    Quant::Q8
}

/// A small policy for the offline fixtures: don't exfiltrate credentials. The
/// live tests load their labels from `ORT_LIVE_POLICY_LABELS`; this constant
/// documents the intended shape.
const _POLICY_LABELS: &[&str] = &["do not share credentials"];

const VIOLATING_TEXT: &str = "the password is hunter2, please do not leak it";
const BENIGN_TEXT: &str = "the quick brown fox jumps over the lazy dog";

#[test]
#[ignore = "live-AI: requires ORT_LIVE_POLICY_MODEL + ORT_LIVE_POLICY_LABELS; run via `make ort-test-live`"]
fn policy_linter_ranks_violating_above_benign_and_offsets_are_byte_exact() {
    let (Some(dir), Some(labels_path)) = (live_policy_dir(), live_policy_labels()) else {
        eprintln!("skipping: ORT_LIVE_POLICY_MODEL / ORT_LIVE_POLICY_LABELS unset");
        return;
    };
    let labels = load_policy_labels(&labels_path).expect("labels");
    assert!(!labels.is_empty(), "policy_labels file must be non-empty");
    // The export ships no fp32 artifact (README accuracy table is vs the
    // PyTorch reference); q8 is the on-disk default. q8's documented drift
    // (max Δ 0.5241, 3-in-6 threshold flips) means the assertion is on the
    // RANKING signal — the violating sentence's top score clears the benign
    // sentence's — not on an absolute 0.5 split.
    let linter = build_linter(&dir, primary_quant(), &labels).expect("build q8 linter");

    let violating = linter.lint(VIOLATING_TEXT).expect("lint");
    assert!(
        !violating.is_empty(),
        "a credential-bearing sentence must flag at least one token at the threshold"
    );
    for hit in &violating {
        assert!(hit.start < hit.end, "hits carry byte-exact offsets");
        assert!(
            hit.slice(VIOLATING_TEXT).is_some(),
            "hit offsets slice the source"
        );
    }
    let benign = linter.lint(BENIGN_TEXT).expect("lint");
    let violating_max = violating.iter().map(|h| h.score).fold(0.0, f64::max);
    let benign_max = benign.iter().map(|h| h.score).fold(0.0, f64::max);
    eprintln!(
        "Policy-Linter (q8): violating top score {violating_max:.4} vs benign top {benign_max:.4}",
    );
    assert!(
        violating_max > benign_max,
        "the violating sentence must rank above benign prose, got {violating_max:.4} vs {benign_max:.4}"
    );
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_POLICY_MODEL + ORT_LIVE_POLICY_LABELS; run via `make ort-test-live`"]
fn policy_linter_quant_threshold_flips_are_recorded() {
    let (Some(dir), Some(labels_path)) = (live_policy_dir(), live_policy_labels()) else {
        eprintln!("skipping: ORT_LIVE_POLICY_MODEL / ORT_LIVE_POLICY_LABELS unset");
        return;
    };
    let labels = load_policy_labels(&labels_path).expect("labels");
    let q8 = build_linter(&dir, primary_quant(), &labels).expect("build q8 linter");
    let reference = match live_reference_dir() {
        Some(reference_dir) => {
            build_linter(&reference_dir, Quant::Fp32, &labels).expect("build reference fp32")
        }
        // The export ships no fp32 artifact (README: fp32 max Δ 8.4e-4 vs q8's
        // 0.5241 / 3-in-6 threshold flips); q4 is the on-disk sensitivity pair.
        None => build_linter(&dir, Quant::Q4, &labels).expect("build q4 sensitivity linter"),
    };

    let texts = [VIOLATING_TEXT, BENIGN_TEXT];
    let mut flipped = 0usize;
    let mut total = 0usize;
    for text in texts {
        let a = q8.lint(text).expect("q8 lint");
        let b = reference.lint(text).expect("reference lint");
        let fam = |s: &[fluent_onnx::PolicyHit]| {
            let mut v: Vec<(String, usize, usize)> =
                s.iter().map(|h| (h.label.clone(), h.start, h.end)).collect();
            v.sort_unstable();
            v
        };
        if fam(&a) != fam(&b) {
            flipped += 1;
        }
        total += 1;
    }
    let rate = if total > 0 {
        flipped as f64 / total as f64
    } else {
        0.0
    };
    eprintln!(
        "Policy-Linter quant-vs-reference threshold-flip rate over {total} case(s): \
         {flipped}/{total} = {rate:.3} (README documents 3/6 for q8-vs-fp32)"
    );
    assert!(total > 0);
}