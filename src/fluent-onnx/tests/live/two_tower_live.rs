//! Live-AI tests for the two-tower Prompt-Router (ROADMAP_20260827_ORT §2.1).
//!
//! Performs REAL ONNX inference on the Prompt-Router export. Compiled only
//! under the `live-ai` feature, `#[ignore]`d, run via `make ort-test-live` /
//! `make test-live`.
//!
//! Env contract (see `tests/live/README.md`):
//! - `ORT_LIVE_PROMPT_ROUTER_MODEL` — the Prompt-Router model directory
//!   (contains `onnx/` with the `.onnx` artifacts, `config.json` with the
//!   `head` block, and `tokenizer.json`).
//! - `ORT_LIVE_PROMPT_ROUTER_REFERENCE` — optional; a second directory for the
//!   q8-vs-reference delta (fp32 when an fp32 export exists; the on-disk q8/q4
//!   pair is the fallback). Absent → the delta test skips.
//!
//! When the model var is absent the tests skip cleanly (early `return`, never
//! panic). The top-1 assertions are a **smoke baseline** — the redirect
//! evidence requires the ≥100-case zero-shot eval corpus (ROADMAP §2.6a), not
//! these 4 cases.

use std::path::{Path, PathBuf};

use fluent_onnx::{OnnxConfig, OnnxTask, Quant, OrtSessionLoader, SessionLoader, TwoTowerWorker};

fn live_router_dir() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_PROMPT_ROUTER_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn live_reference_dir() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_PROMPT_ROUTER_REFERENCE")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Build a two-tower worker for `quant` over the model in `dir`.
fn build_worker(dir: &Path, quant: Quant) -> Result<TwoTowerWorker, fluent_onnx::OrtError> {
    let config = OnnxConfig::new()
        .model_path(dir.join("onnx"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .task(OnnxTask::ZeroShotRouting)
        .quantization(quant)
        .build();
    config.validate()?;
    let handle = OrtSessionLoader.load(&config, "live-prompt-router")?;
    TwoTowerWorker::from_handle(&handle, &config, "live-prompt-router")
}

/// The smoke-baseline label set (route descriptions the router would score
/// against).
const LABELS: &[&str] = &["code", "prose", "translation", "report command"];

/// Four clearly-typed reference cases. Top-1 agreement here is the smoke
/// baseline only — NOT the redirect evidence (that needs the ≥100-case corpus).
const EVAL_CASES: &[(&str, usize)] = &[
    ("write a rust function that parses json from a file", 0),
    ("the quick brown fox jumps over the lazy dog", 1),
    ("translate this sentence into french please", 2),
    ("show me the sales report for last quarter", 3),
];

#[test]
#[ignore = "live-AI: requires ORT_LIVE_PROMPT_ROUTER_MODEL; run via `make ort-test-live`"]
fn two_tower_scores_are_a_normalized_distribution() {
    let Some(dir) = live_router_dir() else {
        eprintln!("skipping: ORT_LIVE_PROMPT_ROUTER_MODEL unset");
        return;
    };
    let worker = build_worker(&dir, Quant::Q8).expect("build q8 prompt-router");
    let labels: Vec<String> = LABELS.iter().map(|s| s.to_string()).collect();
    for (text, _) in EVAL_CASES {
        let scores = worker.score_labels(text, &labels).expect("score labels");
        assert_eq!(scores.len(), labels.len(), "one score per label");
        let sum: f64 = scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "softmax over the cosine head must sum to ~1, got {sum:.6} for {text:?}"
        );
        assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)), "scores in [0,1]");
    }
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_PROMPT_ROUTER_MODEL; run via `make ort-test-live`"]
fn two_tower_top1_agrees_on_reference_cases() {
    let Some(dir) = live_router_dir() else {
        eprintln!("skipping: ORT_LIVE_PROMPT_ROUTER_MODEL unset");
        return;
    };
    let worker = build_worker(&dir, Quant::Q8).expect("build q8 prompt-router");
    let labels: Vec<String> = LABELS.iter().map(|s| s.to_string()).collect();
    for (text, expected) in EVAL_CASES {
        let scores = worker.score_labels(text, &labels).expect("score labels");
        let (argmax, best) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .expect("non-empty scores");
        eprintln!(
            "case {text:?}: top-1 = {} ({best:.3}), expected {}",
            LABELS[argmax],
            LABELS[*expected],
        );
        // Smoke baseline (see the module doc): top-1 agreement on clearly-typed
        // cases, NOT the redirect evidence.
        assert_eq!(argmax, *expected, "top-1 smoke baseline for {text:?}");
    }
}

/// Largest absolute difference in a final probability across the label set.
fn max_abs_delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_PROMPT_ROUTER_MODEL (+ ORT_LIVE_PROMPT_ROUTER_REFERENCE for the fp32 band); run via `make ort-test-live`"]
fn two_tower_q8_vs_reference_delta_is_recorded() {
    let Some(dir) = live_router_dir() else {
        eprintln!("skipping: ORT_LIVE_PROMPT_ROUTER_MODEL unset");
        return;
    };
    let q8 = build_worker(&dir, Quant::Q8).expect("build q8 prompt-router");
    // fp32 reference when one is supplied (the README band is fp32-anchored);
    // otherwise the on-disk q4 artifact measures quant-sensitivity — a
    // different, larger quantity that the README does not document, so the
    // assertion is a sanity floor only.
    let reference = match live_reference_dir() {
        Some(reference_dir) => build_worker(&reference_dir, Quant::Fp32)
            .or_else(|_| build_worker(&reference_dir, Quant::Q4))
            .expect("build reference worker (fp32 preferred, q4 fallback)"),
        None => build_worker(&dir, Quant::Q4).expect("build q4 sensitivity worker"),
    };
    let is_fp32_reference = live_reference_dir()
        .and_then(|d| build_worker(&d, Quant::Fp32).ok())
        .is_some();
    let labels: Vec<String> = LABELS.iter().map(|s| s.to_string()).collect();
    let mut deltas: Vec<f64> = EVAL_CASES
        .iter()
        .map(|(text, _)| {
            let a = q8.score_labels(text, &labels).expect("q8 scores");
            let b = reference.score_labels(text, &labels).expect("reference scores");
            max_abs_delta(&a, &b)
        })
        .collect();
    deltas.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let max = deltas.last().copied().unwrap_or(0.0);
    let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
    eprintln!(
        "q8-vs-reference max-Δ probability over {} cases: mean={mean:.6} max={max:.6} \
         (fp32 reference: {}; README q8-vs-fp32: max Δ 0.0910, top-1 flips 0)",
        deltas.len(),
        is_fp32_reference,
    );
    if is_fp32_reference {
        // README records q8-vs-fp32 max Δ = 0.0910 with 0 top-1 flips across
        // the reference cases. The band leaves a machine-variance margin and
        // still catches a broken head/pooling path.
        assert!(max < 0.25, "q8-vs-fp32 delta exceeded the README band: max Δ = {max:.6}");
    } else {
        // q8-vs-q4 is NOT a README-documented quantity; both artifacts drift
        // from fp32 in different directions. A truly broken pipeline would
        // produce ~random probabilities and near-1.0 deltas — the floor only
        // catches that.
        assert!(max < 0.6, "q8-vs-q4 sanity floor exceeded: max Δ = {max:.6}");
    }
}