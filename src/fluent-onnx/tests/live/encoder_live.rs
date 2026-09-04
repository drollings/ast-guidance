//! Live-AI tests for the ONNX Encoder (`EmbeddingProvider`, ROADMAP §1).
//!
//! Performs REAL ONNX inference. Compiled only under the `live-ai` feature,
//! `#[ignore]`d, run via `make ort-test-live` / `make test-live`.
//!
//! Env contract (see `tests/live/README.md`):
//! - `ORT_LIVE_ENCODER_MODEL` — path to the encoder model directory (containing
//!   `onnx/` with the `.onnx` artifacts and `tokenizer.json`).
//! - `ORT_LIVE_ENCODER_REFERENCE` — optional; a second directory for the
//!   q8-vs-reference drift measurement (fp32 when an fp32 export exists;
//!   the on-disk q8/q4 pair is the fallback). Absent → drift test skips.
//!
//! When the model var is absent the tests skip cleanly (early `return`, never
//! panic). Assertions are structural (dims, non-empty, determinism) plus the
//! drift band the roadmap requires be recorded.

use std::path::{Path, PathBuf};

use fluent_llm::embeddings::EmbeddingProvider;
use fluent_onnx::{OnnxConfig, OnnxTask, Quant, OrtEncoder, OrtSessionLoader, SessionLoader};

fn live_encoder_dir() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_ENCODER_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn live_reference_dir() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_ENCODER_REFERENCE")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Build an encoder session for `quant` over the model in `dir`.
fn build_encoder(dir: &Path, quant: Quant) -> Result<OrtEncoder, fluent_onnx::OrtError> {
    let config = OnnxConfig::new()
        .model_path(dir.join("onnx"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .task(OnnxTask::FillMask)
        .quantization(quant)
        .build();
    config.validate()?;
    let handle = OrtSessionLoader.load(&config, "live-encoder")?;
    fluent_onnx::build_encoder(&config, "live-encoder", &handle)
}

/// A fixed sample spanning plain prose and code-ish text, for drift + latency.
const FIXED_SAMPLE: &[&str] = &[
    "show me the sales report for last quarter",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "The report is due on Friday and must include the revenue breakdown.",
    "a stable content-addressed id is a pure function of the text",
];

#[test]
#[ignore = "live-AI: requires ORT_LIVE_ENCODER_MODEL; run via `make ort-test-live`"]
fn encoder_produces_dims_and_nonempty_embeddings() {
    let Some(dir) = live_encoder_dir() else {
        eprintln!("skipping: ORT_LIVE_ENCODER_MODEL unset");
        return;
    };
    let encoder = build_encoder(&dir, Quant::Q8).expect("build q8 encoder");
    let dims = encoder.dimensions();
    assert!(dims > 0, "dims must be non-zero");
    for text in FIXED_SAMPLE {
        let embedding = encoder.embed(text).expect("embed");
        assert_eq!(embedding.len() as u32, dims, "embedding dims must match");
        assert!(
            embedding.iter().any(|v| *v != 0.0),
            "embedding must be non-zero (fully-masked yields zeros)"
        );
    }
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_ENCODER_MODEL; run via `make ort-test-live`"]
fn encoder_is_deterministic_across_repeat_runs() {
    let Some(dir) = live_encoder_dir() else {
        eprintln!("skipping: ORT_LIVE_ENCODER_MODEL unset");
        return;
    };
    // intra_op_threads=1 (the default) makes the reduction single-threaded, so
    // a fixed input must produce bit-identical output run after run.
    let encoder = build_encoder(&dir, Quant::Q8).expect("build q8 encoder");
    for text in FIXED_SAMPLE {
        let first = encoder.embed(text).expect("first embed");
        let second = encoder.embed(text).expect("second embed");
        assert_eq!(first, second, "determinism: repeat run must be bit-identical");
    }
}

/// Cosine similarity between two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    f64::from(dot) / (f64::from(na).sqrt() * f64::from(nb).sqrt())
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_ENCODER_MODEL (+ ORT_LIVE_ENCODER_REFERENCE for fp32); run via `make ort-test-live`"]
fn q8_vs_reference_cosine_drift_band_is_recorded() {
    let Some(dir) = live_encoder_dir() else {
        eprintln!("skipping: ORT_LIVE_ENCODER_MODEL unset");
        return;
    };
    // The primary is q8 (the default quantization). The reference is the
    // fp32 export when one exists (`ORT_LIVE_ENCODER_REFERENCE`); otherwise
    // the on-disk q4 artifact measures the quant-sensitivity band. The band
    // is RECORDED here — this crate does NOT license q8 globally; each
    // consumer (M5 retrieval vs M6 entity-similarity) gates its own
    // quantization decision in its own acceptance test.
    let q8 = build_encoder(&dir, Quant::Q8).expect("build q8 encoder");
    let reference = match live_reference_dir() {
        Some(reference_dir) => build_encoder(&reference_dir, Quant::Fp32)
            .or_else(|_| build_encoder(&reference_dir, Quant::Q4))
            .expect("build reference encoder (fp32 preferred, q4 fallback)"),
        None => build_encoder(&dir, Quant::Q4).expect("build q4 sensitivity encoder"),
    };

    let mut cosines: Vec<f64> = FIXED_SAMPLE
        .iter()
        .map(|text| {
            let a = q8.embed(text).expect("q8 embed");
            let b = reference.embed(text).expect("reference embed");
            cosine(&a, &b)
        })
        .collect();
    cosines.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let min = cosines.first().copied().unwrap_or(0.0);
    let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
    eprintln!(
        "q8-vs-reference cosine band over {} sentences: min={min:.6} mean={mean:.6} max={:.6}",
        cosines.len(),
        cosines.last().copied().unwrap_or(0.0)
    );
    // Sanity floor only — a sentence embedding that is *anti-correlated*
    // with its reference means the pooling or the quant is broken. The
    // recorded band is the deliverable (q4 is documented as heavily lossy on
    // this architecture); the strict acceptance lives in M5/M6.
    assert!(min > 0.0, "drift band floor: min cosine {min:.6} must be positive");
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_ENCODER_MODEL; run via `make ort-test-live`"]
fn encoder_latency_is_within_budget() {
    let Some(dir) = live_encoder_dir() else {
        eprintln!("skipping: ORT_LIVE_ENCODER_MODEL unset");
        return;
    };
    let encoder = build_encoder(&dir, Quant::Q8).expect("build q8 encoder");
    let start = std::time::Instant::now();
    const N: usize = 8;
    for _ in 0..N {
        for text in FIXED_SAMPLE {
            encoder.embed(text).expect("embed");
        }
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / (N * FIXED_SAMPLE.len()) as u32;
    eprintln!("encoder single-call latency (q8, intra_op_threads=1): {per_call:?}");
    // A 350M encoder forward over a sentence is ms-scale single-threaded; the
    // budget is generous so slow CI machines do not flake. This is a record,
    // not a gate.
    assert!(per_call.as_millis() < 5_000, "latency budget exceeded: {per_call:?}");
}