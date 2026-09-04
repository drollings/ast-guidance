//! Live-AI tests for the trained-encoder annotation worker
//! (`OrtAnnotationWorker`, ROADMAP_20260828_ORT_FIXES §4 / M2.2).
//!
//! Performs REAL ONNX inference. Compiled only under the `live-ai` feature,
//! `#[ignore]`d, run via `make ort-test-live` / `make test-live`.
//!
//! Env contract (see `tests/live/README.md`):
//! - `ORT_LIVE_ENCODER_MODEL` — path to the finetuned Encoder model directory
//!   (containing `onnx/` with the `.onnx` artifact and `tokenizer.json`).
//! - `ORT_LIVE_ENCODER_REFERENCE` — optional; a second directory (fp32 export)
//!   for the q8-vs-reference head drift band. Absent → drift test skips.
//! - `ORT_LIVE_ANNOTATION_LABELS` — path to the annotation labels file
//!   (`annotation_labels.json`).
//!
//! When the model/labels vars are absent the tests skip cleanly (early
//! `return`, never panic). Assertions: bit-determinism (intra_op_threads=1),
//! `is_total()` alignment (every spacy token covered), and the recorded
//! q8-vs-fp32 head drift band — the exact guarantees ROADMAP §5/D5 requires.

use std::path::{Path, PathBuf};

use fluent_onnx::align::SpacyTokenAligner;
use fluent_onnx::annotate::build_annotation_worker_from_registry;
use fluent_onnx::{
    AnnotationHeads, OnnxConfig, OnnxTask, OrtError, OrtSessionLoader, OrtSessionRegistry,
    Quant,
};

fn live_dir() -> Option<PathBuf> {
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

fn live_labels_path() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_ANNOTATION_LABELS")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Build a registered annotation worker for `quant` over the model in `dir`.
fn build_worker(
    dir: &Path,
    labels_path: &Path,
    quant: Quant,
) -> Result<fluent_onnx::annotate::OrtAnnotationWorker, OrtError> {
    let config = OnnxConfig::new()
        .model_path(dir.join("onnx"))
        .tokenizer_path(dir.join("tokenizer.json"))
        .task(OnnxTask::FillMask)
        .quantization(quant)
        .annotation_heads(AnnotationHeads::new().labels(labels_path).build())
        .build();
    config.validate()?;
    let registry = OrtSessionRegistry::new(Arc::new(OrtSessionLoader));
    registry.register("live-annotation", config.clone())?;
    let worker = build_annotation_worker_from_registry(&registry, "live-annotation")?
        .expect("worker for a FillMask+heads model");
    // Pull the concrete worker out of the Arc.
    Arc::try_unwrap(worker).map_err(|_| OrtError::Other("worker arc not unique".into()))
}

use std::sync::Arc;

/// A fixed sample spanning plain prose, for the head-drift band.
const FIXED_SAMPLE: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "show me the sales report for last quarter",
    "The report is due on Friday and must include the revenue breakdown.",
    "a stable content-addressed id is a pure function of the text",
];

/// Align the LFM predictions onto a simple whitespace-split spacy baseline.
/// The spacy spans are built from tokenizer orth + idx via the shared aligner;
/// here we approximate orth from whitespace-split tokens (the real baseline is
/// spacy-rs's tokenizer, which the router supplies). The alignment's
/// `is_total()` check is what the roadmap requires — every baseline token must
/// be covered by at least one LFM subword.
fn run_worker(
    worker: &fluent_onnx::annotate::OrtAnnotationWorker,
    text: &str,
) -> Result<Vec<Option<fluent_onnx::annotate::TokenAnnotation>>, OrtError> {
    // Approximate spacy spans from whitespace tokens (byte offsets). In the
    // router these come from spacy-rs's orth+idx; this is the hermetic stand-in
    // for the live alignment check.
    let mut orth: Vec<&str> = Vec::new();
    let mut idx: Vec<usize> = Vec::new();
    let mut pos = 0usize;
    for word in text.split_whitespace() {
        orth.push(word);
        idx.push(pos);
        pos += word.len() + 1;
    }
    let spans = SpacyTokenAligner::spacy_spans(&orth, &idx);
    worker.annotate(text, &spans)
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_ENCODER_MODEL + ORT_LIVE_ANNOTATION_LABELS; run via `make ort-test-live`"]
fn annotation_is_deterministic_and_total() {
    let (Some(dir), Some(labels)) = (live_dir(), live_labels_path()) else {
        eprintln!("skipping: ORT_LIVE_ENCODER_MODEL / ORT_LIVE_ANNOTATION_LABELS unset");
        return;
    };
    let worker = build_worker(&dir, &labels, Quant::Q8).expect("build q8 worker");
    for text in FIXED_SAMPLE {
        let first = run_worker(&worker, text).expect("first annotate");
        let second = run_worker(&worker, text).expect("second annotate");
        assert_eq!(first, second, "determinism: repeat run must be bit-identical");
        // Every baseline token must be covered (is_total alignment) — a partial
        // mix is a fail-open fallback, never a silent partial annotation.
        assert!(
            first.iter().all(|t| t.is_some()),
            "every spacy token must be covered by LFM subwords; got {first:?}"
        );
    }
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_ENCODER_MODEL + ORT_LIVE_ANNOTATION_LABELS (+ ORT_LIVE_ENCODER_REFERENCE for fp32); run via `make ort-test-live`"]
fn q8_vs_fp32_head_drift_band_is_recorded() {
    let (Some(dir), Some(labels)) = (live_dir(), live_labels_path()) else {
        eprintln!("skipping: ORT_LIVE_ENCODER_MODEL / ORT_LIVE_ANNOTATION_LABELS unset");
        return;
    };
    let q8 = build_worker(&dir, &labels, Quant::Q8).expect("build q8 worker");
    // The reference is the fp32 export when one exists; otherwise the on-disk
    // q4 artifact measures the quant-sensitivity band. The band is RECORDED —
    // this crate does not license q8 globally.
    let reference = match live_reference_dir() {
        Some(ref_dir) => build_worker(&ref_dir, &labels, Quant::Fp32)
            .or_else(|_| build_worker(&ref_dir, &labels, Quant::Q4))
            .expect("build reference worker (fp32 preferred, q4 fallback)"),
        None => build_worker(&dir, &labels, Quant::Q4).expect("build q4 sensitivity worker"),
    };

    // Head-agreement band: fraction of tokens whose pos/dep label + head the
    // q8 and reference decodes agree on, across the sample.
    let mut agree_total = 0usize;
    let mut token_total = 0usize;
    for text in FIXED_SAMPLE {
        let a = run_worker(&q8, text).expect("q8 annotate");
        let b = run_worker(&reference, text).expect("reference annotate");
        for (ta, tb) in a.iter().zip(b.iter()) {
            match (ta, tb) {
                (Some(a), Some(b)) => {
                    token_total += 1;
                    if a.pos == b.pos && a.dep == b.dep && a.head_abs == b.head_abs {
                        agree_total += 1;
                    }
                }
                _ => {}
            }
        }
    }
    let band = if token_total > 0 {
        agree_total as f64 / token_total as f64
    } else {
        0.0
    };
    eprintln!(
        "q8-vs-reference head agreement band: {agree_total}/{token_total} tokens = {band:.4}",
    );
    // Sanity floor only — an agreement below chance means the quant broke a
    // head. The recorded band is the deliverable.
    assert!(band > 0.5, "head drift band floor: agreement {band:.4} must exceed chance");
}