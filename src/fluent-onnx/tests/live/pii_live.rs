//! Live-AI tests for the PII-Detector (`OrtPiiClassifier`, ROADMAP §3.2/§3.4).
//!
//! Performs REAL ONNX inference. Compiled only under the `live-ai` feature,
//! `#[ignore]`d, run via `make ort-test-live` / `make test-live`.
//!
//! Env contract (see `tests/live/README.md`):
//! - `ORT_LIVE_PII_MODEL` — path to the PII-Detector directory (contains the
//!   `model*.onnx` artifacts, `config.json` with `id2label`, `tokenizer.json`).
//! - `ORT_LIVE_PII_REFERENCE` — optional; a second directory for the
//!   quant-vs-reference span-flip measurement (the on-disk fp32 `model.onnx`
//!   is the reference when absent).
//!
//! When the model var is absent the tests skip cleanly. Assertions are the
//! known-PII golden corpus recall + the recorded flip rate (a per-model record,
//! not a global license — the review pre-filter gates its own threshold).

use std::path::{Path, PathBuf};

use fluent_onnx::pii::PiiSpanDetector;
use fluent_onnx::{OnnxConfig, OnnxTask, OrtPiiClassifier, OrtSessionLoader, Quant, SessionLoader};

fn live_pii_dir() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_PII_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn live_reference_dir() -> Option<PathBuf> {
    std::env::var("ORT_LIVE_PII_REFERENCE")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Build a PII classifier session for `quant` over the model in `dir`.
fn build_pii(dir: &Path, quant: Quant) -> Result<OrtPiiClassifier, fluent_onnx::OrtError> {
    let config = OnnxConfig::new()
        .model_path(dir)
        .tokenizer_path(dir.join("tokenizer.json"))
        .task(OnnxTask::TokenClassification)
        .quantization(quant)
        .build();
    config.validate()?;
    let handle = OrtSessionLoader.load(&config, "live-pii")?;
    OrtPiiClassifier::from_handle(&handle, &config, "live-pii")
}

/// Known-PII golden corpus: each case must yield at least one span of the
/// expected label family. All labels are content-typed (no dialing).
const KNOWN_PII_CASES: &[(&str, &str)] = &[
    (
        "contact me at alice@example.com",
        "contact.email",
    ),
    (
        "my ssn is 123-45-6789",
        "identity.ssn",
    ),
    (
        "call 555-123-4567 tonight",
        "contact.phone",
    ),
];

#[test]
#[ignore = "live-AI: requires ORT_LIVE_PII_MODEL; run via `make ort-test-live`"]
fn pii_classifier_recalls_known_pii_golden_corpus() {
    let Some(dir) = live_pii_dir() else {
        eprintln!("skipping: ORT_LIVE_PII_MODEL unset");
        return;
    };
    // fp32 (the binary-gate default for the PII-Detector).
    let classifier = build_pii(&dir, Quant::Fp32).expect("build fp32 classifier");
    // The on-disk kucukkanat PII export ends in the base LM head (vocab-wide
    // logits), so the classifier refuses loudly rather than argmaxing the
    // wrong axis. That is a recorded blocker, not a failure: the test skips
    // until a corrected export (classification head included) is supplied.
    let (first_text, first_family) = KNOWN_PII_CASES[0];
    match classifier.detect(first_text) {
        Err(fluent_onnx::PiiError::Inference(msg)) if msg.contains("logits last dim") => {
            eprintln!("skipping: on-disk PII export is head-less — {msg}");
            eprintln!("         (re-export with the token-classification head to enable)");
            return;
        }
        Err(e) => panic!("unexpected PII detect error: {e}"),
        Ok(spans) => {
            assert!(!spans.is_empty(), "expected a span in {first_text:?}");
            let labels: Vec<&str> = spans.iter().map(|s| s.label.as_str()).collect();
            assert!(
                labels.iter().any(|l| l.contains(first_family)),
                "expected family {first_family:?} among {labels:?}"
            );
        }
    }
    for (text, expected_family) in &KNOWN_PII_CASES[1..] {
        let spans = classifier.detect(text).expect("detect");
        assert!(
            !spans.is_empty(),
            "expected a PII span in {text:?} (family {expected_family:?})"
        );
        let labels: Vec<&str> = spans.iter().map(|s| s.label.as_str()).collect();
        assert!(
            labels.iter().any(|l| l.contains(expected_family)),
            "expected family {expected_family:?} among {labels:?} for {text:?}"
        );
        for span in &spans {
            assert!(span.end > span.start, "spans are non-empty");
            assert!(
                span.slice(text).is_some(),
                "span offsets are byte-exact against the source"
            );
        }
    }
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_PII_MODEL; run via `make ort-test-live`"]
fn pii_classifier_flags_clean_text_sparingly() {
    let Some(dir) = live_pii_dir() else {
        eprintln!("skipping: ORT_LIVE_PII_MODEL unset");
        return;
    };
    let classifier = build_pii(&dir, Quant::Fp32).expect("build fp32 classifier");
    // Head-less exports skip (see `pii_classifier_recalls_known_pii_golden_corpus`).
    if let Err(fluent_onnx::PiiError::Inference(msg)) = classifier.detect("benign probe text") {
        if msg.contains("logits last dim") {
            eprintln!("skipping: on-disk PII export is head-less — {msg}");
            return;
        }
    }
    // Benign prose must not produce a firehose of spans — a sanity ceiling on
    // false positives (the review pre-filter tolerates a few, never a deluge).
    let spans = classifier
        .detect("the quick brown fox jumps over the lazy dog")
        .expect("detect");
    assert!(
        spans.len() <= 2,
        "clean prose flagged too eagerly: {} spans",
        spans.len()
    );
}

#[test]
#[ignore = "live-AI: requires ORT_LIVE_PII_MODEL (+ ORT_LIVE_PII_REFERENCE for a second artifact); run via `make ort-test-live`"]
fn pii_quant_flip_rate_is_recorded() {
    let Some(dir) = live_pii_dir() else {
        eprintln!("skipping: ORT_LIVE_PII_MODEL unset");
        return;
    };
    // The primary is q8; the reference is fp32 (on-disk `model.onnx`) or the
    // `ORT_LIVE_PII_REFERENCE` directory when given. The flip rate is the
    // fraction of corpus spans the two disagree on — RECORDED here, never
    // asserted against a global threshold (the review pre-filter gates its
    // own acceptance with the recorded rate, ROADMAP §3.2).
    let q8 = build_pii(&dir, Quant::Q8).expect("build q8 classifier");
    let reference = match live_reference_dir() {
        Some(reference_dir) => {
            build_pii(&reference_dir, Quant::Fp32).expect("build reference fp32 classifier")
        }
        None => build_pii(&dir, Quant::Fp32).expect("build fp32 classifier"),
    };

    // Head-less exports skip (see `pii_classifier_recalls_known_pii_golden_corpus`).
    let (probe, _) = KNOWN_PII_CASES[0];
    for classifier in [&q8, &reference] {
        if let Err(fluent_onnx::PiiError::Inference(msg)) = classifier.detect(probe) {
            if msg.contains("logits last dim") {
                eprintln!("skipping: on-disk PII export is head-less — {msg}");
                eprintln!("         (the flip rate is unmeasurable until a corrected export lands)");
                return;
            }
        }
    }

    let mut flipped = 0usize;
    let mut total = 0usize;
    for (text, _) in KNOWN_PII_CASES {
        let a = q8.detect(text).expect("q8 detect");
        let b = reference.detect(text).expect("reference detect");
        let fam = |s: &[fluent_onnx::PiiSpan]| {
            let mut v: Vec<(String, usize, usize)> =
                s.iter().map(|x| (x.label.clone(), x.start, x.end)).collect();
            v.sort_unstable();
            v
        };
        let va = fam(&a);
        let vb = fam(&b);
        if va != vb {
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
        "PII quant-vs-reference span-set flip rate over {} cases: {flipped}/{total} = {rate:.3}",
        total
    );
    assert!(total > 0, "the golden corpus must be non-empty");
}