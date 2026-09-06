//! PII detection (ROADMAP_20260827_ORT §3.2): the BILUO decode, the label-map
//! loading, and the onnx PII-Detector behind the [`PiiSpanDetector`] seam.
//!
//! Two implementations justify the trait: [`fluent_llm::backend::RegexPiiDetector`]
//! (wrapping the canonical `fluent_llm::pii_patterns` table — never a
//! duplicated pattern table) and the onnx-gated [`OrtPiiClassifier`] (token
//! classification over the PII-Detector export, BILUO-decoded into
//! char-aligned spans). The BILUO decode and the label-map loading are pure
//! and unit-tested without a model.
//!
//! The trait is the fail-open review pre-filter surface (§3.5): it only ever
//! *adds* candidates (spans a review job records), never drops a job and
//! never blocks the hot path — a detector error surfaces as an empty span set.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use fluent_llm::backend::PiiSpan;

use crate::config::OnnxConfig;
use crate::error::OrtError;

/// Decode per-token BILUO labels into char-aligned spans.
///
/// `offsets` are the per-token byte offsets (from the LFM tokenizer), `labels`
/// the predicted label strings (`"O"`, `"B-contact.email"`, `"I-contact.email"`,
/// `"E-contact.email"`, `"S-contact.email"`), `scores` the per-token
/// confidence. A span's char range runs from its first token's start to its
/// last token's end; its score is the mean of the covered tokens' scores.
/// Pure and unit-testable — the classifier's deterministic half.
#[must_use]
pub fn decode_biluo(
    offsets: &[(usize, usize)],
    labels: &[String],
    scores: &[f64],
) -> Vec<PiiSpan> {
    let n = offsets.len().min(labels.len()).min(scores.len());
    let mut spans: Vec<PiiSpan> = Vec::new();
    // An open span under construction: (start_token, label, accumulated score).
    let mut open: Option<(usize, String, f64)> = None;

    for i in 0..n {
        let label = labels[i].as_str();
        let score = scores[i];
        if label == "O" {
            if let Some((start, kind, acc)) = open.take() {
                push_span(&mut spans, offsets, start, i, &kind, acc, n);
            }
            continue;
        }
        let (prefix, kind) = match label.split_once('-') {
            Some((p, k)) if matches!(p, "B" | "I" | "E" | "S") => (p, k.to_string()),
            // A bare non-O label (no BILUO prefix): treat as a single-token
            // span of that label.
            _ => ("S", label.to_string()),
        };
        match prefix {
            "S" => {
                if let Some((start, kind0, acc)) = open.take() {
                    push_span(&mut spans, offsets, start, i, &kind0, acc, n);
                }
                push_span(&mut spans, offsets, i, i + 1, &kind, score, n);
            }
            "B" => {
                if let Some((start, kind0, acc)) = open.take() {
                    push_span(&mut spans, offsets, start, i, &kind0, acc, n);
                }
                open = Some((i, kind, score));
            }
            "I" => match &mut open {
                Some((_, kind0, acc)) if *kind0 == kind => *acc += score,
                _ => {
                    if let Some((start, kind0, acc)) = open.take() {
                        push_span(&mut spans, offsets, start, i, &kind0, acc, n);
                    }
                    open = Some((i, kind, score));
                }
            },
            "E" => {
                if let Some((start, kind0, acc)) = open.take() {
                    if kind0 == kind {
                        push_span(&mut spans, offsets, start, i + 1, &kind, acc + score, n);
                    } else {
                        push_span(&mut spans, offsets, start, i, &kind0, acc, n);
                        push_span(&mut spans, offsets, i, i + 1, &kind, score, n);
                    }
                } else {
                    push_span(&mut spans, offsets, i, i + 1, &kind, score, n);
                }
            }
            _ => unreachable!("BILUO prefix matched above"),
        }
    }
    if let Some((start, kind, acc)) = open.take() {
        push_span(&mut spans, offsets, start, n, &kind, acc, n);
    }
    spans
}

/// Push a span covering tokens `[start, end)` of `kind` with the summed score,
/// averaging the score over the covered token count. Zero-width spans (all
/// covered tokens zero-offset, e.g. `[CLS]`/`[SEP]`) are skipped.
fn push_span(
    out: &mut Vec<PiiSpan>,
    offsets: &[(usize, usize)],
    start: usize,
    end: usize,
    kind: &str,
    acc: f64,
    n: usize,
) {
    if start >= end || start >= n {
        return;
    }
    let e = end.min(n);
    let span_start = offsets[start].0;
    let span_end = offsets[e - 1].1;
    if span_start >= span_end {
        return;
    }
    let count = (e - start) as f64;
    out.push(PiiSpan::new(span_start, span_end, kind, acc / count));
}

/// Load a label map (`id -> label`) from a JSON file: either an object of
/// `{"0": "O", "1": "B-contact.email", …}` (the HF `id2label` shape) or an
/// array of labels indexed by class id.
pub fn load_id2label(path: &Path) -> Result<HashMap<i64, String>, OrtError> {
    let raw = std::fs::read_to_string(path).map_err(|e| OrtError::ConfigRead {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|e| OrtError::ConfigParse {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
    match value {
        serde_json::Value::Object(map) => {
            let mut out = HashMap::with_capacity(map.len());
            for (key, v) in map {
                let id: i64 = key.parse().map_err(|_| OrtError::ConfigParse {
                    path: path.display().to_string(),
                    detail: format!("id2label key `{key}` is not an integer id"),
                })?;
                let label = v.as_str().ok_or_else(|| OrtError::ConfigParse {
                    path: path.display().to_string(),
                    detail: format!("id2label[{id}] is not a string"),
                })?;
                out.insert(id, label.to_string());
            }
            Ok(out)
        }
        serde_json::Value::Array(items) => items
            .into_iter()
            .enumerate()
            .map(|(id, v)| {
                v.as_str()
                    .map(|s| (id as i64, s.to_string()))
                    .ok_or_else(|| OrtError::ConfigParse {
                        path: path.display().to_string(),
                        detail: format!("id2label[{id}] is not a string"),
                    })
            })
            .collect(),
        _ => Err(OrtError::ConfigParse {
            path: path.display().to_string(),
            detail: "id2label must be a JSON object or array of labels".to_string(),
        }),
    }
}

/// Read the PII label map for a token-classification config: the `label_source`
/// file when configured, else the model `config.json` `id2label`. `None` when
/// no label source resolves (the classifier surfaces a loud build error).
pub fn label_map_for(config: &OnnxConfig) -> Result<Option<HashMap<i64, String>>, OrtError> {
    if let Some(source) = config.label_source.as_ref() {
        return load_id2label(source).map(Some);
    }
    let Some(config_json) = config.config_json_path() else {
        return Ok(None);
    };
    let raw = std::fs::read_to_string(&config_json).map_err(|e| OrtError::ConfigRead {
        path: config_json.display().to_string(),
        detail: e.to_string(),
    })?;
    let parsed: ConfigJsonLabels =
        serde_json::from_str(&raw).map_err(|e| OrtError::ConfigParse {
            path: config_json.display().to_string(),
            detail: e.to_string(),
        })?;
    Ok(parsed.id2label)
}

/// The subset of a model `config.json` the PII label-map reader needs.
#[derive(Debug, Default, Deserialize)]
struct ConfigJsonLabels {
    #[serde(default)]
    id2label: Option<HashMap<i64, String>>,
}

/// The onnx PII-Detector: token classification → per-token label + confidence
/// → BILUO decode into char-aligned spans. Behind the `onnx` feature.
///
/// The export is **stateful** (the LFM2 conv+attention stack carries running
/// `past_conv.*` / `past_key_values.*` caches and requires `position_ids`),
/// so the classifier discovers the input schema from the session and feeds
/// zero-initialized fresh caches per run (never chains runs). The logits'
/// last dim is checked against the `id2label` count: a **head-less export**
/// (the base LM head's vocab-width logits instead of the classification head)
/// surfaces a loud, actionable error — the caller falls back to the regex
/// baseline (fail-open) rather than argmaxing over the wrong axis.
#[cfg(feature = "onnx")]
pub mod ort_pii {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use fluent_llm::backend::{PiiError, PiiSpan, PiiSpanDetector};

    use crate::config::OnnxConfig;
    use crate::error::OrtError;
    use crate::pii::{decode_biluo, label_map_for};
    use crate::session::SessionHandle;
    use crate::tokenizer::LfmTokenizer;

    /// A stateful input that needs a fresh cache tensor per run.
    #[derive(Debug, Clone)]
    struct StateInput {
        name: String,
        /// The declared shape; dim 0 (batch) is dynamic (`0`/negative) and
        /// becomes `1` for a fresh run. Remaining dims are kept as declared
        /// (the kv prefix `seq` dim stays `0` — no chaining).
        declared: Vec<i64>,
    }

    /// Build the fresh-run shape for a state input: batch → 1, everything else as
    /// declared — with symbolic (negative) dims clamped to `0` so an unknown
    /// kv-prefix length stays an empty prefix and the tensor construction never
    /// sees a negative dim. Pure — unit-testable without onnx.
    pub(crate) fn fresh_state_shape(declared: &[i64]) -> Vec<i64> {
        declared
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if i == 0 {
                    1
                } else if d < 0 {
                    0
                } else {
                    d
                }
            })
            .collect()
    }

    /// A token-classification session over the PII-Detector export.
    pub struct OrtPiiClassifier {
        session: Arc<Mutex<ort::session::Session>>,
        tokenizer: Arc<LfmTokenizer>,
        id2label: HashMap<i64, String>,
        requires_position_ids: bool,
        state_inputs: Vec<StateInput>,
    }

    /// Per-token classification output: token offsets, predicted labels, scores.
    type ClassifiedTokens = (Vec<(usize, usize)>, Vec<String>, Vec<f64>);

    impl OrtPiiClassifier {
        /// Build the classifier over an already-loaded registry session handle.
        pub fn from_handle(
            handle: &SessionHandle,
            config: &OnnxConfig,
            model_key: &str,
        ) -> Result<Self, OrtError> {
            let session = handle
                .downcast_arc::<Mutex<ort::session::Session>>()
                .ok_or_else(|| {
                    OrtError::Other("session handle does not hold an ort session".to_string())
                })?;
            let tokenizer_path = config.tokenizer_path.as_ref().ok_or_else(|| {
                OrtError::Other(format!("PII classifier '{model_key}' missing tokenizer_path"))
            })?;
            let tokenizer = LfmTokenizer::from_file(tokenizer_path, config.max_seq_len)?;
            let id2label = label_map_for(config)?.ok_or_else(|| {
                OrtError::Other(format!(
                    "PII model `{model_key}` declares neither a label_source nor a \
                     config.json id2label"
                ))
            })?;

            // Discover the stateful input schema from the session metadata so
            // the classifier is robust to the conv-cache configuration instead
            // of hardcoding a specific `layer_types` layout.
            let guard = session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut requires_position_ids = false;
            let mut state_inputs = Vec::new();
            for input in guard.inputs() {
                let name = input.name();
                if name == "position_ids" {
                    requires_position_ids = true;
                } else if name.starts_with("past_conv.")
                    || name.starts_with("past_key_values.")
                {
                    let declared = input
                        .dtype()
                        .tensor_shape()
                        .map(|s| s.iter().copied().collect::<Vec<i64>>())
                        .unwrap_or_default();
                    state_inputs.push(StateInput {
                        name: name.to_string(),
                        declared,
                    });
                }
            }
            drop(guard);

            Ok(Self {
                session,
                tokenizer,
                id2label,
                requires_position_ids,
                state_inputs,
            })
        }

        /// One forward pass: tokenize → run → per-token `(label, score)`.
        fn classify(&self, text: &str) -> Result<ClassifiedTokens, PiiError> {
            let encoding = self
                .tokenizer
                .encode(text)
                .map_err(|e| PiiError::Inference(e.to_string()))?;
            let seq = encoding.len().max(1);
            let ids: Vec<i64> = encoding.ids.iter().map(|&i| i64::from(i)).collect();
            let mask: Vec<i64> = encoding
                .attention_mask
                .iter()
                .map(|&m| i64::from(m))
                .collect();
            let shape = [1usize, seq];
            let input = ort::value::Tensor::from_array((shape, ids.clone()))
                .map_err(|e| PiiError::Inference(e.to_string()))?;
            let mask_t = ort::value::Tensor::from_array((shape, mask))
                .map_err(|e| PiiError::Inference(e.to_string()))?;

            let mut feed: Vec<(String, ort::value::Value)> = vec![
                ("input_ids".to_string(), input.into()),
                ("attention_mask".to_string(), mask_t.into()),
            ];
            if self.requires_position_ids {
                let positions: Vec<i64> = (0..seq).map(|i| i as i64).collect();
                let pos_t = ort::value::Tensor::from_array((shape, positions))
                    .map_err(|e| PiiError::Inference(e.to_string()))?;
                feed.push(("position_ids".to_string(), pos_t.into()));
            }
            for state in &self.state_inputs {
                let state_shape = fresh_state_shape(&state.declared);
                // Symbolic (negative) dims clamp to 0 so the zero-vector sizing
                // never overflows; the concrete on-disk export has none.
                let zeros: Vec<f32> = vec![0.0; state_shape.iter().map(|&d| d.max(0) as usize).product()];
                let tensor = ort::value::Tensor::from_array((state_shape, zeros))
                    .map_err(|e| PiiError::Inference(e.to_string()))?;
                feed.push((state.name.clone(), tensor.into()));
            }

            let mut guard = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = guard
                .run(feed)
                .map_err(|e| PiiError::Inference(e.to_string()))?;
            let (logits_shape, logits) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| PiiError::Inference(e.to_string()))?;

            let num_labels = self.id2label.len();
            let logit_axis = logits_shape
                .get(logits_shape.len().saturating_sub(1))
                .copied()
                .map_or(0usize, |d| d as usize);
            if logit_axis != num_labels {
                // The on-disk kucukkanat PII-Detector export ends in the base
                // LM head (65536-wide logits) — the token-classification head
                // (`nn.Linear(hidden, num_labels)` from `modeling_phase2_tc.py`)
                // was not exported. Argmaxing over the vocab axis would emit
                // token ids, not labels: refuse loudly and let the caller fall
                // back to the regex baseline (fail-open).
                return Err(PiiError::Inference(format!(
                    "PII export's logits last dim is {logit_axis} but the id2label map has \
                     {num_labels} labels — the export appears to end in the base LM head \
                     rather than the token-classification head; a corrected export (or a \
                     label_source with matching logits) is required"
                )));
            }

            let mut labels = Vec::with_capacity(seq);
            let mut scores = Vec::with_capacity(seq);
            for i in 0..seq {
                let row = &logits[i * num_labels..(i + 1) * num_labels];
                let (argmax, _) = row
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bs), (i, &v)| {
                        if v > bs {
                            (i, v)
                        } else {
                            (bi, bs)
                        }
                    });
                let score = softmax_row(row)[argmax];
                let label = self
                    .id2label
                    .get(&(argmax as i64))
                    .cloned()
                    .unwrap_or_else(|| "O".to_string());
                labels.push(label);
                scores.push(f64::from(score));
            }
            Ok((encoding.offsets.clone(), labels, scores))
        }
    }

    /// Numerically-stable softmax over a row of logits.
    fn softmax_row(row: &[f32]) -> Vec<f32> {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    impl PiiSpanDetector for OrtPiiClassifier {
        fn detect(&self, text: &str) -> Result<Vec<PiiSpan>, PiiError> {
            if text.is_empty() {
                return Ok(Vec::new());
            }
            let (offsets, labels, scores) = self.classify(text)?;
            Ok(decode_biluo(&offsets, &labels, &scores))
        }
    }

    /// Build a `PiiSpanDetector` from the registry's session for `model_key`,
    /// if the model is registered as a `TokenClassification` session. `Ok(None)`
    /// when the model is absent or mis-typed (fail-open).
    pub fn build_pii_classifier(
        registry: &crate::session::OrtSessionRegistry,
        model_key: &str,
    ) -> Result<Option<Arc<dyn PiiSpanDetector>>, OrtError> {
        use crate::config::OnnxTask;
        let Some(config) = registry.config(model_key) else {
            return Ok(None);
        };
        if config.task != OnnxTask::TokenClassification {
            return Ok(None);
        }
        let Some(handle) = registry.ensure_loaded(model_key)? else {
            return Ok(None);
        };
        let classifier = OrtPiiClassifier::from_handle(&handle, &config, model_key)?;
        Ok(Some(Arc::new(classifier)))
    }
}

#[cfg(feature = "onnx")]
pub use ort_pii::{build_pii_classifier, OrtPiiClassifier};

#[cfg(test)]
#[path = "../tests/pii.rs"]
mod tests;
