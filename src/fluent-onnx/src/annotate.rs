//! The trained-encoder annotation rung (ROADMAP_20260828_ORT_FIXES §4 / M2).
//!
//! A finetuned `FillMask` Encoder export carries three per-token heads — UPOS,
//! dependency label, and head position — in the **same graph** as the base
//! `last_hidden_state`. This module turns those head logits into per-token
//! annotations aligned onto a caller-supplied token baseline.
//!
//! The decode + alignment core is **pure and spacy-free**: it consumes plain
//! tensors, the `AnnotationLabels` file, and the caller's `spacy_spans` (byte
//! offsets built from the deterministic tokenizer's orth + idx via
//! [`SpacyTokenAligner::spacy_spans`]). The `OrtAnnotationWorker` (behind the
//! `onnx` feature) is the session glue that produces those tensors; it never
//! imports `spacy-rs`.
//!
//! ## Determinism / quant
//!
//! Sessions run with `intra_op_threads=1` (the config default), so a fixed
//! input is bit-identical run after run. q8 is accepted for annotation (a
//! scoring path, not a binary gate — `Quant::suggested_for(FillMask)` = Q8),
//! with the q8-vs-fp32 drift band recorded on the head outputs in the live
//! test, exactly as the base encoder records its own band.
//!
//! ## Fail-open
//!
//! A spacy token whose LFM subword range is empty (`None` in the returned
//! `Vec<Option<TokenAnnotation>>`) makes the whole rung fall through to the
//! deterministic ArcEager — never a partial mix (ROADMAP §4 alignment).

use std::ops::Range;

use crate::align::SpacyTokenAlignment;
use crate::config::AnnotationLabels;
use crate::error::OrtError;

/// One spacy token's annotation, decoded from the head logits and aligned by
/// byte offset. Plain data — no `spacy-rs` import. `pos`/`dep` are the label
/// strings from the labels file (training order → argmax index).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenAnnotation {
    /// UPOS label (from the labels file's `upos` list).
    pub pos: String,
    /// Dependency label (from the labels file's `dep` list).
    pub dep: String,
    /// Absolute spacy-token index of the predicted head, resolved through the
    /// LFM→spacy alignment. `None` when the head maps to a special token
    /// (`[CLS]`/`[SEP]`) or outside every spacy span — the caller decodes that
    /// as ROOT (`head=0`, `dep="root"`).
    pub head_abs: Option<usize>,
}

/// One LFM token's decoded annotation (before spacy aggregation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LfmAnnotation {
    /// UPOS label (from the labels file's `upos` list).
    pub pos: String,
    /// Dependency label (from the labels file's `dep` list).
    pub dep: String,
    /// Absolute LFM index of the predicted head (argmax over this token's head
    /// row). Whether it is a special token is the caller's concern (via the
    /// alignment's `lfm_to_spacy`).
    pub head_abs: usize,
}

/// The index of the maximum element of a row of logits (the argmax class).
#[must_use]
pub fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bs), (i, &v)| {
            if v > bs {
                (i, v)
            } else {
                (bi, bs)
            }
        })
        .0
}

/// Pure per-LFM-token argmax decode over the three head logit tensors.
///
/// `upos` is `seq * n_pos`, `dep` is `seq * n_dep`, `head` is `seq * seq`
/// (token `i`'s row is a softmax over candidate head positions `j` in the full
/// LFM sequence **including special tokens** — the absolute index, not the
/// spacy index). `seq` is the LFM token count; `n_pos`/`n_dep` come from the
/// labels file. A shape mismatch is a loud error (a mis-exported finetune must
/// never silently decode the wrong axis).
pub fn decode_heads(
    upos: &[f32],
    dep: &[f32],
    head: &[f32],
    labels: &AnnotationLabels,
    seq: usize,
) -> Result<Vec<LfmAnnotation>, OrtError> {
    let n_pos = labels.upos.len();
    let n_dep = labels.dep.len();
    if upos.len() != seq * n_pos {
        return Err(OrtError::Other(format!(
            "upos_logits length {} != seq({seq}) * n_upos({n_pos}) — the export's UPOS head \
             is missing or mis-shaped",
            upos.len()
        )));
    }
    if dep.len() != seq * n_dep {
        return Err(OrtError::Other(format!(
            "dep_logits length {} != seq({seq}) * n_dep({n_dep}) — the export's dep head is \
             missing or mis-shaped",
            dep.len()
        )));
    }
    if head.len() != seq * seq {
        return Err(OrtError::Other(format!(
            "head_logits length {} != seq({seq})^2 — the export's head head is missing or \
             mis-shaped",
            head.len()
        )));
    }
    let mut out = Vec::with_capacity(seq);
    for i in 0..seq {
        let pos_idx = argmax(&upos[i * n_pos..(i + 1) * n_pos]);
        let dep_idx = argmax(&dep[i * n_dep..(i + 1) * n_dep]);
        let head_abs = argmax(&head[i * seq..(i + 1) * seq]);
        out.push(LfmAnnotation {
            pos: labels.upos[pos_idx].clone(),
            dep: labels.dep[dep_idx].clone(),
            head_abs,
        });
    }
    Ok(out)
}

/// Pure argmax-of-mean over a contiguous range of per-token logit rows
/// (ROADMAP §4 alignment: "argmax of the mean logit"). `range` is the LFM
/// subword range covering one spacy token; the logits are averaged across the
/// range's rows and the argmax class index is returned.
#[must_use]
pub fn argmax_of_mean(logits: &[f32], classes: usize, range: Range<usize>) -> usize {
    let mut acc = vec![0.0f32; classes];
    let mut count = 0usize;
    for i in range {
        if (i + 1) * classes > logits.len() {
            break;
        }
        let row = &logits[i * classes..(i + 1) * classes];
        for (a, v) in acc.iter_mut().zip(row.iter()) {
            *a += v;
        }
        count += 1;
    }
    if count > 0 {
        let inv = 1.0 / count as f32;
        for a in &mut acc {
            *a *= inv;
        }
    }
    argmax(&acc)
}

/// Aggregate per-LFM-token head predictions onto the spacy token baseline.
///
/// For each spacy token: pos/dep are the **argmax of the mean logit** over its
/// LFM subword range; the head is the **head row of the first subword**, then
/// resolved through the alignment to an absolute spacy index. A spacy token
/// with an empty LFM range yields `None` — the caller must fall back to
/// ArcEager (never a partial mix). `None` head means the head maps to a
/// special token / outside every spacy span → caller decodes as ROOT.
pub fn aggregate_to_spacy(
    align: &SpacyTokenAlignment,
    upos: &[f32],
    dep: &[f32],
    head: &[f32],
    labels: &AnnotationLabels,
    seq: usize,
) -> Result<Vec<Option<TokenAnnotation>>, OrtError> {
    let lfm = decode_heads(upos, dep, head, labels, seq)?;
    let n_pos = labels.upos.len();
    let n_dep = labels.dep.len();
    let n_spacy = align.spacy_to_lfm.len();
    let mut out = Vec::with_capacity(n_spacy);
    for s in 0..n_spacy {
        let range = align.lfm_range(s);
        if range.is_empty() {
            out.push(None);
            continue;
        }
        let pos_idx = argmax_of_mean(upos, n_pos, range.clone());
        let dep_idx = argmax_of_mean(dep, n_dep, range.clone());
        // Head row of the first subword → absolute LFM head → absolute spacy
        // head (None when the head LFM token maps to nothing / is a special).
        let head_lfm = lfm[range.start].head_abs;
        let head_abs = align.lfm_to_spacy.get(head_lfm).copied().flatten();
        out.push(Some(TokenAnnotation {
            pos: labels.upos[pos_idx].clone(),
            dep: labels.dep[dep_idx].clone(),
            head_abs,
        }));
    }
    Ok(out)
}

#[cfg(feature = "onnx")]
pub mod ort_annotate {
    use std::sync::{Arc, Mutex};

    use crate::align::SpacyTokenAligner;
    use crate::annotate::{aggregate_to_spacy, TokenAnnotation};
    use crate::config::{AnnotationHeads, AnnotationLabels, OnnxConfig, OnnxTask};
    use crate::error::OrtError;
    use crate::session::{OrtSessionRegistry, SessionHandle};
    use crate::tokenizer::LfmTokenizer;

    /// The trained-encoder annotation worker: LFM-tokenize → run the session →
    /// extract `last_hidden_state` + the three head tensors → decode → align
    /// LFM→spacy via [`SpacyTokenAligner`] → one [`TokenAnnotation`] per spacy
    /// token (`None` = empty LFM range → caller falls back).
    ///
    /// Spacy-free by construction: the caller supplies the spacy spans
    /// (byte offsets) and the source text; the worker never imports `spacy-rs`.
    pub struct OrtAnnotationWorker {
        session: Arc<Mutex<ort::session::Session>>,
        tokenizer: Arc<LfmTokenizer>,
        labels: AnnotationLabels,
        heads: AnnotationHeads,
    }

    impl OrtAnnotationWorker {
        /// Build the worker over an already-loaded registry session handle.
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
                OrtError::Other(format!("annotation worker '{model_key}' missing tokenizer_path"))
            })?;
            let tokenizer = LfmTokenizer::from_file(tokenizer_path, config.max_seq_len)?;
            let heads = config.annotation_heads.clone().ok_or_else(|| {
                OrtError::Other(format!(
                    "model `{model_key}` is a FillMask session but declares no annotation_heads"
                ))
            })?;
            let labels = AnnotationLabels::load(&heads.labels)?;
            Ok(Self {
                session,
                tokenizer,
                labels,
                heads,
            })
        }

        /// The closed label vocabularies this worker decodes against.
        #[must_use]
        pub fn labels(&self) -> &AnnotationLabels {
            &self.labels
        }

        /// Annotate `text`, aligning the LFM predictions onto the caller's
        /// `spacy_spans` (byte offsets built from orth + idx via
        /// [`SpacyTokenAligner::spacy_spans`]). The returned vector is aligned
        /// to `spacy_spans`; `None` at index `s` means spacy token `s` had no
        /// covering LFM subwords (caller falls back — never a partial mix).
        pub fn annotate(
            &self,
            text: &str,
            spacy_spans: &[(usize, usize)],
        ) -> Result<Vec<Option<TokenAnnotation>>, OrtError> {
            if text.is_empty() {
                return Ok(spacy_spans.iter().map(|_| None).collect());
            }
            let encoding = self.tokenizer.encode(text)?;
            let seq = encoding.len().max(1);
            let ids: Vec<i64> = encoding.ids.iter().map(|&i| i64::from(i)).collect();
            let mask: Vec<i64> = encoding
                .attention_mask
                .iter()
                .map(|&m| i64::from(m))
                .collect();
            let shape = [1usize, seq];
            let input = ort::value::Tensor::from_array((shape, ids))
                .map_err(|e| OrtError::SessionRun {
                    model: "annotation".to_string(),
                    detail: e.to_string(),
                })?;
            let mask_t = ort::value::Tensor::from_array((shape, mask))
                .map_err(|e| OrtError::SessionRun {
                    model: "annotation".to_string(),
                    detail: e.to_string(),
                })?;

            let mut guard = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = guard
                .run(ort::inputs!["input_ids" => input, "attention_mask" => mask_t])
                .map_err(|e| OrtError::SessionRun {
                    model: "annotation".to_string(),
                    detail: e.to_string(),
                })?;
            let (_, upos) = outputs[self.heads.pos_output.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| OrtError::SessionRun {
                    model: "annotation".to_string(),
                    detail: e.to_string(),
                })?;
            let (_, dep) = outputs[self.heads.dep_output.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| OrtError::SessionRun {
                    model: "annotation".to_string(),
                    detail: e.to_string(),
                })?;
            let (_, head) = outputs[self.heads.head_output.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| OrtError::SessionRun {
                    model: "annotation".to_string(),
                    detail: e.to_string(),
                })?;

            let align = SpacyTokenAligner::align(spacy_spans, &encoding.offsets);
            aggregate_to_spacy(&align, upos, dep, head, &self.labels, seq)
        }
    }

    /// Build the annotation worker from the registry's session for `model_key`,
    /// if the model is registered as a `FillMask` session **with annotation
    /// heads declared**. `Ok(None)` when the model is absent or mis-typed
    /// (fail-open — the caller degrades to the deterministic baseline). Factory
    /// selection mirrors `build_encoder_from_registry`.
    pub fn build_annotation_worker_from_registry(
        registry: &OrtSessionRegistry,
        model_key: &str,
    ) -> Result<Option<Arc<OrtAnnotationWorker>>, OrtError> {
        let Some(config) = registry.config(model_key) else {
            return Ok(None);
        };
        if config.task != OnnxTask::FillMask || config.annotation_heads.is_none() {
            return Ok(None);
        }
        let Some(handle) = registry.ensure_loaded(model_key)? else {
            return Ok(None);
        };
        OrtAnnotationWorker::from_handle(&handle, &config, model_key)
            .map(Arc::new)
            .map(Some)
    }
}

#[cfg(feature = "onnx")]
pub use ort_annotate::{build_annotation_worker_from_registry, OrtAnnotationWorker};

#[cfg(test)]
#[path = "../tests/annotate.rs"]
mod tests;
