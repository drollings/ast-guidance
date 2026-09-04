//! Two-tower zero-shot worker: the shared engine behind the Prompt-Router
//! (cosine head, softmax over route labels) and the Policy-Linter (dot head,
//! sigmoid per token × rule).
//!
//! The two towers (`token_proj`, `rule_proj`) are affine maps over the shared
//! encoder, so **pooling after projecting is exact** (an affine map commutes
//! with a mean) — the graph stays static and the host pools the projected
//! per-token vectors over label/text regions. Regions are recovered from the
//! tokenizer's **byte offsets** against the byte-exact prompt assembled by
//! [`PromptBuilder`] (ROADMAP_20260827_ORT §2.1; the character arithmetic the
//! model was trained on depends on that prompt byte for byte — risk #5).
//!
//! The pooling / scoring math is pure and unit-tested with no model; the
//! session-facing glue lives behind the `onnx` feature.

use std::path::Path;
use common_core::vector_math::cosine_similarity_f32;

#[cfg(feature = "onnx")]
use std::sync::{Arc, Mutex};

use crate::config::OnnxConfig;
use crate::error::OrtError;
#[cfg(feature = "onnx")]
use crate::session::SessionHandle;
#[cfg(feature = "onnx")]
use crate::tokenizer::{LfmEncoding, LfmTokenizer};

/// Version-pinned prompt contract (see the Prompt-Router / Policy-Linter
/// READMEs). Bumping this forces a re-review of the label-region arithmetic
/// and the golden tests.
pub const TWO_TOWER_PROMPT_VERSION: &str = "v1";

/// The head the two-tower graph was trained with, read from the model's
/// `config.json` `head` block (never hardcoded).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TwoTowerHead {
    /// `cosine`: L2-normalise both pooled towers, dot, scale + bias, softmax
    /// across labels (Prompt-Router).
    Cosine { scale: f64, bias: f64 },
    /// `dot`: dot product (no normalisation), scale + bias, sigmoid per
    /// (token, rule) pair (Policy-Linter).
    Dot { scale: f64, bias: f64 },
}

impl TwoTowerHead {
    /// Read and parse the `head` block from the model's `config.json`.
    pub fn from_config_json(config: &OnnxConfig) -> Result<Self, OrtError> {
        let path = config.config_json_path().ok_or_else(|| {
            OrtError::Other(
                "two-tower task requires a config.json with a `head` block; none found next to \
                 the model path"
                    .into(),
            )
        })?;
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| OrtError::ConfigRead {
                path: path.display().to_string(),
                detail: e.to_string(),
            })?;
        let head: RawConfigHead = serde_json::from_str(&raw).map_err(|e| OrtError::ConfigParse {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        let head = head.head.ok_or_else(|| OrtError::ConfigParse {
            path: path.display().to_string(),
            detail: "config.json has no `head` block".to_string(),
        })?;
        Self::from_raw(&head, &path)
    }

    fn from_raw(raw: &RawHead, path: &Path) -> Result<Self, OrtError> {
        let detail = |field: &str| {
            OrtError::ConfigParse {
                path: path.display().to_string(),
                detail: format!("two-tower `head` block missing required field `{field}`"),
            }
        };
        let head = match raw.kind.as_str() {
            "cosine" => TwoTowerHead::Cosine {
                scale: raw.scale.ok_or_else(|| detail("scale"))?,
                bias: raw.bias.unwrap_or(0.0),
            },
            "dot" => TwoTowerHead::Dot {
                scale: raw.scale.ok_or_else(|| detail("scale"))?,
                bias: raw.bias.unwrap_or(0.0),
            },
            other => {
                return Err(OrtError::ConfigParse {
                    path: path.display().to_string(),
                    detail: format!("unknown two-tower head kind `{other}` (expected cosine|dot)"),
                })
            }
        };
        // The exports declare a consistency between the head kind, its
        // `normalize` flag, and its activation. Surface a loud warning when the
        // declaration contradicts the math this crate implements so a
        // mis-declared head is never silently served.
        let (expect_normalize, expect_activation) = match head {
            TwoTowerHead::Cosine { .. } => (true, "softmax"),
            TwoTowerHead::Dot { .. } => (false, "sigmoid"),
        };
        if raw.normalize != expect_normalize
            || (!raw.activation.is_empty() && raw.activation != expect_activation)
        {
            tracing::warn!(
                target: "fluent-onnx",
                path = %path.display(),
                kind = %raw.kind,
                normalize = raw.normalize,
                activation = %raw.activation,
                "two-tower head declaration contradicts the implemented {expect_activation} math",
            );
        }
        Ok(head)
    }
}

/// The raw `head` block from `config.json` (serde view over the exact keys the
/// exports ship). `prefix_heading`/`proj_dim` are consumed by the onnx-gated
/// `TwoTowerWorker`; with the feature off they are deliberately unread.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[allow(dead_code)]
struct RawHead {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    normalize: bool,
    #[serde(default)]
    scale: Option<f64>,
    #[serde(default)]
    bias: Option<f64>,
    #[serde(default)]
    activation: String,
    #[serde(default)]
    prefix_heading: String,
    #[serde(default)]
    proj_dim: Option<u32>,
}

/// The `config.json` wrapper that carries the nested `head` block.
#[derive(Debug, Clone, Default, serde::Deserialize)]
struct RawConfigHead {
    #[serde(default)]
    head: Option<RawHead>,
}

impl RawHead {
    /// The heading line (`"Categories"` / `"Policy"`), defaulting to
    /// `"Categories"` (the Prompt-Router contract) when absent.
    #[cfg(feature = "onnx")]
    fn prefix_heading(&self) -> String {
        if self.prefix_heading.is_empty() {
            "Categories".into()
        } else {
            self.prefix_heading.clone()
        }
    }

    /// The projection dimension of both towers (default 256 for both exports).
    #[cfg(feature = "onnx")]
    fn proj_dim(&self) -> usize {
        self.proj_dim.map_or(256, |d| d as usize)
    }
}

/// The assembled byte-exact prompt plus the byte regions the pooling math
/// needs: each label line's text and the `Text:` region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoTowerPrompt {
    /// The full assembled prompt (the README contract, byte for byte).
    pub prompt: String,
    /// `(start, end)` byte ranges of each label's text within `prompt`.
    pub label_regions: Vec<(usize, usize)>,
    /// `(start, end)` byte range of the `Text:` region's text within `prompt`.
    pub text_region: (usize, usize),
}

/// Assembles the byte-exact two-tower prompt and, from its fixed structure,
/// the byte regions the host-side pooling needs.
///
/// ```text
/// {prefix_heading}:
/// - label one
/// - label two
///
/// Text:
/// <text>
/// ```
pub struct PromptBuilder {
    prefix_heading: String,
}

impl PromptBuilder {
    /// A builder for the given heading line (`"Categories"` or `"Policy"`).
    pub fn new(prefix_heading: impl Into<String>) -> Self {
        Self {
            prefix_heading: prefix_heading.into(),
        }
    }

    /// The full assembled prompt (used by the golden exact-string tests).
    pub fn build(&self, labels: &[String], text: &str) -> String {
        self.assemble(labels, text).prompt
    }

    /// Assemble the prompt and record the label/text byte regions.
    pub fn assemble(&self, labels: &[String], text: &str) -> TwoTowerPrompt {
        let mut prompt = String::new();
        let mut label_regions = Vec::with_capacity(labels.len());
        prompt.push_str(&self.prefix_heading);
        prompt.push(':');
        prompt.push('\n');
        for label in labels {
            prompt.push_str("- ");
            let start = prompt.len();
            prompt.push_str(label);
            let end = prompt.len();
            label_regions.push((start, end));
            prompt.push('\n');
        }
        prompt.push('\n');
        prompt.push_str("Text:\n");
        let text_start = prompt.len();
        prompt.push_str(text);
        let text_region = (text_start, prompt.len());
        TwoTowerPrompt {
            prompt,
            label_regions,
            text_region,
        }
    }
}

/// L2-normalise a vector in place-free form; a zero vector stays zero (so the
/// cosine below never hits 0/0).
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f64 = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
    if norm <= 0.0 {
        return vec![0.0; v.len()];
    }
    let inv = 1.0 / norm as f32;
    v.iter().map(|&x| x * inv).collect()
}

/// Numerically-stable softmax across the input logits.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Logistic sigmoid.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Mean-pool the rows of a `[seq * dims]` flat tensor over the token rows that
/// overlap each `(start, end)` byte region, given the tokenizer's per-token
/// byte offsets. A token overlaps a region when `t_start < r_end && t_end >
/// r_start`; zero-width special tokens (`(0,0)` / `(len,len)`) never match.
/// A region with no covering tokens yields a zero vector.
pub fn pool_regions(
    flat: &[f32],
    seq: usize,
    dims: usize,
    offsets: &[(usize, usize)],
    regions: &[(usize, usize)],
) -> Vec<Vec<f32>> {
    debug_assert_eq!(flat.len(), seq * dims, "flat must be seq*dims");
    debug_assert_eq!(offsets.len(), seq, "offsets must equal seq");
    let mut out = Vec::with_capacity(regions.len());
    for (rs, re) in regions {
        let mut pooled = vec![0.0f32; dims];
        let mut count = 0usize;
        for i in 0..seq {
            let (ts, te) = offsets[i];
            if ts < *re && te > *rs {
                let row = &flat[i * dims..(i + 1) * dims];
                for (p, v) in pooled.iter_mut().zip(row.iter()) {
                    *p += v;
                }
                count += 1;
            }
        }
        if count > 0 {
            let inv = 1.0 / count as f32;
            for p in &mut pooled {
                *p *= inv;
            }
        }
        out.push(pooled);
    }
    out
}

/// Apply the head to a pooled query vector against pooled rule vectors,
/// returning one score per rule.
pub fn score_with_head(
    head: &TwoTowerHead,
    query: &[f32],
    rules: &[Vec<f32>],
) -> Vec<f64> {
    match head {
        TwoTowerHead::Cosine { scale, bias } => {
            let q = l2_normalize(query);
            // `cosine_similarity_f32` is the single canonical similarity
            // (`common-core::vector_math`); the f32→f64 widening here is
            // lossless and keeps the softmax logits in f64, where the
            // exp/sum precision genuinely matters.
            let logits: Vec<f64> = rules
                .iter()
                .map(|r| f64::from(cosine_similarity_f32(&q, &l2_normalize(r))) * *scale + *bias)
                .collect();
            softmax(&logits)
        }
        TwoTowerHead::Dot { scale, bias } => {
            let dot = |r: &[f32]| {
                query
                    .iter()
                    .zip(r)
                    .map(|(&x, &y)| f64::from(x) * f64::from(y))
                    .sum::<f64>()
            };
            rules.iter().map(|r| sigmoid(dot(r) * scale + bias)).collect()
        }
    }
}

/// Per-token × per-rule scores (the Policy-Linter op: dot + scale + bias +
/// sigmoid over the text-region tokens against each pooled rule vector).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenScoreMatrix {
    /// The text-region token surfaces, in order.
    pub tokens: Vec<String>,
    /// Per-token byte offsets **into the original `text`** (not the assembled
    /// prompt — the text region's prompt offset is subtracted), so a consumer
    /// (e.g. the Policy-Linter) can produce char-aligned spans.
    pub offsets: Vec<(usize, usize)>,
    /// `scores[t][r]` — text token `t` against rule `r`.
    pub scores: Vec<Vec<f64>>,
}

/// The `TwoTowerWorker` — a shared two-tower session plus its tokenizer and
/// head. Behind the `onnx` feature.
#[cfg(feature = "onnx")]
pub struct TwoTowerWorker {
    session: Arc<Mutex<ort::session::Session>>,
    tokenizer: Arc<LfmTokenizer>,
    head: TwoTowerHead,
    prefix_heading: String,
    proj_dim: usize,
    model_key: String,
}

#[cfg(feature = "onnx")]
impl TwoTowerWorker {
    /// Build a worker over an already-loaded registry session handle.
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
            OrtError::Other(format!("two-tower worker '{model_key}' missing tokenizer_path"))
        })?;
        let tokenizer = LfmTokenizer::from_file(tokenizer_path, config.max_seq_len)?;
        let head = TwoTowerHead::from_config_json(config)?;
        let raw_head = read_raw_head(config)?;
        let prefix_heading = raw_head.prefix_heading();
        let proj_dim = raw_head.proj_dim();
        Ok(Self {
            session,
            tokenizer,
            head,
            prefix_heading,
            proj_dim,
            model_key: model_key.to_string(),
        })
    }

    /// One forward pass over the assembled prompt, returning `token_proj` and
    /// `rule_proj` as flat `seq * proj_dim` tensors.
    fn run_towers(&self, prompt: &str) -> Result<(LfmEncoding, Vec<f32>, Vec<f32>), OrtError> {
        let encoding = self.tokenizer.encode(prompt)?;
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
                model: self.model_key.clone(),
                detail: e.to_string(),
            })?;
        let mask_t = ort::value::Tensor::from_array((shape, mask))
            .map_err(|e| OrtError::SessionRun {
                model: self.model_key.clone(),
                detail: e.to_string(),
            })?;
        let mut guard = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let outputs = guard
            .run(ort::inputs!["input_ids" => input, "attention_mask" => mask_t])
            .map_err(|e| OrtError::SessionRun {
                model: self.model_key.clone(),
                detail: e.to_string(),
            })?;
        let extract = |name: &str| -> Result<Vec<f32>, OrtError> {
            let (_, tensor) = outputs[name].try_extract_tensor::<f32>().map_err(|e| {
                OrtError::SessionRun {
                    model: self.model_key.clone(),
                    detail: format!("extract {name}: {e}"),
                }
            })?;
            Ok(tensor.to_vec())
        };
        let token_proj = extract("token_proj")?;
        let rule_proj = extract("rule_proj")?;
        Ok((encoding, token_proj, rule_proj))
    }

    /// Score `text` against `labels` (route descriptions). Returns one score
    /// per label through the configured head (cosine/softmax or dot/sigmoid).
    pub fn score_labels(&self, text: &str, labels: &[String]) -> Result<Vec<f64>, OrtError> {
        if labels.is_empty() {
            return Ok(Vec::new());
        }
        let assembled = PromptBuilder::new(&self.prefix_heading).assemble(labels, text);
        let (encoding, token_proj, rule_proj) = self.run_towers(&assembled.prompt)?;
        let seq = encoding.len();
        let dims = self.proj_dim;
        let rule_vecs = pool_regions(
            &rule_proj,
            seq,
            dims,
            &encoding.offsets,
            &assembled.label_regions,
        );
        let query_vecs = pool_regions(
            &token_proj,
            seq,
            dims,
            &encoding.offsets,
            std::slice::from_ref(&assembled.text_region),
        );
        let query = query_vecs.into_iter().next().unwrap_or_default();
        Ok(score_with_head(&self.head, &query, &rule_vecs))
    }

    /// Per-text-token × per-rule sigmoid scores (the Policy-Linter op).
    pub fn token_scores(&self, text: &str, rules: &[String]) -> Result<TokenScoreMatrix, OrtError> {
        let assembled = PromptBuilder::new(&self.prefix_heading).assemble(rules, text);
        let (encoding, token_proj, rule_proj) = self.run_towers(&assembled.prompt)?;
        let seq = encoding.len();
        let dims = self.proj_dim;
        let rule_vecs = pool_regions(
            &rule_proj,
            seq,
            dims,
            &encoding.offsets,
            &assembled.label_regions,
        );
        let (ts, te) = assembled.text_region;
        let mut tokens = Vec::new();
        let mut offsets = Vec::new();
        let mut scores = Vec::new();
        for i in 0..seq {
            let (start, end) = encoding.offsets[i];
            if start < te && end > ts {
                let row = &token_proj[i * dims..(i + 1) * dims];
                let row_scores: Vec<f64> = rule_vecs
                    .iter()
                    .map(|r| {
                        let dot = row
                            .iter()
                            .zip(r)
                            .map(|(&x, &y)| f64::from(x) * f64::from(y))
                            .sum::<f64>();
                        let (scale, bias) = match self.head {
                            TwoTowerHead::Dot { scale, bias }
                            | TwoTowerHead::Cosine { scale, bias } => (scale, bias),
                        };
                        sigmoid(dot * scale + bias)
                    })
                    .collect();
                let surface = if start < end {
                    assembled.prompt[start.min(assembled.prompt.len())
                        ..end.min(assembled.prompt.len())]
                        .to_string()
                } else {
                    String::new()
                };
                tokens.push(surface);
                // Rebase the prompt offsets onto the original text.
                offsets.push((start.saturating_sub(ts), end.saturating_sub(ts)));
                scores.push(row_scores);
            }
        }
        Ok(TokenScoreMatrix {
            tokens,
            offsets,
            scores,
        })
    }

    /// The configured head (for diagnostics and M3 policy-label wiring).
    pub fn head(&self) -> &TwoTowerHead {
        &self.head
    }
}

#[cfg(feature = "onnx")]
fn read_raw_head(config: &OnnxConfig) -> Result<RawHead, OrtError> {
    let path = config.config_json_path().ok_or_else(|| {
        OrtError::Other(
            "two-tower task requires a config.json with a `head` block; none found next to \
             the model path"
                .into(),
        )
    })?;
    let raw = std::fs::read_to_string(&path).map_err(|e| OrtError::ConfigRead {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let head: RawConfigHead = serde_json::from_str(&raw).map_err(|e| OrtError::ConfigParse {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    head.head.ok_or_else(|| OrtError::ConfigParse {
        path: path.display().to_string(),
        detail: "config.json has no `head` block".to_string(),
    })
}

/// Build a `TwoTowerWorker` from the registry's session for `model_key`, if the
/// model is registered and its task is a two-tower task.
#[cfg(feature = "onnx")]
pub fn build_two_tower_from_registry(
    registry: &crate::session::OrtSessionRegistry,
    model_key: &str,
) -> Result<Option<Arc<TwoTowerWorker>>, OrtError> {
    use crate::config::OnnxTask;
    let Some(config) = registry.config(model_key) else {
        return Ok(None);
    };
    if config.task != OnnxTask::ZeroShotRouting && config.task != OnnxTask::ZeroShotTokenMatching
    {
        return Ok(None);
    }
    let Some(handle) = registry.ensure_loaded(model_key)? else {
        return Ok(None);
    };
    let worker = TwoTowerWorker::from_handle(&handle, &config, model_key)?;
    Ok(Some(Arc::new(worker)))
}

/// A single Policy-Linter hit: a text token whose score against a policy rule
/// cleared the threshold. `start`/`end` are byte offsets into the linter's
/// input text (ROADMAP_20260827_ORT §3.1).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PolicyHit {
    /// The token surface.
    pub token: String,
    /// Byte offset of the token's start within the linter input.
    pub start: usize,
    /// Byte offset of the token's end within the linter input.
    pub end: usize,
    /// The policy rule this token matched.
    pub label: String,
    /// The sigmoid score (0..1) of the token against the rule.
    pub score: f64,
}

impl PolicyHit {
    /// Slice the hit's text out of `source` (byte offsets are valid for
    /// `&source[start..end]`). `None` when the offsets are out of bounds.
    #[must_use]
    pub fn slice<'a>(&self, source: &'a str) -> Option<&'a str> {
        source.get(self.start..self.end)
    }
}

/// Pure hit extraction from a [`TokenScoreMatrix`] and its rule labels: every
/// (token, rule) pair whose score clears `threshold`, ordered token-major.
/// Unit-testable without a model (the Policy-Linter's deterministic half).
#[must_use]
pub fn policy_hits_from_matrix(
    matrix: &TokenScoreMatrix,
    labels: &[String],
    threshold: f64,
) -> Vec<PolicyHit> {
    let mut hits = Vec::new();
    for (ti, row) in matrix.scores.iter().enumerate() {
        for (ri, &score) in row.iter().enumerate() {
            if score >= threshold {
                let (start, end) = matrix.offsets.get(ti).copied().unwrap_or((0, 0));
                hits.push(PolicyHit {
                    token: matrix.tokens.get(ti).cloned().unwrap_or_default(),
                    start,
                    end,
                    label: labels.get(ri).cloned().unwrap_or_else(|| format!("rule-{ri}")),
                    score,
                });
            }
        }
    }
    hits
}

/// The Policy-Linter (ROADMAP_20260827_ORT §3.1): a [`TwoTowerWorker`]
/// (dot/sigmoid head) scoring every text token against the config's policy
/// labels, emitting the tokens whose score clears the configured threshold.
/// The label list is a data artifact (`OnnxConfig.policy_labels`, a JSON array
/// of strings — typically the config's blacklist/`RejectPatterns`
/// descriptions), never a code constant. Default quantization `fp32` (the
/// binary-gate rule); q8 only behind a golden-gated threshold.
#[cfg(feature = "onnx")]
pub struct PolicyLinter {
    worker: Arc<TwoTowerWorker>,
    labels: Vec<String>,
    threshold: f64,
}

#[cfg(feature = "onnx")]
impl PolicyLinter {
    /// A linter over an existing worker, its rule labels, and the flag
    /// threshold (a score at/above it is a hit).
    #[must_use]
    pub fn new(worker: Arc<TwoTowerWorker>, labels: Vec<String>, threshold: f64) -> Self {
        Self {
            worker,
            labels,
            threshold,
        }
    }

    /// The rule labels this linter scores against.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The flag threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Lint `text`: score every text token against every rule and return the
    /// hits. Empty when no label cleared the threshold.
    pub fn lint(&self, text: &str) -> Result<Vec<PolicyHit>, OrtError> {
        if self.labels.is_empty() {
            return Ok(Vec::new());
        }
        let matrix = self.worker.token_scores(text, &self.labels)?;
        Ok(policy_hits_from_matrix(&matrix, &self.labels, self.threshold))
    }
}

/// Load the Policy-Linter's rule labels from a JSON file (a JSON array of
/// strings, or an object of `"route"/"description"` pairs whose `description`
/// values are the rules — either shape is accepted).
pub fn load_policy_labels(path: &Path) -> Result<Vec<String>, OrtError> {
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
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| OrtError::ConfigParse {
                        path: path.display().to_string(),
                        detail: "policy_labels must be a JSON array of strings".to_string(),
                    })
            })
            .collect(),
        serde_json::Value::Object(map) => {
            let mut labels = Vec::with_capacity(map.len());
            for (_, v) in map {
                if let Some(s) = v.as_str() {
                    labels.push(s.to_string());
                }
            }
            if labels.is_empty() {
                return Err(OrtError::ConfigParse {
                    path: path.display().to_string(),
                    detail: "policy_labels object yielded no string values".to_string(),
                });
            }
            Ok(labels)
        }
        _ => Err(OrtError::ConfigParse {
            path: path.display().to_string(),
            detail: "policy_labels must be a JSON array of strings".to_string(),
        }),
    }
}

/// Build a `PolicyLinter` from the registry's session for `model_key`, if the
/// model is registered as a `ZeroShotTokenMatching` two-tower session and a
/// `policy_labels` file is configured. `Ok(None)` when the model is absent,
/// mis-typed, or has no `policy_labels` source (fail-open).
#[cfg(feature = "onnx")]
pub fn build_policy_linter_from_registry(
    registry: &crate::session::OrtSessionRegistry,
    model_key: &str,
    threshold: f64,
) -> Result<Option<Arc<PolicyLinter>>, OrtError> {
    use crate::config::OnnxTask;
    let Some(config) = registry.config(model_key) else {
        return Ok(None);
    };
    if config.task != OnnxTask::ZeroShotTokenMatching {
        return Ok(None);
    }
    let Some(labels_path) = config.policy_labels.as_ref() else {
        return Ok(None);
    };
    let Some(handle) = registry.ensure_loaded(model_key)? else {
        return Ok(None);
    };
    let worker = TwoTowerWorker::from_handle(&handle, &config, model_key)?;
    let labels = load_policy_labels(labels_path)?;
    Ok(Some(Arc::new(PolicyLinter::new(Arc::new(worker), labels, threshold))))
}

#[cfg(test)]
#[path = "../tests/two_tower.rs"]
mod tests;
