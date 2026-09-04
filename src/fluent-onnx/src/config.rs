//! ONNX model configuration types — pure, ort-free.
//!
//! This module is the serialization + validation surface of `fluent-onnx`. It
//! depends on nothing ONNX-specific, so the crate compiles and its hermetic
//! config tests run with the `onnx` feature off.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::error::OrtError;

/// The task a declared ONNX model performs. Drives registry typing, accessor
/// selection, and the boot sanity-check against the model's `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnnxTask {
    /// Base bidirectional encoder: `input_ids` + `attention_mask` →
    /// `last_hidden_state` (sentence embeddings via mean-pool).
    #[default]
    FillMask,
    /// Two-tower zero-shot router (Prompt-Router): softmax over route labels.
    ZeroShotRouting,
    /// Two-tower per-token × per-rule scoring (Policy-Linter): sigmoid.
    ZeroShotTokenMatching,
    /// Token classification → BILUO spans (PII-Detector).
    TokenClassification,
    /// Late-interaction retriever (ColBERT): per-token vectors, MaxSim.
    LateInteraction,
    /// Masked-diffusion draft loop (research spike, not a resident worker).
    MaskedDiffusion,
    /// Causal language modeling (autoregressive decode): `input_ids` +
    /// `attention_mask` + position/KV inputs → `logits` over the vocab. This
    /// is the generative path (ROADMAP M1): the `chat_template.jinja` applied
    /// before tokenization, a prefill pass, then a per-token decode loop with
    /// grammar-constrained sampling. It is the one non-encoder task — the
    /// fleet's general-purpose local LLM behind the `ChatBackend` seam.
    CausalLm,
}

impl OnnxTask {
    /// Architecture families in the model's `config.json` (`architectures`)
    /// that a session of this task may legitimately declare.
    pub fn expected_architecture_families(self) -> &'static [&'static str] {
        match self {
            OnnxTask::FillMask | OnnxTask::MaskedDiffusion => &["MaskedLmGraph"],
            OnnxTask::TokenClassification => {
                &["Lfm2BidirP2ForTokenClassification", "MaskedLmGraph"]
            }
            OnnxTask::ZeroShotRouting | OnnxTask::ZeroShotTokenMatching => &["TwoTowerGraph"],
            // sentence-transformers exports carry no `architectures` key.
            OnnxTask::LateInteraction => &["token_embeddings"],
            OnnxTask::CausalLm => &["Lfm2ForCausalLM"],
        }
    }

    /// Output names the model's `config.json` `onnx.outputs` must include for
    /// this task. An empty slice means "no output-name requirement" (the
    /// artifact does not declare an `onnx` block, or the task is unconstrained).
    pub fn required_outputs(self) -> &'static [&'static str] {
        match self {
            OnnxTask::FillMask | OnnxTask::MaskedDiffusion => &["last_hidden_state"],
            OnnxTask::TokenClassification | OnnxTask::CausalLm => &["logits"],
            OnnxTask::ZeroShotRouting | OnnxTask::ZeroShotTokenMatching => {
                &["token_proj", "rule_proj"]
            }
            OnnxTask::LateInteraction => &["token_embeddings"],
        }
    }
}

/// Quantization of an ONNX model. `Fp32` is the reference; `Fp16` is known
/// broken for this architecture (RMSNorm overflow) and must be validated
/// per-model against its fp32 reference before use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quant {
    Fp32,
    Fp16,
    Fp8,
    Q8,
    Q4,
}

impl Quant {
    /// The quantization the roadmap recommends for a task
    /// (ROADMAP_20260827_ORT §1.6): `q8` where available, but **`fp32` for
    /// binary gates** — the PII-Detector and the Policy-Linter make binary
    /// decisions (span/no-span, flag/no-flag), and the README-documented q8
    /// flip rates (PII spans, 3/6 Policy-Linter threshold flips) are
    /// unacceptable for a gate unless a golden-gated threshold is configured
    /// with the flip rate recorded. This is a *suggestion* the operator
    /// overrides; `OnnxConfig::validate` warns when a gate is declared
    /// non-fp32 so a mis-quantized gate is never silent.
    #[must_use]
    pub fn suggested_for(task: OnnxTask) -> Self {
        match task {
            OnnxTask::TokenClassification | OnnxTask::ZeroShotTokenMatching => Quant::Fp32,
            _ => Quant::Q8,
        }
    }

    /// Whether this quantization is the fp32 reference.
    #[must_use]
    pub fn is_fp32(self) -> bool {
        matches!(self, Self::Fp32)
    }
}

fn default_pos_output() -> String {
    "upos_logits".into()
}

fn default_dep_output() -> String {
    "dep_logits".into()
}

fn default_head_output() -> String {
    "head_logits".into()
}

fn default_io_input_ids() -> String {
    "input_ids".into()
}

fn default_io_attention_mask() -> String {
    "attention_mask".into()
}

fn default_io_position_ids() -> String {
    "position_ids".into()
}

fn default_io_past_key_values() -> String {
    "past_key_values".into()
}

fn default_io_logits() -> String {
    "logits".into()
}

fn default_io_kv_cache_dtype() -> String {
    "float32".into()
}

fn default_io_conv_state() -> String {
    "past_conv".into()
}

fn default_io_present() -> String {
    "present".into()
}

fn default_io_present_conv() -> String {
    "present_conv".into()
}

fn default_io_num_key_value_heads() -> usize {
    8
}

fn default_io_head_dim() -> usize {
    64
}

fn default_io_hidden_size() -> usize {
    2048
}

fn default_io_conv_l_cache() -> usize {
    3
}

fn default_io_attention_layers() -> Vec<usize> {
    vec![2, 5, 9, 13, 17, 21, 24, 27]
}

fn default_io_conv_layers() -> Vec<usize> {
    vec![
        0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 25, 26, 28, 29,
    ]
}

/// The generative `CausalLm` IO contract (ROADMAP §5 / M1): the input/output
/// tensor names, KV-cache dtype, and the per-layer conv/KV structure of the
/// transformers.js-style causal-LM export, so the decoder knows exactly which
/// graph inputs to build and which output to read — one place to edit, one
/// place to fail loudly.
///
/// The defaults match the LFM2.5-2.6B export (`Lfm2ForCausalLM`): prefill via
/// `input_ids`/`attention_mask`/`position_ids`, a per-layer conv state
/// (`past_conv.{L}`, shape `[1, hidden_size, conv_L_cache]`) for the 22 conv
/// layers and a per-layer KV cache (`past_key_values.{L}.key/value`, shape
/// `[1, num_key_value_heads, past_len, head_dim]`) for the 8 full-attention
/// layers, the `logits` output, and `kv_cache_dtype` from the model's
/// `transformers.js_config`. The M1 live probe discovered this contract
/// empirically and committed it as the golden fixture
/// (`tests/live/fixtures/lfm25_26b_io.json`) — the fixture is the single
/// source of truth, and these defaults align with it. When a config carries a
/// `None` `LlmIo`, the decoder introspects the graph at load time and derives
/// the contract from the declared inputs; a provided `Some(LlmIo)` is
/// validated against the graph, and a mismatch is a loud load error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Builder)]
#[builder(start_fn = new)]
pub struct LlmIo {
    /// Prefill + decode token-id input tensor (default `"input_ids"`).
    #[serde(default = "default_io_input_ids")]
    #[builder(default = default_io_input_ids())]
    pub input_ids: String,
    /// Attention-mask input tensor (default `"attention_mask"`).
    #[serde(default = "default_io_attention_mask")]
    #[builder(default = default_io_attention_mask())]
    pub attention_mask: String,
    /// Position-id input tensor (default `"position_ids"`).
    #[serde(default = "default_io_position_ids")]
    #[builder(default = default_io_position_ids())]
    pub position_ids: String,
    /// Prefix of the full-attention KV inputs — `{past_key_values}.{L}.key` /
    /// `.{L}.value` (default `"past_key_values"`).
    #[serde(default = "default_io_past_key_values")]
    #[builder(default = default_io_past_key_values())]
    pub past_key_values: String,
    /// Prefix of the conv-state inputs — `{conv_state}.{L}` (default
    /// `"past_conv"`).
    #[serde(default = "default_io_conv_state")]
    #[builder(default = default_io_conv_state())]
    pub conv_state: String,
    /// Prefix of the full-attention KV outputs — `{present}.{L}.key` /
    /// `.{L}.value` (default `"present"`).
    #[serde(default = "default_io_present")]
    #[builder(default = default_io_present())]
    pub present: String,
    /// Prefix of the conv-state outputs — `{present_conv}.{L}` (default
    /// `"present_conv"`).
    #[serde(default = "default_io_present_conv")]
    #[builder(default = default_io_present_conv())]
    pub present_conv: String,
    /// Output tensor holding the next-token logits (default `"logits"`).
    #[serde(default = "default_io_logits")]
    #[builder(default = default_io_logits())]
    pub logits: String,
    /// KV-cache dtype as a tensor-element string (default `"float32"`, matching
    /// `transformers.js_config.kv_cache_dtype`).
    #[serde(default = "default_io_kv_cache_dtype")]
    #[builder(default = default_io_kv_cache_dtype())]
    pub kv_cache_dtype: String,
    /// Number of KV-heads (default `8`).
    #[serde(default = "default_io_num_key_value_heads")]
    #[builder(default = default_io_num_key_value_heads())]
    pub num_key_value_heads: usize,
    /// Attention head dimension (default `64`).
    #[serde(default = "default_io_head_dim")]
    #[builder(default = default_io_head_dim())]
    pub head_dim: usize,
    /// Hidden / conv-state feature dimension (default `2048`).
    #[serde(default = "default_io_hidden_size")]
    #[builder(default = default_io_hidden_size())]
    pub hidden_size: usize,
    /// Conv-state window length (default `3`, `conv_L_cache`).
    #[serde(default = "default_io_conv_l_cache")]
    #[builder(default = default_io_conv_l_cache())]
    pub conv_l_cache: usize,
    /// The full-attention layers (KV-cached), in ascending order (default the
    /// 8 `full_attention` layers of the LFM2.5-2.6B export).
    #[serde(default = "default_io_attention_layers")]
    #[builder(default = default_io_attention_layers())]
    pub attention_layers: Vec<usize>,
    /// The conv layers (conv-state), in ascending order (default the 22 `conv`
    /// layers of the LFM2.5-2.6B export).
    #[serde(default = "default_io_conv_layers")]
    #[builder(default = default_io_conv_layers())]
    pub conv_layers: Vec<usize>,
}

/// The trained-encoder annotation-heads contract (ROADMAP_20260828_ORT_FIXES §4).
///
/// A finetuned `FillMask` Encoder export carries three per-token heads — UPOS,
/// dependency label, and head position — in the **same graph** as the base
/// `last_hidden_state`. Each head's ONNX output tensor name is configurable
/// (bon defaults `"upos_logits"` / `"dep_logits"` / `"head_logits"`), and the
/// closed label vocabularies live in a separate `labels` file
/// (`annotation_labels.json`: `{"upos": [...], "dep": [...]}`) — the single
/// source of truth for logit-index → label, so `fluent-onnx` never imports
/// `spacy-rs`; the labels cross as data. `OnnxConfig::validate` requires the
/// declared head output names to appear in the model's `config.json`
/// `onnx.outputs` and loads/parses the labels file, so a mis-exported finetune
/// fails boot loudly rather than silently serving the base encoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Builder)]
#[builder(start_fn = new)]
pub struct AnnotationHeads {
    /// ONNX output tensor for per-token UPOS logits (default `"upos_logits"`).
    #[serde(default = "default_pos_output")]
    #[builder(default = default_pos_output())]
    pub pos_output: String,
    /// ONNX output tensor for per-token dependency-label logits (default `"dep_logits"`).
    #[serde(default = "default_dep_output")]
    #[builder(default = default_dep_output())]
    pub dep_output: String,
    /// ONNX output tensor for per-token head-position logits (default `"head_logits"`).
    #[serde(default = "default_head_output")]
    #[builder(default = default_head_output())]
    pub head_output: String,
    /// Path to the labels file (`annotation_labels.json`): an object with
    /// `"upos"` and `"dep"` arrays of labels in training order.
    #[builder(into)]
    #[serde(alias = "tokenizer_path")]
    pub labels: PathBuf,
}

/// The closed label vocabularies for the annotation heads, loaded from the
/// `AnnotationHeads::labels` file. `upos` is the 17-label `Upos` set
/// (`src/spacy-rs/src/labels.rs`) in training order; `dep` is the
/// `DepLabelSet` union the finetune was trained on. Pure serde data — it is
/// the single source of truth for logit-index → label, and `fluent-onnx` never
/// imports `spacy-rs`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnnotationLabels {
    /// UPOS labels in training order (`N_UPOS` = this list's length).
    pub upos: Vec<String>,
    /// Dependency labels in training order (`N_DEP` = this list's length).
    pub dep: Vec<String>,
}

impl AnnotationLabels {
    /// Load and parse the labels file from disk.
    pub fn load(path: &Path) -> Result<Self, OrtError> {
        let raw = std::fs::read_to_string(path).map_err(|e| OrtError::ConfigRead {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        let parsed: AnnotationLabels =
            serde_json::from_str(&raw).map_err(|e| OrtError::ConfigParse {
                path: path.display().to_string(),
                detail: e.to_string(),
            })?;
        if parsed.upos.is_empty() || parsed.dep.is_empty() {
            return Err(OrtError::ConfigParse {
                path: path.display().to_string(),
                detail: "labels file must declare non-empty `upos` and `dep` arrays".to_string(),
            });
        }
        Ok(parsed)
    }
}

/// Whether a declared onnx model is resident for the process lifetime.
///
/// `Always` models load at boot into the `OrtSessionRegistry` and are never
/// evicted or unloaded (`/models/unload` refuses them). `Unloadable` models
/// load on first use (`ensure_loaded`) and may be released.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResidencyPolicy {
    Always,
    Unloadable {
        /// Whether the loaded weights may be released after idle.
        weights: bool,
        /// Whether per-call context may be released after idle.
        context: bool,
    },
}

impl ResidencyPolicy {
    /// Derive the policy from the config `resident` knob (`true` → `Always`,
    /// `false` → fully `Unloadable`).
    pub fn from_resident(resident: bool) -> Self {
        if resident {
            Self::Always
        } else {
            Self::Unloadable {
                weights: true,
                context: true,
            }
        }
    }

    /// Whether this policy forbids unloading.
    pub fn is_always(self) -> bool {
        matches!(self, Self::Always)
    }
}

fn default_execution_provider() -> String {
    "cpu".into()
}

fn default_intra_threads() -> usize {
    1
}

fn default_optimization_level() -> String {
    "all".into()
}

fn default_quantization() -> Quant {
    Quant::Q8
}

fn default_resident() -> bool {
    true
}

fn default_max_seq_len() -> usize {
    1024
}

fn default_max_gen_tokens() -> usize {
    512
}

/// Full onnx model declaration for a `ModelEntry`. Pure serde + builder; every
/// field carries a default so an `onnx` block is backward-compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Builder)]
#[builder(start_fn = new)]
pub struct OnnxConfig {
    /// Path to the ONNX model file (or a directory containing a
    /// quant-appropriate `*.onnx` plus `config.json`).
    #[builder(into)]
    #[serde(alias = "weights")]
    pub model_path: PathBuf,
    /// Path to the LFM `tokenizer.json`. `None` falls back to the encoder
    /// role's tokenizer at registry load time (some models — router, PII —
    /// share the base encoder's tokenizer).
    #[serde(default)]
    #[builder(into)]
    pub tokenizer_path: Option<PathBuf>,
    /// What this session does (drives registry typing + boot validation). For a
    /// role-based declaration (`OnnxRoleConfig`) the role supplies the task, so
    /// this is optional and defaults to `FillMask`; a standalone `OnnxConfig`
    /// (legacy `ModelEntry.onnx`) declares it explicitly.
    #[serde(default)]
    #[builder(default = OnnxTask::FillMask)]
    pub task: OnnxTask,
    /// ONNX Runtime execution provider (default `"cpu"`).
    #[serde(default = "default_execution_provider")]
    #[builder(default = default_execution_provider())]
    pub execution_provider: String,
    /// Intra-op threads per session run (default `1` — determinism + thread
    /// budget; parallelism comes from batching, not intra-op threads).
    #[serde(default = "default_intra_threads")]
    #[builder(default = default_intra_threads())]
    pub intra_threads: usize,
    /// Graph optimization level (`disable|basic|extended|all`, default `all`).
    #[serde(default = "default_optimization_level")]
    #[builder(default = default_optimization_level())]
    pub optimization_level: String,
    /// Quantization (default `q8`; `fp32` for binary gates).
    #[serde(default = "default_quantization")]
    #[builder(default = default_quantization())]
    pub quantization: Quant,
    /// `true` → `ResidencyPolicy::Always` (load at boot, never unload).
    #[serde(default = "default_resident")]
    #[builder(default = default_resident())]
    pub resident: bool,
    /// Optional label map for token-classification tasks (PII/token-class
    /// id2label file). `None` falls back to the model's `config.json` map.
    #[serde(default)]
    pub label_source: Option<PathBuf>,
    /// Optional policy-label list for the Policy-Linter (`task:
    /// ZeroShotTokenMatching`): a JSON file of rule strings the linter scores
    /// text tokens against (the config's blacklist descriptions, externalized
    /// so the linter's rule set is a data artifact, not a code constant).
    /// `None` leaves the linter's label list empty (it lints nothing).
    #[serde(default)]
    pub policy_labels: Option<PathBuf>,
    /// Maximum sequence length for padding/truncation (ColBERT: 511).
    #[serde(default = "default_max_seq_len")]
    #[builder(default = default_max_seq_len())]
    pub max_seq_len: usize,
    /// Declared embedding dimensions (default 1024 for the Encoder).
    #[serde(default)]
    pub dims: Option<u32>,
    /// Optional annotation heads for a `FillMask` Encoder export (the finetuned
    /// q8 Encoder with UPOS/dep/head heads). When present, `validate` requires
    /// the head output names in the model's `config.json` `onnx.outputs` and
    /// loads/parses the labels file — a mis-exported finetune fails boot loudly.
    #[serde(default)]
    pub annotation_heads: Option<AnnotationHeads>,
    /// Maximum generated tokens for a `CausalLm` decode (default 512). Only
    /// meaningful for the generative task; ignored by the encoder-style
    /// workers.
    #[serde(default = "default_max_gen_tokens")]
    #[builder(default = default_max_gen_tokens())]
    pub max_gen_tokens: usize,
    /// Resident memory this model occupies once loaded (bytes). `None` →
    /// computed at boot by the registry from the model file + its external-data
    /// siblings, so the residency loop's working-set eviction knows the real
    /// footprint without manual maintenance.
    #[serde(default)]
    pub resident_bytes: Option<u64>,
    /// The generative IO contract for a `CausalLm` session. `None` → the
    /// contract is introspected from the graph at load time (and validated
    /// against it); `Some` is used verbatim, and a mismatch against the graph
    /// is a loud load error.
    #[serde(default)]
    pub llm_io: Option<LlmIo>,
    /// Max context-window size (tokens) for `CausalLm` sessions and any
    /// declared contexts (the `instances` block on an `OnnxRoleConfig`).
    /// `None` (the default) = no cap — today's behavior, byte-identical.
    #[serde(default)]
    pub max_ctx: Option<usize>,
}

impl OnnxConfig {
    /// The `ResidencyPolicy` this config implies.
    pub fn policy(&self) -> ResidencyPolicy {
        ResidencyPolicy::from_resident(self.resident)
    }

    /// The resolved tokenizer path. `None` when the config omits it (the
    /// registry fills it from the encoder role at boot). Panics if called
    /// before resolution — callers should check or use `require_tokenizer_path`.
    pub fn resolved_tokenizer_path(&self) -> Option<&Path> {
        self.tokenizer_path.as_deref()
    }

    /// The model's `config.json` path (a directory's `config.json`, a declared
    /// file's sibling, or one level up for the `onnx/` layout). `None` when no
    /// config.json exists next to the model path — the two-tower `head` block
    /// and the boot sanity-check both discover it through here.
    pub fn config_json_path(&self) -> Option<PathBuf> {
        discover_config_json(&self.model_path)
    }

    /// Resolve the model file for this config: the exact file when
    /// `model_path` names one, otherwise the quant-appropriate file inside the
    /// directory. Returns an error when nothing resolvable exists.
    pub fn resolve_model_file(&self) -> Result<PathBuf, OrtError> {
        let path = Path::new(&self.model_path);
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
        let candidates = quant_file_candidates(self.quantization, path);
        for candidate in candidates {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        Err(OrtError::ModelFileNotFound {
            model: self.model_path.display().to_string(),
        })
    }

    /// Sanity-check the declared `task` against the model's `config.json`
    /// (`architectures` / `onnx.outputs`). A missing config.json (or one
    /// without the relevant keys) is tolerated — the session loader remains
    /// the loud failure point for a genuinely unusable model. A *mismatched*
    /// declaration is a loud error so a mis-tagged model never silently serves
    /// the wrong task.
    pub fn validate(&self) -> Result<(), OrtError> {
        let Some(config_json) = discover_config_json(&self.model_path) else {
            return Ok(());
        };
        let json = std::fs::read_to_string(&config_json).map_err(|e| OrtError::ConfigRead {
            path: config_json.display().to_string(),
            detail: e.to_string(),
        })?;
        let parsed: ModelConfigJson = serde_json::from_str(&json).map_err(|e| {
            OrtError::ConfigParse {
                path: config_json.display().to_string(),
                detail: e.to_string(),
            }
        })?;

        if let Some(architectures) = parsed.architectures.as_ref() {
            let families = self.task.expected_architecture_families();
            if !families.is_empty() {
                let matched = architectures
                    .iter()
                    .any(|a| families.iter().any(|f| a == f));
                if !matched {
                    return Err(OrtError::TaskMismatch {
                        task: format!("{:?}", self.task),
                        declared: architectures.clone(),
                        expected: families.to_vec(),
                    });
                }
            }
        }

        if let Some(outputs) = parsed.onnx.as_ref().map(|o| o.outputs.as_slice()) {
            let mut required: Vec<&str> = self.task.required_outputs().to_vec();
            if let Some(heads) = self.annotation_heads.as_ref() {
                required.push(&heads.pos_output);
                required.push(&heads.dep_output);
                required.push(&heads.head_output);
            }
            for name in required {
                if !outputs.iter().any(|o| o == name) {
                    return Err(OrtError::OutputMismatch {
                        task: format!("{:?}", self.task),
                        missing: name.to_string(),
                        declared: outputs.to_vec(),
                    });
                }
            }
        }

        // The annotation-heads labels file is a declared contract: it must load
        // and parse regardless of whether a config.json was discoverable (the
        // output-name check above is the config.json half of the contract). A
        // mis-exported finetune fails boot loudly, never silently.
        if let Some(heads) = self.annotation_heads.as_ref() {
            let labels = AnnotationLabels::load(&heads.labels)?;
            if labels.upos.len() < 2 || labels.dep.len() < 2 {
                return Err(OrtError::ConfigParse {
                    path: heads.labels.display().to_string(),
                    detail: format!(
                        "annotation labels file must declare at least two `upos` and `dep` \
                         labels (got upos={}, dep={})",
                        labels.upos.len(),
                        labels.dep.len(),
                    ),
                });
            }
        }

        // A binary gate (PII-Detector, Policy-Linter) declared non-fp32 is a
        // loud warning, not a silent acceptance: the README-documented q8 flip
        // rates mean the quantization only ships with a golden-gated threshold
        // and the flip rate recorded (ROADMAP_20260827_ORT §1.6/§3.1).
        let is_gate = matches!(
            self.task,
            OnnxTask::TokenClassification | OnnxTask::ZeroShotTokenMatching
        );
        if is_gate && !self.quantization.is_fp32() {
            tracing::warn!(
                target: "fluent-onnx",
                task = ?self.task,
                quantization = ?self.quantization,
                suggested = ?Quant::suggested_for(self.task),
                "binary gate task declared non-fp32 quantization — q8 flips binary \
                 decisions (PII spans / Policy-Linter threshold); only ship behind a \
                 golden-gated threshold with the flip rate recorded",
            );
        }
        Ok(())
    }
}

/// The subset of a model `config.json` the boot sanity-check reads.
#[derive(Debug, Deserialize)]
struct ModelConfigJson {
    #[serde(default)]
    architectures: Option<Vec<String>>,
    #[serde(default)]
    onnx: Option<OnnxIo>,
}

#[derive(Debug, Deserialize)]
struct OnnxIo {
    #[serde(default)]
    outputs: Vec<String>,
}

/// Candidate model filenames inside a model directory, in preference order for
/// the requested quantization.
fn quant_file_candidates(quant: Quant, dir: &Path) -> Vec<PathBuf> {
    let names: Vec<&str> = match quant {
        Quant::Fp32 | Quant::Fp8 => vec![
            "model.onnx",
            "model_fp32.onnx",
            "onnx/model.onnx",
            "onnx/model_fp32.onnx",
        ],
        Quant::Fp16 => vec![
            "model_fp16.onnx",
            "onnx/model_fp16.onnx",
            "model.onnx",
            "onnx/model.onnx",
        ],
        Quant::Q8 => vec![
            "onnx/model_quantized.onnx",
            "onnx/model_q8.onnx",
            "model_q8.onnx",
            "model_quantized.onnx",
            "model.onnx",
            "onnx/model.onnx",
        ],
        Quant::Q4 => vec![
            "onnx/model_q4.onnx",
            "model_q4.onnx",
            "model.onnx",
            "onnx/model.onnx",
        ],
    };
    names.into_iter().map(|n| dir.join(n)).collect()
}

/// Find the model's `config.json` relative to `model_path`: inside a declared
/// directory, next to a declared file, or one level up (the `onnx/` layout).
fn discover_config_json(model_path: &Path) -> Option<PathBuf> {
    let direct = if model_path.is_dir() {
        model_path.join("config.json")
    } else {
        model_path.parent()?.join("config.json")
    };
    if direct.is_file() {
        return Some(direct);
    }
    let parent = if model_path.is_dir() {
        model_path.join("..")
    } else {
        model_path.parent()?.join("..")
    };
    let up = parent.join("config.json");
    if up.is_file() {
        Some(up)
    } else {
        None
    }
}

// ── Role-based fleet declarations ────────────────────────────────────────────
//
// A top-level `onnx` section keys one optional `OnnxRoleConfig` per role
// (Encoder / PII / Router / Policy / ColBERT). Every role is optional and the
// pipeline is fully functional (pure-deterministic) with none of them loaded.
// The config vocabulary parallels the llama.cpp `ModelEntry` + `default_params`
// surface — resident/pinned residency, run/idle timeouts, and sampling
// `params` merged into dispatch — because the same operator thinks about both
// fleets; the difference is that onnx models run in-process via `ort`, never
// as a spawned `llama-server`.

/// The role an in-process ONNX model serves. Each role maps 1:1 to an
/// [`OnnxTask`] and a stable registry key; declaring the role implies the task
/// (so a role config needs no `task` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnnxRole {
    /// Base bidirectional encoder (optionally with trained annotation heads).
    Encoder,
    /// PII token-classifier (binary gate → prefer fp32).
    Pii,
    /// Two-tower zero-shot prompt router.
    Router,
    /// Two-tower policy-linter (binary gate → prefer fp32).
    Policy,
    /// Late-interaction retriever.
    Colbert,
    /// General-purpose generative causal-LM — the ONNX LLM behind the
    /// `fluent_llm::client::ChatBackend` seam (ROADMAP M1/M2). Nothing
    /// task-specific; the single enrichment/annotation model.
    Llm,
}

impl OnnxRole {
    /// The [`OnnxTask`] this role performs.
    #[must_use]
    pub fn task(self) -> OnnxTask {
        match self {
            OnnxRole::Encoder => OnnxTask::FillMask,
            OnnxRole::Pii => OnnxTask::TokenClassification,
            OnnxRole::Router => OnnxTask::ZeroShotRouting,
            OnnxRole::Policy => OnnxTask::ZeroShotTokenMatching,
            OnnxRole::Colbert => OnnxTask::LateInteraction,
            OnnxRole::Llm => OnnxTask::CausalLm,
        }
    }

    /// The stable key this role's session is registered under in the ort
    /// registry. `OrtSessionRegistry` is keyed by `String`; roles use these
    /// names so a caller can reach a role's session without a model-name
    /// indirection.
    #[must_use]
    pub fn registry_key(self) -> &'static str {
        match self {
            OnnxRole::Encoder => "onnx/encoder",
            OnnxRole::Pii => "onnx/pii",
            OnnxRole::Router => "onnx/router",
            OnnxRole::Policy => "onnx/policy",
            OnnxRole::Colbert => "onnx/colbert",
            OnnxRole::Llm => "onnx/llm",
        }
    }

    /// The recommended quantization for this role (fp32 for binary gates).
    #[must_use]
    pub fn suggested_quantization(self) -> Quant {
        Quant::suggested_for(self.task())
    }

    /// All roles, in declaration order.
    #[must_use]
    pub fn all() -> [OnnxRole; 6] {
        [
            OnnxRole::Encoder,
            OnnxRole::Pii,
            OnnxRole::Router,
            OnnxRole::Policy,
            OnnxRole::Colbert,
            OnnxRole::Llm,
        ]
    }
}

fn default_zero_u64() -> u64 {
    0
}

fn default_onnx_instance_count() -> u32 {
    1
}

/// One declared context window for an onnx role — a self-contained,
/// serde-compatible twin of the router's `InstanceProfile` (the llama
/// `ModelEntry.instances` vocabulary). `fluent-onnx` deliberately does not
/// import router config; the composition root (`ort.rs`) maps this type
/// to/from `InstanceProfile` where the two fleets must be driven uniformly.
///
/// The map key on `OnnxRoleConfig.instances` provides the default context
/// name; `count > 1` expands into sibling contexts named `<key>-0` ..
/// `<key>-{count-1}` sharing the profile's group. `num_ctx` is the context
/// window size; `max_ctx` caps how large `resize` may grow it (a cap below
/// `num_ctx` clamps the window at materialization).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct OnnxInstanceProfile {
    /// Context name; default = the map key (expanded `<name><i>` for count > 1).
    #[serde(default)]
    pub name: Option<String>,
    /// Group; default = context name. count > 1 contexts share this group.
    #[serde(default)]
    pub group: Option<String>,
    /// Number of sibling contexts this profile expands to (1 = single context).
    #[serde(default = "default_onnx_instance_count")]
    pub count: u32,
    /// Context window size in tokens.
    pub num_ctx: u64,
    /// Slots per context; default = inherit the session's parallelism.
    #[serde(default)]
    pub parallel: Option<u32>,
    /// Exempt from idle release / eviction; implies `no_sleep`.
    #[serde(default)]
    pub pinned: bool,
    /// Never auto-sleep (stays warm). `warm` is a friendly serde alias.
    #[serde(default, alias = "warm")]
    pub no_sleep: bool,
    /// >0 = per-context idle timeout seconds; `None` = inherit.
    #[serde(default)]
    pub sleep_idle_seconds: Option<i32>,
    /// Target of a bare `<base>` request.
    #[serde(default)]
    pub default: bool,
    /// Preserve this context's KV across eviction (the resume snapshot half).
    #[serde(default)]
    pub resume: bool,
    /// Sampling params merged into the request body for this context.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    /// Context-size cap (tokens); `None` = inherit (no cap).
    #[serde(default)]
    pub max_ctx: Option<u64>,
}

impl OnnxInstanceProfile {
    /// Apply the `max_ctx` cap at materialization: a cap below `num_ctx`
    /// clamps the context window. A no-op when `max_ctx` is absent (the
    /// default) — byte-identical to today's behavior.
    pub fn apply_max_ctx(&mut self) {
        if let Some(cap) = self.max_ctx {
            if cap < self.num_ctx {
                self.num_ctx = cap;
            }
        }
    }
}

/// A llama.cpp-parallel, role-scoped ONNX model declaration.
///
/// The `model` fields are flattened [`OnnxConfig`] (model/tokenizer paths,
/// provider, threads, quantization, context, heads/labels); the surrounding
/// fields mirror the llama.cpp lifecycle surface. The task is implied by the
/// role and injected by [`OnnxRoleConfig::to_onnx_config`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxRoleConfig {
    /// Exempt from lazy unload/eviction (parallel to a llama instance's
    /// `pinned`). `resident: true` (in the flattened model block) already
    /// forces an `Always` residency; `pinned` additionally forbids release.
    #[serde(default)]
    pub pinned: bool,
    /// Never auto-sleep (parallel to a llama instance's `no_sleep`/`warm`).
    #[serde(default, alias = "warm")]
    pub no_sleep: bool,
    /// Idle seconds after which the session may be released; a positive value
    /// overrides the registry default, `None`/`0` inherits it.
    #[serde(default)]
    pub sleep_idle_seconds: Option<i32>,
    /// Per-run timeout in milliseconds (parallel to `ModelEntry.total_timeout_ms`).
    #[serde(default = "default_zero_u64")]
    pub total_timeout_ms: u64,
    /// Idle timeout in milliseconds (parallel to `ModelEntry.idle_timeout_ms`).
    #[serde(default = "default_zero_u64")]
    pub idle_timeout_ms: u64,
    /// Sampling defaults merged into dispatch bodies (parallel to llama
    /// `params` / `default_params`); per-request values win.
    #[serde(default)]
    pub params: Option<serde_json::Value>,
    /// Declared context windows (the onnx analogue of a llama `ModelEntry`'s
    /// `instances` block): the same profile vocabulary as the llama fleet,
    /// materialized as named `OnnxContextPool` contexts with per-context KV.
    /// `None` (the default) keeps today's single-shot decode — byte-identical
    /// for existing configs (an onnx model with no `instances` block).
    #[serde(default)]
    pub instances: Option<HashMap<String, OnnxInstanceProfile>>,
    /// The model + execution declaration (task overridden by the role).
    #[serde(flatten)]
    pub model: OnnxConfig,
}

impl OnnxRoleConfig {
    /// The flattened [`OnnxConfig`] with the task forced to this role's task.
    ///
    /// `max_ctx` (the global context-size cap) lives on the flattened `model`
    /// block and is carried through unchanged. The role's declared
    /// `instances` block stays on [`OnnxRoleConfig`] (this method consumes
    /// `self`), where the composition root / `OnnxContextPool` builder reads it
    /// directly when materializing named contexts (M4/M6) — the pool consumes
    /// the role config, so nothing is lost in the round-trip.
    #[must_use]
    pub fn to_onnx_config(mut self, role: OnnxRole) -> OnnxConfig {
        self.model.task = role.task();
        self.model
    }
}

/// The top-level role-keyed ONNX fleet. Every role is optional; an all-`None`
/// fleet is fully fail-open (pure-deterministic pipeline).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OnnxFleetConfig {
    #[serde(default)]
    pub encoder: Option<OnnxRoleConfig>,
    #[serde(default)]
    pub pii: Option<OnnxRoleConfig>,
    #[serde(default)]
    pub router: Option<OnnxRoleConfig>,
    #[serde(default)]
    pub policy: Option<OnnxRoleConfig>,
    #[serde(default)]
    pub colbert: Option<OnnxRoleConfig>,
    #[serde(default)]
    pub llm: Option<OnnxRoleConfig>,
}

impl OnnxFleetConfig {
    /// Whether no role is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }

    /// Iterate configured roles in declaration order as `(role, config)`.
    pub fn iter(&self) -> impl Iterator<Item = (OnnxRole, &OnnxRoleConfig)> {
        OnnxRole::all()
            .into_iter()
            .filter_map(|role| self.get(role).map(|cfg| (role, cfg)))
    }

    /// The config for a role, if declared.
    #[must_use]
    pub fn get(&self, role: OnnxRole) -> Option<&OnnxRoleConfig> {
        match role {
            OnnxRole::Encoder => self.encoder.as_ref(),
            OnnxRole::Pii => self.pii.as_ref(),
            OnnxRole::Router => self.router.as_ref(),
            OnnxRole::Policy => self.policy.as_ref(),
            OnnxRole::Colbert => self.colbert.as_ref(),
            OnnxRole::Llm => self.llm.as_ref(),
        }
    }

    /// Whether a role is declared.
    #[must_use]
    pub fn has(&self, role: OnnxRole) -> bool {
        self.get(role).is_some()
    }
}

#[cfg(test)]
#[path = "../tests/config.rs"]
mod tests;
