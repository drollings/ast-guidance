//! `fluent-onnx` — ONNX / `ort` workers for Coral Router + spacy-rs.
//!
//! Two layers:
//!
//! - **Pure, ort-free** (compile with `--no-default-features`): the config
//!   schema (`OnnxTask`, `Quant`, `ResidencyPolicy`, `OnnxConfig`), the
//!   `OrtError` type, and the `OrtSessionRegistry` with its `SessionLoader`
//!   DIP seam. Hermetic tests exercise all of this without a model.
//! - **Feature `onnx` (default-on)**: the real `OrtSessionLoader` and the
//!   workers — `OrtEncoder`, the two-tower `TwoTowerWorker`, PII classifiers,
//!   the token aligner, and the ColBERT retriever.
//!
//! Dependency direction: `fluent-onnx` consumes `fluent-llm`
//! (`EmbeddingProvider`, `EmbeddingError`), `common-core`, `tokenizers`, and
//! `ort`. It MUST NOT import `spacy-rs`, `guidance`, `coral`, or `wasm_ipc` —
//! it consumes plain strings and the router/spacy-rs define the seams it
//! supplies implementations for.

pub mod align;
pub mod annotate;
pub mod colbert;
pub mod config;
pub mod encoder;
pub mod error;
pub mod grammar;
pub mod overlay;
pub mod pii;
pub mod residency;
pub mod session;
pub mod two_tower;

pub use align::{SpacyTokenAligner, SpacyTokenAlignment};
pub use annotate::{
    aggregate_to_spacy, argmax, argmax_of_mean, decode_heads, LfmAnnotation, TokenAnnotation,
};
pub use colbert::{
    l2_normalize, maxsim_score, maxsim_score_normalized, ConceptEncoding, EntitySimilarityHit,
    EntitySimilarityIndex,
};
pub use common_core::vector_math::cosine_similarity_f32;
pub use config::{
    AnnotationHeads, AnnotationLabels, LlmIo, OnnxConfig, OnnxFleetConfig, OnnxRole,
    OnnxRoleConfig, OnnxTask, Quant, ResidencyPolicy,
};
pub use encoder::mean_pool;
pub use error::OrtError;
pub use grammar::{
    BatchPromptGrammar, Grammar, JsonArrayGrammar, JsonField, JsonObjectGrammar, JsonSchema,
    JsonType, TokenVocab, grammar_from_json_schema, is_valid_json_prefix, tokens_for_literal,
};
pub use overlay::{
    OverlayContribution, OverlayError, Residual, ResidualKind, ResidualOverlay, RouteLabel,
};
pub use pii::{decode_biluo, load_id2label, PiiError, PiiSpan, PiiSpanDetector, RegexPiiDetector};
pub use residency::OrtResidencyLoop;
pub use session::{OrtSessionRegistry, ResidencyReportEntry, SessionHandle, SessionLoader};
pub use two_tower::{
    policy_hits_from_matrix, PromptBuilder, TwoTowerHead, TwoTowerPrompt,
};
pub use two_tower::{load_policy_labels, PolicyHit};

#[cfg(feature = "onnx")]
pub mod tokenizer;
#[cfg(feature = "onnx")]
pub mod llm;
#[cfg(feature = "onnx")]
pub mod context;
#[cfg(feature = "onnx")]
pub mod context_pool;
#[cfg(feature = "onnx")]
pub use grammar::HuggingFaceVocab;
#[cfg(feature = "onnx")]
pub use llm::{
    BOS_TOKEN_ID, EOS_TOKEN_ID, VOCAB_SIZE, LlmParams, OnnxChatCompletion, OrtLlmSession,
    apply_chat_template, build_llm_session, build_llm_session_from_handle, sample_next_token,
};
#[cfg(feature = "onnx")]
pub use context::{
    OnnxContext, OnnxContextProfile, OnnxKVCache, PastState, DEFAULT_ONNX_CONTEXT_TOKENS,
};
#[cfg(feature = "onnx")]
pub use context_pool::OnnxContextPool;
#[cfg(feature = "onnx")]
pub use colbert::{bake_entity_index, build_colbert, build_colbert_from_registry, ColbertRetriever};
#[cfg(feature = "onnx")]
pub use encoder::{build_encoder, build_encoder_from_registry, OrtEncoder};
#[cfg(feature = "onnx")]
pub use overlay::{build_prompt_router_overlay, PromptRouterOverlay};
#[cfg(feature = "onnx")]
pub use pii::{build_pii_classifier, OrtPiiClassifier};
#[cfg(feature = "onnx")]
pub use annotate::{build_annotation_worker_from_registry, OrtAnnotationWorker};
#[cfg(feature = "onnx")]
pub use session::OrtSessionLoader;
#[cfg(feature = "onnx")]
pub use two_tower::{
    build_policy_linter_from_registry, build_two_tower_from_registry, PolicyLinter,
    TwoTowerWorker,
};