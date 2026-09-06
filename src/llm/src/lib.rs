//! fluent-llm: LLM HTTP client provider — embeddings, chat completions,
//! prompt utilities, context packing, and request queueing.

pub mod anonymize;
pub mod backend;
pub mod cache;
pub mod client;
pub mod constants;
pub mod context_packer;
pub mod decomposer;
pub mod embeddings;
pub mod embeddings_cache;
pub mod error;
pub mod http_class;
pub mod llm_queue;
pub mod onnx_config;
pub mod onnx_error;
pub mod onnx_session;
pub mod openai;
pub mod parse;
pub mod pii_patterns;
pub mod protocol;
pub mod runtime;
pub mod sse;
pub mod telemetry;
pub mod testutil;
pub mod thinking;
pub mod tokens;
pub mod url;

// Re-export the LLM protocol + queue types from the owned `protocol` module
// (ROADMAP_20260903_LLM M9) so existing
// `use fluent_llm::{LlmConfig, LlmError, ...}` paths keep working unchanged.
pub use protocol::{
    ChatMessage, LlmConfig, LlmError, LlmQueueConfig, LlmRequestQueue, LlmTask,
};

pub use anonymize::{anonymize, build_anonymize_map};
pub use backend::{
    BackendCaps, BackendError, BackendLoader, ContextProfile, EntityLinkScorer,
    InferenceBackend, InferenceCapability, InferenceRegistry, NamedContexts, OverlayContribution,
    OverlayError, PiiError, PiiSpan, PiiSpanDetector, Readiness, RegexPiiDetector, Residual,
    ResidualKind, ResidualOverlay, RouteLabel,
};
pub use onnx_config::{
    AnnotationHeads, AnnotationLabels, LlmIo, OnnxConfig, OnnxFleetConfig, OnnxInstanceProfile,
    OnnxRole, OnnxRoleConfig, OnnxTask, Quant, ResidencyPolicy,
};
pub use onnx_error::OrtError;
pub use onnx_session::{OrtSessionRegistry, ResidencyReportEntry, SessionHandle, SessionLoader};
pub use client::{
    block_on, chat_complete_http, extract_comment_tag, is_blank_or_plausible,
    is_malformed_response, model_name, strip_preamble, ChatBackend, LlmClient,
};
pub use constants::MAX_EMBEDDING_DIMENSIONS;
pub use context_packer::ContextPacker;
pub use decomposer::{Decomposer, DecomposerConfig, LocalDecomposer};
pub use embeddings::{
    create_embedding_provider, BatchEmbedding, EmbeddingError, EmbeddingProvider, NoopEmbedding,
    OllamaEmbedding, OpenAiEmbedding,
};
pub use error::EmbedError;
pub use http_class::{classify_http_status, FailureClass, HttpClass};
pub use parse::{parse_json_response, parse_typed, repair_json, strip_json_fence, JsonParseError};
pub use telemetry::{
    FeatureName, NoopSink, ProviderCategory, TelemetryEvent, TelemetrySink, ToolName, TracingSink,
};
pub use thinking::{strip_think_block, strip_thinking_blocks, StreamingThinkFilter};
pub use runtime::{
    EvictionPolicy, LlmContext, LlmKVCache, LlmResidencyEngine, LlmResidencyRow, LlmRuntime,
    LlmRuntimeError, LlmWeights, MemoryPool, SnapshotMeta,
};
