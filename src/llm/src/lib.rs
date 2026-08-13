//! fluent-llm: LLM HTTP client provider — embeddings, chat completions,
//! prompt utilities, context packing, and request queueing.

pub mod anonymize;
pub mod client;
pub mod constants;
pub mod context_packer;
pub mod decomposer;
pub mod embeddings;
pub mod error;
pub mod http_class;
pub mod llm_queue;
pub mod openai;
pub mod parse;
pub mod pii_patterns;
pub mod url;

// Re-export the LLM protocol + queue types from fluent-concurrency so
// existing `use fluent_llm::{LlmConfig, LlmError, ...}` paths keep working.
pub use fluent_concurrency::llm_queue::{
    ChatMessage, LlmConfig, LlmError, LlmQueueConfig, LlmRequestQueue, LlmTask,
};

pub use anonymize::anonymize;
pub use client::{
    block_on, chat_complete_http, extract_comment_tag, is_blank_or_plausible,
    is_malformed_response, model_name, strip_preamble, strip_think_block, ChatBackend, LlmClient,
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
pub use parse::{parse_json_response, parse_typed, strip_json_fence, JsonParseError};
