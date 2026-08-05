---
name: embedding-providers
description: Pluggable embedding provider trait that converts text to dense float vectors for semantic search. Supports Ollama (local), OpenAI-compatible APIs, and a no-op keyword-only fallback. Embeds CachedEmbeddingProvider<T> for in-memory LRU caching and LlmRequestQueue for controlled concurrency.
anchors:
  - EmbeddingProvider
  - CachedEmbeddingProvider
  - create_embedding_provider
  - OllamaEmbedding
  - OpenAiEmbedding
  - NoopEmbedding
  - BatchEmbedding
  - LlmRequestQueue
---

# Embedding Providers

A `dyn EmbeddingProvider: Send + Sync` trait that decouples vector search from the underlying embedding model, allowing the codebase to work offline (keyword-only) or with any OpenAI-compatible embedding API.

## Providers

| Name | Type | Default model | Dimensions |
|------|------|---------------|------------|
| `ollama` | Local HTTP | `nomic-embed-text` | 768 |
| `openai` | HTTPS API | `text-embedding-3-small` | 1536 |
| `ollama:<model>` | Ollama with custom model | configurable | configurable |
| `custom:<url>` | OpenAI-compatible | configurable | configurable |
| `none` | No-op | — | 0 (keyword fallback) |

## CachedEmbeddingProvider

A generic wrapper (`llm/src/embeddings.rs`) that adds an in-memory LRU cache
around any `EmbeddingProvider`:

```rust
pub struct CachedEmbeddingProvider<T: EmbeddingProvider> {
    inner: T,
    cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
}
```

- `cache_key` / `cache_lookup` / `cache_store` methods are shared by both
  `OllamaEmbedding` and `OpenAiEmbedding` via this generic wrapper.
- Created automatically in `create_embedding_provider` for all provider types.

## Event-Driven Queue

`LlmRequestQueue` (`llm/src/llm_queue.rs`) wraps
`guidance_concurrency_queue::EventQueue<LlmTask>` to provide bounded MPMC
concurrency for LLM requests:

```rust
pub struct LlmRequestQueue {
    inner: Arc<EventQueue<LlmTask>>,
}

impl LlmRequestQueue {
    pub fn submit(&self, messages: Vec<ChatMessage>, config: LlmConfig) -> Result<String, LlmError>;
    pub fn submit_async(&self, messages: Vec<ChatMessage>, config: LlmConfig) -> impl Future<Output = Result<String, LlmError>>;
}
```

- `LlmClient` holds `Option<Arc<LlmRequestQueue>>` — preferred path uses queue
  for backpressure; direct HTTP fallback available for backward compat.
- `EmbeddingProvider` impls use the same queue architecture internally.
- Retry policies: `None`, `Fixed { max_attempts, backoff_ms }`,
  `Exponential { max_attempts, base_ms, max_ms }`.

## Key files

- `llm/src/embeddings.rs` — `EmbeddingProvider` trait, `CachedEmbeddingProvider<T>`,
  `OllamaEmbedding`, `OpenAiEmbedding`, `NoopEmbedding`, `BatchEmbedding`,
  `create_embedding_provider`
- `llm/src/llm_queue.rs` — `LlmRequestQueue`, `LlmTask`
- `concurrency-queue/src/event_queue.rs` — `EventQueue<T>` (bounded MPMC channel)
- `concurrency-queue/src/config.rs` — `QueueConfig`, `RetryPolicy`

## Example

```rust
use fluent_llm::embeddings::{create_embedding_provider, EmbeddingProvider};

let provider = create_embedding_provider(
    "ollama",
    Some("nomic-embed-text"),
    Some("http://localhost:11434"),
    None,
    768,
)
.unwrap();

assert_eq!(provider.name(), "ollama");
let vec = provider.embed("hello world").unwrap();
assert_eq!(vec.len(), 768);
```

## Semantic Deviations

- **`dyn Trait` + `Box`** replaces Zig's explicit `{ptr, vtable}` struct pattern — Rust's `Box<dyn EmbeddingProvider>` is the idiomatic trait-object dispatch
- **`Send + Sync` bounds** replace Zig's `thread_id` assertions — the Rust trait requires `Send + Sync` for safe multi-threaded use instead of runtime thread-ID checks
- **`reqwest` HTTP** replaces `std.http.Client` — blocking HTTP via `reqwest::blocking::Client` for sync calls, async `reqwest::Client` for the embedding path
- **`serde_json`** replaces `std.json` — JSON serialization/deserialization for request/response bodies
- **`thiserror`** replaces Zig error unions — `EmbeddingError` is a typed error enum with `#[derive(thiserror::Error)]`
- **`CachedEmbeddingProvider<T>`** replaces Zig's `content_hash_with_model` — the Rust version wraps the caching logic generically rather than embedding it in each provider impl
- **Event queue** replaces `std.Thread.Pool` — `LlmRequestQueue` with configurable retry policies replaces ad-hoc thread spawning

## Zig reference

See `../doc/capabilities/embedding-providers/CAPABILITY.md` in the Zig guidance source tree for the original vtable-based `EmbeddingProvider` with content-hash caching.
