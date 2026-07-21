//! Test stubs for coral cache reactor tests.
//!
//! Provides deterministic, seedable implementations of `EmbeddingProvider`,
//! `Decomposer`, and `ChatBackend` for unit testing without LLM dependencies.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use guidance_llm::client::ChatBackend;
use guidance_llm::decomposer::Decomposer;
use guidance_llm::embeddings::{BatchEmbedding, EmbeddingError, EmbeddingProvider};
use guidance_llm::{ChatMessage, LlmError};

// ---------------------------------------------------------------------------
// StubEmbedder — deterministic embedding provider
// ---------------------------------------------------------------------------

/// Deterministic embedder that returns fixed-dimension vectors derived from
/// the input text hash. Useful for testing L4 semantic routing without an
/// actual embedding service.
pub struct StubEmbedder {
    dims: u32,
    call_count: Arc<Mutex<u64>>,
}

impl StubEmbedder {
    pub fn new(dims: u32) -> Self {
        Self {
            dims,
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Returns the number of `embed` calls made through this stub.
    pub fn call_count(&self) -> u64 {
        *self.call_count.lock().unwrap()
    }
}

impl EmbeddingProvider for StubEmbedder {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn dimensions(&self) -> u32 {
        self.dims
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        *self.call_count.lock().unwrap() += 1;
        // Deterministic: each dimension gets a value derived from text bytes
        let mut vec = Vec::with_capacity(self.dims as usize);
        let bytes = text.as_bytes();
        for i in 0..self.dims as usize {
            let byte = bytes.get(i % bytes.len()).copied().unwrap_or(0);
            vec.push(f32::from(byte) / 255.0);
        }
        Ok(vec)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        let dims = self.dims as usize;
        let mut flat = Vec::with_capacity(texts.len() * dims);
        for text in texts {
            let emb = self.embed(text)?;
            flat.extend(emb);
        }
        Ok(BatchEmbedding {
            flat,
            count: texts.len(),
            dims,
        })
    }
}

// ---------------------------------------------------------------------------
// StubDecomposer — deterministic task decomposition
// ---------------------------------------------------------------------------

/// Deterministic decomposer that returns pre-configured subtask lists for
/// known queries, and falls back to `vec![task]` for unknown queries.
pub struct StubDecomposer {
    responses: HashMap<String, Vec<String>>,
}

impl StubDecomposer {
    pub fn new(responses: HashMap<String, Vec<String>>) -> Self {
        Self { responses }
    }
}

impl Decomposer for StubDecomposer {
    fn decompose(&self, task: &str) -> Vec<String> {
        self.responses
            .get(task)
            .cloned()
            .unwrap_or_else(|| vec![task.to_string()])
    }
}

// ---------------------------------------------------------------------------
// StubChatBackend — deterministic chat responses
// ---------------------------------------------------------------------------

/// Deterministic chat backend that returns canned responses in order.
/// When all canned responses are exhausted, returns `Err(LlmError::NoResponse)`.
pub struct StubChatBackend {
    responses: Mutex<VecDeque<String>>,
}

impl StubChatBackend {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }

    /// Create a backend that always returns the same response.
    pub fn always(response: impl Into<String>) -> Self {
        Self::new(vec![response.into()])
    }

    /// Create a backend that always returns an error.
    pub fn always_err(error: impl Into<String>) -> StubFailingChatBackend {
        StubFailingChatBackend {
            error: error.into(),
        }
    }
}

impl ChatBackend for StubChatBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        let mut queue = self.responses.lock().unwrap();
        queue.pop_front().ok_or(LlmError::NoResponse)
    }
}

// ---------------------------------------------------------------------------
// StubFailingChatBackend — always errors
// ---------------------------------------------------------------------------

/// Chat backend that always returns an error. Used to test error propagation
/// (e.g., L5 `FrontierError` instead of `CacheMiss`).
pub struct StubFailingChatBackend {
    error: String,
}

impl ChatBackend for StubFailingChatBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        Err(LlmError::Api(self.error.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_embedder_returns_deterministic_vectors() {
        let emb = StubEmbedder::new(4);
        let v1 = emb.embed("hello").unwrap();
        let v2 = emb.embed("hello").unwrap();
        assert_eq!(v1, v2);
        assert_eq!(v1.len(), 4);
        assert_eq!(emb.call_count(), 2);
    }

    #[test]
    fn stub_decomposer_returns_configured_subtasks() {
        let mut responses = HashMap::new();
        responses.insert(
            "complex task".to_string(),
            vec!["sub1".to_string(), "sub2".to_string()],
        );
        let dec = StubDecomposer::new(responses);
        assert_eq!(dec.decompose("complex task"), vec!["sub1", "sub2"]);
        assert_eq!(dec.decompose("unknown"), vec!["unknown"]);
    }

    #[test]
    fn stub_chat_backend_returns_canned_responses() {
        let backend = StubChatBackend::new(vec!["first".into(), "second".into()]);
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "test".into(),
        }];
        assert_eq!(backend.chat_complete(&msgs).unwrap(), "first");
        assert_eq!(backend.chat_complete(&msgs).unwrap(), "second");
        assert!(backend.chat_complete(&msgs).is_err());
    }

    #[test]
    fn stub_chat_backend_always() {
        let backend = StubChatBackend::always("canned");
        let msgs = vec![];
        assert_eq!(backend.chat_complete(&msgs).unwrap(), "canned");
    }

    #[test]
    fn stub_failing_chat_backend_always_errors() {
        let backend = StubChatBackend::always_err("boom");
        let msgs = vec![];
        let err = backend.chat_complete(&msgs).unwrap_err();
        assert!(err.to_string().contains("boom"));
    }
}
