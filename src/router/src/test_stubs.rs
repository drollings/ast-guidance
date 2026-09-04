use std::collections::VecDeque;
use std::sync::Mutex;

use common_core::sync::lock;
use fluent_llm::client::ChatBackend;
use fluent_llm::{BatchEmbedding, ChatMessage, EmbeddingError, EmbeddingProvider, LlmError};
use fluent_wvr::prelude::*;

use crate::pipeline_types::{PipelineStage, StageDecision, StageVerdict};

pub struct StubChatBackend {
    responses: Mutex<VecDeque<String>>,
}

impl StubChatBackend {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }

    pub fn always(response: impl Into<String>) -> Self {
        Self::new(vec![response.into()])
    }
}

impl ChatBackend for StubChatBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        let mut queue = lock(&self.responses);
        queue.pop_front().ok_or(LlmError::NoResponse)
    }
}

/// A `ChatBackend` that counts every call and always returns a canned response.
/// The count-calls pattern (mirrors `config::builder::tests::RecordingBackend`)
/// for asserting "derived exactly once" in LOD/view tests.
pub struct CountingBackend {
    calls: std::sync::atomic::AtomicUsize,
    response: String,
}

impl CountingBackend {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
            response: response.into(),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl ChatBackend for CountingBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(self.response.clone())
    }
}

// ── Test stage stubs for DAG pipeline tests ──────────────────────────────

/// A minimal `WorkUnit` stage that always passes, returning a
/// `StageDecision` with `Passed` verdict and a configurable reason.
pub struct SimplePassStage {
    name: ArcIntern<str>,
    reason: String,
}

impl SimplePassStage {
    pub fn new(stage_name: &str, reason: &str) -> Self {
        Self {
            name: ArcIntern::from(stage_name.to_string()),
            reason: reason.to_string(),
        }
    }
}

impl WorkUnit for SimplePassStage {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        WorkOutput::typed(
            "passed",
            &StageDecision {
                stage: PipelineStage::DeterministicPreFilter,
                verdict: StageVerdict::Passed,
                score: None,
                reason: self.reason.clone(),
                latency_ms: 0,
                metadata: serde_json::Value::Object(Default::default()),
            },
        )
    }
}

impl_fieldless!(SimplePassStage);

impl Describable for SimplePassStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}
impl_component!(SimplePassStage);

/// A `WorkUnit` stage that fails N times then succeeds.  Used by
/// `RetryClassifier` tests.
pub struct FailingStage {
    name: ArcIntern<str>,
    failures_remaining: Mutex<usize>,
}

impl FailingStage {
    pub fn new(stage_name: &str, failure_count: usize) -> Self {
        Self {
            name: ArcIntern::from(stage_name.to_string()),
            failures_remaining: Mutex::new(failure_count),
        }
    }
}

impl WorkUnit for FailingStage {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let mut remaining = self.failures_remaining.lock().unwrap();
        if *remaining > 0 {
            *remaining -= 1;
            WorkOutput::typed(
                "fallback",
                &StageDecision {
                    stage: PipelineStage::Classifier,
                    verdict: StageVerdict::Passed,
                    score: Some(1.0),
                    reason: format!(
                        "parse error: attempt would be {} more failures",
                        *remaining + 1
                    ),
                    latency_ms: 0,
                    metadata: serde_json::json!({"fallback": true}),
                },
            )
        } else {
            WorkOutput::typed(
                "classified",
                &StageDecision {
                    stage: PipelineStage::Classifier,
                    verdict: StageVerdict::Passed,
                    score: Some(0.95),
                    reason: "intent=code, action=route".into(),
                    latency_ms: 0,
                    metadata: serde_json::json!({"intent": "code", "fallback": false}),
                },
            )
        }
    }
}

impl_fieldless!(FailingStage);

impl Describable for FailingStage {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}
impl_component!(FailingStage);

// ── Deterministic test embedder (DRY: shared by charts store/select/plan) ──

/// FNV-1a over lowercase word tokens into a fixed-dimension vector,
/// L2-normalized. Deterministic and collision-tolerant — enough for tests
/// that need cosine similarity without a live embedding endpoint.
pub struct HashEmbedder {
    dims: usize,
}

impl HashEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl EmbeddingProvider for HashEmbedder {
    fn name(&self) -> &'static str {
        "test-hash"
    }

    fn dimensions(&self) -> u32 {
        self.dims as u32
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut vec = vec![0.0f32; self.dims];
        for token in text.split(|c: char| !c.is_ascii_alphanumeric()) {
            if token.is_empty() {
                continue;
            }
            let h = common_core::hash::fnv1a64(token.to_ascii_lowercase().as_bytes());
            let bucket = (h % self.dims as u64) as usize;
            vec[bucket] += 1.0;
        }
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut vec {
                *x /= norm;
            }
        }
        Ok(vec)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        let mut flat = Vec::new();
        for t in texts {
            flat.extend_from_slice(&self.embed(t)?);
        }
        Ok(BatchEmbedding {
            flat,
            count: texts.len(),
            dims: self.dims,
        })
    }
}
