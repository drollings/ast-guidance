use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

use common_core::sync::lock;
use fluent_llm::backend::{BackendCaps, InferenceBackend, InferenceRegistry};
use fluent_llm::client::ChatBackend;
use fluent_llm::runtime::LlmWeights;
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

/// Maps a requested instance to the marker text the stub answers with.
type StubResponder = Arc<dyn Fn(Option<&str>) -> &'static str + Send + Sync>;

/// A stub [`InferenceBackend`] serving exactly one key, so registry routing
/// can be told apart per backend by marker text. The responder maps the
/// requested instance to the marker text — the two resolution shapes
/// `local_backend` (`None`) / `local_backend_for_instance` (`Some`) produce.
pub struct StubInferenceBackend {
    id: &'static str,
    key: String,
    respond: StubResponder,
}

impl StubInferenceBackend {
    /// A stub serving `key`, answering through `respond` (instance → marker).
    pub fn with_responder(
        id: &'static str,
        key: impl Into<String>,
        respond: impl Fn(Option<&str>) -> &'static str + Send + Sync + 'static,
    ) -> Self {
        Self {
            id,
            key: key.into(),
            respond: Arc::new(respond),
        }
    }

    /// A stub serving `key` with distinct default/instance markers (`None`
    /// instance → `marker`, `Some` instance → `instance_marker`).
    pub fn named(
        id: &'static str,
        key: impl Into<String>,
        marker: &'static str,
        instance_marker: &'static str,
    ) -> Self {
        Self::with_responder(id, key, move |instance| {
            if instance.is_some() {
                instance_marker
            } else {
                marker
            }
        })
    }

    /// A stub serving `key` with one marker for both resolution shapes.
    pub fn fixed(id: &'static str, key: impl Into<String>, marker: &'static str) -> Self {
        Self::named(id, key, marker, marker)
    }

    /// A single-backend registry holding this stub, ready to install on a
    /// config via `RouterConfig::set_inference_registry`.
    pub fn into_registry(self) -> Arc<RwLock<InferenceRegistry>> {
        let mut registry = InferenceRegistry::new();
        registry.register(Arc::new(self));
        Arc::new(RwLock::new(registry))
    }
}

struct StubMarkerBackend {
    text: String,
}

impl ChatBackend for StubMarkerBackend {
    fn chat_complete(&self, _m: &[ChatMessage]) -> Result<String, LlmError> {
        Ok(self.text.clone())
    }
}

impl FieldAccess for StubInferenceBackend {
    fn set_field(&mut self, name: &str, _v: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(name.into()))
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(name.into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for StubInferenceBackend {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
}

impl WorkUnit for StubInferenceBackend {
    fn name(&self) -> &str {
        self.id
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("stub"))
    }
}

impl_component!(StubInferenceBackend);

impl InferenceBackend for StubInferenceBackend {
    fn backend_id(&self) -> &'static str {
        self.id
    }
    fn model_keys(&self) -> Vec<String> {
        vec![self.key.clone()]
    }
    fn weights(&self, _key: &str) -> Option<Arc<dyn LlmWeights>> {
        None
    }
    fn chat_backend(
        &self,
        key: &str,
        instance: Option<&str>,
    ) -> Option<Arc<dyn ChatBackend>> {
        if key != self.key {
            return None;
        }
        let text = (self.respond)(instance);
        Some(Arc::new(StubMarkerBackend { text: text.into() }))
    }
    fn capabilities(&self) -> BackendCaps {
        BackendCaps::default()
    }
}

// ── Shared onnx decode doubles (single home for the grammar-seam fakes) ──

/// Fixed token-id → text map for hermetic grammar tests.
#[cfg(feature = "onnx")]
pub struct StubVocab {
    tokens: Vec<String>,
}

#[cfg(feature = "onnx")]
impl StubVocab {
    /// A vocab from an ordered token list (`id` = position).
    pub fn from_list(tokens: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            tokens: tokens.iter().map(|s| s.to_string()).collect(),
        })
    }
}

#[cfg(feature = "onnx")]
impl fluent_onnx::TokenVocab for StubVocab {
    fn token_text(&self, id: u32) -> Option<String> {
        self.tokens.get(id as usize).cloned()
    }
}

/// A fake decode runner recording whether each call was grammar-constrained.
#[cfg(feature = "onnx")]
pub struct RecordingRunner {
    /// One entry per call: `true` when a grammar was supplied.
    pub calls: Mutex<Vec<bool>>,
    output: String,
}

#[cfg(feature = "onnx")]
impl RecordingRunner {
    /// A runner answering every call with `output`.
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            output: output.into(),
        }
    }
}

#[cfg(feature = "onnx")]
impl crate::ort::OnnxLlmRunner for RecordingRunner {
    fn complete(
        &self,
        _messages: &[ChatMessage],
        grammar: Option<&mut (dyn fluent_onnx::Grammar + 'static)>,
        _max_tokens: Option<usize>,
        _params: fluent_onnx::LlmParams,
    ) -> Result<String, LlmError> {
        lock(&self.calls).push(grammar.is_some());
        Ok(self.output.clone())
    }
}
