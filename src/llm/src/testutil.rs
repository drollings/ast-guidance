//! Shared hermetic doubles for inference tests.
//!
//! The single home for stub backends: a canned-response [`StubBackend`], an
//! order-recording [`CountingStubBackend`], and a canned-handle
//! [`StubSessionLoader`]. Test-only by convention; hermetic (no model, no
//! network).

use std::sync::{Arc, Mutex};

use common_core::sync::lock;
use fluent_wvr::prelude::*;

use crate::backend::{BackendCaps, InferenceBackend, Readiness};
use crate::client::ChatBackend;
use crate::onnx_config::OnnxConfig;
use crate::onnx_error::OrtError;
use crate::onnx_session::{SessionHandle, SessionLoader};
use crate::{ChatMessage, LlmError};

struct StubChat {
    response: String,
}

impl ChatBackend for StubChat {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        Ok(self.response.clone())
    }
}

/// A canned-response backend serving a fixed set of model keys.
pub struct StubBackend {
    id: &'static str,
    keys: Vec<String>,
    response: String,
}

impl StubBackend {
    /// A stub registered under `id`, serving `keys` with `response`.
    pub fn new(id: impl Into<String>, keys: Vec<&str>, response: &str) -> Self {
        Self {
            id: Box::leak(id.into().into_boxed_str()),
            keys: keys.into_iter().map(str::to_string).collect(),
            response: response.to_string(),
        }
    }

    /// All model keys this stub serves.
    pub fn served_keys(&self) -> &[String] {
        &self.keys
    }
}

impl FieldAccess for StubBackend {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Ok(())
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound("stub".into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for StubBackend {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
}

impl WorkUnit for StubBackend {
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

fluent_wvr::impl_component!(StubBackend);

impl InferenceBackend for StubBackend {
    fn backend_id(&self) -> &'static str {
        self.id
    }
    fn model_keys(&self) -> Vec<String> {
        self.keys.clone()
    }
    fn weights(&self, _key: &str) -> Option<Arc<dyn crate::runtime::LlmWeights>> {
        None
    }
    fn chat_backend(
        &self,
        key: &str,
        _instance: Option<&str>,
    ) -> Option<Arc<dyn crate::client::ChatBackend>> {
        if self.keys.iter().any(|k| k == key) {
            Some(Arc::new(StubChat {
                response: self.response.clone(),
            }))
        } else {
            None
        }
    }
    fn capabilities(&self) -> BackendCaps {
        BackendCaps::default()
    }
}

/// A stub backend recording every served-key consultation, in order.
/// Non-candidate keys are never recorded: the registry pre-filters by model
/// key before any construction call.
pub struct CountingStubBackend {
    inner: StubBackend,
    failed: bool,
    consults: Mutex<Vec<String>>,
}

impl CountingStubBackend {
    /// A healthy counting stub serving `keys` with `response`.
    pub fn new(id: impl Into<String>, keys: Vec<&str>, response: &str) -> Self {
        Self {
            inner: StubBackend::new(id, keys, response),
            failed: false,
            consults: Mutex::new(Vec::new()),
        }
    }

    /// A stub reporting [`Readiness::Failed`]; the registry skips it with
    /// that failure as the recorded cause.
    pub fn failed(id: impl Into<String>, keys: Vec<&str>) -> Self {
        Self {
            inner: StubBackend::new(id, keys, ""),
            failed: true,
            consults: Mutex::new(Vec::new()),
        }
    }

    /// Consultation log snapshot, in occurrence order.
    pub fn consults(&self) -> Vec<String> {
        lock(&self.consults).clone()
    }
}

impl FieldAccess for CountingStubBackend {
    fn set_field(&mut self, _name: &str, _value: &str) -> Result<(), FieldError> {
        Ok(())
    }
    fn get_field(&self, _name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound("stub".into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for CountingStubBackend {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
}

impl WorkUnit for CountingStubBackend {
    fn name(&self) -> &str {
        self.inner.id
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

fluent_wvr::impl_component!(CountingStubBackend);

impl InferenceBackend for CountingStubBackend {
    fn backend_id(&self) -> &'static str {
        self.inner.id
    }
    fn model_keys(&self) -> Vec<String> {
        self.inner.keys.clone()
    }
    fn weights(&self, _key: &str) -> Option<Arc<dyn crate::runtime::LlmWeights>> {
        None
    }
    fn chat_backend(
        &self,
        key: &str,
        _instance: Option<&str>,
    ) -> Option<Arc<dyn crate::client::ChatBackend>> {
        if self.inner.keys.iter().any(|k| k == key) {
            lock(&self.consults).push(key.to_string());
            if self.failed {
                return None;
            }
            Some(Arc::new(StubChat {
                response: self.inner.response.clone(),
            }))
        } else {
            None
        }
    }
    fn capabilities(&self) -> BackendCaps {
        BackendCaps::default()
    }
    fn readiness(&self, key: &str) -> Readiness {
        if self.failed && self.inner.keys.iter().any(|k| k == key) {
            Readiness::Failed(format!("stub failure for '{key}'"))
        } else {
            Readiness::Unloaded
        }
    }
}

/// A canned-handle session loader: every load succeeds without touching `ort`.
#[derive(Debug, Clone, Default)]
pub struct StubSessionLoader;

impl SessionLoader for StubSessionLoader {
    fn load(&self, _config: &OnnxConfig, _model_key: &str) -> Result<SessionHandle, OrtError> {
        Ok(SessionHandle::new("stub"))
    }
}
