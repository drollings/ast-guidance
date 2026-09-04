//! `OnnxContextPool` — the onnx half of "one weights load, N named context
//! windows" (ROADMAP M2).
//!
//! Wraps one loaded [`crate::llm::OrtLlmSession`] (the shared
//! `Arc<Mutex<Session>>` + tokenizer + [`crate::config::LlmIo`]) with a
//! registry of named [`crate::context::OnnxContext`]s. The llama fork serves
//! this contract over HTTP (`/instances`); onnx serves it in-process: contexts
//! are created on demand, interleave on the shared session (their KV lives
//! per-context), and can be resized, destroyed, touched, and rendered into the
//! shared residency envelope.

use std::sync::Arc;

use common_core::registry::ConcurrentRegistry;
use fluent_llm::runtime::{LlmResidencyRow, LlmRuntime};

use crate::context::{OnnxContext, OnnxContextProfile};
use crate::error::OrtError;
use crate::llm::OrtLlmSession;

/// One loaded weights instance serving N named context windows. Contexts share
/// the session (and its `Mutex`-serialized graph runs) but each owns its own
/// KV, so interleaving is safe.
pub struct OnnxContextPool {
    /// One weights load: the shared decode surface for every context.
    session: Arc<OrtLlmSession>,
    /// The public model key (e.g. `"onnx/llm"`) — the prefix of each context's
    /// `context_key` in the residency envelope.
    model_key: String,
    /// The named contexts (create-on-demand, idempotent).
    contexts: ConcurrentRegistry<String, OnnxContext>,
}

impl OnnxContextPool {
    /// Build the pool over one loaded session. The pool is `Arc`-shared with
    /// every dispatch path that needs to create/reach contexts.
    pub fn new(session: Arc<OrtLlmSession>, model_key: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            session,
            model_key: model_key.into(),
            contexts: ConcurrentRegistry::new(),
        })
    }

    /// The shared decode session (for `prefill`/`decode_step`).
    pub fn session(&self) -> &Arc<OrtLlmSession> {
        &self.session
    }

    /// The public model key.
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    /// Ensure a named context exists, creating it on demand (mirrors the llama
    /// fork's `ensure_instance`). Idempotent: a second resolve returns the same
    /// context. A profile with `n_ctx == 0` inherits the pool default.
    pub fn ensure_context(&self, name: &str, profile: OnnxContextProfile) -> Arc<OnnxContext> {
        self.contexts.resolve_or_create(name.to_string(), move |key| {
            OnnxContext::new(key.clone(), profile)
        })
    }

    /// The named context, if it is currently materialized.
    pub fn context(&self, name: &str) -> Option<Arc<OnnxContext>> {
        self.contexts.get(&name.to_string())
    }

    /// Resize a named context, bounded by its `max_ctx` (growth past the cap is
    /// refused). An unknown context is a loud error.
    pub fn resize(&self, name: &str, n_ctx: u64) -> Result<(), OrtError> {
        let ctx = self.context(name).ok_or_else(|| {
            OrtError::Other(format!("onnx context {name} not found"))
        })?;
        ctx.resize(n_ctx)
            .map_err(|e| OrtError::Other(e.to_string()))
    }

    /// Destroy a named context, freeing its KV. Returns the removed context, if
    /// it was materialized.
    pub fn destroy(&self, name: &str) -> Option<Arc<OnnxContext>> {
        let removed = self.contexts.remove(&name.to_string());
        if let Some(ctx) = &removed {
            ctx.clear();
        }
        removed
    }

    /// Record dispatch recency on a named context (no-op for an unknown name).
    pub fn touch(&self, name: &str) {
        if let Some(ctx) = self.context(name) {
            ctx.touch();
        }
    }

    /// Number of materialized contexts.
    pub fn len(&self) -> usize {
        self.contexts.len()
    }

    /// Whether no context is materialized.
    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    /// The materialized context keys.
    pub fn context_keys(&self) -> Vec<String> {
        self.contexts.keys()
    }

    /// The residency view: one [`LlmResidencyRow`] per live context —
    /// `runtime: Onnx`, `vram_bytes: 0`, RAM-resident. Model bytes are omitted
    /// (the `OnnxWeights` adapter reports the shared weights footprint in M4).
    pub fn residency_rows(&self) -> Vec<LlmResidencyRow> {
        let mut rows = Vec::with_capacity(self.contexts.len());
        for key in self.contexts.keys() {
            let Some(ctx) = self.context(&key) else {
                continue;
            };
            let context_bytes = ctx.kv_bytes();
            rows.push(LlmResidencyRow {
                context_key: format!("{}:{key}", self.model_key),
                group: ctx.group().to_string(),
                n_ctx: ctx.n_ctx(),
                parallel: 1,
                pinned: ctx.pinned(),
                resume: ctx.resume(),
                state: if ctx.seq_len() > 0 {
                    "loaded".into()
                } else {
                    "sleeping".into()
                },
                runtime: LlmRuntime::Onnx,
                model_bytes: 0,
                context_bytes,
                compute_bytes: 0,
                total_bytes: context_bytes,
                vram_bytes: 0,
                last_used: ctx.last_used(),
            });
        }
        rows
    }
}

#[cfg(test)]
#[path = "../tests/context_pool.rs"]
mod tests;
