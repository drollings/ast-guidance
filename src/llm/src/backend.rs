//! Backend plugin layer: one base trait, one multi-backend registry, and
//! the neutral consumer types shared by every backend.
//!
//! Each inference fleet (a child-process pool, an in-process session set, or
//! a stub in tests) presents the same [`InferenceBackend`] interface to the
//! router regardless of where it was assembled. The router holds an
//! [`InferenceRegistry`] of type-erased backends and routes each request to
//! the first backend that serves its model key, via the shared
//! `first_accept_in_order_sync` fallback combinator. New backends are added
//! as new types behind the trait, never as new branches in dispatch.
//!
//! The PII / overlay / entity-link consumer types also live here so backend
//! consumers never import a backend implementation crate for a plain data
//! shape. Backend crates re-export these paths for compatibility.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use common_core::registry::KeyedRegistry;
use fluent_wvr::Component;

use crate::client::ChatBackend;
use crate::embeddings::EmbeddingProvider;
use crate::runtime::{LlmContext, LlmWeights};

// ─── Capabilities / readiness ───────────────────────────────────────────────

/// What a backend can do. All flags default to false; each backend adapter
/// reports the flags its transport actually honors.
// Capability flags are the intentional shape here (one bool per transport
// capability, all false by default); the struct stays a plain data bag.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendCaps {
    /// Serves named context windows with per-context KV state.
    pub named_contexts: bool,
    /// Persists KV snapshots across eviction.
    pub kv_snapshot: bool,
    /// Constrains decoding with a supplied JSON schema.
    pub grammar_constrained: bool,
    /// Serves embeddings as well as chat completions.
    pub embeddings: bool,
    /// Serves streaming completions.
    pub streaming: bool,
}

impl BackendCaps {
    /// A backend serving only chat completions over named contexts with KV.
    #[must_use]
    pub fn chat_with_contexts() -> Self {
        Self {
            named_contexts: true,
            kv_snapshot: true,
            ..Self::default()
        }
    }
}

/// Load state of one model key on a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Readiness {
    /// Resident and serving.
    Loaded,
    /// Load in flight; callers should fall through to the next backend.
    Loading,
    /// Known but not resident; load on demand.
    Unloaded,
    /// Load failed; carries the failure for diagnostics.
    Failed(String),
}

/// Declarative shape of a named context window. Mirrors the onnx context
/// profile field-for-field without importing it, so callers describe a
/// window once for any backend that serves named contexts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextProfile {
    /// The context's group.
    pub group: String,
    /// The allocated context window in tokens (`0` → the backend default).
    pub n_ctx: u64,
    /// The context-size cap (`None` = no cap).
    pub max_ctx: Option<u64>,
    /// Whether the context is pinned (never evicted).
    pub pinned: bool,
    /// Whether the context is resume-marked (KV preserved across eviction).
    pub resume: bool,
}

impl Default for ContextProfile {
    fn default() -> Self {
        Self {
            group: "default".into(),
            n_ctx: 0,
            max_ctx: None,
            pinned: false,
            resume: false,
        }
    }
}

// ─── Backend loader seam ────────────────────────────────────────────────────

/// How a backend loads a model key. Generalizes the session-loader seam:
/// the real loader spawns or maps the weights, tests inject a stub.
pub trait BackendLoader: Send + Sync {
    /// Load the model key (spawn + probe, or session load).
    fn load(
        &self,
        model_key: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>>;
    /// Probe load state without loading.
    fn probe_ready(&self, model_key: &str) -> Readiness;
}

/// A backend load failure.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Transport or spawn failure before serving.
    #[error("backend load failed: {0}")]
    Load(String),
    /// The requested key is unknown to this backend.
    #[error("backend key not found: {0}")]
    NotFound(String),
}

// ─── Base trait ─────────────────────────────────────────────────────────────

/// The base interface every inference backend implements.
///
/// This is the `Component + domain ops` split: [`Component`] carries the
/// orchestration surface (field access, schema, work unit) and this trait
/// carries the inference surface. The router stores
/// `Arc<dyn InferenceBackend>` and never branches on implementation type.
pub trait InferenceBackend: Component {
    /// Short identifier: `"llama"`, `"onnx"`, … Used as the registry key.
    fn backend_id(&self) -> &'static str;
    /// Model keys this backend serves (e.g. `["lfm2.5-2.6b"]`, `["onnx/llm"]`).
    fn model_keys(&self) -> Vec<String>;
    /// The shared weights handle for a served key, if resident or loadable.
    fn weights(&self, key: &str) -> Option<Arc<dyn LlmWeights>>;
    /// The chat transport for a served key and optional instance.
    fn chat_backend(
        &self,
        key: &str,
        instance: Option<&str>,
    ) -> Option<Arc<dyn ChatBackend>>;
    /// The embedding transport for a served key. Defaults to `None` —
    /// only backends that serve embeddings override this.
    fn embed_provider(&self, _key: &str) -> Option<Arc<dyn EmbeddingProvider>> {
        None
    }
    /// What this backend's transport honors.
    fn capabilities(&self) -> BackendCaps;
    /// Load state of one model key. Defaults to `Unloaded` — only backends
    /// tracking residency override this.
    fn readiness(&self, _key: &str) -> Readiness {
        Readiness::Unloaded
    }
}

/// Backends serving multiple named context windows per weights instance,
/// each with its own KV cache.
pub trait NamedContexts: InferenceBackend {
    /// Ensure the named context exists under the model key, creating it
    /// on demand from `profile`.
    fn ensure_context(
        &self,
        key: &str,
        profile: ContextProfile,
    ) -> Arc<dyn LlmContext>;
}

// ─── Registry ───────────────────────────────────────────────────────────────

/// Multi-backend registry: one entry per `backend_id`, each a type-erased
/// backend handle. Routes across many backends — there is no single-active
/// selection; every registered backend is consulted in key order until one
/// serves the requested model key.
pub struct InferenceRegistry {
    backends: KeyedRegistry<String, Arc<dyn InferenceBackend>>,
}

impl InferenceRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            backends: KeyedRegistry::new(),
        }
    }

    /// Register a backend under its [`InferenceBackend::backend_id`].
    ///
    /// # Panics
    ///
    /// Panics on duplicate `backend_id`. Duplicate registration is a
    /// programming error, not a runtime condition.
    pub fn register(&mut self, backend: Arc<dyn InferenceBackend>) {
        let id = backend.backend_id().to_string();
        assert!(
            !self.backends.contains(&id),
            "duplicate inference backend registration: '{id}'"
        );
        self.backends.insert(id, backend);
    }

    /// Remove the backend registered under `backend_id`.
    pub fn unregister(&mut self, backend_id: &str) -> Option<Arc<dyn InferenceBackend>> {
        self.backends.remove(backend_id)
    }

    /// Look up a backend by its id.
    pub fn get(&self, backend_id: &str) -> Option<Arc<dyn InferenceBackend>> {
        self.backends.get(backend_id).cloned()
    }

    /// All registered backend ids, in sorted order.
    pub fn backend_ids(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }

    /// Route a chat request: the first backend (in registry key order)
    /// serving `key` wins. `None` when no backend serves the key.
    ///
    /// Two gates run before any construction call: backends whose model keys
    /// do not contain the key's base are not candidates (no probe, no call),
    /// and candidates reporting [`Readiness::Failed`] are skipped with that
    /// failure as the recorded cause. A miss falls through to the next
    /// candidate; every fallback therefore carries a genuine cause.
    pub fn route_chat(
        &self,
        key: &str,
        instance: Option<&str>,
    ) -> Option<Arc<dyn ChatBackend>> {
        let base = base_key(key);
        let rungs: Vec<Arc<dyn InferenceBackend>> =
            self.backends.values().map(Arc::clone).collect();
        fluent_concurrency::ladder::first_accept_in_order_sync(
            rungs,
            |backend| {
                if !is_candidate(&backend, base) {
                    return Ok::<_, std::convert::Infallible>(None);
                }
                if matches!(backend.readiness(key), Readiness::Failed(_)) {
                    return Ok(None);
                }
                Ok(backend.chat_backend(key, instance))
            },
            |_| false,
        )
        .ok()
        .flatten()
    }

    /// Route an embedding request: the first backend (in registry key order)
    /// serving `key` wins. `None` when no backend serves embeddings for it.
    /// Same candidate + readiness gates as [`Self::route_chat`].
    pub fn route_embed(&self, key: &str) -> Option<Arc<dyn EmbeddingProvider>> {
        let base = base_key(key);
        let rungs: Vec<Arc<dyn InferenceBackend>> =
            self.backends.values().map(Arc::clone).collect();
        fluent_concurrency::ladder::first_accept_in_order_sync(
            rungs,
            |backend| {
                if !is_candidate(&backend, base) {
                    return Ok::<_, std::convert::Infallible>(None);
                }
                if matches!(backend.readiness(key), Readiness::Failed(_)) {
                    return Ok(None);
                }
                Ok(backend.embed_provider(key))
            },
            |_| false,
        )
        .ok()
        .flatten()
    }

    /// Number of registered backends.
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

/// The base model key of a possibly-qualified inference key
/// (`base:qualifier` → `base`; a bare or malformed key passes through).
/// Mirrors the router's qualified-key split with identical semantics so the
/// registry can pre-filter candidates without importing router config.
fn base_key(key: &str) -> &str {
    match key.split_once(':') {
        Some((base, qualifier)) if !base.is_empty() && !qualifier.is_empty() => base,
        _ => key,
    }
}

/// Whether the backend is a routing candidate for `key`: the key's base must
/// be one of its served model keys. Non-candidates are skipped without any
/// probe or construction call, so unregistered keys resolve to `None` with
/// zero backend consultations.
fn is_candidate(backend: &Arc<dyn InferenceBackend>, base: &str) -> bool {
    backend.model_keys().iter().any(|k| k == base)
}

impl Default for InferenceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for InferenceRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceRegistry")
            .field("backends", &self.backend_ids())
            .finish()
    }
}

// ─── Capability mediation ───────────────────────────────────────────────────

/// Capability token mediating registry access, mirroring the memory-plugin
/// pattern: holders route through the shared registry; an empty registry
/// resolves to `None` (graceful no-op) rather than an error.
#[derive(Clone)]
pub struct InferenceCapability {
    registry: Arc<std::sync::RwLock<InferenceRegistry>>,
}

impl InferenceCapability {
    /// Bind a capability to the shared registry.
    pub fn new(registry: Arc<std::sync::RwLock<InferenceRegistry>>) -> Self {
        Self { registry }
    }

    /// Route a chat request through the shared registry.
    pub fn route_chat(
        &self,
        key: &str,
        instance: Option<&str>,
    ) -> Option<Arc<dyn ChatBackend>> {
        let registry = common_core::sync::lock_read(&self.registry);
        registry.route_chat(key, instance)
    }

    /// Route an embedding request through the shared registry.
    pub fn route_embed(&self, key: &str) -> Option<Arc<dyn EmbeddingProvider>> {
        let registry = common_core::sync::lock_read(&self.registry);
        registry.route_embed(key)
    }
}

// ─── Neutral consumer types ─────────────────────────────────────────────────
// Moved here so backend consumers share one definition. Backend crates keep
// re-exporting these paths; the shapes below are the single source of truth.

/// A detected PII span: byte offsets into the scanned text, the label (the
/// regex pattern name or the token-classification label, e.g.
/// `"credential.password"`), and the detector's confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiiSpan {
    /// Byte offset of the span's start within the scanned text.
    pub start: usize,
    /// Byte offset of the span's end within the scanned text.
    pub end: usize,
    /// The PII label (`ssn`, `credential.api_key`, …).
    pub label: String,
    /// The detector's confidence (regex baseline: `1.0`; classifier: the
    /// softmax probability of the span's tokens, averaged).
    pub score: f64,
}

impl PiiSpan {
    /// A span from byte offsets and a label, with the given confidence.
    #[must_use]
    pub fn new(start: usize, end: usize, label: impl Into<String>, score: f64) -> Self {
        Self {
            start,
            end,
            label: label.into(),
            score,
        }
    }

    /// Slice the span's text out of `source`. `None` when out of bounds.
    #[must_use]
    pub fn slice<'a>(&self, source: &'a str) -> Option<&'a str> {
        source.get(self.start..self.end)
    }
}

/// A PII detection failure. The review pre-filter treats every error as an
/// empty span set (fail-open — never a job drop and never a rejection).
#[derive(Debug, thiserror::Error)]
pub enum PiiError {
    #[error("PII classifier inference failed: {0}")]
    Inference(String),
}

/// The PII detection seam: `dyn` at the request boundary, concrete impls
/// inside.
pub trait PiiSpanDetector: Send + Sync {
    /// Detect PII spans in `text`. Errors are fail-open at the boundary —
    /// the caller maps an error to an empty span set, never to a job drop.
    fn detect(&self, text: &str) -> Result<Vec<PiiSpan>, PiiError>;
}

/// The deterministic PII baseline: wraps the canonical
/// [`crate::pii_patterns`] table (never a duplicated pattern list) and
/// reports every regex match as a `PiiSpan` at the pattern's byte offsets
/// with confidence `1.0`. Always available — no model, no feature gate,
/// fail-open by construction.
#[derive(Debug, Clone, Default)]
pub struct RegexPiiDetector;

impl PiiSpanDetector for RegexPiiDetector {
    fn detect(&self, text: &str) -> Result<Vec<PiiSpan>, PiiError> {
        let mut spans = Vec::new();
        for pattern in crate::pii_patterns::pii_patterns() {
            for m in pattern.regex.find_iter(text) {
                spans.push(PiiSpan::new(m.start(), m.end(), pattern.name, 1.0));
            }
        }
        Ok(spans)
    }
}

/// What kind of parse residual a sentence (or span) carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualKind {
    /// A sentence whose parse was uncertain below the confidence floor.
    Disambiguation,
    /// A PII-shaped span.
    PiiSpan,
    /// A PROPN span with no resolved entity.
    EntityLink,
    /// A parse whose dependency structure wants correction.
    ParseCorrection,
    /// A span worth a concept-level summary.
    ConceptSummary,
}

/// A deterministic parse residual: the sentence/span the deterministic layer
/// was unsure about, plus byte span and structured context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Residual {
    pub kind: ResidualKind,
    /// Optional byte span into the source request text.
    pub span: Option<(usize, usize)>,
    /// The sentence or span text the overlay scores.
    pub text: String,
    /// Structured context the producer attached.
    #[serde(default)]
    pub meta: serde_json::Value,
}

impl Residual {
    /// A disambiguation residual over a sentence.
    #[must_use]
    pub fn disambiguation(sentence: impl Into<String>) -> Self {
        Self {
            kind: ResidualKind::Disambiguation,
            span: None,
            text: sentence.into(),
            meta: serde_json::Value::Object(Default::default()),
        }
    }
}

/// The result of running an overlay over a residual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayContribution {
    pub kind: ResidualKind,
    /// The overlay's primary score, when the contribution is score-shaped.
    pub score: Option<f64>,
    /// Structured payload (e.g. `{"route_hints": [...]}` for disambiguation).
    #[serde(default)]
    pub payload: serde_json::Value,
}

/// An overlay failure. The overlay stage treats every error as skip-and-log
/// (fail-open enrichment, never a gate).
#[derive(Debug, thiserror::Error)]
pub enum OverlayError {
    #[error("overlay inference failed: {0}")]
    Inference(String),
    #[error("overlay rejected the residual: {0}")]
    Rejected(String),
}

/// A `ResidualOverlay` consumes residuals of one [`ResidualKind`] and produces
/// contributions. `dyn` at the request boundary.
pub trait ResidualOverlay: Send + Sync {
    /// The residual kind this overlay consumes.
    fn kind(&self) -> ResidualKind;

    /// Score the residual. Errors are fail-open at the stage boundary.
    fn run(&self, residual: &Residual) -> Result<OverlayContribution, OverlayError>;
}

/// A route the disambiguation overlay scores against: its config key and the
/// description the prompt line is built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteLabel {
    pub route: String,
    pub description: String,
}

/// The entity-link scoring seam: given a span text, return ranked
/// `(entity_id, score)` candidates. Generic over the entity-id type so this
/// neutral crate stays free of domain imports; consumers instantiate it with
/// their interlingua id (e.g. `EntityLinkScorer<InterlinguaId>` at the router
/// seam). A scorer that yields no candidates is fail-open.
pub type EntityLinkScorer<EntityId = u64> =
    Arc<dyn Fn(&str) -> Vec<(EntityId, f64)> + Send + Sync>;

#[cfg(test)]
#[path = "../tests/pii.rs"]
mod tests;
