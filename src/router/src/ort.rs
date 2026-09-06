//! Ort ONNX registry composition (ROADMAP_20260827_ORT §0.4/§0.5).
//!
//! The router is the composition root: it turns `ModelEntry.onnx` blocks into
//! a boot-built `OrtSessionRegistry`, validates exclusivity and task-vs-model
//! sanity at boot, and hands the registry to the server (management refusals,
//! chart embedder). The llama.cpp supervisor, sidecar, and `/instances` never
//! touch onnx models — they are served by the registry alone.

use std::sync::Arc;

use fluent_llm::backend::{BackendCaps, InferenceBackend, Readiness};
use fluent_llm::client::ChatBackend;
#[cfg(feature = "onnx")]
use fluent_llm::runtime::LlmWeights;
use fluent_llm::onnx_error::OrtError;
use fluent_llm::onnx_session::OrtSessionRegistry;
use fluent_wvr::prelude::*;

use crate::config::RouterConfig;
#[cfg(feature = "onnx")]
use crate::instances::manager::resume_snapshot_name;

#[cfg(feature = "onnx")]
use tracing::info;

/// Convenience alias for the shared onnx registry.
pub type OrtRegistry = Arc<OrtSessionRegistry>;

/// Build the disambiguation overlays for the overlay stage
/// (ROADMAP_20260827_ORT §2.5): one `PromptRouterOverlay` per `overlay_models`
/// key whose registered onnx model is a `ZeroShotRouting` two-tower session,
/// each scored against the given route descriptions. The router (composition
/// root) supplies the route labels from `RoutingConfig`.
#[cfg(feature = "onnx")]
pub fn disambiguation_overlays<S: std::hash::BuildHasher>(
    registry: &OrtRegistry,
    model_keys: &[String],
    routes: &std::collections::HashMap<String, crate::config::RouteRef, S>,
) -> Result<Vec<Arc<dyn fluent_llm::backend::ResidualOverlay>>, OrtError> {
    let labels: Vec<fluent_llm::backend::RouteLabel> = routes
        .iter()
        .map(|(name, r)| fluent_llm::backend::RouteLabel {
            route: name.clone(),
            description: r.description.clone(),
        })
        .collect();
    let mut overlays = Vec::new();
    for key in model_keys {
        let Some(overlay) = fluent_onnx::build_prompt_router_overlay(registry, key, labels.clone())?
        else {
            continue;
        };
        tracing::info!(
            target: "router.ort",
            model = %key,
            route_labels = labels.len(),
            "prompt-router overlay built",
        );
        overlays.push(overlay);
    }
    Ok(overlays)
}

/// No-ort build: no overlays can be built — the builder skips the overlay
/// stage (fail-open).
#[cfg(not(feature = "onnx"))]
pub fn disambiguation_overlays<S: std::hash::BuildHasher>(
    _registry: &OrtRegistry,
    model_keys: &[String],
    _routes: &std::collections::HashMap<String, crate::config::RouteRef, S>,
) -> Result<Vec<Arc<dyn fluent_llm::backend::ResidualOverlay>>, OrtError> {
    if !model_keys.is_empty() {
        tracing::warn!(
            target: "router.ort",
            models = ?model_keys,
            "overlay models declared but this build has the `onnx` feature off; \
             overlay stage skipped (fail-open)",
        );
    }
    Ok(Vec::new())
}

/// Build the ort encoder for a model and wrap it in a `CachedEmbeddingProvider`
/// (the router is the composition root; caching is deliberately not done
/// inside fluent-onnx). `Ok(None)` when the model is not registered as a
/// `FillMask` encoder.
#[cfg(feature = "onnx")]
pub fn onnx_chart_embedder(
    registry: &OrtRegistry,
    model_key: &str,
) -> Result<Option<Arc<dyn fluent_llm::embeddings::EmbeddingProvider>>, OrtError> {
    let encoder = fluent_onnx::build_encoder_from_registry(registry, model_key)?;
    let Some(encoder) = encoder else {
        return Ok(None);
    };
    let cached = fluent_llm::embeddings::CachedEmbeddingProvider::new(encoder);
    Ok(Some(Arc::new(cached)))
}

/// ColBERT two-stage reranker: builds a `ColbertRetriever` from the registry
/// for `model_key` (must be a `LateInteraction` task). The caller wraps it
/// in a `ColbertReranker` and uses it for MaxSim-based candidate re-ranking.
///
/// Returns `Ok(Some(retriever))` when available; `Ok(None)` when the model
/// is not registered or not a `LateInteraction` task.
#[cfg(feature = "onnx")]
pub fn onnx_colbert_reranker(
    registry: &OrtRegistry,
    model_key: &str,
) -> Result<Option<fluent_onnx::ColbertRetriever>, OrtError> {
    fluent_onnx::build_colbert_from_registry(registry, model_key)
}

/// No-ort build: no ColBERT reranker available.
#[cfg(not(feature = "onnx"))]
pub fn onnx_colbert_reranker(
    _registry: &OrtRegistry,
    _model_key: &str,
) -> Result<Option<()>, OrtError> {
    Ok(None)
}

/// The decode seam behind the onnx `ChatBackend` (ROADMAP M2.1). Decouples the
/// backend's grammar-wiring + limiter from the concrete `OrtLlmSession` decode
/// so hermetic tests inject a fake decoder (no real model). `complete` runs one
/// chat call: chat-template render → tokenize → prefill → (optionally
/// grammar-constrained) decode to text.
#[cfg(feature = "onnx")]
pub(crate) trait OnnxLlmRunner: Send + Sync {
    fn complete(
        &self,
        messages: &[fluent_llm::ChatMessage],
        grammar: Option<&mut (dyn fluent_onnx::Grammar + 'static)>,
        max_tokens: Option<usize>,
        params: fluent_onnx::LlmParams,
    ) -> Result<String, fluent_llm::LlmError>;

    /// Run the chat call onto a named context (ROADMAP M6): the same
    /// template→decode flow, but the KV persists in `ctx` so a follow-up call
    /// continues from where the previous stopped. Default = single-shot.
    fn complete_on_context(
        &self,
        _ctx: &fluent_onnx::OnnxContext,
        messages: &[fluent_llm::ChatMessage],
        grammar: Option<&mut (dyn fluent_onnx::Grammar + 'static)>,
        max_tokens: Option<usize>,
        params: fluent_onnx::LlmParams,
    ) -> Result<String, fluent_llm::LlmError> {
        self.complete(messages, grammar, max_tokens, params)
    }
}

/// The real runner over an `OrtLlmSession` (wraps [`fluent_onnx::OnnxChatCompletion`]).
#[cfg(feature = "onnx")]
struct RealOnnxRunner {
    session: Arc<fluent_onnx::OrtLlmSession>,
}

#[cfg(feature = "onnx")]
impl OnnxLlmRunner for RealOnnxRunner {
    fn complete(
        &self,
        messages: &[fluent_llm::ChatMessage],
        grammar: Option<&mut (dyn fluent_onnx::Grammar + 'static)>,
        max_tokens: Option<usize>,
        params: fluent_onnx::LlmParams,
    ) -> Result<String, fluent_llm::LlmError> {
        fluent_onnx::OnnxChatCompletion::new(self.session.clone())
            .complete(messages, grammar, max_tokens, params)
            .map_err(|e| fluent_llm::LlmError::Api(e.to_string()))
    }

    fn complete_on_context(
        &self,
        ctx: &fluent_onnx::OnnxContext,
        messages: &[fluent_llm::ChatMessage],
        grammar: Option<&mut (dyn fluent_onnx::Grammar + 'static)>,
        max_tokens: Option<usize>,
        params: fluent_onnx::LlmParams,
    ) -> Result<String, fluent_llm::LlmError> {
        fluent_onnx::OnnxChatCompletion::new(self.session.clone())
            .complete_on_context(ctx, messages, grammar, max_tokens, params)
            .map_err(|e| fluent_llm::LlmError::Api(e.to_string()))
    }
}

/// The onnx generative `ChatBackend` (ROADMAP M2.1/M2.2): a `CausalLm` session
/// served behind the `fluent_llm::client::ChatBackend` seam, with optional
/// grammar-constrained structured decoding.
///
/// - `chat_complete` → free text (no grammar — summarization / free generation).
/// - `chat_complete_with_extras` reads `extras["response_format"]["schema"]` and
///   builds a [`fluent_onnx::JsonObjectGrammar`] / [`fluent_onnx::JsonArrayGrammar`]
///   from it (the llama-fork `response_format` vocabulary), so a constrained
///   caller's output is structurally valid by construction. An unrepresentable
///   schema degrades to free text (the caller's post-hoc validator is the
///   backstop — the grammar is a strictness improvement, never a gate).
///
/// Every decode is bounded by a [`fluent_concurrency::pool::Limiter`] (the
/// router's thread-budget rule): a CPU decode never starves a GPU dispatch.
#[cfg(feature = "onnx")]
pub(crate) struct OnnxChatBackend {
    runner: Arc<dyn OnnxLlmRunner>,
    vocab: Arc<dyn fluent_onnx::TokenVocab>,
    params: fluent_onnx::LlmParams,
    limiter: fluent_concurrency::pool::Limiter,
    /// The role's shared context pool, when this backend is context-bound
    /// (ROADMAP M6). `None` keeps the single-shot decode — byte-identical for
    /// an onnx role with no declared `instances` block.
    pool: Option<Arc<fluent_onnx::OnnxContextPool>>,
    /// The bound context name (the role's pool context). `extras["instance"]`
    /// overrides it per call; `None` + a pool still allows dynamic targeting.
    context: Option<String>,
    /// Context-name → profile resolution (the role's `instances` block).
    profile_for: Arc<dyn Fn(&str) -> fluent_onnx::OnnxContextProfile + Send + Sync>,
}

#[cfg(feature = "onnx")]
impl OnnxChatBackend {
    /// Build a single-shot backend (no named contexts).
    pub(crate) fn new(
        runner: Arc<dyn OnnxLlmRunner>,
        vocab: Arc<dyn fluent_onnx::TokenVocab>,
        params: fluent_onnx::LlmParams,
    ) -> Self {
        Self {
            runner,
            vocab,
            params,
            limiter: fluent_concurrency::pool::Limiter::new(1),
            pool: None,
            context: None,
            profile_for: Arc::new(|name| fluent_onnx::OnnxContextProfile {
                group: name.to_string(),
                n_ctx: 0,
                max_ctx: None,
                pinned: false,
                resume: false,
            }),
        }
    }

    /// Build a context-bound backend (ROADMAP M6): decodes run on the role's
    /// named contexts (created on demand) so their KV persists across calls.
    pub(crate) fn with_contexts(
        runner: Arc<dyn OnnxLlmRunner>,
        vocab: Arc<dyn fluent_onnx::TokenVocab>,
        params: fluent_onnx::LlmParams,
        pool: Arc<fluent_onnx::OnnxContextPool>,
        context: Option<String>,
        profile_for: Arc<dyn Fn(&str) -> fluent_onnx::OnnxContextProfile + Send + Sync>,
    ) -> Self {
        Self {
            runner,
            vocab,
            params,
            limiter: fluent_concurrency::pool::Limiter::new(1),
            pool: Some(pool),
            context,
            profile_for,
        }
    }

    /// Run a single-shot decode through the limiter. `grammar` is `None` for
    /// free text.
    fn run_limited(
        &self,
        messages: &[fluent_llm::ChatMessage],
        mut grammar: Option<Box<dyn fluent_onnx::Grammar>>,
    ) -> Result<String, fluent_llm::LlmError> {
        let runner = Arc::clone(&self.runner);
        let params = self.params;
        let messages = messages.to_vec();
        // The sync `complete` runs in the `FnOnce` (before the async block), so
        // the grammar borrow never crosses an await boundary.
        self.limiter.run_sync(move || {
            let result = runner.complete(&messages, grammar.as_deref_mut(), None, params);
            async move { result }
        })
    }

    /// Run a decode onto a named context through the limiter (ROADMAP M6).
    fn run_on_context(
        &self,
        ctx: &Arc<fluent_onnx::OnnxContext>,
        messages: &[fluent_llm::ChatMessage],
        mut grammar: Option<Box<dyn fluent_onnx::Grammar>>,
    ) -> Result<String, fluent_llm::LlmError> {
        let runner = Arc::clone(&self.runner);
        let params = self.params;
        let messages = messages.to_vec();
        let ctx_arc = Arc::clone(ctx);
        self.limiter.run_sync(move || {
            let result =
                runner.complete_on_context(&ctx_arc, &messages, grammar.as_deref_mut(), None, params);
            async move { result }
        })
    }

    /// Select the target context for this call: `extras["instance"]` if
    /// present, else the backend's bound `context` (the pool context).
    fn target_context(&self, extras: &serde_json::Value) -> Option<String> {
        extras
            .get("instance")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| self.context.clone())
    }

    /// The extras-driven decode (ROADMAP M6): a targeted context gets its KV
    /// restored from `extras["snapshot"]` before decode and re-snapshotted when
    /// `resume`/`save_snapshot` asks, mirroring the llama `snapshot`/`instance`/
    /// `id_slot` request fields. Absent a pool/context → single-shot.
    fn run_with_extras(
        &self,
        messages: &[fluent_llm::ChatMessage],
        extras: &serde_json::Value,
        grammar: Option<Box<dyn fluent_onnx::Grammar>>,
    ) -> Result<String, fluent_llm::LlmError> {
        let Some(pool) = &self.pool else {
            return self.run_limited(messages, grammar);
        };
        let Some(name) = self.target_context(extras) else {
            return self.run_limited(messages, grammar);
        };
        let ctx = pool.ensure_context(&name, (self.profile_for)(&name));
        // ROADMAP M7 §1 resize-to-demand: a request that declares a bigger
        // context need than the context's allocated `n_ctx` grows the context
        // to fit (up to `max_ctx`); a need beyond `max_ctx` fails with the same
        // error shape a too-large llama request gets today (no new fail-open).
        if let Some(need) = extras.get("num_ctx").and_then(serde_json::Value::as_u64) {
            if need > ctx.n_ctx() {
                if let Some(cap) = ctx.max_ctx() {
                    if need > cap {
                        return Err(fluent_llm::LlmError::Api(format!(
                            "requested context {need} exceeds max_ctx {cap} for onnx context {name}"
                        )));
                    }
                }
                pool.resize(&name, need)
                    .map_err(|e| fluent_llm::LlmError::Api(e.to_string()))?;
            }
        }
        // Restore a prior snapshot before continuing (the resume half).
        if let Some(snap) = extras.get("snapshot").and_then(serde_json::Value::as_str) {
            ctx.kv_cache()
                .restore_sync(snap)
                .map_err(|e| fluent_llm::LlmError::Api(e.to_string()))?;
        }
        let result = self.run_on_context(&ctx, messages, grammar);
        // Snapshot the advanced KV on completion when resume/save asks.
        let save = extras
            .get("save_snapshot")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                let is_resume = extras
                    .get("resume")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                is_resume.then(|| resume_snapshot_name(&name))
            });
        if let Some(snap) = save {
            let _ = ctx.kv_cache().save_sync(&snap);
        }
        result
    }
}

#[cfg(feature = "onnx")]
impl fluent_llm::client::ChatBackend for OnnxChatBackend {
    fn chat_complete(&self, messages: &[fluent_llm::ChatMessage]) -> Result<String, fluent_llm::LlmError> {
        self.run_limited(messages, None)
    }

    fn chat_complete_with_extras(
        &self,
        messages: &[fluent_llm::ChatMessage],
        extras: &serde_json::Value,
    ) -> Result<String, fluent_llm::LlmError> {
        // The llama-fork `response_format.schema` vocabulary: a structured
        // caller declares the output shape; we constrain decode to it. An
        // absent/unrepresentable schema → free text (fail-open to the caller's
        // post-hoc validator).
        let grammar = extras
            .get("response_format")
            .and_then(|rf| rf.get("schema"))
            .and_then(|schema| {
                fluent_onnx::grammar_from_json_schema(schema, Arc::clone(&self.vocab))
            });
        self.run_with_extras(messages, extras, grammar)
    }
}

/// Build the onnx generative `ChatBackend` for `key` from the boot registry
/// (ROADMAP M2.1). Returns `None` when `key` is not a registered `CausalLm`
/// session — fail-open, with a loud warning when the key *is* registered but
/// mistyped (a mis-configured generative role is never silent).
#[cfg(feature = "onnx")]
pub fn onnx_chat_backend(
    registry: &OrtRegistry,
    key: &str,
) -> Result<Option<Arc<dyn fluent_llm::client::ChatBackend>>, OrtError> {
    use fluent_onnx::HuggingFaceVocab;

    let Some(session) = fluent_onnx::build_llm_session(registry, key)? else {
        if registry.is_registered(key) {
            tracing::warn!(
                target: "router.ort",
                model = %key,
                "onnx_chat_backend: key registered but not a CausalLm session — \
                 no onnx backend (fail-open to HTTP/deterministic)",
            );
        }
        return Ok(None);
    };
    // Grammar-constrained decodes use the same tokenizer the session was built
    // with (the vocab the decode loop reasons over).
    let vocab: Arc<dyn fluent_onnx::TokenVocab> =
        Arc::new(HuggingFaceVocab::new(session.tokenizer().inner().clone()));
    let max_gen_tokens = session.max_gen_tokens();
    let runner: Arc<dyn OnnxLlmRunner> = Arc::new(RealOnnxRunner {
        session: Arc::new(session),
    });
    let backend = OnnxChatBackend::new(runner, vocab, fluent_onnx::LlmParams::default());
    tracing::info!(
        target: "router.ort",
        model = %key,
        max_gen_tokens = max_gen_tokens,
        "onnx generative ChatBackend built",
    );
    Ok(Some(Arc::new(backend)))
}

/// Build a **context-bound** onnx `ChatBackend` for `weights`' named context
/// `name` (ROADMAP M6). Loads the role's context pool on first use (the onnx
/// lazy-residency load point — `ensure_pool` calls `build_llm_session`, which
/// loads the session through the registry) and binds the backend to that
/// context, so decodes persist their KV there. `name == pool_context` for the
/// role's default dispatch point; `local_backend_for_instance` targets any
/// declared context.
#[cfg(feature = "onnx")]
pub fn onnx_context_backend(
    weights: &OnnxWeights,
    name: &str,
) -> Result<Option<Arc<dyn fluent_llm::client::ChatBackend>>, OrtError> {
    use fluent_onnx::HuggingFaceVocab;

    let pool = weights
        .ensure_pool()
        .map_err(|e| {
            // Supervisor-containment parity for onnx (ROADMAP M6 §4): a session
            // that fails to load on first use is a LOUD error, not a silent
            // fall-through — the registry's own `ensure_loaded` already logged
            // the underlying load failure.
            tracing::error!(
                target: "router.ort",
                model = %weights.model_key(),
                context = %name,
                error = %e,
                "onnx context pool load failed on first use — onnx backend unavailable (fail-open)",
            );
            OrtError::Other(format!("onnx context pool load failed: {e}"))
        })?;
    let session = pool.session();
    let vocab: Arc<dyn fluent_onnx::TokenVocab> =
        Arc::new(HuggingFaceVocab::new(session.tokenizer().inner().clone()));
    let runner: Arc<dyn OnnxLlmRunner> = Arc::new(RealOnnxRunner {
        session: session.clone(),
    });
    let role = weights.role().clone();
    let profile_for: Arc<dyn Fn(&str) -> fluent_onnx::OnnxContextProfile + Send + Sync> =
        Arc::new(move |name| onnx_role_profile_for(&role, name));
    let backend = OnnxChatBackend::with_contexts(
        runner,
        vocab,
        fluent_onnx::LlmParams::default(),
        pool.clone(),
        Some(name.to_string()),
        profile_for,
    );
    tracing::info!(
        target: "router.ort",
        model = %pool.model_key(),
        context = %name,
        "onnx context-bound ChatBackend built (KV persists on the named context)",
    );
    Ok(Some(Arc::new(backend)))
}

/// The role's **pool context** — the largest non-default `instances` group,
/// mirroring `ModelEntry::pool_qualifier` for the llama fleet (ROADMAP M6 §3.5).
pub fn onnx_pool_context(role: &fluent_llm::onnx_config::OnnxRoleConfig) -> Option<String> {
    let instances = role.instances.as_ref()?;
    if instances.is_empty() {
        return None;
    }
    let first = instances.iter().next()?;
    let first_group = first.1.group.clone().unwrap_or_else(|| first.0.clone());
    // 1. The non-default profile with the largest sibling count (ties resolve
    //    to the first encountered in deterministic map order).
    let mut best_key: Option<String> = None;
    let mut best_group: Option<String> = None;
    let mut best_count: u32 = 0;
    for (key, p) in instances.iter().filter(|(_, p)| !p.default) {
        let c = p.count.max(1);
        if best_key.is_none() || c > best_count {
            best_key = Some(key.clone());
            best_group = Some(p.group.clone().unwrap_or_else(|| key.clone()));
            best_count = c;
        }
    }
    if let Some(g) = best_group {
        return Some(g);
    }
    // 2. The default profile's group.
    if let Some((key, p)) = instances.iter().find(|(_, p)| p.default) {
        return Some(p.group.clone().unwrap_or_else(|| key.clone()));
    }
    // 3. The single group shared by all profiles.
    if instances
        .iter()
        .all(|(k, p)| p.group.as_deref().unwrap_or(k.as_str()) == first_group)
    {
        Some(first_group)
    } else {
        None
    }
}

/// No-ort build: no onnx `ChatBackend` exists (fail-open). A configured onnx
/// LLM is worth a loud warning so an `onnx`-less build never silently drops the
/// generative role.
#[cfg(not(feature = "onnx"))]
pub fn onnx_chat_backend(
    _registry: &OrtRegistry,
    key: &str,
) -> Result<Option<Arc<dyn fluent_llm::client::ChatBackend>>, OrtError> {
    if !key.is_empty() {
        tracing::warn!(
            target: "router.ort",
            model = %key,
            "onnx_chat_backend requested but this build has the `onnx` feature off — \
             no onnx backend (fail-open)",
        );
    }
    Ok(None)
}

/// ColBERT-based chart candidate reranker: uses MaxSim late-interaction
/// scoring to re-rank HNSW candidates by fine-grained token-level relevance.
#[cfg(feature = "onnx")]
pub struct ColbertChartReranker {
    retriever: fluent_onnx::ColbertRetriever,
    cache: Arc<fluent_onnx::colbert::CachedColbert>,
}

#[cfg(feature = "onnx")]
impl ColbertChartReranker {
    pub fn new(retriever: fluent_onnx::ColbertRetriever) -> Self {
        Self {
            retriever,
            cache: Arc::new(fluent_onnx::colbert::CachedColbert::new(1024)),
        }
    }
}

#[cfg(feature = "onnx")]
impl crate::charts::select::ColbertRerank for ColbertChartReranker {
    fn rerank_candidates(
        &self,
        query: &str,
        candidates: &[(String, f64)],
        doc_texts: &dyn Fn(&str) -> Option<String>,
    ) -> Vec<(String, f64)> {
        use fluent_onnx::colbert::maxsim_score;

        let query_tokens = match self.retriever.encode_query(query) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    target: "router.ort",
                    error = %e,
                    "colbert query encoding failed — falling back to HNSW order"
                );
                return candidates.to_vec();
            }
        };
        if query_tokens.is_empty() {
            return candidates.to_vec();
        }
        let query_refs: Vec<&[f32]> = query_tokens.iter().map(Vec::as_slice).collect();

        let mut scored: Vec<(String, f64)> = candidates
            .iter()
            .filter_map(|(name, _hnsw_score)| {
                let text = doc_texts(name)?;
                let cache_key = format!("{}:{}", self.retriever.name(), name);
                let doc_tokens = if let Some(cached) = self.cache.get(&cache_key) {
                    cached
                } else {
                    let tokens = self.retriever.encode(&text).ok()?;
                    self.cache.insert(cache_key, tokens.clone());
                    tokens
                };
                let doc_refs: Vec<&[f32]> = doc_tokens.iter().map(Vec::as_slice).collect();
                let colbert_score = f64::from(maxsim_score(&query_refs, &doc_refs));
                // Blend: ColBERT score replaces the HNSW score for re-ranking.
                Some((name.clone(), colbert_score))
            })
            .collect();

        // Sort by descending ColBERT score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored.is_empty() {
            candidates.to_vec()
        } else {
            scored
        }
    }
}

/// Build the trained-encoder annotation seam (ROADMAP_20260827_ORT §4.4 /
/// ROADMAP_20260828_ORT_FIXES M2.3): an `EncoderFetchSync` closure that runs
/// the trained-encoder annotation worker over the doc's orth/idx and produces
/// an `AnnotationSet` aligned by construction (record `text` = spacy orth).
///
/// `Ok(Some(closure))` when a registered `FillMask` encoder with declared
/// `annotation_heads` is available; `Ok(None)` when no such encoder is
/// configured (fail-open — the encoder rung is absent from the ladder, exactly
/// the pre-M2 behavior). The closure returns `Err` on any `None`-aligned
/// token, unknown label, or session error, so the ladder falls through to
/// ArcEager (never a partial mix, never a wrong enum).
#[cfg(feature = "onnx")]
pub fn nlp_encoder_fetch(
    registry: &OrtRegistry,
    encoder_model: &str,
) -> Result<Option<spacy_rs::pipeline::EncoderFetchSync>, OrtError> {
    use spacy_rs::pipeline::{AnnotateError, EncoderFetchSync};
    use spacy_rs::{AnnotationSet, Doc, Lemmatizer};

    let Some(worker) = fluent_onnx::build_annotation_worker_from_registry(registry, encoder_model)?
    else {
        // No registered FillMask+heads encoder: the encoder rung is absent
        // (identical to the pre-M2 behavior). A declared-but-untyped model is
        // worth a warning so a mis-configured encoder is never silent.
        if registry.is_registered(encoder_model) {
            tracing::warn!(
                target: "router.ort",
                model = %encoder_model,
                "encoder_model configured but not a registered FillMask session with \
                 annotation_heads — encoder rung absent (fail-open to ArcEager)",
            );
        }
        return Ok(None);
    };

    // The deterministic rule lemmatizer supplies the lemma field (spacy-rs's
    // rule base form). The encoder predicts pos/dep/head only; lemma stays
    // deterministic (VISION: the model annotates *given* tokens, it does not
    // own lexical data).
    let lemmatizer = Lemmatizer::english_rule();
    let closure: EncoderFetchSync = Arc::new(move |doc: &Doc| -> Result<AnnotationSet, AnnotateError> {
        // The doc's reconstructed text is the byte source both the LFM
        // tokenizer and the spacy spans index (orth + idx → spacy_spans).
        let text = doc.text();
        let orth: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
        let idx: Vec<usize> = (0..doc.len()).map(|i| doc.token(i).idx as usize).collect();
        let spans = fluent_onnx::SpacyTokenAligner::spacy_spans(&orth, &idx);

        // Session error → Err → the ladder falls through to ArcEager.
        let annotations = worker
            .annotate(&text, &spans)
            .map_err(|e| AnnotateError::Encoder(e.to_string()))?;
        map_annotations(doc, &lemmatizer, &annotations)
    });

    tracing::info!(
        target: "router.ort",
        model = %encoder_model,
        "encoder annotation seam built (trained UPOS/dep/head rung active)",
    );

    Ok(Some(closure))
}

/// Map `OrtAnnotationWorker` output onto an [`spacy_rs::AnnotationSet`] for
/// `doc`, aligned **by construction** (record `text` = spacy orth). Pure — no
/// session. `pos`/`dep` are the label strings; `head` is the F8 relative
/// offset (`head_abs − i`); a `None` head (special / outside every spacy span)
/// is decoded as ROOT (`dep="root"`, `head=0`); the lemma comes from the
/// deterministic rule [`spacy_rs::Lemmatizer`].
///
/// Returns `Err` on any `None`-aligned token or unknown label, so the ladder
/// falls through to ArcEager — never a partial mix, never a wrong enum.
#[cfg(feature = "onnx")]
fn map_annotations(
    doc: &spacy_rs::Doc,
    lemmatizer: &spacy_rs::Lemmatizer,
    annotations: &[Option<fluent_onnx::TokenAnnotation>],
) -> Result<spacy_rs::AnnotationSet, spacy_rs::pipeline::AnnotateError> {
    use spacy_rs::pipeline::AnnotateError;
    use spacy_rs::{AnnotationRecord, AnnotationSet, Upos};

    if annotations.len() != doc.len() {
        return Err(AnnotateError::Encoder(format!(
            "encoder produced {} annotations for {} spacy tokens",
            annotations.len(),
            doc.len()
        )));
    }
    let mut records = Vec::with_capacity(doc.len());
    for (i, t) in annotations.iter().enumerate() {
        let orth = doc.token_text(i);
        let t = t.as_ref().ok_or_else(|| {
            AnnotateError::Encoder(format!(
                "spacy token {i} has no covering LFM subword — encoder rung falls back"
            ))
        })?;
        // Unknown label → Err → ArcEager (the 7-check gate would reject it
        // anyway; fail-open before a wrong enum is ever written).
        let pos: Upos = t
            .pos
            .parse()
            .map_err(|_| AnnotateError::Encoder(format!("unknown UPOS label {:?}", t.pos)))?;
        let lemma = lemmatizer
            .lemmatize(&orth, pos, 0)
            .into_iter()
            .next()
            .unwrap_or_else(|| orth.to_ascii_lowercase());
        // A head that maps to a special token / outside every spacy span is
        // decoded as ROOT (head=0, dep="root"); otherwise the F8 relative
        // offset is the absolute spacy head minus the token's own index.
        let (dep, head) = match t.head_abs {
            Some(h) => (t.dep.clone(), (h as i32) - (i as i32)),
            None => ("root".to_string(), 0),
        };
        records.push(AnnotationRecord {
            text: orth,
            pos: t.pos.clone(),
            tag: String::new(),
            dep,
            head,
            lemma,
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        });
    }
    Ok(AnnotationSet(records))
}

/// No-ort build: the encoder annotation seam is unavailable (fail-open).
#[cfg(not(feature = "onnx"))]
pub fn nlp_encoder_fetch(
    _registry: &OrtRegistry,
    encoder_model: &str,
) -> Result<Option<spacy_rs::pipeline::EncoderFetchSync>, OrtError> {
    if !encoder_model.is_empty() {
        tracing::warn!(
            target: "router.ort",
            model = %encoder_model,
            "encoder model declared but this build has the `onnx` feature off — \
             encoder rung unavailable (fail-open)",
        );
    }
    Ok(None)
}

/// Build the review worker's PII pre-filter (ROADMAP_20260827_ORT §3.2/§3.3):
/// the ort PII-Detector when `pii_model` names a registered
/// `TokenClassification` session, else the deterministic `RegexPiiDetector`
/// when `auto_enqueue` requires a pre-filter. Fail-open: a mis-declared or
/// absent model degrades to the regex baseline (never a boot error, never a
/// silent no-pre-filter when `auto_enqueue` is on).
///
/// M6: the shared tail of both cfg definitions below — the deterministic
/// regex baseline when `auto_enqueue` requires a pre-filter, else no
/// pre-filter. The cfg split itself stays (the onnx classifier type does not
/// exist without the feature); only the repeated tail is factored out.
fn regex_baseline_or_none(auto_enqueue: bool) -> Option<Arc<dyn fluent_llm::backend::PiiSpanDetector>> {
    if auto_enqueue {
        Some(Arc::new(fluent_llm::RegexPiiDetector))
    } else {
        None
    }
}

#[cfg(feature = "onnx")]
pub fn pii_prefilter(
    registry: Option<&OrtRegistry>,
    pii_model: Option<&str>,
    auto_enqueue: bool,
) -> Result<Option<Arc<dyn fluent_llm::backend::PiiSpanDetector>>, OrtError> {
    let regex_fallback = || regex_baseline_or_none(auto_enqueue);
    match pii_model {
        Some(key) => {
            if let Some(registry) = registry {
                if let Some(detector) = fluent_onnx::build_pii_classifier(registry, key)? {
                    tracing::info!(
                        target: "router.ort",
                        model = %key,
                        "ort PII-Detector pre-filter built",
                    );
                    Ok(Some(detector))
                } else {
                    tracing::warn!(
                        target: "router.ort",
                        model = %key,
                        "pii_model configured but not a registered TokenClassification \
                         session — falling back to the regex PII baseline",
                    );
                    Ok(if auto_enqueue { regex_fallback() } else { None })
                }
            } else {
                tracing::warn!(
                    target: "router.ort",
                    model = %key,
                    "pii_model configured but no onnx registry is available — falling back to \
                     the regex PII baseline",
                );
                Ok(if auto_enqueue { regex_fallback() } else { None })
            }
        }
        None => Ok(if auto_enqueue { regex_fallback() } else { None }),
    }
}

/// No-ort build: the ort classifier is unavailable, so the pre-filter is the
/// deterministic `RegexPiiDetector` when `auto_enqueue` requires one (fail-open).
/// M6: shares the `regex_baseline_or_none` tail with the onnx definition above.
#[cfg(not(feature = "onnx"))]
pub fn pii_prefilter(
    _registry: Option<&OrtRegistry>,
    pii_model: Option<&str>,
    auto_enqueue: bool,
) -> Result<Option<Arc<dyn fluent_llm::backend::PiiSpanDetector>>, OrtError> {
    if pii_model.is_some() {
        tracing::warn!(
            target: "router.ort",
            pii_model = pii_model,
            "pii_model declared but this build has the `onnx` feature off — using the \
             regex PII baseline",
        );
    }
    Ok(regex_baseline_or_none(auto_enqueue))
}

/// Build the ColBERT-backed entity-link scorer (ROADMAP_20260828_ORT_FIXES
/// M3.2): bakes an [`fluent_onnx::EntitySimilarityIndex`] over the concept
/// store's labels at boot, then returns an [`EntityLinkScorer`] closure that
/// encodes a span via the retriever, looks it up against the baked index, and
/// resolves each hit's canonical → `InterlinguaId` through the store.
///
/// Fail-open: no model registered / not a `LateInteraction` task / no labels →
/// an empty scorer (yields no candidates, identical to the pre-M3 stub). A
/// lookup-time encoding error also degrades to "no candidates" (loud warn) —
/// never a block, never a drop.
#[cfg(feature = "onnx")]
pub fn colbert_entity_scorer(
    registry: &OrtRegistry,
    model_key: &str,
    concept_store: &Arc<dyn fluent_concept::ConceptStore>,
    threshold: f64,
) -> Result<crate::server::entity_link::EntityLinkScorer, OrtError> {
    use fluent_onnx::colbert::{bake_entity_index, EntitySimilarityIndex};
    use fluent_types::InterlinguaId;

    let empty: crate::server::entity_link::EntityLinkScorer = Arc::new(|_text| Vec::new());
    let Some(retriever) = fluent_onnx::build_colbert_from_registry(registry, model_key)? else {
        if registry.is_registered(model_key) {
            tracing::warn!(
                target: "router.ort",
                model = %model_key,
                "entity-link scorer model configured but not a registered LateInteraction \
                 session — empty scorer (fail-open)",
            );
        }
        return Ok(empty);
    };

    // Bake once at boot over the concept store's labels: (namespace, canonical,
    // label) triples. An absent label (None) falls back to the canonical name.
    let concepts: Vec<(String, String, String)> = concept_store
        .iter_ids()
        .filter_map(|id| concept_store.get(id).ok())
        .map(|meta| {
            let namespace = format!("{:?}", meta.namespace);
            let label = meta.label.unwrap_or_else(|| meta.canonical_name.clone());
            (namespace, meta.canonical_name, label)
        })
        .collect();
    let index: EntitySimilarityIndex = bake_entity_index(&retriever, &concepts, threshold as f32)?;
    tracing::info!(
        target: "router.ort",
        model = %model_key,
        concepts = concepts.len(),
        threshold = threshold,
        "entity-link ColBERT scorer baked (data-time)",
    );

    let store = Arc::clone(concept_store);
    Ok(Arc::new(move |text: &str| -> Vec<(InterlinguaId, f64)> {
        let query_tokens = match retriever.encode_query(text) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    target: "router.ort",
                    error = %e,
                    "entity-link query encoding failed — no candidates (fail-open)",
                );
                return Vec::new();
            }
        };
        score_span(&query_tokens, &index, store.as_ref())
    }))
}

/// Score already-encoded query tokens against a baked [`fluent_onnx::EntitySimilarityIndex`]
/// and map each above-threshold hit's canonical → `InterlinguaId` through the
/// store. Pure (no session) — the encode step is the only model-dependent part
/// of the entity-link scorer, so the mapping is hermetically testable.
#[cfg(feature = "onnx")]
fn score_span(
    query_tokens: &[Vec<f32>],
    index: &fluent_onnx::EntitySimilarityIndex,
    store: &dyn fluent_concept::ConceptStore,
) -> Vec<(fluent_types::InterlinguaId, f64)> {
    if query_tokens.is_empty() {
        return Vec::new();
    }
    let query_refs: Vec<&[f32]> = query_tokens.iter().map(Vec::as_slice).collect();
    index
        .lookup(&query_refs)
        .into_iter()
        .filter_map(|hit| {
            store
                .resolve_name(&hit.canonical)
                .ok()
                .map(|id| (id, f64::from(hit.score)))
        })
        .collect()
}

/// No-ort build: the ColBERT scorer is unavailable — empty scorer (fail-open).
#[cfg(not(feature = "onnx"))]
pub fn colbert_entity_scorer(
    _registry: &OrtRegistry,
    model_key: &str,
    _concept_store: &Arc<dyn fluent_concept::ConceptStore>,
    _threshold: f64,
) -> Result<crate::server::entity_link::EntityLinkScorer, OrtError> {
    if !model_key.is_empty() {
        tracing::warn!(
            target: "router.ort",
            model = %model_key,
            "entity-link scorer model declared but this build has the `onnx` feature off — \
             empty scorer (fail-open)",
        );
    }
    Ok(Arc::new(|_text| Vec::new()))
}

/// Build the `OrtSessionRegistry` from the role-based ONNX fleet in `config`.
///
/// Each configured role (`encoder`/`pii`/`router`/`policy`/`colbert`) is
/// registered under its stable `OnnxRole::registry_key()` with the task implied
/// by the role. Lifecycle semantics parallel the llama.cpp fleet:
///
/// - `resident: true` (Always) roles load at boot (loud error on a missing
///   model file — the registry refuses to boot half-configured).
/// - `resident: false` (Unloadable) roles register lazily and load on first
///   use.
/// - Task-vs-`config.json` mismatches are loud boot errors.
///
/// Absent ONNX config yields `None` (fully fail-open — the pipeline is
/// pure-deterministic). When the crate is built without the `onnx` feature,
/// `None` is returned.
#[cfg(feature = "onnx")]
pub fn build_onnx_registry(
    config: &RouterConfig,
) -> Result<Option<Arc<OrtSessionRegistry>>, OrtError> {
    let Some(fleet) = config.onnx.as_ref() else {
        return Ok(None);
    };
    if fleet.is_empty() {
        return Ok(None);
    }
    // Resolve tokenizer paths: roles like router and PII often share the
    // encoder's tokenizer. When a role has no tokenizer_path, inherit it
    // from the encoder role. This keeps the simplified config format concise.
    // The generative `Llm` role is exempt — its tokenizer is its own (the
    // model's `tokenizer.json`), never the encoder's.
    let encoder_tokenizer = fleet.encoder.as_ref()
        .and_then(|e| e.model.tokenizer_path.clone());
    let registry = OrtSessionRegistry::new(Arc::new(fluent_onnx::OrtSessionLoader));
    for (role, role_cfg) in fleet.iter() {
        let mut resolved_cfg = role_cfg.clone();
        if resolved_cfg.model.tokenizer_path.is_none()
            && role != fluent_llm::onnx_config::OnnxRole::Llm
        {
            if let Some(ref tok) = encoder_tokenizer {
                resolved_cfg.model.tokenizer_path = Some(tok.clone());
                info!(
                    target: "router.ort",
                    role = ?role,
                    tokenizer = %tok.display(),
                    "tokenizer_path inherited from encoder role",
                );
            }
        }
        let model = resolved_cfg.to_onnx_config(role);
        model.validate()?;
        info!(
            target: "router.ort",
            role = ?role,
            key = role.registry_key(),
            task = ?model.task,
            resident = model.resident,
            pinned = role_cfg.pinned,
            sleep_idle_seconds = role_cfg.sleep_idle_seconds,
            model_path = %model.model_path.display(),
            "onnx role registered",
        );
        // Register with the full residency lifecycle so the onnx residency
        // loop honors the role's `pinned` / `sleep_idle_seconds` directly from
        // the registry entry — no parallel table.
        let policy = model.policy();
        registry.register_with_lifecycle(
            role.registry_key().to_string(),
            model,
            policy,
            role_cfg.pinned,
            role_cfg.sleep_idle_seconds,
        )?;
    }
    Ok(Some(Arc::new(registry)))
}

/// The default idle threshold (seconds) after which an onnx session may be
/// released, when the role's `sleep_idle_seconds` is absent or zero. Feeds
/// the shared residency engine; per-role overrides come from each entry's
/// `sleep_idle_seconds` (registered by `build_onnx_registry`).
pub const DEFAULT_SLEEP_IDLE_SECONDS: i32 = 30;

// ── ROADMAP M4: the onnx half of the shared LlmWeights contract ─────────────
//
// `OnnxWeights` presents one in-process onnx role through the shared
// `fluent_llm::runtime::LlmWeights` surface (the same contract the llama
// adapters in `instances/traits.rs` implement). It wraps the role's generative
// `OnnxContextPool` (built **lazily** — never forces a session load) and the
// `OrtSessionRegistry` entry. The residency engine (M5), the unified
// `/instances` + `/v1/models` facade, `ps`, and `POST /models/unload` drive
// the onnx fleet through it with the `LruLargest` eviction ordering.

/// Map an onnx registry error onto the runtime-agnostic error type. A release
/// refusal (`Always`/pinned) surfaces as [`LlmRuntimeError::UnloadRefused`].
#[cfg(feature = "onnx")]
fn map_ort_err(e: &OrtError) -> fluent_llm::runtime::LlmRuntimeError {
    let s = e.to_string();
    if s.contains("refuses unload") {
        fluent_llm::runtime::LlmRuntimeError::UnloadRefused(s)
    } else {
        fluent_llm::runtime::LlmRuntimeError::Other(s)
    }
}

/// The `LlmWeights` implementor for one onnx role (ROADMAP M4). Wraps the
/// role's `OnnxContextPool` (generative roles) + the registry entry. A loaded
/// generative role with no materialized contexts synthesizes the single
/// `default` context row — the onnx analogue of a plain llama model's
/// footprint. Non-generative roles render no context rows (today's
/// invisibility — their weights still count toward the RAM working set).
#[cfg(feature = "onnx")]
pub struct OnnxWeights {
    model_key: String,
    registry: Arc<OrtSessionRegistry>,
    role: fluent_llm::onnx_config::OnnxRoleConfig,
    pool: std::sync::Mutex<Option<Arc<fluent_onnx::OnnxContextPool>>>,
}

#[cfg(feature = "onnx")]
impl OnnxWeights {
    /// Build the adapter over a registered role's registry entry + role config
    /// (the `instances` block drives `ensure_context` profiles).
    pub fn new(
        model_key: String,
        registry: Arc<OrtSessionRegistry>,
        role: fluent_llm::onnx_config::OnnxRoleConfig,
    ) -> Self {
        Self {
            model_key,
            registry,
            role,
            pool: std::sync::Mutex::new(None),
        }
    }

    /// The role config (the `instances` block + `max_ctx` drive context
    /// profiles).
    pub(crate) fn role(&self) -> &fluent_llm::onnx_config::OnnxRoleConfig {
        &self.role
    }

    /// Whether the registry reports the session loaded.
    fn is_loaded(&self) -> bool {
        self.registry
            .residency_report()
            .iter()
            .any(|r| r.key == self.model_key && r.loaded)
    }

    /// Whether the role's registered session is generative (CausalLm — the only
    /// task that serves named contexts).
    fn is_generative(&self) -> bool {
        self.registry
            .config(&self.model_key)
            .is_some_and(|c| c.task == fluent_llm::onnx_config::OnnxTask::CausalLm)
    }

    /// Build the role's context pool on first use (loads the session through
    /// `build_llm_session` — the M6 lazy-residency load point). Idempotent.
    fn ensure_pool(&self) -> Result<Arc<fluent_onnx::OnnxContextPool>, fluent_llm::runtime::LlmRuntimeError> {
        {
            let guard = common_core::sync::lock(&self.pool);
            if let Some(pool) = guard.as_ref() {
                return Ok(Arc::clone(pool));
            }
        }
        let session = fluent_onnx::build_llm_session(&self.registry, &self.model_key)
            .map_err(|e| map_ort_err(&e))?
            .ok_or_else(|| {
                fluent_llm::runtime::LlmRuntimeError::NotLoaded(format!(
                    "onnx model {} is not a CausalLm session",
                    self.model_key
                ))
            })?;
        let pool = fluent_onnx::OnnxContextPool::new(Arc::new(session), self.model_key.clone());
        *common_core::sync::lock(&self.pool) = Some(Arc::clone(&pool));
        Ok(pool)
    }

    /// The `OnnxContextProfile` for a named context: the role's declared
    /// `instances` block (name-or-key match, `max_ctx` applied), else a default
    /// profile inheriting the role's global `max_ctx` and pin.
    fn profile_for(&self, name: &str) -> fluent_onnx::OnnxContextProfile {
        onnx_role_profile_for(&self.role, name)
    }
}

/// Resolve an onnx role's named-context profile (ROADMAP M6): the role's
/// `instances` block (name-or-key match, `max_ctx` applied), else a default
/// profile inheriting the role's global `max_ctx` and pin. Shared by
/// `OnnxWeights::ensure_context` and the context-bound `OnnxChatBackend`.
#[cfg(feature = "onnx")]
fn onnx_role_profile_for(role: &fluent_llm::onnx_config::OnnxRoleConfig, name: &str) -> fluent_onnx::OnnxContextProfile {
    if let Some((key, p)) = role.instances.as_ref().and_then(|m| {
        m.iter()
            .find(|(k, p)| p.name.as_deref() == Some(name) || k.as_str() == name)
    }) {
        let mut p = p.clone();
        p.apply_max_ctx();
        let group = p.group.clone().unwrap_or_else(|| key.clone());
        fluent_onnx::OnnxContextProfile {
            group,
            n_ctx: p.num_ctx,
            max_ctx: p.max_ctx.or_else(|| role.model.max_ctx.map(|c| c as u64)),
            pinned: p.pinned,
            resume: p.resume,
        }
    } else {
        fluent_onnx::OnnxContextProfile {
            group: name.to_string(),
            n_ctx: 0, // the pool default
            max_ctx: role.model.max_ctx.map(|c| c as u64),
            pinned: role.pinned,
            resume: false,
        }
    }
}

#[cfg(feature = "onnx")]
#[async_trait::async_trait]
impl fluent_llm::runtime::LlmWeights for OnnxWeights {
    fn model_key(&self) -> &str {
        &self.model_key
    }

    fn weights_bytes(&self) -> u64 {
        self.registry.resident_bytes(&self.model_key).unwrap_or(0)
    }

    fn pinned(&self) -> bool {
        self.registry.is_pinned(&self.model_key)
    }

    fn refuse_unload(&self) -> bool {
        // `Always` residency OR pinned — the exact onnx residency rule
        // (`policy.is_always() || pinned` is never released/evicted). Only the
        // shared engine consumes this (the admin unload path reads the
        // registry's `refuses_unload` directly, so this does not affect it).
        self.registry.refuses_unload(&self.model_key) || self.registry.is_pinned(&self.model_key)
    }

    fn is_loaded(&self) -> bool {
        OnnxWeights::is_loaded(self)
    }

    fn sleep_idle_seconds(&self) -> Option<i32> {
        self.registry.sleep_idle_seconds(&self.model_key)
    }

    async fn ensure_loaded(&self) -> Result<(), fluent_llm::runtime::LlmRuntimeError> {
        let _ = self
            .registry
            .ensure_loaded(&self.model_key)
            .map_err(|e| map_ort_err(&e))?;
        Ok(())
    }

    async fn unload(&self) -> Result<(), fluent_llm::runtime::LlmRuntimeError> {
        self.registry.release(&self.model_key).map_err(|e| map_ort_err(&e))?;
        // The session is gone: drop the pool's contexts (their KV dies with it).
        let pool = common_core::sync::lock(&self.pool).take();
        if let Some(pool) = pool {
            for key in pool.context_keys() {
                pool.destroy(&key);
            }
        }
        Ok(())
    }

    fn touch(&self) {
        self.registry.touch(&self.model_key);
    }

    fn last_used(&self) -> i64 {
        self.registry.last_used_of(&self.model_key).unwrap_or(0)
    }

    async fn residency_rows(&self) -> Vec<fluent_llm::runtime::LlmResidencyRow> {
        // A cold/unloaded role is invisible (today's rule); only generative
        // roles serve context rows.
        if !self.is_loaded() || !self.is_generative() {
            return Vec::new();
        }
        let pool = common_core::sync::lock(&self.pool).clone();
        if let Some(pool) = pool {
            let rows = pool.residency_rows();
            if !rows.is_empty() {
                return rows;
            }
        }
        // A loaded generative role with no materialized contexts synthesizes
        // the single `default` row (the onnx analogue of a plain llama model).
        let resident = self.weights_bytes();
        vec![fluent_llm::runtime::LlmResidencyRow {
            context_key: format!("{}:default", self.model_key),
            group: "default".into(),
            n_ctx: fluent_onnx::DEFAULT_ONNX_CONTEXT_TOKENS,
            parallel: 1,
            pinned: self.registry.is_pinned(&self.model_key),
            resume: false,
            state: "loaded".into(),
            runtime: fluent_llm::runtime::LlmRuntime::Onnx,
            model_bytes: resident,
            context_bytes: 0,
            compute_bytes: 0,
            total_bytes: 0,
            vram_bytes: 0,
            last_used: self.registry.last_used_of(&self.model_key).unwrap_or(0),
        }]
    }

    fn context(&self, name: &str) -> Option<Arc<dyn fluent_llm::runtime::LlmContext>> {
        let pool = common_core::sync::lock(&self.pool).clone()?;
        let ctx = pool.context(name)?;
        Some(ctx as Arc<dyn fluent_llm::runtime::LlmContext>)
    }

    async fn ensure_context(
        &self,
        name: &str,
    ) -> Result<Arc<dyn fluent_llm::runtime::LlmContext>, fluent_llm::runtime::LlmRuntimeError> {
        let pool = self.ensure_pool()?;
        let profile = self.profile_for(name);
        let ctx = pool.ensure_context(name, profile);
        Ok(ctx as Arc<dyn fluent_llm::runtime::LlmContext>)
    }

    fn eviction_policy(&self) -> fluent_llm::runtime::EvictionPolicy {
        fluent_llm::runtime::EvictionPolicy::LruLargest
    }
}

/// Build one `OnnxWeights` per registered onnx role (ROADMAP M4). The role's
/// `instances` block (from `config.onnx`) drives `ensure_context` profiles;
/// the registry supplies the lifecycle + residency data. Onnx rows appear on
/// the unified `/instances`/`/v1/models` facades only when the onnx fleet is
/// configured.
#[cfg(feature = "onnx")]
pub fn onnx_weights_impls(
    registry: &OrtRegistry,
    config: &RouterConfig,
) -> Vec<Arc<dyn fluent_llm::runtime::LlmWeights>> {
    let mut out = Vec::new();
    let Some(fleet) = config.onnx.as_ref() else {
        return out;
    };
    for (role, role_cfg) in fleet.iter() {
        let key = role.registry_key().to_string();
        if !registry.is_registered(&key) {
            continue;
        }
        out.push(Arc::new(OnnxWeights::new(key, Arc::clone(registry), role_cfg.clone())));
    }
    out
}

/// No-ort build: no onnx weights can exist — the fleet degrades to nothing
/// (fail-open, byte-identical to today's onnx invisibility).
#[cfg(not(feature = "onnx"))]
pub fn onnx_weights_impls(
    _registry: &OrtRegistry,
    _config: &RouterConfig,
) -> Vec<Arc<dyn fluent_llm::runtime::LlmWeights>> {
    Vec::new()
}

/// No-ort build: the ONNX fleet degrades to nothing (fail-open), but its
/// presence is worth a loud warning so a config that expects ort wiring is not
/// silently degraded.
#[cfg(not(feature = "onnx"))]
pub fn build_onnx_registry(
    config: &RouterConfig,
) -> Result<Option<Arc<OrtSessionRegistry>>, OrtError> {
    if let Some(fleet) = config.onnx.as_ref() {
        if !fleet.is_empty() {
            let roles: Vec<&'static str> = fluent_llm::onnx_config::OnnxRole::all()
                .iter()
                .filter(|r| fleet.has(**r))
                .map(|r| r.registry_key())
                .collect();
            tracing::warn!(
                target: "router.ort",
                roles = ?roles,
                "onnx roles declared but this build has the `onnx` feature off; \
                 treating them as unavailable (fail-open)",
            );
        }
    }
    Ok(None)
}

/// The onnx inference backend: a thin [`InferenceBackend`] adapter over the
/// boot-built onnx registry for the generative (`onnx/llm`) role.
/// Single-shot dispatch delegates to [`onnx_chat_backend`], named-context
/// dispatch to [`onnx_context_backend`] over the role's `OnnxWeights`, and
/// `weights()` serves that same `OnnxWeights` — the adapter adds routing,
/// never a second session or construction path.
pub struct OnnxBackend {
    llm_key: String,
    registry: OrtRegistry,
    single_shot: Option<Arc<dyn ChatBackend>>,
    pool_context: Option<String>,
    has_instances: bool,
    #[cfg(feature = "onnx")]
    llm_weights: Option<Arc<OnnxWeights>>,
}

impl OnnxBackend {
    /// Build the adapter from the boot registry + config. `None` when no
    /// generative llm role is configured (fail-open — the backend simply is
    /// not registered).
    pub fn from_config(config: &RouterConfig, registry: &OrtRegistry) -> Option<Self> {
        let llm_key = fluent_llm::onnx_config::OnnxRole::Llm.registry_key().to_string();
        let role = config.onnx.as_ref()?.llm.as_ref()?.clone();
        let single_shot = onnx_chat_backend(registry, &llm_key).ok().flatten();
        #[cfg(feature = "onnx")]
        let llm_weights = Some(Arc::new(OnnxWeights::new(
            llm_key.clone(),
            Arc::clone(registry),
            role.clone(),
        )));
        let has_instances = role
            .instances
            .as_ref()
            .is_some_and(|m| !m.is_empty());
        let pool_context = onnx_pool_context(&role);
        if single_shot.is_some() {
            tracing::info!(
                target: "router.ort",
                model = %llm_key,
                has_instances = has_instances,
                "onnx generative backend wired as the default local backend",
            );
        }
        Some(Self {
            llm_key,
            registry: Arc::clone(registry),
            single_shot,
            pool_context,
            has_instances,
            #[cfg(feature = "onnx")]
            llm_weights,
        })
    }

    /// The role's single-shot backend (the default dispatch point when the
    /// role declares no `instances` block). The composition root wires this
    /// into the server's explicit onnx-llm seam.
    pub fn single_shot(&self) -> Option<Arc<dyn ChatBackend>> {
        self.single_shot.clone()
    }
}

impl FieldAccess for OnnxBackend {
    fn set_field(&mut self, name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(name.into()))
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        Err(FieldError::NotFound(name.into()))
    }
    fn field_names(&self) -> &'static [&'static str] {
        &[]
    }
}

impl Describable for OnnxBackend {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "description": "onnx generative inference backend",
        })
    }
}

impl WorkUnit for OnnxBackend {
    fn name(&self) -> &str {
        "onnx"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("onnx backend adapter"))
    }
}

fluent_wvr::impl_component!(OnnxBackend);

impl InferenceBackend for OnnxBackend {
    fn backend_id(&self) -> &'static str {
        "onnx"
    }
    fn model_keys(&self) -> Vec<String> {
        vec![self.llm_key.clone()]
    }
    fn weights(&self, key: &str) -> Option<Arc<dyn fluent_llm::runtime::LlmWeights>> {
        #[cfg(feature = "onnx")]
        {
            if key != self.llm_key || !self.registry.is_registered(key) {
                return None;
            }
            self.llm_weights.clone().map(|w| w as _)
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = key;
            None
        }
    }
    fn chat_backend(
        &self,
        key: &str,
        instance: Option<&str>,
    ) -> Option<Arc<dyn ChatBackend>> {
        if key != self.llm_key {
            return None;
        }
        match instance {
            #[cfg(feature = "onnx")]
            Some(name) => self
                .llm_weights
                .as_ref()
                .and_then(|w| onnx_context_backend(w, name).ok().flatten()),
            #[cfg(not(feature = "onnx"))]
            Some(_) => None,
            None if self.has_instances => {
                let ctx = self.pool_context.clone()?;
                #[cfg(feature = "onnx")]
                {
                    self.llm_weights
                        .as_ref()
                        .and_then(|w| onnx_context_backend(w, &ctx).ok().flatten())
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = ctx;
                    None
                }
            }
            None => self.single_shot.clone(),
        }
    }
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            named_contexts: true,
            kv_snapshot: true,
            grammar_constrained: cfg!(feature = "onnx"),
            ..BackendCaps::default()
        }
    }
    fn readiness(&self, key: &str) -> Readiness {
        if key != self.llm_key {
            return Readiness::Unloaded;
        }
        if self
            .registry
            .residency_report()
            .iter()
            .any(|r| r.key == self.llm_key && r.loaded)
        {
            return Readiness::Loaded;
        }
        // Registered but lazy, or (no-ort builds) never loadable: known but
        // not resident. Unregistered keys stay fail-open `Unloaded` too.
        Readiness::Unloaded
    }
}

#[cfg(test)]
#[path = "../tests/ort.rs"]
mod tests;
