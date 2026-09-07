//! Pipeline builder - constructs pipeline stages from `RouterConfig`.
//! Separated from `config.rs` to keep the configuration types focused
//! on data definition rather than orchestration.
//!
//! # Seams
//!
//! This file bundles four builder facades that a future split could separate
//! into submodules when any grows a second consumer or a dedicated test suite:
//!
//! 1. **Pipeline build** (`impl RouterConfig::build_pipeline`,
//!    `PipelineParams`, `build_classification_engine`) — the two-stage
//!    deterministic/classifier pipeline construction.
//! 2. **Escalation build** (`build_escalation_ladders`,
//!    `escalation_backends`) — the per-group ladder/backend assembly.
//! 3. **`LlmClient` DIP factory** (`build_llm_client` /
//!    `frontier_api_client`) — client construction from a shared `reqwest`
//!    handle and `api_key_env`.
//! 4. **Ledger/coordinator build** (`build_ledger`, `build_coordinator`) —
//!    ledger + agent coordinator wiring.
//!
//! Today they are kept together because they all live on `impl RouterConfig`
//! and share `resolve_classifier_model_key` / `classifier_intelligence` /
//! `build_classifier_client` helpers; the `#[allow(clippy::too_many_arguments)]`
//! on the pipeline/engine constructors is a builder-shape, not a call-shape.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common_core::config::load_json_or_default;
use fluent_llm::client::ChatBackend;
use fluent_llm::{LlmClient, LlmConfig};

use super::{
    default_true, refine_policy::RouterRefinePolicy, strip_declaration_params, ModelEntry,
    RejectPatterns, RouterConfig,
};
use crate::pipeline::PipelineOrchestrator;
use crate::score_matrix::ScoreMatrix;
use crate::stages::classifier::ClassifierBackendResolver;
use crate::target_match::{TargetBackends, TargetMatcher};

/// In-group target-matching policy for a pipeline (-4.6 of the routing
/// roadmap). `SelfAssess` (default) runs the VISION ladder: each candidate
/// target self-assesses the prompt and defers to the next, more-intelligent
/// group member when the assessed complexity exceeds its `intelligence`.
/// `Static` restores today's behavior - the cheapest qualifying model is
/// picked at route-resolution time with no self-assessment calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TargetMatchMode {
    /// Run the per-candidate self-assessment ladder for 2+ member groups
    /// (single-member groups resolve statically, byte-identical to today).
    #[default]
    #[serde(rename = "self_assess")]
    SelfAssess,
    /// Pick the cheapest qualifying model at resolution time (no LLM calls).
    #[serde(rename = "static")]
    Static,
}

/// NLP annotation ordering (ROADMAP_20260827_ORT §2.7). `LlmFirst` (the
/// default) preserves today's behavior: the LLM annotation rung runs first
/// when a fetch is wired, with ArcEager/rule beneath. `DeterministicFirst`
/// skips the up-front annotation LLM call and lets the `overlay` stage consult
/// models on residuals — it is only reachable together with non-empty
/// `overlay_models` (the two changes are inseparable in config).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NlpOrdering {
    /// LLM annotation rung first when a fetch is wired (today's ladder).
    #[default]
    #[serde(rename = "llm_first")]
    LlmFirst,
    /// ArcEager baseline; models consulted on residuals via overlays.
    #[serde(rename = "deterministic_first")]
    DeterministicFirst,
}

/// A1a: cohesive NLP view derived from the flat `PipelineParams` fields.
/// The three seams (`nlp` stage, `overlay` stage, encoder) are now accessed
/// through this single view — the flat fields remain for serde backward
/// compat but all pipeline logic goes through `nlp_config()` / `overlay_enabled()`.
#[derive(Debug, Clone)]
pub struct NlpConfig {
    pub enabled: bool,
    pub ordering: NlpOrdering,
    pub refine_policy: Option<RouterRefinePolicy>,
    pub encoder_model: Option<String>,
}

/// Named pipeline parameters. Pipelines are stored as a map keyed by name.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct PipelineParams {
    #[serde(default = "default_true")]
    pub deterministic_prefilter: bool,
    /// Milestone 6: run the deterministic NLP parse stage (`spacy-rs`) on the
    /// request text and publish the per-sentence routing signals under the
    /// `"nlp_parse"` handoff key. When the resolved classifier backend is
    /// available, the stage also attempts the LLM annotation rung (full UD
    /// deps) before falling back to the deterministic star parse. Default
    /// `false` — strictly additive.
    #[serde(default)]
    pub nlp: bool,
    #[serde(default = "default_true")]
    pub classifier: bool,
    #[serde(default = "default_true")]
    pub router: bool,
    #[serde(default = "default_coherence_threshold")]
    pub coherence_threshold: f64,
    #[serde(default)]
    pub classifier_model: Option<String>,
    /// Bounds the number of concurrently executing classifier LLM calls for
    /// this pipeline. `None` defaults to `available_parallelism()`.
    #[serde(default)]
    pub classifier_max_concurrency: Option<usize>,
    #[serde(default)]
    pub blacklist: Option<String>,
    #[serde(default)]
    pub score_matrix: Option<ScoreMatrix>,
    /// When `true` and a `score_matrix` is configured, the matrix's
    /// top-scoring route **decides** the dispatch target (weighted selection
    /// over the four score axes) instead of the LLM's `action`/`target` being
    /// metadata-only. Coherence/safety thresholds and the `reject` action stay
    /// as hard gates that run first. Default `false` so existing behavior
    /// and goldens are untouched until a deployment opts in.
    #[serde(default)]
    pub score_matrix_authoritative: bool,
    /// Maximum retry attempts for the classifier when its LLM response fails
    /// JSON parsing (`0` = disabled, the default - existing behavior is
    /// unchanged). When `> 0`, the classifier stage is wrapped in a
    /// `RetryClassifier` that re-executes it with escalating corrective
    /// prompts on `metadata.fallback = true`.
    #[serde(default)]
    pub classifier_retry_max: u32,
    /// Escalating corrective system prompts used on each retry attempt (the
    /// last prompt is reused when retries exceed the list length). Defaults to
    /// two stock prompts that demand strict JSON.
    #[serde(default = "default_classifier_retry_prompts")]
    pub classifier_retry_prompts: Vec<String>,
    /// In-group target-matching policy (-4.6). `SelfAssess` (default) runs the
    /// target-matching ladder for 2+ member groups; `Static` restores today's
    /// cheapest-qualifying pick.
    #[serde(default)]
    pub target_match: TargetMatchMode,
    /// Per-self-assessment wall-clock budget for the target-matching ladder.
    /// Defaults to `DEFAULT_TOTAL_TIMEOUT_MS` (the shared timeout constant).
    #[serde(default = "default_target_match_timeout_ms")]
    pub target_match_timeout_ms: u64,
    /// Model keys whose onnx `ZeroShotRouting` sessions back the overlay
    /// stage's disambiguation scoring. Non-empty enables the overlay stage.
    /// A legacy `"overlay": true` key in JSON is ignored by serde (unknown
    /// field); the migration path is to set `overlay_models`.
    #[serde(default)]
    pub overlay_models: Vec<String>,
    /// Bounds concurrent overlay `run` calls (default 2).
    #[serde(default)]
    pub overlay_max_concurrency: Option<usize>,
    /// The parse-confidence floor below which a sentence (or the doc) yields a
    /// disambiguation residual. Defaults to the shared disambiguation floor.
    #[serde(default = "default_overlay_disambiguation_floor")]
    pub overlay_disambiguation_floor: f64,
    /// NLP annotation ordering (default `llm_first` — today's behavior,
    /// unchanged for existing `nlp: true` deployments). `deterministic_first`
    /// is only reachable together with overlay models.
    #[serde(default)]
    pub nlp_ordering: NlpOrdering,
    /// Optional redirect gate (OFF by default): when set AND the zero-shot
    /// eval corpus gate passes, a top overlay hint at/above the threshold can
    /// turn into a `Rerouted` classifier verdict. Inert until the ≥100-case
    /// golden corpus lands (see ROADMAP_20260827_ORT §2.6a).
    #[serde(default)]
    pub overlay_redirect_threshold: Option<f64>,
    /// The ort encoder model key for the trained-encoder annotation rung
    /// (ROADMAP_20260827_ORT §4.4). When set and the model is registered in
    /// the ort registry, the NlpStage runs the encoder between the LLM and
    /// ArcEager rungs. `None` (default) disables the encoder rung — the
    /// ladder is unchanged from today's behavior.
    #[serde(default)]
    pub encoder_model: Option<String>,
    /// Refinement policy for the deterministic-first annotation ladder
    /// (ROADMAP_20260831_ARCEAGER M4.1). When `Some`, this DTO is converted
    /// `Into<spacy_rs::RefinePolicy>` verbatim; when `None` (default), the
    /// effective policy follows `nlp_ordering` — `LlmFirst` ⇒ `Always`,
    /// `DeterministicFirst` ⇒ `OnUncertain` with default thresholds — so
    /// existing configs are byte-identical. Every `refine_on_*` flag is
    /// independently tunable without a code change.
    #[serde(default)]
    pub refine_policy: Option<RouterRefinePolicy>,
}

fn default_target_match_timeout_ms() -> u64 {
    fluent_llm::constants::DEFAULT_TOTAL_TIMEOUT_MS
}

/// Default parse-confidence floor for the overlay disambiguation residual
/// ([A] producer self-doubt axis). Locked by the calibration corpus in
/// `tests/overlay_calibration.rs`: 50 known-quality sentences + 20
/// confident-but-wrong controls (which must NOT yield residuals, proving
/// [A] is not correctness). Residual yield 30/50, 0 false positives on
/// the 40 confident controls at this value.
pub const OVERLAY_DISAMBIGUATION_FLOOR_DEFAULT: f64 = 0.5;

fn default_overlay_disambiguation_floor() -> f64 {
    OVERLAY_DISAMBIGUATION_FLOOR_DEFAULT
}

/// Default redirect gate ([B] task-value axis). Locked to OFF (`None`) by
/// the calibration corpus in `tests/overlay_calibration.rs`: 100
/// route-labeled prompts + 20 adversarial-nearby pairs (which must NOT
/// redirect). Precision on controls is 100% only because nothing redirects
/// while OFF; any arming requires re-calibration to 100% first.
pub const OVERLAY_REDIRECT_THRESHOLD_DEFAULT: Option<f64> = None;

fn default_classifier_retry_prompts() -> Vec<String> {
    vec![
        "Your previous output failed JSON parsing. Respond with ONLY a single valid JSON \
         object matching the requested schema - no prose, no markdown fences, no trailing text."
            .into(),
        "Your previous output was still not valid JSON. Output exactly one JSON object with \
         the required fields and nothing else."
            .into(),
    ]
}

impl Default for PipelineParams {
    fn default() -> Self {
        Self {
            deterministic_prefilter: true,
            nlp: false,
            classifier: true,
            router: true,
            coherence_threshold: default_coherence_threshold(),
            classifier_model: None,
            classifier_max_concurrency: None,
            blacklist: None,
            score_matrix: None,
            score_matrix_authoritative: false,
            classifier_retry_max: 0,
            classifier_retry_prompts: default_classifier_retry_prompts(),
            target_match: TargetMatchMode::SelfAssess,
            target_match_timeout_ms: default_target_match_timeout_ms(),
            overlay_models: Vec::new(),
            overlay_max_concurrency: None,
            overlay_disambiguation_floor: default_overlay_disambiguation_floor(),
            nlp_ordering: NlpOrdering::LlmFirst,
            overlay_redirect_threshold: OVERLAY_REDIRECT_THRESHOLD_DEFAULT,
            encoder_model: None,
            refine_policy: None,
        }
    }
}

impl PipelineParams {
    /// Cohesive NLP view (A1a): single struct for the `nlp` stage's seams.
    #[must_use]
    pub fn nlp_config(&self) -> NlpConfig {
        NlpConfig {
            enabled: self.nlp,
            ordering: self.nlp_ordering,
            refine_policy: self.refine_policy,
            encoder_model: self.encoder_model.clone(),
        }
    }

    /// Overlay is enabled when `overlay_models` is non-empty: the single
    /// named predicate for the derived flag.
    #[must_use]
    pub fn overlay_enabled(&self) -> bool {
        !self.overlay_models.is_empty()
    }
}

fn default_coherence_threshold() -> f64 {
    0.70
}

/// Default classifier concurrency cap: the machine's available parallelism,
/// never fewer than 1 worker.
fn default_classifier_concurrency() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().max(1))
}

/// NLP dependencies threaded into the pipeline builder from the composition
/// root (ROADMAP_20260828_ORT M1.1/M1.2). The concept store (built over the
/// shared ledger connection **before** pipeline build) backs the interlingua
/// resolver; the optional strings path wires the durable `StringStore` (G9).
/// `Default` is fully fail-open: no resolver, in-memory strings — byte-identical
/// to today's `NlpPipeline::en_default()`.
#[derive(Clone, Default)]
pub struct NlpDeps {
    /// The shared concept registry over the ledger connection. `Some` wires
    /// `InterlinguaResolver` into the NLP pipeline so `NlpStage` stamps
    /// interlingua ids; `None` disables interlingua stamping (with a `warn!`).
    pub concept_store: Option<Arc<dyn fluent_concept::ConceptStore>>,
    /// Optional durable StringStore path: the pipeline's vocab is loaded from
    /// (`en_default_with_strings`) and persisted (`persist_strings`) to this
    /// path. `None` keeps the in-memory vocab.
    pub strings_path: Option<PathBuf>,
}

impl std::fmt::Debug for NlpDeps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NlpDeps")
            .field("concept_store", &self.concept_store.is_some())
            .field("strings_path", &self.strings_path)
            .finish()
    }
}

/// Canonical build deps for a named pipeline (Plan B F4): one struct carries
/// the three optional seams that previously required three overloads
/// (`classifier_backend`, `onnx`, `nlp_deps`). Old overloads now delegate
/// here (DRY — ordering + fallback logic lives in exactly one place).
#[derive(Default)]
pub struct PipelineBuildDeps<'a> {
    pub classifier_backend: Option<Arc<dyn ChatBackend>>,
    pub onnx: Option<&'a crate::ort::OrtRegistry>,
    pub nlp_deps: NlpDeps,
}

/// Build the `spacy-rs` NLP pipeline for a named pipeline from `deps`: a
/// vocab (loaded from `deps.strings_path` when configured, else fresh), a
/// tokenizer, and — when a concept store is present — an
/// [`InterlinguaResolver`] over it so the pipeline stamps interlingua ids.
/// Fail-open: `None` deps produce the deterministic baseline with no resolver.
fn build_nlp_pipeline(
    deps: &NlpDeps,
) -> Result<Arc<spacy_rs::NlpPipeline>, spacy_rs::PipelineError> {
    let vocab = match deps.strings_path.as_deref() {
        Some(path) => Arc::new(spacy_rs::vocab::Vocab::load_or_empty(
            path,
            spacy_rs::lang::en::lexicon_config(),
        )),
        None => Arc::new(spacy_rs::vocab::Vocab::new(spacy_rs::lang::en::lexicon_config())),
    };
    let tokenizer = spacy_rs::lang::en::tokenizer(Arc::clone(&vocab))?;
    let resolver = deps.concept_store.clone().map(|concepts| {
        Arc::new(spacy_rs::InterlinguaResolver::new(
            concepts,
            Arc::clone(vocab.strings()),
        ))
    });
    match resolver {
        Some(resolver) => spacy_rs::NlpPipeline::new_with_resolver(
            vocab,
            tokenizer,
            spacy_rs::AnnotationValidator::new(),
            Some(resolver),
        )
        .map(Arc::new),
        None => spacy_rs::NlpPipeline::new(
            vocab,
            tokenizer,
            spacy_rs::AnnotationValidator::new(),
        )
        .map(Arc::new),
    }
}

impl RouterConfig {
    pub fn load_reject_patterns(path: &str) -> RejectPatterns {
        load_json_or_default::<RejectPatterns>(Path::new(path))
    }

    /// Build a standalone `NlpPipeline` for the `arc_ready` spacy overlay,
    /// threading the same `InterlinguaResolver` (via `nlp_deps.concept_store`)
    /// as the request-time pipelines. `None` on build error (fail-open — the
    /// spacy overlay just isn't wired).
    pub fn overlay_nlp_pipeline(&self, nlp_deps: &NlpDeps) -> Option<Arc<spacy_rs::NlpPipeline>> {
        build_nlp_pipeline(nlp_deps).ok()
    }

    pub fn routing_config(&self) -> super::RoutingConfig {
        // The system prompt is always derived from the root classifier
        // node's children (M3c: the flat `system_prompt` field is gone).
        let system_prompt = self
            .classification
            .as_ref()
            .and_then(super::ClassificationTree::derive_system_prompt)
            .unwrap_or_default();
        // Effective instance maps: entries declaring none inherit the whole
        // fleet-default map (the R1 fallback), materialized into this derived
        // view so every routing path below resolves through one code path.
        // The authoritative `RouterConfig` is untouched (round-trips clean;
        // supervision keeps reading the originals).
        let models = self
            .models
            .iter()
            .map(|(key, entry)| {
                let mut effective = entry.clone();
                if effective.instances.is_none() {
                    effective
                        .instances
                        .clone_from(&self.default_params.instances);
                }
                (key.clone(), effective)
            })
            .collect();
        super::RoutingConfig {
            routes: self.routes_view(),
            models,
            model_groups: self.model_groups.clone(),
            system_prompt,
            safety_threshold: self.safety_threshold,
            default_route: self.default_route.clone(),
            score_matrix: None,
            onnx_keys: self.onnx_role_keys(),
            roles: self.roles.clone(),
        }
    }

    /// The registry keys of the configured in-process onnx roles (e.g.
    /// `onnx/llm`). A `model_groups` member that names one is a valid dispatch
    /// target served by the onnx `ChatBackend`. Empty when no onnx fleet (or
    /// this build's `onnx` feature) is configured.
    pub fn onnx_role_keys(&self) -> std::collections::BTreeSet<String> {
        self.onnx
            .as_ref()
            .map(|fleet| {
                fleet
                    .iter()
                    .map(|(role, _)| role.registry_key().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn build_named_pipeline(&self, name: &str) -> Option<PipelineOrchestrator> {
        self.build_named_pipeline_with_backend(name, None)
    }

    pub fn build_named_pipeline_with_backend(
        &self,
        name: &str,
        classifier_backend: Option<Arc<dyn ChatBackend>>,
    ) -> Option<PipelineOrchestrator> {
        self.build_named_pipeline_with_backend_and_onnx(name, classifier_backend, None)
    }

    /// Build a named pipeline with an injected classifier backend and the boot
    /// ort registry (ROADMAP_20260827_ORT §2.5). The registry supplies the
    /// overlay `Arc`s for `PipelineParams.overlay_models`; `None` (or absent
    /// onnx config) is fully fail-open — the overlay stage is skipped.
    ///
    /// No `NlpDeps` are supplied, so the NLP pipeline is built fail-open (no
    /// interlingua resolver, in-memory strings) — equivalent to
    /// `NlpPipeline::en_default()`. Composition roots that have built a concept
    /// store should call [`Self::build_named_pipeline_with_backend_onnx_and_nlp`].
    pub fn build_named_pipeline_with_backend_and_onnx(
        &self,
        name: &str,
        classifier_backend: Option<Arc<dyn ChatBackend>>,
        onnx: Option<&crate::ort::OrtRegistry>,
    ) -> Option<PipelineOrchestrator> {
        self.build_named_pipeline_with_backend_onnx_and_nlp(
            name,
            classifier_backend,
            onnx,
            &NlpDeps::default(),
        )
    }

    /// Canonical single-point pipeline build (Plan B F4): the three overloads
    /// above delegate here. `deps` bundles the three optional seams.
    pub fn build_named_pipeline_with_deps(
        &self,
        name: &str,
        deps: PipelineBuildDeps<'_>,
    ) -> Option<PipelineOrchestrator> {
        self.build_named_pipeline_with_backend_onnx_and_nlp(
            name,
            deps.classifier_backend,
            deps.onnx,
            &deps.nlp_deps,
        )
    }

    /// [`Self::build_named_pipeline_with_backend_and_onnx`] plus the NLP
    /// dependencies (concept store + optional durable strings path,
    /// ROADMAP_20260828_ORT M1.2). When `nlp: true` and a concept store is
    /// present, the NLP pipeline is built with an `InterlinguaResolver` so the
    /// `NlpStage` stamps interlingua ids; otherwise it fails open with a
    /// `warn!`.
    ///
    /// This is the sole implementation; the other `build_named_pipeline_*`
    /// overloads delegate through [`Self::build_named_pipeline_with_deps`] (DRY).
    pub fn build_named_pipeline_with_backend_onnx_and_nlp(
        &self,
        name: &str,
        classifier_backend: Option<Arc<dyn ChatBackend>>,
        onnx: Option<&crate::ort::OrtRegistry>,
        nlp_deps: &NlpDeps,
    ) -> Option<PipelineOrchestrator> {
        let params = self.pipelines.get(name)?;
        let mut orchestrator = PipelineOrchestrator::builder();

        // Overlay is enabled exactly when `overlay_models` is non-empty.
        // `DeterministicFirst` requires overlay models.
        let nlp_cfg = params.nlp_config();
        // `DeterministicFirst` and overlay are inseparable: the new ordering
        // never ships without the residual-consultation machinery that
        // justifies it. A config that sets `deterministic_first` without
        // overlay models is a loud warning and falls back to today's `LlmFirst`.
        let nlp_ordering = if nlp_cfg.ordering == NlpOrdering::DeterministicFirst
            && !params.overlay_enabled()
        {
            tracing::warn!(
                target: "router.config",
                pipeline = %name,
                "nlp_ordering=deterministic_first requires overlay_models to be non-empty; \
                 falling back to llm_first (today's behavior)",
            );
            NlpOrdering::LlmFirst
        } else {
            nlp_cfg.ordering
        };

        if params.deterministic_prefilter {
            if let Some(ref blacklist_path) = params.blacklist {
                let reject_patterns = Self::load_reject_patterns(blacklist_path);
                orchestrator = orchestrator.push(Arc::new(
                    crate::stages::deterministic::DeterministicPreFilter::from_config(
                        &reject_patterns,
                    ),
                ));
            } else {
                orchestrator = orchestrator.push(Arc::new(
                    crate::stages::deterministic::DeterministicPreFilter::new(),
                ));
            }
        }

        if nlp_cfg.enabled {
            // The NLP parse stage (ROADMAP_20260831_ARCEAGER M4): the pipeline
            // is built with an `InterlinguaResolver` when the boot supplied a
            // concept store (M1.2); with `None` it fails open to `en_default()`.
            // The annotation LLM call is now a **refiner** gated by a
            // `RefinePolicy` rather than an up-front rung. `LlmFirst` maps to
            // `Always` (today's behavior); `DeterministicFirst` maps to
            // `OnUncertain` — deterministic base + confidence-and-task-value-gated
            // refinement — instead of `None`.
            if nlp_deps.concept_store.is_none() {
                tracing::warn!(
                    target: "router.config",
                    pipeline = %name,
                    "nlp enabled without a concept store — interlingua ids disabled",
                );
            }
            match build_nlp_pipeline(nlp_deps) {
                Ok(pipeline) => {
                    // Effective refine policy: explicit DTO wins (converted); otherwise
                    // it follows `nlp_ordering` (M4.1). `LlmFirst` preserves
                    // today's always-consult behavior; `DeterministicFirst`
                    // consults only on uncertainty or routing incompleteness.
                    let refine_policy: spacy_rs::RefinePolicy = nlp_cfg
                        .refine_policy
                        .map_or_else(
                            || match nlp_ordering {
                                NlpOrdering::LlmFirst => spacy_rs::RefinePolicy {
                                    mode: spacy_rs::RefineMode::Always,
                                    ..spacy_rs::RefinePolicy::default()
                                },
                                NlpOrdering::DeterministicFirst => spacy_rs::RefinePolicy {
                                    mode: spacy_rs::RefineMode::OnUncertain,
                                    ..spacy_rs::RefinePolicy::default()
                                },
                            },
                            Into::into,
                        );
                    // The annotation backend (refiner): the injected classifier
                    // backend, else the resolved classifier client, else the
                    // onnx LLM backend (M2.6). Present for both orderings — the
                    // policy decides when it is consulted.
                    let fetch = classifier_backend
                        .clone()
                        .or_else(|| build_classifier_client(self, name, params))
                        .or_else(|| self.onnx_llm_backend())
                        .map(crate::stages::nlp::annotation_fetch);
                    let llm_rung = fetch.is_some();
                    // Build the trained-encoder annotation seam: when the
                    // `encoder_model` knob is set and the `encoder` role is
                    // registered, the ladder attempts it between LLM and
                    // ArcEager.
                    let encoder = nlp_cfg
                        .encoder_model
                        .as_deref()
                        .map(|_| fluent_llm::onnx_config::OnnxRole::Encoder.registry_key())
                        .and_then(|role_key| {
                            onnx.and_then(|reg| {
                                match crate::ort::nlp_encoder_fetch(reg, role_key) {
                                    Ok(enc) => enc,
                                    Err(e) => {
                                        tracing::warn!(
                                            target: "router.config",
                                            pipeline = %name,
                                            model = %role_key,
                                            error = %e,
                                            "encoder role build failed — encoder rung \
                                             unavailable (fail-open)",
                                        );
                                        None
                                    }
                                }
                            })
                        });
                    let encoder_rung = encoder.is_some();
                    orchestrator = orchestrator.push(Arc::new(
                        crate::stages::nlp::NlpStage::with_strings(
                            pipeline,
                            fetch,
                            encoder,
                            nlp_deps.strings_path.clone(),
                        )
                        .with_refine_policy(refine_policy),
                    ));
                    tracing::info!(
                        target: "router.config",
                        pipeline = %name,
                        llm_rung = llm_rung,
                        encoder_rung = encoder_rung,
                        interlingua = nlp_deps.concept_store.is_some(),
                        nlp_ordering = ?nlp_ordering,
                        refine_policy = ?refine_policy.mode,
                        "NLP parse stage enabled"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        target: "router.config",
                        pipeline = %name,
                        error = %e,
                        "NLP pipeline build failed; pipeline dropped"
                    );
                    return None;
                }
            }
        }

        // The overlay stage sits between nlp and classifier: it consumes the
        // parse residuals and publishes route hints the classifier merges.
        // Absent onnx config (or an unresolvable overlay model) skips the
        // stage with a warning — fail-open. The overlay disambiguation is
        // served by the `router` role (a zero-shot two-tower model); the
        // per-pipeline `overlay_models` knob (when non-empty) enables it (A1a).
        if params.overlay_enabled() {
            let router_keys = [fluent_llm::onnx_config::OnnxRole::Router.registry_key().to_string()];
            let overlays = if let Some(registry) = onnx {
                let routing = self.routing_config();
                match crate::ort::disambiguation_overlays(
                    registry,
                    &router_keys,
                    &routing.routes,
                ) {
                    Ok(overlays) => overlays,
                    Err(e) => {
                        tracing::warn!(
                            target: "router.config",
                            pipeline = %name,
                            error = %e,
                            "overlay role build failed — overlay stage skipped (fail-open)",
                        );
                        Vec::new()
                    }
                }
            } else {
                tracing::warn!(
                    target: "router.config",
                    pipeline = %name,
                    overlay_models = ?params.overlay_models,
                    "overlay enabled but no onnx registry is available — overlay stage \
                     skipped (fail-open)",
                );
                Vec::new()
            };
            if overlays.is_empty() {
                tracing::warn!(
                    target: "router.config",
                    pipeline = %name,
                    overlay_models = ?params.overlay_models,
                    "overlay enabled but no overlay model resolved — overlay stage skipped \
                     (fail-open)",
                );
            } else {
                let selector = crate::stages::overlay::ResidualSelector::new(
                    params.overlay_disambiguation_floor,
                );
                orchestrator = orchestrator.push(Arc::new(crate::stages::overlay::OverlayStage::new(
                    selector,
                    overlays,
                    params.overlay_max_concurrency,
                )));
                tracing::info!(
                    target: "router.config",
                    pipeline = %name,
                    overlay_models = ?params.overlay_models,
                    overlay_disambiguation_floor = params.overlay_disambiguation_floor,
                    "overlay stage inserted between nlp and classifier",
                );
            }
        }

        if params.classifier {
            let injected_backend = classifier_backend.is_some();
            // Real mode keeps the resolved classifier key and re-resolves the
            // backend per request; the injected mock path serves its frozen
            // backend with no resolver installed.
            let backend_resolver =
                (!injected_backend).then(|| classifier_backend_resolver(self));
            let routing_config = self.routing_config();
            let classifier_intel = classifier_intelligence(self, params);
            let classifier_model = resolve_classifier_model_key(self, params)
                .map_or_else(|| "unknown".into(), str::to_string);
            let client: Arc<dyn ChatBackend> = if let Some(backend) = classifier_backend {
                tracing::info!(target: "router.config", pipeline = %name, backend = "mock/transcript", "classifier using injected backend");
                backend
            } else if let Some(client) = build_classifier_client(self, name, params) {
                tracing::info!(target: "router.config", pipeline = %name, "classifier using real LLM client");
                client
            } else {
                tracing::warn!(
                    target: "router.config",
                    pipeline = %name,
                    classifier_model = resolve_classifier_model_key(self, params)
                        .unwrap_or("(none)"),
                    "classifier enabled but no model resolved — classifier stage \
                     skipped (fail-open); pipeline runs without classification",
                );
                // Build the pipeline stages accumulated so far (deterministic
                // prefilter, NLP, overlay) without the classifier and return.
                return Some(orchestrator.build());
            };
            let max_concurrency = params
                .classifier_max_concurrency
                .unwrap_or_else(default_classifier_concurrency);
            let limiter = Arc::new(fluent_concurrency::pool::Limiter::new(max_concurrency));
            tracing::debug!(target: "router.config", pipeline = %name, classifier_max_concurrency = max_concurrency, "classifier concurrency limiter constructed");

            // Target-matching ladder: built only when the pipeline opts in
            // (`target_match: "self_assess"`). The injected mock/transcript
            // backend is the matcher's `default` covering every key absent from
            // the per-key map (test mode: the map is empty, so every candidate
            // routes through the injected backend); real mode builds one
            // dedicated `LlmClient` per group member via the single `local_backend`
            // factory (DIP) and uses the classifier client as defense-in-depth
            // default for keys outside all groups.
            let target_matcher = if params.target_match == TargetMatchMode::SelfAssess {
                let backends = if injected_backend {
                    TargetBackends::new(HashMap::new(), Arc::clone(&client))
                } else {
                    TargetBackends::new(self.target_backends(), Arc::clone(&client))
                };
                tracing::debug!(
                    target: "router.config",
                    pipeline = %name,
                    target_backends = backends.len(),
                    target_match_timeout_ms = params.target_match_timeout_ms,
                    "target-matching ladder enabled (self-assess)",
                );
                Some(TargetMatcher::new(
                    backends,
                    Arc::clone(&limiter),
                    params.target_match_timeout_ms,
                ))
            } else {
                tracing::debug!(
                    target: "router.config",
                    pipeline = %name,
                    "target-matching ladder disabled (static)",
                );
                None
            };

            // Unparseable classifier responses are dumped to
            // `<log_dir>/classifier_failures/` for review (diagnostic corpus
            // that drives repair heuristics). Mock/injected backends never
            // dump — canned transcripts cannot produce real model output.
            let failure_dir = if injected_backend {
                None
            } else {
                Some(self.logging.log_dir.clone())
            };

            let stage = if let Some(tree) = &self.classification {
                // Classification tree drives the classifier stage. The
                // injected backend (mock/transcript) is always the default
                // client; per-node model backends are only built in real mode.
                // The target-matching ladder is shared with the flat path -
                // the engine resolves 2+ member group terminals through it.
                let engine = build_classification_engine(
                    self,
                    tree,
                    routing_config.clone(),
                    Arc::clone(&client),
                    Arc::clone(&limiter),
                    params.coherence_threshold,
                    !injected_backend,
                    target_matcher.clone(),
                    backend_resolver.clone(),
                );
                tracing::info!(
                    target: "router.config",
                    pipeline = %name,
                    tree_models = ?tree.classifier_model_keys(),
                    "classifier stage driven by classification tree",
                );
                crate::stages::classifier::ClassifierStage::with_tree(
                    client,
                    routing_config,
                    params.coherence_threshold,
                    params.score_matrix.clone(),
                    params.score_matrix_authoritative,
                    classifier_intel,
                    classifier_model,
                    limiter,
                    Arc::new(engine),
                    target_matcher,
                    self.classifier_failure_policy,
                    failure_dir,
                    backend_resolver.clone(),
                )
            } else {
                crate::stages::classifier::ClassifierStage::new(
                    client,
                    routing_config,
                    params.coherence_threshold,
                    params.score_matrix.clone(),
                    params.score_matrix_authoritative,
                    classifier_intel,
                    classifier_model,
                    limiter,
                    target_matcher,
                    self.classifier_failure_policy,
                    failure_dir,
                    backend_resolver.clone(),
                )
            };
            // When configured, wrap the classifier in the retry decorator
            // so parse/LLM failures re-run with escalating corrective prompts.
            // Both push through the orchestrator's typed producer path so the
            // dispatch target reaches the typed store without a JSON round-trip.
            let classifier: Arc<crate::stages::classifier::ClassifierStage> = Arc::new(stage);
            if params.classifier_retry_max > 0 {
                let retry_max = params.classifier_retry_max as usize;
                let retry_prompts = params.classifier_retry_prompts.clone();
                tracing::info!(
                    target: "router.config",
                    pipeline = %name,
                    classifier_retry_max = params.classifier_retry_max,
                    retry_prompt_count = retry_prompts.len(),
                    "classifier wrapped in RetryClassifier",
                );
                let retry: Arc<crate::stages::retry_classifier::RetryClassifier> =
                    Arc::new(crate::stages::retry_classifier::RetryClassifier::new(
                        classifier,
                        retry_max,
                        retry_prompts,
                    ));
                let producer_stage = Arc::clone(&retry);
                let producer: crate::pipeline::StageProducer = Arc::new(move |ctx, prior| {
                    producer_stage.evaluate_with_target(ctx, prior)
                });
                orchestrator = orchestrator.push_with_producer(retry, producer);
            } else {
                let producer_stage = Arc::clone(&classifier);
                let producer: crate::pipeline::StageProducer = Arc::new(move |ctx, prior| {
                    producer_stage.evaluate_with_target(ctx, prior)
                });
                orchestrator = orchestrator.push_with_producer(classifier, producer);
            }
        } else if classifier_backend.is_some() {
            tracing::warn!(
                target: "router.config",
                pipeline = %name,
                "classifier backend was provided but classifier is disabled for this pipeline"
            );
        }

        Some(orchestrator.build())
    }

    pub fn build_all_pipelines(&self) -> HashMap<String, Arc<PipelineOrchestrator>> {
        self.build_all_pipelines_with_backend(None)
    }

    pub fn build_all_pipelines_with_backend(
        &self,
        classifier_backend: Option<&Arc<dyn ChatBackend>>,
    ) -> HashMap<String, Arc<PipelineOrchestrator>> {
        self.build_all_pipelines_with_backend_and_onnx(classifier_backend, None)
    }

    /// Build every pipeline, supplying the boot ort registry to the overlay
    /// wiring (ROADMAP_20260827_ORT §2.5).
    pub fn build_all_pipelines_with_backend_and_onnx(
        &self,
        classifier_backend: Option<&Arc<dyn ChatBackend>>,
        onnx: Option<&crate::ort::OrtRegistry>,
    ) -> HashMap<String, Arc<PipelineOrchestrator>> {
        self.build_all_pipelines_with_backend_onnx_and_nlp(classifier_backend, onnx, &NlpDeps::default())
    }

    /// Canonical single-point build-all (Plan B F4) — delegates to the
    /// named-pipeline canonical above.
    #[allow(clippy::needless_pass_by_value)]
    pub fn build_all_pipelines_with_deps(
        &self,
        deps: PipelineBuildDeps<'_>,
    ) -> HashMap<String, Arc<PipelineOrchestrator>> {
        self.build_all_pipelines_with_backend_onnx_and_nlp(
            deps.classifier_backend.as_ref(),
            deps.onnx,
            &deps.nlp_deps,
        )
    }

    /// [`Self::build_all_pipelines_with_backend_and_onnx`] plus the NLP
    /// dependencies (ROADMAP_20260828_ORT M1.2). Composition roots that built a
    /// concept store before pipeline build thread it here so `nlp: true`
    /// pipelines resolve interlingua ids end-to-end.
    pub fn build_all_pipelines_with_backend_onnx_and_nlp(
        &self,
        classifier_backend: Option<&Arc<dyn ChatBackend>>,
        onnx: Option<&crate::ort::OrtRegistry>,
        nlp_deps: &NlpDeps,
    ) -> HashMap<String, Arc<PipelineOrchestrator>> {
        let mut map = HashMap::new();
        let mut dropped = Vec::new();
        let pipeline_count = self.pipelines.len();
        let has_mock = classifier_backend.is_some();
        tracing::info!(target: "router.config", pipeline_count = pipeline_count, mock_backend = has_mock, classifier_model = ?self.classifier_model, default_route = %self.default_route, "building pipelines");
        for name in self.pipelines.keys() {
            let backend_for_pipeline = classifier_backend.cloned();
            if let Some(pipeline) = self.build_named_pipeline_with_deps(
                name,
                PipelineBuildDeps {
                    classifier_backend: backend_for_pipeline,
                    onnx,
                    nlp_deps: nlp_deps.clone(),
                },
            ) {
                map.insert(name.clone(), Arc::new(pipeline));
            } else {
                dropped.push(name.clone());
                let params = &self.pipelines[name];
                tracing::warn!(
                    target: "router.config",
                    pipeline = %name,
                    configured_classifier = ?params.classifier_model.as_deref(),
                    resolved_classifier = ?resolve_classifier_model_key(self, params),
                    "pipeline not built - classifier model unresolved or invalid",
                );
            }
        }
        if !dropped.is_empty() {
            tracing::error!(
                target: "router.config",
                built = map.len(),
                configured = pipeline_count,
                dropped = ?dropped,
                "some configured pipelines were not built",
            );
        }
        tracing::info!(target: "router.config", built = map.len(), "pipelines built");
        map
    }

    pub fn route_pipeline_names(&self, model_name: &str) -> Vec<String> {
        self.routes_view()
            .get(model_name)
            .map_or_else(|| vec!["default".into()], |r| r.pipelines.clone())
    }
}

/// Resolve the classifier model key from config, following the priority:
/// 1. Pipeline-level `classifier_model`
/// 2. Root-level `classifier_model`
/// 3. Root `classification` classifier node's `model` (tree configs boot
///    without a flat classifier key)
/// 4. First model in the `fast` model group
fn resolve_classifier_model_key<'a>(
    config: &'a RouterConfig,
    params: &'a PipelineParams,
) -> Option<&'a str> {
    params
        .classifier_model
        .as_deref()
        .or(config.classifier_model.as_deref())
        .or_else(|| {
            config
                .classification
                .as_ref()
                .and_then(super::ClassificationTree::root_classifier_model)
        })
        .or_else(|| {
            config
                .model_groups
                .get("fast")
                .and_then(|group| group.models().first())
                .map(String::as_str)
        })
}

/// Return the classifier model's intelligence rating, or 0 if not found.
fn classifier_intelligence(config: &RouterConfig, params: &PipelineParams) -> u8 {
    resolve_classifier_model_key(config, params)
        .and_then(|k| {
            let (base, _) = crate::config::split_model_key(k);
            config.models.get(base)
        })
        .map_or(0, |m| m.intelligence)
}

/// Per-request classifier backend resolution over the live config: the
/// single `local_backend` factory (never a second one), so endpoint rewrites
/// and lazy loads after boot are always current. The shared config snapshot
/// stays current because nothing mutates the models map after the pipeline
/// build, while the inference registry inside it is live-shared and observes
/// every post-build registration. One map lookup plus an `LlmClient` build
/// per request (no I/O).
fn classifier_backend_resolver(config: &RouterConfig) -> ClassifierBackendResolver {
    let shared = Arc::new(config.clone());
    Arc::new(move |key: &str| shared.local_backend(key))
}

/// Build a classifier LLM client from the model config.
///
/// # DIP note
/// This factory is the **only** place in the crate that constructs a concrete
/// `LlmClient`.  The rest of the pipeline receives `Arc<dyn ChatBackend>` and
/// is oblivious to the concrete implementation.  There is exactly one
/// `ChatBackend` implementation today (`LlmClient`); if a second appears,
/// the factory can inject it without touching pipeline code.
fn build_classifier_client(
    config: &RouterConfig,
    _name: &str,
    params: &PipelineParams,
) -> Option<Arc<dyn ChatBackend>> {
    let model_key = resolve_classifier_model_key(config, params)?;
    config.local_backend(model_key)
}

/// Build the classification-tree engine for a pipeline.
///
/// `default_client` (the injected mock/transcript backend or the real
/// classifier client) serves every classifier node whose `model` key has no
/// dedicated backend. When `use_per_node_backends` is true (real mode only -
/// never when a backend was injected for mock/transcript runs), a dedicated
/// `LlmClient` is built for each distinct classifier-node model key that
/// differs from the resolved classifier model.
fn build_classification_engine(
    config: &RouterConfig,
    tree: &super::ClassificationTree,
    routing: super::RoutingConfig,
    default_client: Arc<dyn ChatBackend>,
    limiter: Arc<fluent_concurrency::pool::Limiter>,
    coherence_threshold: f64,
    use_per_node_backends: bool,
    target_matcher: Option<TargetMatcher>,
    backend_resolver: Option<ClassifierBackendResolver>,
) -> crate::stages::tree::ClassificationEngine {
    let default_params = PipelineParams::default();
    let default_model_key = resolve_classifier_model_key(config, &default_params);
    let mut clients = HashMap::new();
    if use_per_node_backends {
        for key in tree.classifier_model_keys() {
            if default_model_key == Some(key.as_str()) {
                continue;
            }
            if let Some(backend) = config.local_backend(&key) {
                clients.insert(key, backend);
            }
        }
    }
    crate::stages::tree::ClassificationEngine::new(
        tree.clone(),
        routing,
        default_client,
        clients,
        limiter,
        coherence_threshold,
        target_matcher,
        backend_resolver,
    )
}

/// Build the HTTP `LlmClient` for a `models` key — the llama construction
/// half of `RouterConfig::local_backend`, shared with the `LlamaBackend`
/// inference adapter so the construction site stays singular. `None` for an
/// unknown base key. Never touches the onnx resolver (fail-open): onnx keys
/// have no `models` entry and fall out here naturally.
///
/// A role name resolves to its head candidate's backend, qualified to the
/// single inference point (`resolve_inference_point`); literal keys behave
/// exactly as before.
pub(crate) fn llama_chat_backend_for_key(
    models: &HashMap<String, ModelEntry>,
    roles: &HashMap<String, crate::config::RoleEntry>,
    key: &str,
    default_instances: Option<&HashMap<String, crate::config::InstanceProfile>>,
) -> Option<Arc<dyn ChatBackend>> {
    let resolved = crate::config::role_head_key(models, roles, key, true)?;
    let (base, _) = crate::config::split_model_key(&resolved);
    let entry = models.get(base)?;
    let base_name = entry.name.as_deref().unwrap_or(base);
    // One qualifier resolver for every path: explicit qualifier, else the
    // role's instance point, else the entry default (over the fleet-default
    // map when the entry declares none), else bare.
    let qualifier =
        crate::config::resolve_inference_point(models, roles, key, None, default_instances);
    let model = match &qualifier {
        Some(qualifier) => format!("{base_name}:{qualifier}"),
        None => base_name.to_string(),
    };
    // The pool is an instance/group of the model, so its sampling
    // params (e.g. the swarm work pool's temperature) reach the body.
    let params = qualifier
        .as_deref()
        .and_then(|q| entry.instance_params_for_with(q, default_instances))
        .or_else(|| entry.params.clone().map(strip_declaration_params));
    let llm_config = LlmConfig::new()
        .api_url(entry.endpoint.clone())
        .model(model)
        .timeout_ms(entry.total_timeout_ms)
        .maybe_extra_body_params(params)
        .build();
    Some(Arc::new(LlmClient::with_config(llm_config)))
}

/// Build the HTTP `LlmClient` for a specific named inference point
/// (`<base>:<instance_or_group>`) — the llama construction half of
/// `RouterConfig::local_backend_for_instance`, shared with the
/// `LlamaBackend` adapter. `None` for an unknown key or instance.
/// Fleet-wide default profiles (`default_params.instances`) back entries
/// that declare none, so a hoisted profile is addressable by name.
/// A role name resolves to its head candidate's entry first.
pub(crate) fn llama_chat_backend_for_instance(
    models: &HashMap<String, ModelEntry>,
    roles: &HashMap<String, crate::config::RoleEntry>,
    key: &str,
    instance_or_group: &str,
    default_instances: Option<&HashMap<String, crate::config::InstanceProfile>>,
) -> Option<Arc<dyn ChatBackend>> {
    let resolved = crate::config::role_head_key(models, roles, key, true)?;
    let (base, _) = crate::config::split_model_key(&resolved);
    let entry = models.get(base)?;
    // Resolve the named profile; an unknown instance name -> None.
    entry
        .instance_profiles_with(default_instances)
        .into_iter()
        .find(|p| p.name.as_deref() == Some(instance_or_group))?;
    let base_name = entry.name.as_deref().unwrap_or(base);
    let model = format!("{base_name}:{instance_or_group}");
    let params = entry
        .instance_params_for_with(instance_or_group, default_instances)
        .unwrap_or_else(|| strip_declaration_params(serde_json::Value::Null));
    let llm_config = LlmConfig::new()
        .api_url(entry.endpoint.clone())
        .model(model)
        .timeout_ms(entry.total_timeout_ms)
        .maybe_extra_body_params(Some(params))
        .build();
    Some(Arc::new(LlmClient::with_config(llm_config)))
}

impl RouterConfig {
    /// Build the escalation ladder for every model group that configures one
    /// (`model_groups[g].escalation`). Groups without a ladder (or without a
    /// frontier endpoint) are absent - dispatch falls back to
    /// `fallback_completion` as before.
    ///
    /// The ladders are keyed by group name; `RoutingTarget.group` resolves
    /// which one a failed local chain escalates through
    pub fn build_escalation_ladders(
        &self,
        http_client: &reqwest::Client,
    ) -> HashMap<String, Arc<crate::dispatch::escalation::Ladder>> {
        use crate::dispatch::backend::OpenAiChatBackend;
        use crate::dispatch::escalation::{EscalationBackends, Ladder};

        let mut ladders = HashMap::new();
        for (group, group_cfg) in &self.model_groups {
            let Some(ladder_cfg) = group_cfg.escalation() else {
                continue;
            };
            let Some(frontier) = &ladder_cfg.frontier else {
                continue;
            };
            let frontier_client = frontier_api_client(http_client, frontier.api_key_env.as_deref());
            let backends = EscalationBackends {
                frontier: Arc::new(OpenAiChatBackend::new(
                    frontier_client,
                    frontier.endpoint.clone(),
                )),
                decomposer: ladder_cfg
                    .decomposer_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
                assembler: ladder_cfg
                    .assembler_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
                classifier: ladder_cfg
                    .classifier_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
                draft: ladder_cfg
                    .draft_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
                judge: ladder_cfg
                    .judge_model
                    .as_deref()
                    .and_then(|k| self.local_backend(k)),
            };
            tracing::info!(
                target: "router.config",
                group = %group,
                modes = ?ladder_cfg.modes,
                frontier_model = %frontier.model,
                "escalation ladder built",
            );
            ladders.insert(
                group.clone(),
                Arc::new(Ladder::new(ladder_cfg.clone(), backends)),
            );
        }
        ladders
    }

    /// Build a sync local-model `ChatBackend` from a `models` key - the single
    /// `LlmClient` construction site shared by the classifier and the
    /// escalation ladder's local roles (DIP: exactly one concrete
    /// `ChatBackend` factory in the crate). A role name resolves to its head
    /// candidate, qualified to the single inference point
    /// (`resolve_inference_point`); a literal key behaves exactly as before.
    /// When the point is `None` the id is bare `<base>` (upstream models,
    /// byte-identical to today). Declaration-only params are stripped.
    ///
    /// **ONNX branch:** a key served by a registered inference backend (the
    /// generative `onnx/llm` key via the boot-installed registry) yields that
    /// backend. Absent registry / unregistered key → `None` (fail-open).
    /// HTTP `models` keys are unchanged.
    pub fn local_backend(&self, key: &str) -> Option<Arc<dyn ChatBackend>> {
        if let Some(backend) = self.inference_registry.as_ref().and_then(|r| {
            common_core::sync::lock_read(r).route_chat(key, None)
        }) {
            return Some(backend);
        }
        llama_chat_backend_for_key(
            &self.models,
            &self.roles,
            key,
            self.default_params.instances.as_ref(),
        )
    }

    /// The onnx LLM role's registry key when the generative role is configured
    /// (`config.onnx.llm`). This is the **default wiring point**: the fallback
    /// backend for the LLM annotation rung, the review worker, the ledger
    /// summarizer/tier worker, and the chart selector model when their
    /// respective explicit keys are absent. `None` when no generative onnx
    /// model is declared.
    pub fn onnx_llm_key(&self) -> Option<String> {
        self.onnx.as_ref()?.llm.as_ref()?;
        Some(fluent_llm::onnx_config::OnnxRole::Llm.registry_key().to_string())
    }

    /// The onnx generative `ChatBackend` (the `onnx/llm` role) via the
    /// single factory, for the default-wiring call sites. `None` when no onnx
    /// LLM is registered (fail-open).
    pub fn onnx_llm_backend(&self) -> Option<Arc<dyn ChatBackend>> {
        let key = fluent_llm::onnx_config::OnnxRole::Llm.registry_key();
        self.local_backend(key)
    }

    /// Install the shared inference registry (composition root, once at boot).
    /// `local_backend` / `local_backend_for_instance` resolve through the
    /// registry's backends first. The registry stays behind its lock so
    /// later-booting backends (llama, whose pool builds after the first
    /// resolvers run) can still register.
    pub fn set_inference_registry(
        &mut self,
        registry: Arc<std::sync::RwLock<fluent_llm::backend::InferenceRegistry>>,
    ) {
        self.inference_registry = Some(registry);
    }

    /// Build a `ChatBackend` for a specific named inference point
    /// (`<base>:<instance_or_group>`) of a `models` key, reusing the single
    /// `LlmClient` factory (DIP - same construction site as `local_backend`).
    /// Used by the ledger summarizer (`<base>:ledger`) and any on-demand
    /// scratch route (`<base>:scratch`), which must target a named instance
    /// rather than the entry's default dispatch point.
    ///
    /// D4 param merging: the matching instance profile's `params` are overlaid
    /// onto the entry `params` (profile wins) so instance-level sampling knobs
    /// (e.g. `scratch`'s `temperature: 0.4`) actually reach the body; the
    /// merged object is then `strip_declaration_params`'d. Returns `None` when
    /// the key is unknown or the named instance does not exist.
    pub fn local_backend_for_instance(
        &self,
        key: &str,
        instance_or_group: &str,
    ) -> Option<Arc<dyn ChatBackend>> {
        // The shared registry (when installed by the composition root) serves
        // both fleets: onnx keys yield context-bound backends, llama keys the
        // named-instance `LlmClient`. Falls through to the legacy path below
        // when no backend serves the key.
        if let Some(backend) = self.inference_registry.as_ref().and_then(|r| {
            common_core::sync::lock_read(r).route_chat(key, Some(instance_or_group))
        }) {
            return Some(backend);
        }
        llama_chat_backend_for_instance(
            &self.models,
            &self.roles,
            key,
            instance_or_group,
            self.default_params.instances.as_ref(),
        )
    }

    /// Build the ledger `Summarizer`'s DIP backend - the ledger
    /// Summarizer's only construction site. Resolves the ledger model key
    /// (the `ledger` section's `model`, else the classifier model key), then
    /// targets the named `ledger` instance via `local_backend_for_instance`.
    /// When no llama `ledger` instance resolves (or no explicit key is set),
    /// it falls back to the onnx LLM backend (ROADMAP M2.6) — the generative
    /// onnx model is the default enrichment backend, config-driven via
    /// `config.onnx.llm`. Returns `None` when no backend resolves.
    pub fn summarizer_for_ledger(&self) -> Option<crate::summarization::Summarizer> {
        let ledger = self.ledger.as_ref()?;
        let backend = self
            .ledger_enrichment_backend(ledger.model.as_deref().or(self.classifier_model.as_deref()))?;
        Some(crate::summarization::Summarizer::new(
            backend,
            ledger.max_summary_tokens,
        ))
    }

    /// Build the `LedgerTierWorker`'s DIP backend - the tier worker's only
    /// construction site. Reuses the same `LlmClient` factory and the same
    /// `<base>:ledger` named-instance target as `summarizer_for_ledger` (no
    /// second HTTP client). `tier_model` (if given) wins over the ledger
    /// section's `model`, then the classifier model key. Falls back to the onnx
    /// LLM backend (ROADMAP M2.6) when no llama `ledger` instance resolves.
    /// Returns `None` when no backend resolves.
    pub fn ledger_tier_backend(
        &self,
        tier_model: Option<&str>,
    ) -> Option<Arc<dyn ChatBackend>> {
        let ledger = self.ledger.as_ref()?;
        let key = tier_model
            .or(ledger.model.as_deref())
            .or(self.classifier_model.as_deref());
        self.ledger_enrichment_backend(key)
    }

    /// Resolve a ledger enrichment backend (ROADMAP M2.6): the explicit key's
    /// `<base>:ledger` named instance when one exists, else the onnx LLM
    /// backend (the default enrichment model, config-driven). `None` when
    /// neither resolves.
    fn ledger_enrichment_backend(
        &self,
        explicit: Option<&str>,
    ) -> Option<Arc<dyn ChatBackend>> {
        if let Some(key) = explicit {
            if let Some(backend) = self.local_backend_for_instance(key, "ledger") {
                return Some(backend);
            }
        }
        self.onnx_llm_backend()
    }

    /// Build the tier worker's `TierConfig` from the `ledger` section.
    /// Queue capacity and max concurrency use the worker defaults; the LOD
    /// char caps and batch/poll knobs come from config. `None` when no `ledger`
    /// section is present.
    pub fn ledger_tier_config(&self) -> Option<crate::ledger::tiering::TierConfig> {
        let ledger = self.ledger.as_ref()?;
        Some(crate::ledger::tiering::TierConfig {
            lod4_max_chars: ledger.lod4_max_chars,
            lod5_max_chars: ledger.lod5_max_chars,
            batch_size: ledger.tier_batch_size,
            poll_interval_ms: ledger.tier_poll_interval_ms,
            credit_limit: ledger.tier_credit_limit,
            credit_more_after: ledger.tier_credit_more_after,
            ..Default::default()
        })
    }

    /// Build the `LedgerAgentCoordinator` from the `ledger.orchestrator`
    /// section — the coordinator's only construction site. `None` when the
    /// coordinator is not enabled (or no ledger section is present), so the
    /// server's dispatch path is untouched unless a deployment opts in.
    ///
    /// Takes the already-composed shared dependencies (`store`, `sessions`,
    /// `kv`, `tiers`, `backend`) — the composition root (`main.rs`) owns their
    /// lifetimes. The prompt budget and role flow from config; the KV policy
    /// is the section's `kv_policy`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_ledger_coordinator(
        &self,
        store: Arc<crate::node_store::ContentNodeStore>,
        sessions: Arc<crate::dag_session::SessionRegistry>,
        kv: crate::kv_cache::SnapshotStore,
        tiers: Arc<crate::ledger::tiering::LedgerTierWorker>,
        backend: Arc<dyn ChatBackend>,
    ) -> Option<crate::ledger::orchestrator::LedgerAgentCoordinator> {
        let section = self.ledger.as_ref()?.orchestrator.clone();
        if !section.enabled {
            return None;
        }
        let config = crate::ledger::orchestrator::OrchestratorConfig {
            kv_policy: section.kv_policy,
            budget: crate::ledger::prompt::PromptBudget::new(section.prompt_budget_chars),
            lod_spec: crate::ledger::prompt::LodSpec::full(),
            role: section.role,
        };
        let mut coordinator = crate::ledger::orchestrator::LedgerAgentCoordinator::new(
            store,
            sessions,
            kv,
            tiers,
            crate::ledger::prompt::LedgerPromptAssembler,
            backend,
            config,
        );
        // Opt-in KV-affinity scheduler: when `affinity_cap` is set, attach an
        // `AffinityScheduler` so the active session's turns get a priority
        // bonus (minimize context switches) while starved sessions age up.
        if let Some(cap) = section.affinity_cap {
            tracing::info!(
                target: "router.config",
                affinity_cap = cap,
                "ledger-agent KV-affinity scheduler attached",
            );
            coordinator = coordinator.with_affinity(
                crate::ledger::orchestrator::LedgerAgentCoordinator::build_affinity_scheduler(cap),
            );
        }
        Some(coordinator)
    }

    /// Build the target-matching ladder's per-candidate backend set (DIP -
    /// reuses the private `local_backend` helper, the single `LlmClient`
    /// factory; no second construction site).
    ///
    /// Iterates every model key referenced by any `model_groups` member and
    /// maps it to its dedicated `ChatBackend`. Role members fan out to their
    /// candidate keys (availability sentinels build nothing, exactly as
    /// unknown keys today). The matcher's `default` (for keys absent from the
    /// map) is supplied by the caller: the injected mock/transcript backend
    /// when one is provided, otherwise a real client (defense in depth - every
    /// real group member has a dedicated backend, so the default is only
    /// reached for a key outside all groups).
    pub fn target_backends(&self) -> HashMap<String, Arc<dyn ChatBackend>> {
        let mut backends = HashMap::new();
        let routing = self.routing_config();
        for group_key in self.model_groups.keys() {
            for key in routing.role_expanded_members(group_key) {
                if key == "last" || key == "any" {
                    continue;
                }
                if let Some(backend) = self.local_backend(&key) {
                    backends.insert(key, backend);
                }
            }
        }
        backends
    }
}

/// A reqwest client for the frontier backend: the shared client by default,
/// or a per-ladder client carrying the `Bearer` token from `api_key_env`
/// (when the variable is set and resolvable).
fn frontier_api_client(shared: &reqwest::Client, api_key_env: Option<&str>) -> reqwest::Client {
    let Some(env) = api_key_env else {
        return shared.clone();
    };
    let Ok(key) = std::env::var(env) else {
        tracing::warn!(
            target: "router.config",
            env = %env,
            "frontier api_key_env set but unreadable - falling back to shared client (no auth header)",
        );
        return shared.clone();
    };
    let Ok(auth) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")) else {
        return shared.clone();
    };
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(reqwest::header::AUTHORIZATION, auth);
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .unwrap_or_else(|_| shared.clone())
}
#[cfg(test)]
#[path = "../../tests/config_builder.rs"]
mod tests;
#[cfg(test)]
#[path = "../../tests/backend_registry.rs"]
mod backend_registry_tests;
#[cfg(test)]
#[path = "../../tests/routing_fallback_golden.rs"]
mod routing_fallback_tests;
