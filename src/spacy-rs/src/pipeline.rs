//! The pipeline composition (walkthrough §5, §10.3-10.5; roadmap §4).
//!
//! A single [`NlpPipeline`] runs the deterministic tokenizer head, then walks
//! the **annotation ladder** — a two-phase design (ROADMAP_20260831_ARCEAGER
//! §2.3):
//!
//! 1. **Base phase** (deterministic, unconditional): `first_accept_in_order`
//!    over `[ArcEagerRung, RuleRung]`. Always produces a validated
//!    [`AnnotationResult`].
//! 2. **Refine phase** (model, conditional): when [`should_refine`] returns
//!    `true`, `first_accept_in_order` over model refiners `[EncoderRung,
//!    LlmRung]`. A refiner that produces a validated, non-regressing result is
//!    adopted; otherwise the base is kept.
//!
//! Mapping to the monorepo idioms (walkthrough §12):
//!
//! | spaCy concept | Monorepo idiom |
//! |---|---|
//! | `Pipe` / `TrainablePipe` (`predict`/`set_annotations`) | `Component`/`WorkUnit` stages; `predict` = pure `execute`, `set_annotations` = [`crate::llm::attach`] (the mutate step) |
//! | Factory registry + `add_pipe` ordering | `DependencyGraph<ArcIntern<str>>` stage graph; waves from `ready_nodes` |
//! | Orchestration (`Language.__call__`, `pipe`) | [`NlpPipeline::process_async`] / [`NlpPipeline::annotate_batch_async`] |
//! | Annotation fallback ladder (§10.3) | `first_accept_in_order` over the [`AnnotationRung`]s |
//! | Parallel annotation (§10.5) | `ResultPool` fan-out in `annotate_batch_async` |
//!
//! The tokenizer is the **factory-injected head**, not a pipeline component
//! (`language.py:221`), so the stage graph starts after tokenization.
//!
//! # Compatibility surface
//!
//! [`NlpPipelineConfig`] derives `FieldAccess`/`Describable`/`bon::Builder` as
//! the scaffold for the Coral Router control plane (roadmap §5): the
//! MCP/control plane will describe and configure the NLP pipeline by name
//! once the router integration lands (Milestone 6). No in-tree consumer calls
//! it today — marked scaffold, not dead code.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use bon::Builder;
use fluent_concurrency::batch::{SupervisedBatch, SupervisedBatchEvent};
use fluent_concurrency::ladder::{first_accept_in_order, first_accept_in_order_sync};
use fluent_concurrency::pool::ResultPool;
use fluent_dag::dep_graph::DependencyGraph;
use fluent_wvr::prelude::*;
use fluent_wvr::Runtime;
use internment::ArcIntern;
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use crate::arc_eager::{ArcEagerAnnotator, ArcEagerRung};
use crate::routing::{InterlinguaSignal, RoutingSignal};
use crate::doc::Doc;
use crate::error::SpacyError;
use crate::interlingua::InterlinguaResolver;
use crate::lang;
use crate::lemmatizer::Lemmatizer;
use crate::llm::{
    apply_with, attach, AnnotationRecord, AnnotationResult, AnnotationSet, AnnotationSource,
};
use crate::sentencizer::Sentencizer;
use crate::tokenizer::Tokenizer;
use crate::validate::{AnnotationError, AnnotationValidator};
use crate::vocab::Vocab;

// ─────────────────────────────────────────────────────────────────────────
// Stage-graph assets (the declared `depends`/`provides` edges)
// ─────────────────────────────────────────────────────────────────────────

static TOKENS: std::sync::LazyLock<ArcIntern<str>> =
    std::sync::LazyLock::new(|| ArcIntern::from("tokens"));
static ANNOTATIONS: std::sync::LazyLock<ArcIntern<str>> =
    std::sync::LazyLock::new(|| ArcIntern::from("annotations"));
static VALIDATED: std::sync::LazyLock<ArcIntern<str>> =
    std::sync::LazyLock::new(|| ArcIntern::from("validated"));
static ANNOTATED_DOC: std::sync::LazyLock<ArcIntern<str>> =
    std::sync::LazyLock::new(|| ArcIntern::from("annotated_doc"));
static FRAMED: std::sync::LazyLock<ArcIntern<str>> =
    std::sync::LazyLock::new(|| ArcIntern::from("framed"));
static INTERLINGUA: std::sync::LazyLock<ArcIntern<str>> =
    std::sync::LazyLock::new(|| ArcIntern::from("interlingua_resolved"));
static SENTS: std::sync::LazyLock<ArcIntern<str>> =
    std::sync::LazyLock::new(|| ArcIntern::from("sents"));

static ANNOTATE_DEPS: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [TOKENS.clone()]);
static ANNOTATE_PROVIDES: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [ANNOTATIONS.clone()]);
static VALIDATE_DEPS: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [ANNOTATIONS.clone()]);
static VALIDATE_PROVIDES: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [VALIDATED.clone()]);
static ATTACH_DEPS: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [VALIDATED.clone()]);
static ATTACH_PROVIDES: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [ANNOTATED_DOC.clone()]);
/// The resolve stage runs only after frames are available (ROADMAP M3 — the
/// DAG is `annotate → validate → attach → frame → resolve → sentencize`).
static RESOLVE_DEPS: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [FRAMED.clone()]);
static RESOLVE_PROVIDES: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [INTERLINGUA.clone()]);
static FRAME_DEPS: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [ANNOTATED_DOC.clone()]);
static FRAME_PROVIDES: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [FRAMED.clone()]);
static YAGO_RESOLVE_DEPS: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [ANNOTATED_DOC.clone()]);
static YAGO_RESOLVE_PROVIDES: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [ArcIntern::from("yago_resolved")]);
static SENTENCIZE_DEPS: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [INTERLINGUA.clone()]);
static SENTENCIZE_PROVIDES: std::sync::LazyLock<[ArcIntern<str>; 1]> =
    std::sync::LazyLock::new(|| [SENTS.clone()]);

/// Per-request state shared between the orchestrator and the stages through
/// the typed `WorkContext.outputs` channel (an `Arc<Mutex<..>>` handoff — the
/// WasmComponent interior-mutability pattern). Stages never hold request
/// state; the orchestrator owns the cell.
#[derive(Default)]
pub struct PipelineState {
    /// The tokenized doc (the shared canvas). Seeded by the orchestrator,
    /// mutated by the attach stage.
    pub doc: Option<Doc>,
    /// The annotations accepted by the ladder. Written by the annotate stage.
    pub annotations: Option<AnnotationSet>,
    /// The ladder's full handoff (provenance + confidence, §9.1). Seeded by
    /// the orchestrator when the ladder ran; consumed by the resolve stage.
    pub annotation: Option<AnnotationResult>,
    /// Set once the validate stage has passed the 7-check gate.
    pub validated: bool,
    /// Collision notes surfaced by the resolve stage (for audit metadata).
    pub interlingua_notes: Vec<crate::interlingua::CollisionNote>,
    /// The deterministic frames extracted by the frame stage (ROADMAP M3).
    pub frames: Vec<crate::frame::Frame>,
    /// The typed ambiguities surfaced by the frame stage.
    pub ambiguities: Vec<crate::frame::AmbiguityEntry>,
    /// The provisional-vs-permanent keys minted from the frames. Only
    /// permanent keys are persisted to the ledger/graph.
    pub frame_keys: Vec<crate::frame::FrameKey>,
    /// Semantic plausibility computed by `YagoResolveStage` (separate from oracle margins).
    pub semantic_plausibility: Option<f64>,
}

/// The typed handoff key under which the shared state lives in the context.
const STATE_KEY: &str = "spacy.pipeline_state";
/// The structured handoff key for the ladder winner's JSON (§10.1 reply).
const ANNOTATION_JSON_KEY: &str = "annotation_json";

// ─────────────────────────────────────────────────────────────────────────
// Pipeline stages (each a `Component`/`WorkUnit`)
// ─────────────────────────────────────────────────────────────────────────

/// Stage 1 — reads the ladder's raw JSON reply and parses it into typed
/// annotations. Pure (`predict`); provides the `"annotations"` asset.
#[derive(Debug, Clone, Default)]
pub struct AnnotateStage;

impl WorkUnit for AnnotateStage {
    fn name(&self) -> &str {
        "annotate"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &*ANNOTATE_DEPS
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &*ANNOTATE_PROVIDES
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let state = ctx
            .get::<Arc<Mutex<PipelineState>>>(STATE_KEY)
            .ok_or_else(|| WorkError::Dependency(STATE_KEY.to_string()))?;
        let json = ctx
            .structured::<String>(ANNOTATION_JSON_KEY)
            .map_err(|e| WorkError::Execution(e.to_string()))?;
        let set =
            AnnotationSet::parse_json(&json).map_err(|e| WorkError::Execution(e.to_string()))?;
        let mut state = state.lock().expect("pipeline state lock");
        state.annotations = Some(set);
        Ok(WorkOutput::ok("annotations parsed"))
    }
}

impl_fieldless!(AnnotateStage);
impl Describable for AnnotateStage {
    fn describe(&self) -> serde_json::Value {
        json!({
            "name": "annotate",
            "depends": ["tokens"],
            "provides": ["annotations"],
            "purity": "pure predict: parse the ladder winner's JSON into annotations"
        })
    }
}
impl_component!(AnnotateStage);

/// Stage 2 — the 7-check gate. Depends on `"annotations"`; on rejection it
/// returns `WorkError::Execution`, so the supervisor cancels the attach
/// dependent (the "reject, never partially apply" contract). Pure.
#[derive(Debug, Clone)]
pub struct ValidateStage {
    validator: Arc<AnnotationValidator>,
}

impl ValidateStage {
    /// A stage gated by `validator`.
    #[must_use]
    pub fn new(validator: Arc<AnnotationValidator>) -> Self {
        Self { validator }
    }
}

impl WorkUnit for ValidateStage {
    fn name(&self) -> &str {
        "validate"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &*VALIDATE_DEPS
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &*VALIDATE_PROVIDES
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let state = ctx
            .get::<Arc<Mutex<PipelineState>>>(STATE_KEY)
            .ok_or_else(|| WorkError::Dependency(STATE_KEY.to_string()))?;
        let mut state = state.lock().expect("pipeline state lock");
        let doc = state
            .doc
            .as_ref()
            .ok_or_else(|| WorkError::Dependency("pipeline doc missing".into()))?;
        let set = state
            .annotations
            .as_ref()
            .ok_or_else(|| WorkError::Dependency("annotations missing".into()))?;
        self.validator
            .validate(doc, set)
            .map_err(|e| WorkError::Execution(format!("validation rejected: {e}")))?;
        state.validated = true;
        Ok(WorkOutput::ok("annotations validated"))
    }
}

impl_fieldless!(ValidateStage);
impl Describable for ValidateStage {
    fn describe(&self) -> serde_json::Value {
        json!({
            "name": "validate",
            "depends": ["annotations"],
            "provides": ["validated"],
            "purity": "pure predict: the §10.2 deterministic gate"
        })
    }
}
impl_component!(ValidateStage);

/// Stage 3 — the `set_annotations` mutate step: writes the validated records
/// into the shared canvas and rebuilds the dependency tree. Depends on
/// `"validated"`, provides `"annotated_doc"`.
#[derive(Debug, Clone, Default)]
pub struct AttachStage;

impl WorkUnit for AttachStage {
    fn name(&self) -> &str {
        "attach"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &*ATTACH_DEPS
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &*ATTACH_PROVIDES
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let state = ctx
            .get::<Arc<Mutex<PipelineState>>>(STATE_KEY)
            .ok_or_else(|| WorkError::Dependency(STATE_KEY.to_string()))?;
        let mut state = state.lock().expect("pipeline state lock");
        let set = state
            .annotations
            .clone()
            .ok_or_else(|| WorkError::Dependency("annotations missing".into()))?;
        let doc = state
            .doc
            .as_mut()
            .ok_or_else(|| WorkError::Dependency("pipeline doc missing".into()))?;
        attach(doc, &set).map_err(|e| WorkError::Execution(e.to_string()))?;
        Ok(WorkOutput::ok("annotations attached"))
    }
}

impl_fieldless!(AttachStage);
impl Describable for AttachStage {
    fn describe(&self) -> serde_json::Value {
        json!({
            "name": "attach",
            "depends": ["validated"],
            "provides": ["annotated_doc"],
            "purity": "set_annotations mutate step: write records + rebuild the tree"
        })
    }
}
impl_component!(AttachStage);

/// Stage 3.5 — deterministic frame extraction (ROADMAP M3). Reads the attached
/// doc and the ladder's oracle margins, derives the [`crate::frame::Frame`]s +
/// ambiguities, and mints provisional-vs-permanent [`crate::frame::FrameKey`]s
/// into the shared state. PURE over the concept store (boot-only registration
/// invariant, C2). Depends on `"annotated_doc"`, provides `"framed"` — the
/// resolve stage runs only after frames are available (the DAG is
/// `annotate → validate → attach → frame → resolve → sentencize`).
#[derive(Debug, Clone)]
pub struct FrameStage {
    extractor: crate::frame::FrameExtractor,
}

impl FrameStage {
    /// A stage extracting frames with `resolver` and its shared concept store.
    #[must_use]
    pub fn new(resolver: Arc<InterlinguaResolver>) -> Self {
        let concept_store = Arc::clone(resolver.concepts());
        let extractor = crate::frame::FrameExtractor::new(resolver, concept_store);
        Self { extractor }
    }

    /// The extractor backing this stage.
    #[must_use]
    pub fn extractor(&self) -> &crate::frame::FrameExtractor {
        &self.extractor
    }
}

impl WorkUnit for FrameStage {
    fn name(&self) -> &str {
        "frame"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &*FRAME_DEPS
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &*FRAME_PROVIDES
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let state = ctx
            .get::<Arc<Mutex<PipelineState>>>(STATE_KEY)
            .ok_or_else(|| WorkError::Dependency(STATE_KEY.to_string()))?;
        let mut state = state.lock().expect("pipeline state lock");
        let doc = state
            .doc
            .as_ref()
            .ok_or_else(|| WorkError::Dependency("pipeline doc missing".into()))?;
        let margins = state
            .annotation
            .as_ref()
            .and_then(|a| a.oracle_margins.clone());
        let analysis = self.extractor.extract(doc, margins.as_deref());
        let keys = self.extractor.keys(doc, &analysis);
        state.frames = analysis.frames;
        state.ambiguities = analysis.ambiguities;
        state.frame_keys = keys;
        Ok(WorkOutput::ok("frames extracted"))
    }
}

impl_fieldless!(FrameStage);
impl Describable for FrameStage {
    fn describe(&self) -> serde_json::Value {
        json!({
            "name": "frame",
            "depends": ["annotated_doc"],
            "provides": ["framed"],
            "purity": "deterministic: derive frames + ambiguities, mint keys (boot-only concept store)"
        })
    }
}
impl_component!(FrameStage);

/// Stage 4 — resolve each token's lemma hash into an [`InterlinguaId`] and
/// stamp per-token confidence (ROADMAP §11.2). **PURE and read-only** over
/// the [`ConceptStore`](crate::concept_store::ConceptStore) (C2): all
/// registration happens at boot; corrections happen in the review worker. The
/// only shared-state writes are the `TokenRecord` fields this stage owns.
/// Depends on `"annotated_doc"`, provides `"interlingua_resolved"`.
#[derive(Debug, Clone)]
pub struct ResolveStage {
    resolver: Arc<InterlinguaResolver>,
}

impl ResolveStage {
    /// A stage stamping ids with `resolver`.
    #[must_use]
    pub fn new(resolver: Arc<InterlinguaResolver>) -> Self {
        Self { resolver }
    }
}

impl WorkUnit for ResolveStage {
    fn name(&self) -> &str {
        "resolve"
    }
fn depends(&self) -> &[ArcIntern<str>] {
        &*RESOLVE_DEPS
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &*RESOLVE_PROVIDES
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let state = ctx
            .get::<Arc<Mutex<PipelineState>>>(STATE_KEY)
            .ok_or_else(|| WorkError::Dependency(STATE_KEY.to_string()))?;
        let mut state = state.lock().expect("pipeline state lock");
        // Clone the confidence out first so the doc borrow below is exclusive.
        let confidence = state
            .annotation
            .as_ref()
            .and_then(|a| a.token_confidence().map(ToOwned::to_owned));
        let doc = state
            .doc
            .as_mut()
            .ok_or_else(|| WorkError::Dependency("pipeline doc missing".into()))?;
        let notes = self.resolver.resolve_doc(doc, confidence.as_deref());
        state.interlingua_notes = notes;
        Ok(WorkOutput::ok("interlingua ids stamped"))
    }
}

impl_fieldless!(ResolveStage);
impl Describable for ResolveStage {
    fn describe(&self) -> serde_json::Value {
        json!({
            "name": "resolve",
            "depends": ["framed"],
            "provides": ["interlingua_resolved"],
            "purity": "read-only: stamp interlingua ids + confidence (boot-only registration, C2)"
        })
    }
}
impl_component!(ResolveStage);

/// Stage 4 — the deterministic sentencizer: marks `sent_start` from the
/// punctuation rules on the attached doc (walkthrough §8.2). Depends on
/// `"annotated_doc"`, provides `"sents"`. Honors a pre-existing annotation
/// unless the sentencizer's `overwrite` flag is set (spaCy's
/// `BACKWARD_OVERWRITE`).
#[derive(Debug, Clone)]
pub struct SentencizeStage {
    sentencizer: Sentencizer,
}

impl SentencizeStage {
    /// A stage running `sentencizer` over the attached doc.
    #[must_use]
    pub fn new(sentencizer: Sentencizer) -> Self {
        Self { sentencizer }
    }

    /// The stage's sentencizer.
    #[must_use]
    pub fn sentencizer(&self) -> &Sentencizer {
        &self.sentencizer
    }
}

impl WorkUnit for SentencizeStage {
    fn name(&self) -> &str {
        "sentencize"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &*SENTENCIZE_DEPS
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &*SENTENCIZE_PROVIDES
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let state = ctx
            .get::<Arc<Mutex<PipelineState>>>(STATE_KEY)
            .ok_or_else(|| WorkError::Dependency(STATE_KEY.to_string()))?;
        let mut state = state.lock().expect("pipeline state lock");
        let doc = state
            .doc
            .as_mut()
            .ok_or_else(|| WorkError::Dependency("pipeline doc missing".into()))?;
        let guesses = self.sentencizer.predict(doc);
        self.sentencizer.set_annotations(doc, &guesses);
        Ok(WorkOutput::ok("sentence boundaries set"))
    }
}

impl_fieldless!(SentencizeStage);
impl Describable for SentencizeStage {
    fn describe(&self) -> serde_json::Value {
        json!({
            "name": "sentencize",
            "depends": ["annotated_doc"],
            "provides": ["sents"],
            "purity": "deterministic sentencizer: punctuation-rule sent_start"
        })
    }
}
impl_component!(SentencizeStage);

// ─────────────────────────────────────────────────────────────────────────
// The stage graph executor
// ─────────────────────────────────────────────────────────────────────────

/// The deterministic stage DAG (`annotate → validate → attach`) executed in
/// waves of ready nodes under `SupervisedBatch`. Registering only ready
/// stages per wave gives dependency-aware ordering and cancellation: a
/// validation rejection cancels the attach dependent, never a partial apply.
pub struct StagePipeline {
    graph: DependencyGraph<ArcIntern<str>>,
    stages: HashMap<ArcIntern<str>, Arc<dyn Component>>,
}

impl StagePipeline {
    /// The default stage graph (`annotate → validate → attach → sentencize`)
    /// with no resolve stage.
    pub fn new(
        validator: AnnotationValidator,
        sentencizer: Sentencizer,
    ) -> Result<Self, PipelineError> {
        Self::new_with_resolver(validator, sentencizer, None)
    }

    /// The stage graph, optionally extended with the deterministic `frame`
    /// stage and the read-only `resolve` stage
    /// (`annotate → validate → attach → frame → resolve → sentencize`) when a
    /// resolver is supplied (ROADMAP §11.2, M3). The frame stage mints the
    /// structural frames/keys between attach and resolve.
    pub fn new_with_resolver(
        validator: AnnotationValidator,
        sentencizer: Sentencizer,
        resolver: Option<Arc<InterlinguaResolver>>,
    ) -> Result<Self, PipelineError> {
        let mut graph = DependencyGraph::new();
        let annotate = ArcIntern::from("annotate");
        let validate = ArcIntern::from("validate");
        let attach = ArcIntern::from("attach");
        let sentencize = ArcIntern::from("sentencize");
        graph
            .register(&annotate, &*ANNOTATE_DEPS, &*ANNOTATE_PROVIDES)
            .map_err(|e| PipelineError::StageGraph(e.to_string()))?;
        graph
            .register(&validate, &*VALIDATE_DEPS, &*VALIDATE_PROVIDES)
            .map_err(|e| PipelineError::StageGraph(e.to_string()))?;
        graph
            .register(&attach, &*ATTACH_DEPS, &*ATTACH_PROVIDES)
            .map_err(|e| PipelineError::StageGraph(e.to_string()))?;
        let mut stages: HashMap<ArcIntern<str>, Arc<dyn Component>> = HashMap::from([
            (annotate, Arc::new(AnnotateStage) as Arc<dyn Component>),
            (
                validate,
                Arc::new(ValidateStage::new(Arc::new(validator))) as Arc<dyn Component>,
            ),
            (attach, Arc::new(AttachStage) as Arc<dyn Component>),
        ]);
        if let Some(resolver) = resolver {
            // YagoResolveStage: attach → yago_resolve → frame → resolve (Alt C)
            let yago = ArcIntern::from("yago_resolve");
            graph
                .register(&yago, &*YAGO_RESOLVE_DEPS, &*YAGO_RESOLVE_PROVIDES)
                .map_err(|e| PipelineError::StageGraph(e.to_string()))?;
            stages.insert(
                yago,
                Arc::new(crate::yago_resolve::YagoResolveStage::new(Arc::clone(resolver.concepts()))) as Arc<dyn Component>,
            );
            let frame = ArcIntern::from("frame");
            // Frame now depends on yago_resolved, not directly on annotated_doc
            let frame_deps = [ArcIntern::from("yago_resolved")];
            graph
                .register(&frame, &frame_deps, &*FRAME_PROVIDES)
                .map_err(|e| PipelineError::StageGraph(e.to_string()))?;
            stages.insert(
                frame,
                Arc::new(FrameStage::new(Arc::clone(&resolver))) as Arc<dyn Component>,
            );
            let resolve = ArcIntern::from("resolve");
            graph
                .register(&resolve, &*RESOLVE_DEPS, &*RESOLVE_PROVIDES)
                .map_err(|e| PipelineError::StageGraph(e.to_string()))?;
            stages.insert(resolve, Arc::new(ResolveStage::new(resolver)) as Arc<dyn Component>);
        }
        graph
            .register(&sentencize, &*SENTENCIZE_DEPS, &*SENTENCIZE_PROVIDES)
            .map_err(|e| PipelineError::StageGraph(e.to_string()))?;
        stages.insert(sentencize, Arc::new(SentencizeStage::new(sentencizer)) as Arc<dyn Component>);
        Ok(Self { graph, stages })
    }

    /// Run the stage graph over the shared `state`. `annotation_json` is the
    /// ladder winner's raw §10.1 reply, seeded to the annotate stage. The
    /// `tokens` asset is pre-satisfied (the tokenizer is the pipe head), so
    /// the first wave runs the annotate stage alone.
    pub async fn run(
        &self,
        state: &Arc<Mutex<PipelineState>>,
        annotation_json: String,
        rt: Arc<dyn Runtime>,
        caps: CapabilitySet,
    ) -> Result<(), PipelineError> {
        let mut satisfied: HashSet<ArcIntern<str>> = HashSet::from([TOKENS.clone()]);
        let mut completed: HashSet<ArcIntern<str>> = HashSet::new();
        loop {
            let ready: Vec<Arc<dyn Component>> = self
                .graph
                .ready_nodes(&satisfied)
                .into_iter()
                .filter(|n| !completed.contains(n))
                .filter_map(|n| self.stages.get(&n).cloned())
                .collect();
            if ready.is_empty() {
                break;
            }

            let mut batch = SupervisedBatch::new(Arc::clone(&rt), caps.clone());
            for stage in &ready {
                let mut ctx = WorkContext::for_unit_in_batch(&rt, &caps, |_| {});
                ctx.set(STATE_KEY, Arc::clone(state));
                if stage.name() == "annotate" {
                    ctx.set_structured(ANNOTATION_JSON_KEY, &annotation_json);
                }
                batch
                    .register_with_context(stage.clone(), ctx)
                    .map_err(|e| PipelineError::StageRegistration(e.to_string()))?;
            }
            let summary = batch.await;

            if let Some(SupervisedBatchEvent::Panicked { name, .. }) = summary.panicked.first() {
                return Err(PipelineError::Stage(PipelineStageFailure::Panicked(
                    name.to_string(),
                )));
            }
            if let Some(SupervisedBatchEvent::Failed { name, error }) = summary.failed.first() {
                return Err(PipelineError::Stage(PipelineStageFailure::Failed(
                    name.to_string(),
                    error.to_string(),
                )));
            }
            if let Some(SupervisedBatchEvent::Cancelled { name, .. }) = summary.cancelled.first() {
                return Err(PipelineError::Stage(PipelineStageFailure::Cancelled(
                    name.to_string(),
                )));
            }
            for ev in &summary.completed {
                if let SupervisedBatchEvent::Completed { name, .. } = ev {
                    completed.insert(name.clone());
                    if let Some(provides) = self.graph.provides_of(name) {
                        satisfied.extend(provides.iter().cloned());
                    }
                }
            }
        }
        Ok(())
    }

    /// The declared stage names in dependency order.
    #[must_use]
    pub fn stage_names(&self) -> Vec<String> {
        self.graph.nodes().iter().map(ToString::to_string).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The annotation ladder (§10.3): deterministic base → gated model refine
// ─────────────────────────────────────────────────────────────────────────

/// A transport-seam error from an LLM fetch.
#[derive(Debug, Error)]
pub enum AnnotateError {
    /// The LLM call itself failed (network, timeout, empty reply).
    #[error("LLM fetch failed: {0}")]
    Fetch(String),
    /// The trained-encoder rung failed (tokenization, session run, or the
    /// host-side head could not produce labels for the aligned spans).
    #[error("encoder rung failed: {0}")]
    Encoder(String),
    /// The reply was parsed or rejected by the gate.
    #[error("annotation not accepted: {0}")]
    Rejected(#[from] AnnotationError),
}

/// The live-LLM seam: given the deterministic tokenizer's orth list
/// (§10.1 — "the LLM is *given* the token list"), return the raw §10.1 JSON
/// reply. The closure takes owned token texts and returns an owned future, so
/// implementors may prompt with the tokens and await a real endpoint (see the
/// live-ai test).
pub type LlmFetch = Arc<
    dyn Fn(Vec<String>) -> Pin<Box<dyn Future<Output = Result<String, AnnotateError>> + Send>>
        + Send
        + Sync,
>;

/// The **synchronous** LLM annotation seam — the `LlmFetch` analog for sync
/// callers (the Coral Router's `NlpStage` runs inside a synchronous pipeline
/// stage, so it cannot await). Returns the raw §10.1 annotation JSON, or an
/// error the sync ladder swallows in favor of the deterministic base.
pub type LlmFetchSync =
    Arc<dyn Fn(Vec<String>) -> Result<String, AnnotateError> + Send + Sync>;

/// The **trained-encoder** annotation seam (ROADMAP_20260827_ORT §4.2) —
/// symmetric to [`LlmFetchSync`] but consuming the whole [`Doc`]: the encoder
/// needs the deterministic orth list *and* its `idx` byte offsets to align LFM
/// predictions onto the spacy-rs baseline. Returns an [`AnnotationSet`] whose
/// record `text` fields equal the spacy orth **by construction** (validator
/// check 1), or an error the ladder swallows in favor of ArcEager. The closure
/// is whole-doc (full re-annotation); the span-scoped residual variant is the
/// separate [`EncoderResidualFetch`] seam (M2.3).
pub type EncoderFetchSync =
    Arc<dyn Fn(&Doc) -> Result<AnnotationSet, AnnotateError> + Send + Sync>;

/// One annotation rung: attempt to produce an accepted annotation for `doc`,
/// or skip (`Ok(None)`) / fail (`Err`). `run` consumes the boxed rung so the
/// returned future owns it — no borrow across the ladder's awaits. The
/// returned value carries provenance + confidence ([`AnnotationResult`], §9.1)
/// so downstream routing can see which rung won and how confident it is (F7).
pub trait AnnotationRung: Send + Sync {
    /// Try to annotate `doc`. `Ok(None)` skips to the next rung.
    fn run<'a>(
        self: Box<Self>,
        doc: &'a Doc,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    >;
}

// ─────────────────────────────────────────────────────────────────────────
// Refine policy (ROADMAP_20260831_ARCEAGER §2.1)
// ─────────────────────────────────────────────────────────────────────────

/// Whether (and when) the model refiners are consulted after the
/// deterministic base. Pure decision data; no I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefineMode {
    /// Never consult models (today's `deterministic_first` = deterministic
    /// only). The base parse is final.
    #[default]
    Off,
    /// Consult refiners only when the base parse is uncertain or
    /// routing-incomplete (`should_refine` below).
    OnUncertain,
    /// Always consult refiners after the base (today's LLM-first behavior).
    Always,
}

fn default_min_overall() -> f64 { 0.7 }
fn default_min_role_coverage() -> f64 { 0.5 }
fn default_min_token_score() -> f64 { 0.5 }
fn default_true() -> bool { true }
fn default_unresolved_token_threshold() -> f64 { 0.3 }

/// The decision policy for when the model refiners run after the
/// deterministic base parse. Pure data — no I/O, no side effects.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct RefinePolicy {
    /// Whether refiners are off, conditional, or always-on.
    #[serde(default)]
    pub mode: RefineMode,
    // -- confidence-based triggers (parser self-doubt) --
    /// Parse-level floor on `ParseConfidence.overall`.
    #[serde(default = "default_min_overall")]
    pub min_overall: f64,
    /// Floor on `ParseConfidence.role_coverage` (fraction of {nsubj,dobj}).
    #[serde(default = "default_min_role_coverage")]
    pub min_role_coverage: f64,
    /// Refine when any oracle margin is a near-tie (attachment ambiguity).
    #[serde(default = "default_true")]
    pub refine_on_ties: bool,
    /// Per-token floor for the span-scoped focus (M2). Tokens scoring below
    /// this are the "focus" the refiner is asked to reconsider.
    #[serde(default = "default_min_token_score")]
    pub min_token_score: f64,
    // -- task-value triggers (routing relevance, independent of confidence) --
    /// Refine when a routing-critical role (predicate/subject/object per
    /// `RoutingSignal`) is present structurally but its `interlingua` id is
    /// unresolved.
    #[serde(default = "default_true")]
    pub refine_on_unresolved_critical_role: bool,
    /// Refine when a PROPN token has no resolved entity id.
    #[serde(default = "default_true")]
    pub refine_on_unresolved_propn: bool,
    /// Refine when `resolve_doc` surfaced a `CollisionNote` for any token in
    /// the sentence.
    #[serde(default = "default_true")]
    pub refine_on_collision_note: bool,
    /// Fraction of tokens whose lemma id is `InterlinguaId(0)` above which
    /// the task-value `UnresolvedPropn` trigger fires. 0.0 = any token (old
    /// behavior, do not use); 1.0 = never; default 0.3. Task-value axis.
    #[serde(default = "default_unresolved_token_threshold")]
    pub unresolved_token_threshold: f64,
}

impl Default for RefinePolicy {
    fn default() -> Self {
        Self {
            mode: RefineMode::Off,
            min_overall: 0.7,
            min_role_coverage: 0.5,
            refine_on_ties: true,
            min_token_score: 0.5,
            refine_on_unresolved_critical_role: true,
            refine_on_unresolved_propn: true,
            refine_on_collision_note: true,
            unresolved_token_threshold: default_unresolved_token_threshold(),
        }
    }
}

/// The fine-grained confidence trigger that fired (M5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceReason {
    /// `ParseConfidence.overall < min_overall`.
    Overall,
    /// `ParseConfidence.role_coverage < min_role_coverage`.
    RoleCoverage,
    /// `oracle_tie_count > 0` with `refine_on_ties`.
    Ties,
}

/// The fine-grained task-value trigger that fired (M5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskValueReason {
    /// A routing-critical role (predicate, subject, or direct object) is
    /// structurally present but interlingua-unresolved.
    UnresolvedCriticalRole,
    /// A token carries the `InterlinguaId(0)` none-sentinel.
    UnresolvedPropn,
    /// `CollisionNote` surfaced for the sentence.
    Collision,
}

/// Why the refine decision fired (M5.1).  The discriminants mirror the
/// `should_refine` truth table — the bool is just `reason != NoTrigger`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefineReason {
    /// No trigger — keep the base.
    NoTrigger,
    /// `RefineMode::Always` — unconditional.
    AlwaysPolicy,
    /// A confidence (parser self-doubt) trigger.
    Confidence(ConfidenceReason),
    /// A task-value (routing-relevance) trigger.
    TaskValue(TaskValueReason),
}

impl RefineReason {
    /// Stable string key for metrics / logging / StageMetadata.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoTrigger => "no_trigger",
            Self::AlwaysPolicy => "always_policy",
            Self::Confidence(ConfidenceReason::Overall) => "confidence_overall",
            Self::Confidence(ConfidenceReason::RoleCoverage) => "confidence_role_coverage",
            Self::Confidence(ConfidenceReason::Ties) => "confidence_ties",
            Self::TaskValue(TaskValueReason::UnresolvedCriticalRole) => {
                "task_value_unresolved_role"
            }
            Self::TaskValue(TaskValueReason::UnresolvedPropn) => "task_value_unresolved_propn",
            Self::TaskValue(TaskValueReason::Collision) => "task_value_collision",
        }
    }

    /// Parse the flat string produced by [`Self::as_str`].
    #[must_use]
    pub fn from_flat_str(s: &str) -> Option<Self> {
        match s {
            "no_trigger" => Some(Self::NoTrigger),
            "always_policy" => Some(Self::AlwaysPolicy),
            "confidence_overall" => Some(Self::Confidence(ConfidenceReason::Overall)),
            "confidence_role_coverage" => Some(Self::Confidence(ConfidenceReason::RoleCoverage)),
            "confidence_ties" => Some(Self::Confidence(ConfidenceReason::Ties)),
            "task_value_unresolved_role" => {
                Some(Self::TaskValue(TaskValueReason::UnresolvedCriticalRole))
            }
            "task_value_unresolved_propn" => {
                Some(Self::TaskValue(TaskValueReason::UnresolvedPropn))
            }
            "task_value_collision" => Some(Self::TaskValue(TaskValueReason::Collision)),
            _ => None,
        }
    }
}

/// Pure decision with a reason (M5.1): the first trigger in evaluation
/// order, or `NoTrigger`.  `AlwaysPolicy` wins before any other check;
/// confidence triggers are evaluated before task-value triggers — the same
/// order `should_refine` uses, so the bool and the reason agree.
///
/// **Scope:** `signal`/`routing` are a single sentence's interlingua frame.
/// For a multi-sentence document, use [`refine_reason_aggregated`] which
/// aggregates `token_ids` across all sentences and triggers on *any* sentence's
/// unresolved critical role. The `unresolved_token_threshold` is evaluated
/// **aggregated across the whole document** (total unresolved / total tokens),
/// not per-sentence, so a second paragraph's OOVs are not ignored. The
/// single-sentence `refine_reason` is retained for hermetic unit tests; the
/// ladder's hot path uses the aggregated variant.
/// Inner implementation with one ordered trigger chain:
///
/// `Always → confidence Overall → confidence RoleCoverage → confidence Ties →`
/// `task-value UnresolvedCriticalRole (any sentence) → task-value UnresolvedPropn (aggregated) →`
/// `task-value Collision → NoTrigger`
///
/// Confidence triggers are producer self-doubt (`overall/role_coverage/ties`);
/// task-value triggers are outcome-wrong-for-task (`unresolved/collision`).
fn refine_reason_inner(
    base: &AnnotationResult,
    signals: &[(RoutingSignal, InterlinguaSignal)],
    policy: RefinePolicy,
) -> RefineReason {
    match policy.mode {
        RefineMode::Off => RefineReason::NoTrigger,
        RefineMode::Always => RefineReason::AlwaysPolicy,
        RefineMode::OnUncertain => {
            if let Some(ref pc) = base.parse_confidence {
                if pc.overall < policy.min_overall {
                    return RefineReason::Confidence(ConfidenceReason::Overall);
                }
                if pc.role_coverage < policy.min_role_coverage {
                    return RefineReason::Confidence(ConfidenceReason::RoleCoverage);
                }
                if policy.refine_on_ties && pc.oracle_tie_count > 0 {
                    return RefineReason::Confidence(ConfidenceReason::Ties);
                }
            }
            if policy.refine_on_unresolved_critical_role {
                for (routing, signal) in signals {
                    if signal.predicate_id.is_none() {
                        return RefineReason::TaskValue(TaskValueReason::UnresolvedCriticalRole);
                    }
                    if routing.subject.is_some() && signal.subject_id.is_none() {
                        return RefineReason::TaskValue(TaskValueReason::UnresolvedCriticalRole);
                    }
                    if routing.direct_object.is_some() && signal.direct_object_id.is_none() {
                        return RefineReason::TaskValue(TaskValueReason::UnresolvedCriticalRole);
                    }
                }
            }
            if policy.refine_on_unresolved_propn {
                let n: usize = signals.iter().map(|(_, s)| s.token_ids.len()).sum();
                let unresolved: usize = signals
                    .iter()
                    .flat_map(|(_, s)| s.token_ids.iter())
                    .filter(|id| id.as_u64() == 0)
                    .count();
                if n > 0 && (unresolved as f64 / n as f64) > policy.unresolved_token_threshold {
                    return RefineReason::TaskValue(TaskValueReason::UnresolvedPropn);
                }
            }
            if policy.refine_on_collision_note && base.collision_count > 0 {
                return RefineReason::TaskValue(TaskValueReason::Collision);
            }
            RefineReason::NoTrigger
        }
    }
}

#[must_use]
pub fn refine_reason(
    base: &AnnotationResult,
    signal: &InterlinguaSignal,
    routing: &RoutingSignal,
    policy: RefinePolicy,
) -> RefineReason {
    refine_reason_inner(base, &[(routing.clone(), signal.clone())], policy)
}

/// Pure decision: should the model refiners be consulted for this base?
///
/// `true` for `Always`; `false` for `Off`. For `OnUncertain`, `true` iff
/// EITHER a confidence trigger fires (`overall < min_overall ||
/// role_coverage < min_role_coverage || (refine_on_ties && tie_count > 0)`)
/// OR a task-value trigger fires (a routing-critical role structurally
/// present but interlingua-unresolved; a PROPN with no entity id; a
/// surfaced `CollisionNote`) — each task-value check gated by its own
/// `refine_on_*` flag.
///
/// `routing` provides the structural role presence (subject/object strings
/// populated by `extract_routing_signals`); `signal` provides the
/// interlingua resolution state. A role is "present but unresolved" when
/// the routing signal carries it and the interlingua signal does not.
///
/// **Scope:** single-sentence; for multi-sentence use
/// [`should_refine_aggregated`].
pub fn should_refine(
    base: &AnnotationResult,
    signal: &InterlinguaSignal,
    routing: &RoutingSignal,
    policy: RefinePolicy,
) -> bool {
    refine_reason(base, signal, routing, policy) != RefineReason::NoTrigger
}

/// Aggregated variant for a multi-sentence document: `token_ids` fraction is
/// computed over the whole document (total unresolved / total tokens), and an
/// unresolved critical role in *any* sentence triggers `UnresolvedCriticalRole`.
/// The ladder's hot path uses this; the single-sentence `should_refine` is
/// retained for hermetic tests. `signals` is `&[(RoutingSignal,
/// InterlinguaSignal)]` in document order.
#[must_use]
pub fn should_refine_aggregated(
    base: &AnnotationResult,
    signals: &[(RoutingSignal, InterlinguaSignal)],
    policy: RefinePolicy,
) -> bool {
    refine_reason_aggregated(base, signals, policy) != RefineReason::NoTrigger
}

/// Aggregated `RefineReason` for a multi-sentence document.
///
/// See [`refine_reason`] for trigger order. The `UnresolvedPropn` threshold
/// is evaluated **aggregated across the whole document**; the
/// `UnresolvedCriticalRole` check triggers if *any* sentence has a
/// structurally present but unresolved role.
#[must_use]
pub fn refine_reason_aggregated(
    base: &AnnotationResult,
    signals: &[(RoutingSignal, InterlinguaSignal)],
    policy: RefinePolicy,
) -> RefineReason {
    refine_reason_inner(base, signals, policy)
}

fn refine_focus_inner(
    base: &AnnotationResult,
    signals: &[(RoutingSignal, InterlinguaSignal)],
    policy: RefinePolicy,
) -> Vec<usize> {
    let mut focus = Vec::new();
    if let Some(ref pc) = base.parse_confidence {
        for (i, &score) in pc.token_scores.iter().enumerate() {
            if score < policy.min_token_score {
                focus.push(i);
            }
        }
        for (i, &margin) in pc.oracle_margins.iter().enumerate() {
            if margin == 0.0 && !focus.contains(&i) {
                focus.push(i);
            }
        }
    }
    if policy.refine_on_unresolved_critical_role || policy.refine_on_unresolved_propn {
        let n: usize = signals.iter().map(|(_, s)| s.token_ids.len()).sum();
        let unresolved: usize = signals
            .iter()
            .flat_map(|(_, s)| s.token_ids.iter())
            .filter(|id| id.as_u64() == 0)
            .count();
        let above_threshold = n > 0 && (unresolved as f64 / n as f64) > policy.unresolved_token_threshold;
        if above_threshold {
            let mut offset = 0;
            for (_, signal) in signals {
                for (i, id) in signal.token_ids.iter().enumerate() {
                    if id.as_u64() == 0 && !focus.contains(&(offset + i)) {
                        focus.push(offset + i);
                    }
                }
                offset += signal.token_ids.len();
            }
        }
    }
    focus.sort_unstable();
    focus.dedup();
    focus
}

/// Aggregated token indices the span-scoped refiner should reconsider for a
/// multi-sentence document: indices are document-global (0..total_tokens).
/// Confidence and margin indices are document-global; unresolved tokens are
/// included when the **aggregated** `unresolved_fraction` exceeds the
/// threshold. Single-sentence callers use [`refine_focus`].
#[must_use]
pub fn refine_focus_aggregated(
    base: &AnnotationResult,
    signals: &[(RoutingSignal, InterlinguaSignal)],
    policy: RefinePolicy,
) -> Vec<usize> {
    refine_focus_inner(base, signals, policy)
}

/// The token indices the span-scoped refiner should be asked to reconsider —
/// derived from `ParseConfidence.token_scores` (< `min_token_score`), the
/// near-tie margin positions, and the tokens whose interlingua lemma id is
/// unresolved (the `InterlinguaId(0)` none-sentinel in `signal.token_ids`,
/// gated by the task-value flags). Empty for a fully-confident,
/// routing-resolved base.
///
/// **Scope:** single-sentence; for multi-sentence use [`refine_focus_aggregated`].
pub fn refine_focus(
    base: &AnnotationResult,
    signal: &InterlinguaSignal,
    policy: RefinePolicy,
) -> Vec<usize> {
    // Single-sentence is the one-element slice case of the aggregated inner — same threshold semantics (document-wide)
    // The dummy RoutingSignal is not consulted for focus (only token_ids), so an empty default suffices.
    let dummy = crate::routing::RoutingSignal {
        sentence: String::new(),
        predicate: String::new(),
        subject: None,
        direct_object: None,
        indirect_object: None,
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec![],
        lemmas: vec![],
        pos: vec![],
        deps: vec![],
        heads: vec![],
        interlingua: None,
    };
    refine_focus_inner(base, &[(dummy, signal.clone())], policy)
}

/// Fraction of routing-critical roles (predicate, subject, direct object)
/// that are both structurally present AND interlingua-resolved. Used only
/// as a non-regression gate on refiner adoption — never as a replacement
/// for the 7-check validator, which remains the sole well-formedness
/// arbiter. A refined result must pass BOTH gates: valid, and not worse.
///
/// `routing` provides structural presence (predicate always present,
/// subject/direct_object `Option`); `signal` provides resolution state.
/// When no critical role is structurally present, coverage is defined as
/// `1.0` (nothing to resolve → not a regression).
pub fn frame_coverage(routing: &RoutingSignal, signal: &InterlinguaSignal) -> f64 {
    let mut present = 0u32;
    let mut resolved = 0u32;
    // Predicate is always structurally present (RoutingSignal.predicate is a
    // non-empty String for a non-empty sentence).
    present += 1;
    if signal.predicate_id.is_some_and(|id| id.as_u64() != 0) {
        resolved += 1;
    }
    if routing.subject.is_some() {
        present += 1;
        if signal.subject_id.is_some_and(|id| id.as_u64() != 0) {
            resolved += 1;
        }
    }
    if routing.direct_object.is_some() {
        present += 1;
        if signal.direct_object_id.is_some_and(|id| id.as_u64() != 0) {
            resolved += 1;
        }
    }
    if present == 0 {
        1.0
    } else {
        f64::from(resolved) / f64::from(present)
    }
}

/// Backward-compat shim for callers that only have an `InterlinguaSignal`
/// (tests, calibration). Treats all three roles as structurally present
/// (the pre-F3 `total = 3` shape) so existing golden values are preserved
/// when no `RoutingSignal` is available. New code should call
/// [`frame_coverage`] with the routing signal.
pub fn frame_coverage_signal(signal: &InterlinguaSignal) -> f64 {
    let mut present_and_resolved = 0u32;
    let total = 3u32;
    if signal.predicate_id.is_some_and(|id| id.as_u64() != 0) {
        present_and_resolved += 1;
    }
    if signal.subject_id.is_some_and(|id| id.as_u64() != 0) {
        present_and_resolved += 1;
    }
    if signal
        .direct_object_id
        .is_some_and(|id| id.as_u64() != 0)
    {
        present_and_resolved += 1;
    }
    f64::from(present_and_resolved) / f64::from(total)
}

// ─────────────────────────────────────────────────────────────────────────
// Per-reason trigger-rate counters (M5.4) — instance-owned, lightweight
// atomics, no alloc, lock-free. Owned by the router's `NlpStage` (A3a);
// the pipeline itself is pure and never touches metrics — it returns the
// `RefineReason` and the caller records it. `spacy-rs` keeps the types
// but no global state.
// ─────────────────────────────────────────────────────────────────────────

/// Snapshot of per-reason refine trigger counts (M5.4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefineMetricsSnapshot {
    pub no_trigger: u64,
    pub always_policy: u64,
    pub confidence_overall: u64,
    pub confidence_role_coverage: u64,
    pub confidence_ties: u64,
    pub task_value_unresolved_role: u64,
    pub task_value_unresolved_propn: u64,
    pub task_value_collision: u64,
}

impl RefineMetricsSnapshot {
    /// Total decisions observed.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.no_trigger
            + self.always_policy
            + self.confidence_overall
            + self.confidence_role_coverage
            + self.confidence_ties
            + self.task_value_unresolved_role
            + self.task_value_unresolved_propn
            + self.task_value_collision
    }
}

/// Instance-owned, lock-free per-reason counters (A3a). The router's
/// `NlpStage` owns one `Arc<RefineMetrics>` and records each ladder
/// decision; `spacy-rs` tests construct a local instance.
#[derive(Debug, Default)]
pub struct RefineMetrics {
    no_trigger: std::sync::atomic::AtomicU64,
    always: std::sync::atomic::AtomicU64,
    conf_overall: std::sync::atomic::AtomicU64,
    conf_role: std::sync::atomic::AtomicU64,
    conf_ties: std::sync::atomic::AtomicU64,
    task_role: std::sync::atomic::AtomicU64,
    task_propn: std::sync::atomic::AtomicU64,
    task_collision: std::sync::atomic::AtomicU64,
}

impl RefineMetrics {
    /// Create zeroed metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one refine decision (lock-free).
    pub fn record(&self, reason: RefineReason) {
        match reason {
            RefineReason::NoTrigger => {
                self.no_trigger.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            RefineReason::AlwaysPolicy => {
                self.always.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            RefineReason::Confidence(ConfidenceReason::Overall) => {
                self.conf_overall.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            RefineReason::Confidence(ConfidenceReason::RoleCoverage) => {
                self.conf_role.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            RefineReason::Confidence(ConfidenceReason::Ties) => {
                self.conf_ties.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            RefineReason::TaskValue(TaskValueReason::UnresolvedCriticalRole) => {
                self.task_role.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            RefineReason::TaskValue(TaskValueReason::UnresolvedPropn) => {
                self.task_propn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            RefineReason::TaskValue(TaskValueReason::Collision) => {
                self.task_collision.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Snapshot of the per-reason counters.
    #[must_use]
    pub fn snapshot(&self) -> RefineMetricsSnapshot {
        RefineMetricsSnapshot {
            no_trigger: self.no_trigger.load(std::sync::atomic::Ordering::Relaxed),
            always_policy: self.always.load(std::sync::atomic::Ordering::Relaxed),
            confidence_overall: self.conf_overall.load(std::sync::atomic::Ordering::Relaxed),
            confidence_role_coverage: self.conf_role.load(std::sync::atomic::Ordering::Relaxed),
            confidence_ties: self.conf_ties.load(std::sync::atomic::Ordering::Relaxed),
            task_value_unresolved_role: self.task_role.load(std::sync::atomic::Ordering::Relaxed),
            task_value_unresolved_propn: self.task_propn.load(std::sync::atomic::Ordering::Relaxed),
            task_value_collision: self.task_collision.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero (test helper).
    pub fn reset(&self) {
        self.no_trigger.store(0, std::sync::atomic::Ordering::Relaxed);
        self.always.store(0, std::sync::atomic::Ordering::Relaxed);
        self.conf_overall.store(0, std::sync::atomic::Ordering::Relaxed);
        self.conf_role.store(0, std::sync::atomic::Ordering::Relaxed);
        self.conf_ties.store(0, std::sync::atomic::Ordering::Relaxed);
        self.task_role.store(0, std::sync::atomic::Ordering::Relaxed);
        self.task_propn.store(0, std::sync::atomic::Ordering::Relaxed);
        self.task_collision.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The refiner interface (ROADMAP_20260831_ARCEAGER §2.2)
// ─────────────────────────────────────────────────────────────────────────

/// A model rung that refines an EXISTING validated base parse.
/// `Ok(None)` keeps the base (fallback — never worse). Adoption requires the
/// refined set to pass the 7-check gate AND not regress `frame_coverage`
/// versus the base. Provenance re-stamped to the producing source.
pub trait AnnotationRefiner: Send + Sync {
    /// Refine `base` for `doc`, focusing on `focus` token indices.
    /// `Ok(None)` keeps the base (gate rejected or nothing to change).
    fn refine<'a>(
        self: Box<Self>,
        doc: &'a Doc,
        base: &'a AnnotationResult,
        focus: &'a [usize],
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    >;
}

/// Adapter: [`LlmRung`] as a full-reannotation [`AnnotationRefiner`].
/// Ignores `base`/`focus` — re-annotates the whole doc via the LLM fetch,
/// then gates the result. `Ok(None)` when the gate rejects (base kept).
struct LlmRefiner {
    fetch: LlmFetch,
    validator: Arc<AnnotationValidator>,
}

impl AnnotationRefiner for LlmRefiner {
    fn refine<'a>(
        self: Box<Self>,
        doc: &'a Doc,
        _base: &'a AnnotationResult,
        _focus: &'a [usize],
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    > {
        Box::pin(async move {
            let tokens: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
            let json = match (self.fetch)(tokens).await {
                Ok(j) => j,
                Err(_) => return Ok(None),
            };
            let set = match AnnotationSet::parse_json(&json) {
                Ok(s) => s,
                Err(_) => return Ok(None),
            };
            if self.validator.validate(doc, &set).is_err() {
                return Ok(None);
            }
            Ok(Some(AnnotationResult::new(set, AnnotationSource::Llm)))
        })
    }
}

/// Adapter: [`EncoderRung`] as a full-reannotation [`AnnotationRefiner`].
/// Ignores `base`/`focus` — re-annotates the whole doc via the encoder
/// closure, then gates the result. `Ok(None)` when the gate rejects.
struct EncoderRefiner {
    encoder: EncoderFetchSync,
    validator: Arc<AnnotationValidator>,
}

impl AnnotationRefiner for EncoderRefiner {
    fn refine<'a>(
        self: Box<Self>,
        doc: &'a Doc,
        _base: &'a AnnotationResult,
        _focus: &'a [usize],
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    > {
        Box::pin(async move {
            let set = match (self.encoder)(doc) {
                Ok(s) => s,
                Err(_) => return Ok(None),
            };
            if self.validator.validate(doc, &set).is_err() {
                return Ok(None);
            }
            Ok(Some(AnnotationResult::new(
                set,
                AnnotationSource::Encoder,
            )))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Span-scoped refinement (ROADMAP_20260831_ARCEAGER M2)
// ─────────────────────────────────────────────────────────────────────────

/// The payload handed to a span-scoped refine seam: the canonical
/// [`crate::llm::LlmRefinePrompt`] text, the deterministic orth list, and the
/// focus token indices. Implementors send `prompt` to the model and return
/// the raw corrections-object reply.
#[derive(Debug, Clone)]
pub struct LlmRefineRequest {
    pub prompt: String,
    pub tokens: Vec<String>,
    pub focus: Vec<usize>,
}

/// The **synchronous** span-scoped refine seam — the [`LlmRefineRung`]
/// analog of [`LlmFetchSync`].
///
/// Seam decision (M2.2): a *new* type rather than reusing `LlmFetchSync`
/// with a composite payload. `LlmFetchSync`'s contract is
/// "tokens → full §10.1 annotation array" and its consumers attach the
/// `AnnotationSet` schema to the request; the refine reply is a different
/// wire shape (a corrections object). One seam type with two switchable
/// shapes would surprise every implementor and the grammar-constrained
/// backends; a dedicated seam keeps each contract honest.
pub type LlmRefineFetchSync =
    Arc<dyn Fn(LlmRefineRequest) -> Result<String, AnnotateError> + Send + Sync>;

/// The span-scoped **encoder residual** seam (M2.3): given the doc and the
/// focus indices, produce field-level corrections for those tokens only.
/// Unlike [`EncoderFetchSync`] (a whole-doc re-annotation), the residual
/// never proposes edits outside the focus and returns corrections, not a
/// record set. Not configured by default — only wired when a task-specific
/// encoder head exposes per-token residuals.
pub type EncoderResidualFetch =
    Arc<dyn Fn(&Doc, &[usize]) -> Result<Vec<crate::review::Correction>, AnnotateError> + Send + Sync>;

/// The model seams available to the refine phase beyond the whole-doc
/// [`LlmFetchSync`] / [`EncoderFetchSync`] closures. All-`None` (the
/// default) preserves the M1 behavior exactly: refinement stays
/// full-reannotation through the existing seams.
#[derive(Clone, Default)]
pub struct RefineSeams {
    /// Span-scoped LLM corrections (M2.2) — the `OnUncertain` default LLM
    /// refiner when wired.
    pub llm_focused: Option<LlmRefineFetchSync>,
    /// Span-scoped encoder residual (M2.3) — opt-in only.
    pub encoder_residual: Option<EncoderResidualFetch>,
    /// Span-level detail cache (M6.1) — content-addressed `Vec<Correction>`
    /// keyed by `span_key(doc, focus)`. Shared across async workers and sync
    /// calls (read-through before the model, write-through after). `None` by
    /// default (no caching). When present, a cache hit skips the model call
    /// entirely and replays the cached corrections through the same
    /// `adopt_corrections` gate.
    pub span_cache: Option<std::sync::Arc<dyn crate::cache::SpanCache>>,
}

/// Amend `base` with review-shaped `corrections`, restricted to `focus`
/// tokens, gated through the 7-check validator, and re-stamped to the
/// producing `source`. `None` when the correction set is empty, nothing
/// lands inside the focus, nothing actually applies, or the gate rejects —
/// so the caller keeps the base (fallback — never worse).
///
/// The base's `token_confidence` / `parse_confidence` / `oracle_margins`
/// ride onto the amended result untouched: the refiner improves only what it
/// touched (M2.2), and confidence vectors stay aligned with the token
/// indices (which never change — only field values do).
fn adopt_corrections(
    doc: &Doc,
    base: &AnnotationResult,
    focus: &[usize],
    corrections: &[crate::review::Correction],
    validator: &AnnotationValidator,
    source: AnnotationSource,
) -> Option<AnnotationResult> {
    if corrections.is_empty() {
        return None;
    }
    let scoped: Vec<crate::review::Correction> = corrections
        .iter()
        .filter(|c| focus.contains(&c.token_index))
        .cloned()
        .collect();
    if scoped.is_empty() {
        return None;
    }
    let mut records = base.records().records().to_vec();
    if crate::review::apply_edits(&mut records, &scoped) == 0 {
        return None;
    }
    let set = AnnotationSet(records);
    if validator.validate(doc, &set).is_err() {
        return None;
    }
    let mut result = AnnotationResult::new(set, source)
        .with_confidence(base.token_confidence.clone(), base.parse_confidence.clone());
    result.oracle_margins = base.oracle_margins.clone();
    result.collision_count = base.collision_count;
    Some(result)
}

/// The span-scoped LLM refinement rung (M2.2): shows the base parse and the
/// focus indices, asks for corrections only for the focused tokens, amends
/// the base, and gates the result. Any fetch/parse/gate failure — or an
/// empty correction set — yields `Ok(None)` so the base is kept.
///
/// The fetch is a sync closure called inline from the async refiner — the
/// same documented precedent as [`EncoderRefiner`] (a bounded model call at
/// the caller's own seam; the router bounds it with a `Limiter`).
///
/// M6.1: when a `span_cache` is wired the rung first probes the
/// content-addressed cache (`span_key(doc, focus)`). A hit replays the cached
/// `Vec<Correction>` through the same `adopt_corrections` gate — no model
/// call, no cost. A miss calls the model and write-throughs the returned
/// corrections on success.
pub struct LlmRefineRung {
    fetch: LlmRefineFetchSync,
    validator: Arc<AnnotationValidator>,
    span_cache: Option<std::sync::Arc<dyn crate::cache::SpanCache>>,
}

impl LlmRefineRung {
    /// A focused refiner over `fetch`, gated by `validator`.
    #[must_use]
    pub fn new(fetch: LlmRefineFetchSync, validator: Arc<AnnotationValidator>) -> Self {
        Self {
            fetch,
            validator,
            span_cache: None,
        }
    }

    /// Attach a span cache (M6.1). When present the rung checks the cache
    /// before the model and write-throughs on a successful refine.
    #[must_use]
    pub fn with_cache(
        mut self,
        cache: std::sync::Arc<dyn crate::cache::SpanCache>,
    ) -> Self {
        self.span_cache = Some(cache);
        self
    }
}

impl AnnotationRefiner for LlmRefineRung {
    fn refine<'a>(
        self: Box<Self>,
        doc: &'a Doc,
        base: &'a AnnotationResult,
        focus: &'a [usize],
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    > {
        Box::pin(async move {
            Ok(refine_llm_scoped_with_cache(
                doc, base, focus, &self.fetch, &self.validator, self.span_cache.as_deref(),
            ))
        })
    }
}

/// The shared focused-LLM refinement body — the single implementation both
/// the async [`LlmRefineRung`] and the sync ladder consume (DRY).
#[allow(dead_code)]
fn refine_llm_scoped(
    doc: &Doc,
    base: &AnnotationResult,
    focus: &[usize],
    fetch: &LlmRefineFetchSync,
    validator: &AnnotationValidator,
) -> Option<AnnotationResult> {
    refine_llm_scoped_with_cache(doc, base, focus, fetch, validator, None)
}

/// Cache-aware variant (M6.1): probes `span_cache` before the model and
/// write-throughs on success.
fn refine_llm_scoped_with_cache(
    doc: &Doc,
    base: &AnnotationResult,
    focus: &[usize],
    fetch: &LlmRefineFetchSync,
    validator: &AnnotationValidator,
    span_cache: Option<&dyn crate::cache::SpanCache>,
) -> Option<AnnotationResult> {
    if focus.is_empty() {
        return None; // nothing flagged — the base stands (Ok(None) keeps it)
    }
    // M6.1 read-through: a cached correction replays without a model call.
    if let Some(cache) = span_cache {
        let key = crate::cache::span_key(doc, focus);
        if let Some(cached) = cache.get(key) {
            if let Some(hit) = adopt_corrections(doc, base, focus, &cached, validator, AnnotationSource::Llm) {
                return Some(hit);
            }
        }
    }
    let tokens: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
    let prompt = crate::llm::LlmRefinePrompt::prompt(&tokens, base, focus);
    let reply = (fetch)(LlmRefineRequest {
        prompt,
        tokens,
        focus: focus.to_vec(),
    })
    .ok()?;
    let review = crate::review::ParseReview::parse_json(&reply).ok()?;
    let result = adopt_corrections(
        doc,
        base,
        focus,
        &review.corrections,
        validator,
        AnnotationSource::Llm,
    );
    // M6.1 write-through: cache the corrections that produced a valid refine.
    if let (Some(cache), Some(_)) = (span_cache, &result) {
        let key = crate::cache::span_key(doc, focus);
        cache.put(key, review.corrections.clone());
    }
    result
}

/// The span-scoped encoder residual rung (M2.3): asks the injected residual
/// head for corrections on the focus tokens only, amends the base, and gates
/// the result. `Ok(None)` on any failure or empty residual keeps the base.
///
/// M6.1: the same span-cache read/write-through as `LlmRefineRung`, but for
/// the encoder residual seam.
pub struct EncoderResidualRung {
    fetch: EncoderResidualFetch,
    validator: Arc<AnnotationValidator>,
    span_cache: Option<std::sync::Arc<dyn crate::cache::SpanCache>>,
}

impl EncoderResidualRung {
    /// A focused encoder refiner over `fetch`, gated by `validator`.
    #[must_use]
    pub fn new(fetch: EncoderResidualFetch, validator: Arc<AnnotationValidator>) -> Self {
        Self {
            fetch,
            validator,
            span_cache: None,
        }
    }

    /// Attach a span cache (M6.1).
    #[must_use]
    pub fn with_cache(
        mut self,
        cache: std::sync::Arc<dyn crate::cache::SpanCache>,
    ) -> Self {
        self.span_cache = Some(cache);
        self
    }
}

impl AnnotationRefiner for EncoderResidualRung {
    fn refine<'a>(
        self: Box<Self>,
        doc: &'a Doc,
        base: &'a AnnotationResult,
        focus: &'a [usize],
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    > {
        Box::pin(async move {
            if focus.is_empty() {
                return Ok(None);
            }
            // M6.1 read-through for encoder residual as well.
            if let Some(cache) = &self.span_cache {
                let key = crate::cache::span_key(doc, focus);
                if let Some(cached) = cache.get(key) {
                    if let Some(hit) =
                        adopt_corrections(doc, base, focus, &cached, &self.validator, AnnotationSource::Encoder)
                    {
                        return Ok(Some(hit));
                    }
                }
            }
            let corrections = match (self.fetch)(doc, focus) {
                Ok(c) => c,
                Err(_) => return Ok(None),
            };
            let result = adopt_corrections(
                doc,
                base,
                focus,
                &corrections,
                &self.validator,
                AnnotationSource::Encoder,
            );
            if let (Some(cache), Some(_)) = (&self.span_cache, &result) {
                let key = crate::cache::span_key(doc, focus);
                cache.put(key, corrections.clone());
            }
            Ok(result)
        })
    }
}

/// Which seam shape each refine-phase slot runs. The single slot-selection
/// decision both the async [`refiner_order`] and the sync ladder consume
/// (DRY — M2.4: `OnUncertain` prefers the focused variants, `Always` keeps
/// the full re-annotation adapters, today's behavior).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderSlot {
    Off,
    Full,
    Residual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlmSlot {
    Off,
    Full,
    Focused,
}

fn refine_slots(
    seams: &RefineSeams,
    has_encoder: bool,
    has_fetch: bool,
    mode: RefineMode,
) -> (EncoderSlot, LlmSlot) {
    if mode == RefineMode::Always {
        return (
            if has_encoder { EncoderSlot::Full } else { EncoderSlot::Off },
            if has_fetch { LlmSlot::Full } else { LlmSlot::Off },
        );
    }
    let encoder = if seams.encoder_residual.is_some() {
        EncoderSlot::Residual
    } else if has_encoder {
        EncoderSlot::Full
    } else {
        EncoderSlot::Off
    };
    let llm = if seams.llm_focused.is_some() {
        LlmSlot::Focused
    } else if has_fetch {
        LlmSlot::Full
    } else {
        LlmSlot::Off
    };
    (encoder, llm)
}

/// The LLM rung: fetch JSON, parse it, and accept only if it passes the gate.
#[derive(Clone)]
pub struct LlmRung {
    fetch: LlmFetch,
    validator: Arc<AnnotationValidator>,
}

impl LlmRung {
    /// A rung that fetches and gates the live LLM reply.
    #[must_use]
    pub fn new(fetch: LlmFetch, validator: Arc<AnnotationValidator>) -> Self {
        Self { fetch, validator }
    }

    /// Convert into a full-reannotation [`AnnotationRefiner`]. The refiner
    /// re-annotates the whole doc (ignoring `base`/`focus`) and returns
    /// `Ok(None)` when the gate rejects, so the base is kept.
    #[must_use]
    pub fn into_refiner(self) -> Box<dyn AnnotationRefiner> {
        Box::new(LlmRefiner {
            fetch: self.fetch,
            validator: self.validator,
        })
    }
}

impl AnnotationRung for LlmRung {
    fn run<'a>(
        self: Box<Self>,
        doc: &'a Doc,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    > {
        Box::pin(async move {
            let tokens: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
            let json = (self.fetch)(tokens).await?;
            let set = AnnotationSet::parse_json(&json)?;
            self.validator.validate(doc, &set)?;
            Ok(Some(AnnotationResult::new(set, AnnotationSource::Llm)))
        })
    }
}

/// The trained-encoder rung (ROADMAP_20260827_ORT §4.2): the injected encoder
/// closure maps the doc's orth/idx onto LFM predictions and returns an
/// [`AnnotationSet`] aligned by construction. Accepted only if it passes the
/// gate; any failure falls through to the deterministic rungs (never empty).
///
/// The closure is a blocking CPU call (an ort session run). The async rung
/// calls it inline — a single ms-scale forward — because the closure itself is
/// expected to be internally bounded (the router builds it with a swarm-wide
/// `Limiter` capping concurrent ort runs at 2–4, the red-team thread-budget
/// rule); the production annotation path is the *sync* ladder
/// ([`NlpPipeline::process_sync_with_confidence`]), where the router bounds
/// the call at its own call site.
#[derive(Clone)]
pub struct EncoderRung {
    encoder: EncoderFetchSync,
    validator: Arc<AnnotationValidator>,
}

impl EncoderRung {
    /// A rung that runs the encoder closure and gates its output.
    #[must_use]
    pub fn new(encoder: EncoderFetchSync, validator: Arc<AnnotationValidator>) -> Self {
        Self { encoder, validator }
    }

    /// Convert into a full-reannotation [`AnnotationRefiner`]. The refiner
    /// re-annotates the whole doc (ignoring `base`/`focus`) and returns
    /// `Ok(None)` when the gate rejects, so the base is kept.
    #[must_use]
    pub fn into_refiner(self) -> Box<dyn AnnotationRefiner> {
        Box::new(EncoderRefiner {
            encoder: self.encoder,
            validator: self.validator,
        })
    }
}

impl AnnotationRung for EncoderRung {
    fn run<'a>(
        self: Box<Self>,
        doc: &'a Doc,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    > {
        Box::pin(async move {
            let set = (self.encoder)(doc)?;
            self.validator.validate(doc, &set)?;
            Ok(Some(AnnotationResult::new(set, AnnotationSource::Encoder)))
        })
    }
}

/// The deterministic rule rung: always produces a valid (if coarse) parse, so
/// the ladder never exhausts without an answer (the escalation philosophy:
/// "a fallback always lands on a model that will answer the request").
#[derive(Debug, Clone)]
pub struct RuleRung {
    rule: Arc<RuleAnnotator>,
}

impl AnnotationRung for RuleRung {
    fn run<'a>(
        self: Box<Self>,
        doc: &'a Doc,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AnnotationResult>, AnnotateError>> + Send + 'a,
        >,
    > {
        Box::pin(async move {
            let set = self.rule.annotate(doc);
            Ok(Some(AnnotationResult::new(set, AnnotationSource::RuleRung)))
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The deterministic rule annotator (§10.3 rung 2 — replaces the M5
// placeholder)
// ─────────────────────────────────────────────────────────────────────────

/// The deterministic rule annotator: a per-sentence star parse with lexeme-
/// flag POS and rule lemmatizer.
///
/// For each sentence (from the [`Sentencizer`]'s punctuation boundaries) it
/// picks one ROOT — the last alphabetic non-stop token, else the first token
/// of the sentence (the minimal star fallback for degenerate sentences) — and
/// attaches every other token to it. POS comes from the lexeme flags
/// (`PUNCT`/`NUM`/`PROPN`/`X`) unless a [`GenesisIndex`](crate::genesis::GenesisIndex)
/// has promoted a correction for that orth (M6.2 rule genesis), lemma from the
/// rule [`Lemmatizer`]. This is what the ladder runs when the LLM rung fails
/// or is absent; every field it writes is the same shape the LLM writes, so
/// downstream never knows which rung produced the annotations.
pub struct RuleAnnotator {
    sentencizer: Sentencizer,
    lemmatizer: Lemmatizer,
    genesis: Option<std::sync::Arc<dyn crate::genesis::GenesisIndex>>,
}

impl std::fmt::Debug for RuleAnnotator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleAnnotator")
            .field("sentencizer", &self.sentencizer)
            .field("lemmatizer", &self.lemmatizer)
            .field("has_genesis", &self.genesis.is_some())
            .finish()
    }
}

impl RuleAnnotator {
    /// A rule annotator over the English defaults (rule lemmatizer + default
    /// sentencizer).
    #[must_use]
    pub fn en_default() -> Self {
        Self {
            sentencizer: Sentencizer::new(),
            lemmatizer: Lemmatizer::english_rule(),
            genesis: None,
        }
    }

    /// A rule annotator over a custom sentencizer/lemmatizer.
    #[must_use]
    pub fn new(sentencizer: Sentencizer, lemmatizer: Lemmatizer) -> Self {
        Self {
            sentencizer,
            lemmatizer,
            genesis: None,
        }
    }

    /// Attach a genesis index (M6.2). When present, `annotate` consults it
    /// before the heuristic `pos_of` — a promoted correction becomes
    /// deterministic data.
    #[must_use]
    pub fn with_genesis(mut self, genesis: std::sync::Arc<dyn crate::genesis::GenesisIndex>) -> Self {
        self.genesis = Some(genesis);
        self
    }

    /// The attached genesis index, if any.
    #[must_use]
    pub fn genesis(&self) -> Option<&std::sync::Arc<dyn crate::genesis::GenesisIndex>> {
        self.genesis.as_ref()
    }

    /// The sentencizer providing sentence boundaries.
    #[must_use]
    pub fn sentencizer(&self) -> &Sentencizer {
        &self.sentencizer
    }

    /// The rule lemmatizer.
    #[must_use]
    pub fn lemmatizer(&self) -> &Lemmatizer {
        &self.lemmatizer
    }

    /// The deterministic parse for `doc`.
    #[must_use]
    pub fn annotate(&self, doc: &Doc) -> AnnotationSet {
        let len = doc.len();
        let starts = self.sentencizer.predict(doc);

        // Partition into sentences at each start.
        let mut sentences: Vec<(usize, usize)> = Vec::new();
        let mut cur = 0usize;
        for (i, &is_start) in starts.iter().enumerate() {
            if is_start && i != 0 {
                sentences.push((cur, i));
                cur = i;
            }
        }
        if cur < len {
            sentences.push((cur, len));
        }
        if sentences.is_empty() {
            sentences.push((0, len));
        }

        // One ROOT per sentence; every other token attaches to it.
        let mut heads = vec![0i32; len];
        for (start, end) in sentences {
            let root = (start..end)
                .rev()
                .find(|&i| {
                    let flags = doc.token(i).lexeme.flags;
                    flags.is_alpha() && !flags.is_punct() && !flags.is_stop()
                })
                .unwrap_or(start);
            for (i, slot) in heads[start..end].iter_mut().enumerate() {
                let i = start + i;
                *slot = if i == root { 0 } else { root as i32 - i as i32 };
            }
        }

        let records = (0..len)
            .map(|i| {
                let text = doc.token_text(i);
                let flags = doc.token(i).lexeme.flags;
                // M6.2: genesis overrides heuristic POS when promoted.
                let pos = if let Some(genesis) = &self.genesis {
                    let norm = text.to_ascii_lowercase();
                    genesis.get_pos(&norm).unwrap_or_else(|| Self::pos_of(flags))
                } else {
                    Self::pos_of(flags)
                };
                // M6.2: genesis overrides heuristic NER when promoted.
                let ent_type = if let Some(genesis) = &self.genesis {
                    let norm = text.to_ascii_lowercase();
                    genesis
                        .get_ner(&norm)
                        .map(|t| t.to_string())
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                let ent_iob = if ent_type.is_empty() {
                    String::new()
                } else {
                    "U".to_string()
                };
                let lemma = self.lemmatizer.lemmatize(&text, pos, 0);
                let lemma = lemma.first().cloned().unwrap_or_else(|| text.to_ascii_lowercase());
                let is_root = heads[i] == 0;
                AnnotationRecord {
                    text,
                    pos: pos.to_string(),
                    tag: String::new(),
                    dep: if is_root { "root" } else { "dep" }.to_string(),
                    head: heads[i],
                    lemma,
                    morph: String::new(),
                    ent_iob,
                    ent_type,
                }
            })
            .collect();
        AnnotationSet(records)
    }

    /// The deterministic coarse POS from lexeme flags (the §10.3 "suffix-rule
    /// guess"). Punct/num/propn are exact from the flags; lowercase alphabetic
    /// tokens default to NOUN so the rule lemmatizer's plural / third-singular
    /// reduction (the highest-value suffix rules) actually fires. Whitespace
    /// maps to `X` (the 17-tag contract excludes `space`).
    fn pos_of(flags: crate::lexeme::LexemeFlags) -> crate::labels::Upos {
        use crate::labels::Upos;
        if flags.is_punct() {
            Upos::Punct
        } else if flags.is_digit() {
            Upos::Num
        } else if flags.is_upper() {
            Upos::Propn
        } else if flags.is_alpha() {
            Upos::Noun
        } else {
            Upos::X
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The pipeline orchestrator
// ─────────────────────────────────────────────────────────────────────────

/// Errors surfaced by the orchestrator.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// The tokenizer failed.
    #[error("tokenization failed: {0}")]
    Tokenize(#[from] SpacyError),
    /// The stage graph could not be built.
    #[error("stage graph construction failed: {0}")]
    StageGraph(String),
    /// A stage could not be registered in the supervised batch.
    #[error("stage registration failed: {0}")]
    StageRegistration(String),
    /// A pipeline stage failed, panicked, or was cancelled.
    #[error("pipeline stage failure: {0}")]
    Stage(#[from] PipelineStageFailure),
    /// Every annotation rung was exhausted without an accepted answer.
    #[error("annotation ladder exhausted: {0}")]
    Ladder(String),
    /// The ladder winner's JSON could not be re-serialized.
    #[error("annotation serialization failed: {0}")]
    Serialize(String),
    /// The pipeline produced no doc (attach never ran).
    #[error("pipeline produced no doc")]
    NoDoc,
    /// Attaching validated annotations failed.
    #[error("annotation attach failed: {0}")]
    Apply(#[from] AnnotationError),
}

/// A stage-level failure discriminated by the `SupervisedBatch` event
/// taxonomy (`completed` is not a failure).
#[derive(Debug, Error)]
pub enum PipelineStageFailure {
    #[error("stage {0} failed")]
    Failed(String, String),
    #[error("stage {0} panicked")]
    Panicked(String),
    #[error("stage {0} cancelled by the supervisor")]
    Cancelled(String),
}

/// The NLP pipeline: deterministic tokenizer head + annotation ladder + the
/// stage DAG, composed from fluent-monorepo primitives.
pub struct NlpPipeline {
    vocab: Arc<Vocab>,
    tokenizer: Tokenizer,
    validator: Arc<AnnotationValidator>,
    rule: Arc<RuleAnnotator>,
    stages: StagePipeline,
    resolver: Option<Arc<InterlinguaResolver>>,
    /// M6.1 span cache (shared across ladders, read-through view over the
    /// ledger sqlite when wired by the router; `None` by default).
    span_cache: Option<std::sync::Arc<dyn crate::cache::SpanCache>>,
    /// M6.2 genesis index (POS/NER rule genesis; `None` by default).
    genesis: Option<std::sync::Arc<dyn crate::genesis::GenesisIndex>>,
}

impl NlpPipeline {
    /// Build a pipeline over `vocab`/`tokenizer` gated by `validator`, with
    /// the default rule annotator and sentencizer stage.
    pub fn new(
        vocab: Arc<Vocab>,
        tokenizer: Tokenizer,
        validator: AnnotationValidator,
    ) -> Result<Self, PipelineError> {
        Self::new_with_resolver(vocab, tokenizer, validator, None)
    }

    /// Build a pipeline optionally wired to an [`InterlinguaResolver`]: when
    /// present, the stage DAG gains the read-only `resolve` stage and the sync
    /// path stamps interlingua ids + confidence (ROADMAP §10.4/§11.8).
    pub fn new_with_resolver(
        vocab: Arc<Vocab>,
        tokenizer: Tokenizer,
        validator: AnnotationValidator,
        resolver: Option<Arc<InterlinguaResolver>>,
    ) -> Result<Self, PipelineError> {
        let stages =
            StagePipeline::new_with_resolver(validator.clone(), Sentencizer::new(), resolver.clone())?;
        Ok(Self {
            vocab,
            tokenizer,
            validator: Arc::new(validator),
            rule: Arc::new(RuleAnnotator::en_default()),
            stages,
            resolver,
            span_cache: None,
            genesis: None,
        })
    }

    /// Attach a span cache (M6.1). The cache is shared across the sync/async
    /// ladders; the router wires it as a read-through view over the ledger
    /// sqlite. `SpanCache::invalidate` is called when a `CorrectionIndex`
    /// write for the same span lands.
    #[must_use]
    pub fn with_span_cache(mut self, cache: std::sync::Arc<dyn crate::cache::SpanCache>) -> Self {
        self.span_cache = Some(cache);
        self
    }

    /// Attach a genesis index (M6.2). A promoted POS/NER correction becomes
    /// deterministic data consulted by the rule annotator on the next run.
    #[must_use]
    pub fn with_genesis(mut self, genesis: std::sync::Arc<dyn crate::genesis::GenesisIndex>) -> Self {
        // Rebuild the rule annotator with the new genesis handle (first-wins).
        self.rule = std::sync::Arc::new(
            RuleAnnotator::en_default().with_genesis(std::sync::Arc::clone(&genesis)),
        );
        self.genesis = Some(genesis);
        self
    }

    /// The span cache, if wired.
    #[must_use]
    pub fn span_cache(&self) -> Option<&std::sync::Arc<dyn crate::cache::SpanCache>> {
        self.span_cache.as_ref()
    }

    /// The genesis index, if wired.
    #[must_use]
    pub fn genesis(&self) -> Option<&std::sync::Arc<dyn crate::genesis::GenesisIndex>> {
        self.genesis.as_ref()
    }

    /// Record a refiner correction as genesis evidence (M6.2). Call after a
    /// successful refine to let the correction count toward promotion.
    pub fn record_genesis(&self, corrections: &[crate::review::Correction], doc: &Doc) {
        if let Some(genesis) = &self.genesis {
            for c in corrections {
                if c.token_index < doc.len() {
                    let norm = doc.token_text(c.token_index).to_ascii_lowercase();
                    genesis.record(c, &norm);
                }
            }
        }
    }

    /// A pipeline over English defaults (the generated en data).
    pub fn en_default() -> Result<Self, PipelineError> {
        let vocab = Arc::new(Vocab::new(lang::en::lexicon_config()));
        let tokenizer = lang::en::tokenizer(vocab.clone())?;
        Self::new(vocab, tokenizer, AnnotationValidator::new())
    }

    /// A pipeline over English defaults whose **string store is pre-loaded**
    /// from `strings_path` (or empty when absent) — the load-at-startup side
    /// of the durable StringStore (§5, M7.8). Interned lemma/orth strings
    /// survive restarts, so hash→InterlinguaId resolution is stable.
    pub fn en_default_with_strings(strings_path: &Path) -> Result<Self, PipelineError> {
        let vocab = Arc::new(Vocab::load_or_empty(
            strings_path,
            lang::en::lexicon_config(),
        ));
        let tokenizer = lang::en::tokenizer(vocab.clone())?;
        Self::new(vocab, tokenizer, AnnotationValidator::new())
    }

    /// Persist the pipeline's string store (the persist-after-annotate side
    /// of M7.8). Call after an annotation pass so newly interned lemmas are
    /// durable.
    pub fn persist_strings(&self, path: &Path) -> Result<(), std::io::Error> {
        self.vocab.save(path)
    }

    /// The fully synchronous deterministic path — tokenize → sync ladder →
    /// attach → sentencize. No tokio runtime, no `SupervisedBatch`: built for
    /// synchronous callers (the Coral Router's `NlpStage` runs inside a sync
    /// pipeline stage, honoring the WorkUnit purity contract).
    ///
    /// When `fetch` is `Some`, the **deterministic base always runs first**
    /// (ArcEager → Rule). The LLM annotation rung is only consulted as a
    /// refiner when [`RefinePolicy::mode`] is [`RefineMode::Always`] or
    /// [`RefineMode::OnUncertain`]. Any fetch/parse/gate failure falls back to
    /// the deterministic base, so the call never fails on annotation quality.
    /// The returned doc carries `sent_start` boundaries and is ready for
    /// [`crate::routing::extract_routing_signals`].
    pub fn process_sync(
        &self,
        text: &str,
        fetch: Option<&LlmFetchSync>,
    ) -> Result<Doc, PipelineError> {
        self.process_sync_with_refine(text, fetch, None, &RefineSeams::default(), None, RefinePolicy::default())
            .map(|(doc, _)| doc)
    }

    /// The synchronous deterministic path plus the ladder's full handoff
    /// (provenance + confidence, §9.1) — the seam the router's `NlpStage` uses
    /// to publish the parse's confidence alongside the doc (C1). The doc is
    /// resolved (interlingua ids stamped) when a resolver is wired.
    ///
    /// `encoder` (ROADMAP_20260827_ORT §4.2) is the trained-encoder seam:
    /// when present, the refine phase attempts it before the LLM. Its output
    /// must pass the same 7-check gate; any failure falls back to the
    /// deterministic base (never empty).
    pub fn process_sync_with_confidence(
        &self,
        text: &str,
        fetch: Option<&LlmFetchSync>,
        encoder: Option<&EncoderFetchSync>,
        policy: RefinePolicy,
    ) -> Result<(Doc, AnnotationResult), PipelineError> {
        self.process_sync_with_refine(
            text,
            fetch,
            encoder,
            &RefineSeams::default(),
            None,
            policy,
        )
    }

    /// [`Self::process_sync_with_confidence`] plus the span-scoped refine
    /// seams (ROADMAP_20260831_ARCEAGER M2.4) and an explicit `resolver`
    /// override (used by the batch path; `None` uses the pipeline's own).
    pub fn process_sync_with_refine(
        &self,
        text: &str,
        fetch: Option<&LlmFetchSync>,
        encoder: Option<&EncoderFetchSync>,
        seams: &RefineSeams,
        resolver: Option<&Arc<InterlinguaResolver>>,
        policy: RefinePolicy,
    ) -> Result<(Doc, AnnotationResult), PipelineError> {
        self.process_sync_with_refine_and_reason(text, fetch, encoder, seams, resolver, policy)
            .map(|(doc, result, _)| (doc, result))
    }

    /// Like [`Self::process_sync_with_refine`] but also returns the refine
    /// decision reason (M5.1) — the `RefineReason` the ladder evaluated on the
    /// base parse.  Surfaced through `NlpStage`'s confidence summary.
    pub fn process_sync_with_refine_and_reason(
        &self,
        text: &str,
        fetch: Option<&LlmFetchSync>,
        encoder: Option<&EncoderFetchSync>,
        seams: &RefineSeams,
        resolver: Option<&Arc<InterlinguaResolver>>,
        policy: RefinePolicy,
    ) -> Result<(Doc, AnnotationResult, RefineReason), PipelineError> {
        let resolver = resolver.or(self.resolver.as_ref());
        let mut doc = self.tokenizer.tokenize(text)?;
        let (mut result, reason) = run_ladder_sync(
            &doc,
            fetch,
            encoder,
            seams,
            &self.validator,
            &self.rule,
            resolver,
            policy,
        );
        crate::llm::attach(&mut doc, result.records()).map_err(PipelineError::Apply)?;
        self.rule.sentencizer().process(&mut doc);
        if let Some(resolver) = resolver {
            let notes = resolver.resolve_doc(&mut doc, result.token_confidence());
            result.collision_count = notes
                .iter()
                .filter(|n| matches!(n, crate::interlingua::CollisionNote::Collision { .. }))
                .count();
        }
        Ok((doc, result, reason))
    }

    /// The pipeline's vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// The pipeline's validator.
    #[must_use]
    pub fn validator(&self) -> &AnnotationValidator {
        &self.validator
    }

    /// The rule annotator (the deterministic terminal rung).
    #[must_use]
    pub fn rule(&self) -> &RuleAnnotator {
        &self.rule
    }

    /// The stage DAG.
    #[must_use]
    pub fn stages(&self) -> &StagePipeline {
        &self.stages
    }

    /// The hermetic JSON path: tokenize → parse → validate → attach, routing
    /// through the stage DAG. No ladder, no async LLM — fully deterministic.
    pub async fn annotate_json(
        &self,
        text: &str,
        json: &str,
        rt: Arc<dyn Runtime>,
        caps: CapabilitySet,
    ) -> Result<Doc, PipelineError> {
        let doc = self.tokenizer.tokenize(text)?;
        let state = Arc::new(Mutex::new(PipelineState {
            doc: Some(doc),
            ..PipelineState::default()
        }));
        self.stages.run(&state, json.to_string(), rt, caps).await?;
        let state = state.lock().expect("pipeline state lock");
        state.doc.clone().ok_or(PipelineError::NoDoc)
    }

    /// The full path: tokenize, walk the annotation ladder (two-phase:
    /// deterministic base + conditional model refinement), then run the stage
    /// DAG under `SupervisedBatch`. `fetch` is the live-LLM seam; when absent
    /// or when the policy is `Off`, the ladder is just the deterministic base.
    pub async fn process_async(
        &self,
        text: &str,
        fetch: Option<LlmFetch>,
        rt: Arc<dyn Runtime>,
        caps: CapabilitySet,
    ) -> Result<Doc, PipelineError> {
        let doc = self.tokenizer.tokenize(text)?;
        let result = self
            .run_ladder(&doc, fetch, None, &RefineSeams::default(), RefinePolicy::default())
            .await?;
        let json = serde_json::to_string(result.records())
            .map_err(|e| PipelineError::Serialize(e.to_string()))?;
        let state = Arc::new(Mutex::new(PipelineState {
            doc: Some(doc),
            annotation: Some(result),
            ..PipelineState::default()
        }));
        self.stages.run(&state, json, rt, caps).await?;
        let state = state.lock().expect("pipeline state lock");
        state.doc.clone().ok_or(PipelineError::NoDoc)
    }

    /// Fan out the ladder over N docs via `ResultPool` (the §10.5 parallel
    /// annotation path), then attach each validated set. The `fetch` seam is
    /// shared by all workers; the pool bounds concurrent LLM calls. The
    /// optional `encoder` seam is forwarded to the ladder (ROADMAP_20260827_ORT
    /// §4.4).
    pub async fn annotate_batch_async(
        &self,
        texts: &[&str],
        fetch: Option<LlmFetch>,
        encoder: Option<EncoderFetchSync>,
        rt: Arc<dyn Runtime>,
        concurrency: usize,
        queue_capacity: usize,
        policy: RefinePolicy,
    ) -> Result<Vec<Doc>, PipelineError> {
        self.annotate_batch_with_refine(
            texts,
            fetch,
            encoder,
            &RefineSeams::default(),
            rt,
            concurrency,
            queue_capacity,
            policy,
        )
        .await
    }

    /// [`Self::annotate_batch_async`] plus the span-scoped refine seams
    /// (ROADMAP_20260831_ARCEAGER M2.4). The seams are shared `Arc`s across
    /// the pool workers (same pattern as the fetch/encoder seams).
    pub async fn annotate_batch_with_refine(
        &self,
        texts: &[&str],
        fetch: Option<LlmFetch>,
        encoder: Option<EncoderFetchSync>,
        seams: &RefineSeams,
        rt: Arc<dyn Runtime>,
        concurrency: usize,
        queue_capacity: usize,
        policy: RefinePolicy,
    ) -> Result<Vec<Doc>, PipelineError> {
        let docs: Vec<Doc> = texts
            .iter()
            .map(|t| self.tokenizer.tokenize(t))
            .collect::<Result<_, _>>()?;

        let validator = Arc::clone(&self.validator);
        let rule = Arc::clone(&self.rule);
        let resolver = self.resolver.clone();
        let seams = seams.clone();
        let pool = ResultPool::new(
            Arc::clone(&rt),
            concurrency,
            queue_capacity,
            move |(doc, fetch, encoder): (
                Doc,
                Option<LlmFetch>,
                Option<EncoderFetchSync>,
            )| {
                let validator = Arc::clone(&validator);
                let rule = Arc::clone(&rule);
                let resolver = resolver.clone();
                let seams = seams.clone();
                let policy = policy;
                async move {
                    let result = run_ladder_for(
                        &doc,
                        fetch,
                        encoder.as_ref(),
                        &seams,
                        &validator,
                        &rule,
                        resolver.as_ref(),
                        policy,
                    )
                    .await?;
                    Ok((doc, result))
                }
            },
        );

        let mut out = Vec::with_capacity(docs.len());
        for doc in docs {
            let (doc, result) = pool
                .submit((doc, fetch.clone(), encoder.clone()))
                .await
                .map_err(|e| PipelineError::Ladder(pool_error(e)))?;
            let mut doc = doc;
            apply_with(&mut doc, result.records(), &self.validator)?;
            // The resolver is a shared `Arc` (pure, no lock) so the batch
            // workers never contend; stamp ids on each attached doc.
            if let Some(resolver) = &self.resolver {
                resolver.resolve_doc(&mut doc, result.token_confidence());
            }
            out.push(doc);
        }
        Ok(out)
    }

    /// The annotation ladder for a single doc.
    async fn run_ladder(
        &self,
        doc: &Doc,
        fetch: Option<LlmFetch>,
        encoder: Option<&EncoderFetchSync>,
        seams: &RefineSeams,
        policy: RefinePolicy,
    ) -> Result<AnnotationResult, PipelineError> {
        run_ladder_for(
            doc,
            fetch,
            encoder,
            seams,
            &self.validator,
            &self.rule,
            self.resolver.as_ref(),
            policy,
        )
        .await
    }

    /// The shared resolver, when wired.
    #[must_use]
    pub fn resolver(&self) -> Option<&Arc<InterlinguaResolver>> {
        self.resolver.as_ref()
    }
}

/// The frame-completeness view of a parse — §2.3's
/// `signal = extract_routing_signals(doc, base)`. The ladder runs on the raw
/// tokenized doc (attach and resolve happen after it), so `should_refine`
/// and the adoption gate would otherwise see every role as unresolved. This
/// helper attaches the parse onto a **scratch clone** of the doc, sentencizes
/// it, and — when a resolver is wired — stamps the interlingua ids there
/// (pure/read-only over the store, C2), then extracts the first sentence's
/// `(RoutingSignal, InterlinguaSignal)` pair plus the surfaced collision
/// count. The real doc is never mutated by the decision path.
///
/// Returns `(routing, interlingua, collisions)`; defaults (all-`None` roles,
/// empty ids) for a doc with no extractable sentence.
///
/// For multi-sentence documents, the `unresolved_token_threshold` is evaluated
/// **aggregated across the whole document** via [`parse_views`] + the
/// `*_aggregated` helpers — `parse_view` remains the single-sentence view used
/// for the `frame_coverage` regression gate.
fn parse_view(
    doc: &Doc,
    result: &AnnotationResult,
    rule: &RuleAnnotator,
    resolver: Option<&Arc<InterlinguaResolver>>,
) -> (RoutingSignal, InterlinguaSignal, usize) {
    let mut scratch = doc.clone();
    if crate::llm::attach(&mut scratch, result.records()).is_err() {
        return (default_routing_signal(), default_interlingua_signal(), 0);
    }
    rule.sentencizer().process(&mut scratch);
    let collisions = match resolver {
        Some(r) => r
            .resolve_doc(&mut scratch, result.token_confidence())
            .iter()
            .filter(|n| matches!(n, crate::interlingua::CollisionNote::Collision { .. }))
            .count(),
        None => 0,
    };
    match crate::routing::extract_routing_signals(&scratch).into_iter().next() {
        Some(signal) => {
            let interlingua = signal
                .interlingua
                .clone()
                .unwrap_or_else(default_interlingua_signal);
            (signal, interlingua, collisions)
        }
        None => (default_routing_signal(), default_interlingua_signal(), collisions),
    }
}

/// Like [`parse_view`] but returns **all** sentence signals (document-global).
/// The `unresolved_token_threshold` is evaluated over the concatenated
/// `token_ids` of every sentence; `UnresolvedCriticalRole` triggers if any
/// sentence has a structurally present but unresolved role.
fn parse_views(
    doc: &Doc,
    result: &AnnotationResult,
    rule: &RuleAnnotator,
    resolver: Option<&Arc<InterlinguaResolver>>,
) -> (Vec<(RoutingSignal, InterlinguaSignal)>, usize) {
    let mut scratch = doc.clone();
    if crate::llm::attach(&mut scratch, result.records()).is_err() {
        return (Vec::new(), 0);
    }
    rule.sentencizer().process(&mut scratch);
    let collisions = match resolver {
        Some(r) => r
            .resolve_doc(&mut scratch, result.token_confidence())
            .iter()
            .filter(|n| matches!(n, crate::interlingua::CollisionNote::Collision { .. }))
            .count(),
        None => 0,
    };
    let signals = crate::routing::extract_routing_signals(&scratch)
        .into_iter()
        .map(|s| {
            let inter = s.interlingua.clone().unwrap_or_else(default_interlingua_signal);
            (s, inter)
        })
        .collect();
    (signals, collisions)
}

fn default_interlingua_signal() -> InterlinguaSignal {
    InterlinguaSignal {
        predicate_id: None,
        subject_id: None,
        direct_object_id: None,
        indirect_object_id: None,
        concept_ids: vec![],
        token_ids: vec![],
        confidence: None,
    }
}

fn default_routing_signal() -> RoutingSignal {
    RoutingSignal {
        sentence: String::new(),
        predicate: String::new(),
        subject: None,
        direct_object: None,
        indirect_object: None,
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec![],
        lemmas: vec![],
        pos: vec![],
        deps: vec![],
        heads: vec![],
        interlingua: None,
    }
}

/// Build the ordered list of refiners from the available model seams.
/// Encoder first, then LLM — the single source of refiner ordering (DRY).
/// Both the async and sync ladders consume the same slot decision
/// ([`refine_slots`]), so the ordering and the focused/full selection are
/// decided in exactly one place: `Always` keeps the full-reannotation
/// adapters (today's behavior); `OnUncertain` prefers the focused variants
/// when the corresponding [`RefineSeams`] is wired, falling back to the
/// full adapters otherwise.
fn refiner_order(
    encoder: Option<EncoderFetchSync>,
    fetch: Option<LlmFetch>,
    seams: &RefineSeams,
    mode: RefineMode,
    validator: &Arc<AnnotationValidator>,
) -> Vec<Box<dyn AnnotationRefiner>> {
    let (encoder_slot, llm_slot) = refine_slots(seams, encoder.is_some(), fetch.is_some(), mode);
    let mut refiners: Vec<Box<dyn AnnotationRefiner>> = Vec::with_capacity(2);
    match encoder_slot {
        EncoderSlot::Residual => {
            let mut rung = EncoderResidualRung::new(
                seams.encoder_residual.clone().expect("slot implies seam"),
                Arc::clone(validator),
            );
            if let Some(cache) = &seams.span_cache {
                rung = rung.with_cache(std::sync::Arc::clone(cache));
            }
            refiners.push(Box::new(rung));
        }
        EncoderSlot::Full => {
            refiners.push(Box::new(EncoderRefiner {
                encoder: encoder.expect("slot implies seam"),
                validator: Arc::clone(validator),
            }));
        }
        EncoderSlot::Off => {}
    }
    match llm_slot {
        LlmSlot::Focused => {
            let mut rung = LlmRefineRung::new(
                seams.llm_focused.clone().expect("slot implies seam"),
                Arc::clone(validator),
            );
            if let Some(cache) = &seams.span_cache {
                rung = rung.with_cache(std::sync::Arc::clone(cache));
            }
            refiners.push(Box::new(rung));
        }
        LlmSlot::Full => {
            refiners.push(Box::new(LlmRefiner {
                fetch: fetch.expect("slot implies seam"),
                validator: Arc::clone(validator),
            }));
        }
        LlmSlot::Off => {}
    }
    refiners
}

/// The two-phase annotation ladder (ROADMAP_20260831_ARCEAGER §2.3):
///
/// 1. **Base phase** (deterministic, unconditional): `first_accept_in_order`
///    over `[ArcEager, Rule]`. Always produces a validated `AnnotationResult`.
/// 2. **Refine phase** (model, conditional): when `should_refine` (its
///    frame-completeness view taken via [`parse_view`] over the base), runs
///    `first_accept_in_order` over the model refiners `[Encoder, Llm]` —
///    focused variants when the [`RefineSeams`] are wired and the mode is
///    `OnUncertain`, full re-annotation otherwise. A refiner that returns a
///    validated, non-regressing result is adopted; otherwise the base is
///    kept.
#[allow(clippy::too_many_arguments)]
async fn run_ladder_for(
    doc: &Doc,
    fetch: Option<LlmFetch>,
    encoder: Option<&EncoderFetchSync>,
    seams: &RefineSeams,
    validator: &Arc<AnnotationValidator>,
    rule: &Arc<RuleAnnotator>,
    resolver: Option<&Arc<InterlinguaResolver>>,
    policy: RefinePolicy,
) -> Result<AnnotationResult, PipelineError> {
    // ── Base phase: deterministic, unconditional ──
    let annotator = Arc::new(ArcEagerAnnotator::en_default(Arc::clone(doc.vocab())));
    let eager = ArcEagerRung::new(Arc::clone(&annotator), Arc::clone(validator));
    let rule_rung = RuleRung {
        rule: Arc::clone(rule),
    };
    let base_rungs: Vec<Box<dyn AnnotationRung>> = vec![Box::new(eager), Box::new(rule_rung)];
    let base = first_accept_in_order(base_rungs, |rung| rung.run(doc), |_| false)
        .await
        .map_err(|e| PipelineError::Ladder(e.to_string()))?
        .ok_or_else(|| PipelineError::Ladder("no rung produced an answer".into()))?;

    // ── Refine phase: model, conditional on policy ──
    // (A3a: pipeline is pure — it returns the reason, the caller records)
    // R2 scope: aggregated across whole multi-sentence document.
    if policy.mode == RefineMode::Off {
        return Ok(base);
    }
    let (all_signals, collisions) = parse_views(doc, &base, rule, resolver);
    // Fallback to single-view for frame_coverage denominator (first sentence)
    let (signal, interlingua_signal) = all_signals
        .first()
        .cloned()
        .unwrap_or_else(|| (default_routing_signal(), default_interlingua_signal()));
    let mut decision_base = base.clone();
    decision_base.collision_count = collisions;
    let reason = if all_signals.is_empty() {
        refine_reason(&decision_base, &interlingua_signal, &signal, policy)
    } else {
        refine_reason_aggregated(&decision_base, &all_signals, policy)
    };
    if reason == RefineReason::NoTrigger {
        return Ok(base);
    }

    let focus = if all_signals.is_empty() {
        refine_focus(&base, &interlingua_signal, policy)
    } else {
        refine_focus_aggregated(&base, &all_signals, policy)
    };
    // A5a: routing-aware coverage — a refined result that drops a
    // structurally present role is a regression even if the denominator
    // shrinks. The base coverage uses the base's routing presence.
    let base_coverage = frame_coverage(&signal, &interlingua_signal);
    let refiners = refiner_order(encoder.cloned(), fetch, seams, policy.mode, validator);
    if refiners.is_empty() {
        return Ok(base);
    }

    for refiner in refiners {
        if let Some(refined) = refiner
            .refine(doc, &base, &focus)
            .await
            .map_err(|e| PipelineError::Ladder(e.to_string()))?
        {
            let (_, refined_interlingua, _) = parse_view(doc, &refined, rule, resolver);
            // A5a: compare against base routing presence — dropping a
            // structurally present role must be a regression even if the
            // refined parse no longer declares that role (denominator would
            // otherwise shrink and mask the regression).
            if frame_coverage(&signal, &refined_interlingua) < base_coverage {
                continue; // routing regression — try next refiner, then base
            }
            return Ok(refined);
        }
    }
    Ok(base) // fallback keeps the base
}

/// The synchronous two-phase ladder walk (ROADMAP_20260831_ARCEAGER §2.3):
///
/// 1. **Base phase**: `first_accept_in_order_sync` over `[ArcEager, Rule]`.
/// 2. **Refine phase** (conditional): when `should_refine`, tries model
///    refiners via `first_accept_in_order_sync` with the same adoption gate.
///
/// Reuses `should_refine` / `refine_slots` / `refine_focus` /
/// `frame_coverage` / `parse_view` / `refine_llm_scoped` — no duplicated
/// decision logic; the slot selection and the focused-refinement body are
/// the same code the async walk runs. The sync ladder never errors / never
/// exhausts (the rule rung is infallible).
#[allow(clippy::too_many_arguments)]
fn run_ladder_sync(
    doc: &Doc,
    fetch: Option<&LlmFetchSync>,
    encoder: Option<&EncoderFetchSync>,
    seams: &RefineSeams,
    validator: &Arc<AnnotationValidator>,
    rule: &Arc<RuleAnnotator>,
    resolver: Option<&Arc<InterlinguaResolver>>,
    policy: RefinePolicy,
) -> (AnnotationResult, RefineReason) {
    /// One ordered rung of the sync ladder, owned (each closure invocation
    /// consumes it — no per-rung clones).
    enum SyncRung {
        Llm(LlmFetchSync),
        LlmFocused(LlmRefineFetchSync),
        Encoder(EncoderFetchSync),
        EncoderResidual(EncoderResidualFetch),
        Eager(Arc<ArcEagerAnnotator>),
        Rule(Arc<RuleAnnotator>),
    }

    let annotator = Arc::new(ArcEagerAnnotator::en_default(Arc::clone(doc.vocab())));

    // ── Base phase: deterministic, unconditional ──
    let base_rungs: Vec<SyncRung> = vec![
        SyncRung::Eager(annotator),
        SyncRung::Rule(Arc::clone(rule)),
    ];
    let base = first_accept_in_order_sync(
        base_rungs,
        |rung| match rung {
            SyncRung::Eager(annotator) => {
                let accepted = annotator
                    .annotate_with_confidence(doc)
                    .ok()
                    .map(|(result, _)| result)
                    .filter(|result| validator.validate(doc, result.records()).is_ok());
                Ok(accepted)
            }
            SyncRung::Rule(rule) => {
                let set = rule.annotate(doc);
                Ok(Some(AnnotationResult::new(
                    set,
                    AnnotationSource::RuleRung,
                )))
            }
            _ => unreachable!("base phase only has Eager and Rule"),
        },
        |_: &AnnotateError| false,
    )
    .expect("the sync ladder never errors — rungs only skip (Ok(None))")
    .expect("the rule rung is infallible, so the ladder never exhausts");

    // ── Refine phase: model, conditional on policy ──
    // (A3a: pipeline is pure — reason returned, caller records)
    // R2 scope: aggregated across whole multi-sentence document.
    if policy.mode == RefineMode::Off {
        let reason = RefineReason::NoTrigger;
        return (base, reason);
    }
    let (all_signals, collisions) = parse_views(doc, &base, rule, resolver);
    let (signal, interlingua_signal) = all_signals
        .first()
        .cloned()
        .unwrap_or_else(|| (default_routing_signal(), default_interlingua_signal()));
    let mut decision_base = base.clone();
    decision_base.collision_count = collisions;
    let reason = if all_signals.is_empty() {
        refine_reason(&decision_base, &interlingua_signal, &signal, policy)
    } else {
        refine_reason_aggregated(&decision_base, &all_signals, policy)
    };
    if reason == RefineReason::NoTrigger {
        return (base, reason);
    }

    let focus = if all_signals.is_empty() {
        refine_focus(&base, &interlingua_signal, policy)
    } else {
        refine_focus_aggregated(&base, &all_signals, policy)
    };
    // A5a: routing-aware coverage — same as async ladder.
    let base_coverage = frame_coverage(&signal, &interlingua_signal);

    // The same slot decision the async walk makes (refine_slots) — full vs
    // focused, encoder vs LLM, in one place.
    let (encoder_slot, llm_slot) = refine_slots(seams, encoder.is_some(), fetch.is_some(), policy.mode);
    let mut sync_refiners: Vec<SyncRung> = Vec::with_capacity(2);
    match encoder_slot {
        EncoderSlot::Residual => {
            sync_refiners.push(SyncRung::EncoderResidual(
                seams.encoder_residual.clone().expect("slot implies seam"),
            ));
        }
        EncoderSlot::Full => {
            sync_refiners.push(SyncRung::Encoder(
                encoder.cloned().expect("slot implies seam"),
            ));
        }
        EncoderSlot::Off => {}
    }
    match llm_slot {
        LlmSlot::Focused => {
            sync_refiners.push(SyncRung::LlmFocused(
                seams.llm_focused.clone().expect("slot implies seam"),
            ));
        }
        LlmSlot::Full => {
            sync_refiners.push(SyncRung::Llm(
                fetch.cloned().expect("slot implies seam"),
            ));
        }
        LlmSlot::Off => {}
    }
    if sync_refiners.is_empty() {
        return (base, reason);
    }

    let span_cache = seams.span_cache.clone();
    for rung in sync_refiners {
        let maybe_refined: Option<AnnotationResult> = match rung {
            SyncRung::Llm(fetch) => {
                let tokens: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
                let accepted = fetch(tokens)
                    .ok()
                    .and_then(|json| AnnotationSet::parse_json(&json).ok())
                    .and_then(|set| validator.validate(doc, &set).ok().map(|()| set));
                accepted.map(|set| AnnotationResult::new(set, AnnotationSource::Llm))
            }
            SyncRung::LlmFocused(fetch) => refine_llm_scoped_with_cache(
                doc,
                &base,
                &focus,
                &fetch,
                validator,
                span_cache.as_deref(),
            ),
            SyncRung::Encoder(encoder) => {
                let accepted = encoder(doc)
                    .ok()
                    .and_then(|set| validator.validate(doc, &set).ok().map(|()| set));
                accepted.map(|set| AnnotationResult::new(set, AnnotationSource::Encoder))
            }
            SyncRung::EncoderResidual(fetch) => {
                if focus.is_empty() {
                    None
                } else if let Some(cache) = &span_cache {
                    let key = crate::cache::span_key(doc, &focus);
                    if let Some(cached) = cache.get(key) {
                        adopt_corrections(
                            doc, &base, &focus, &cached, validator, AnnotationSource::Encoder,
                        )
                    } else {
                        let corrections = match (fetch)(doc, &focus) {
                            Ok(c) => Some(c),
                            Err(_) => None,
                        };
                        corrections.and_then(|corrections| {
                            let result = adopt_corrections(
                                doc, &base, &focus, &corrections, validator,
                                AnnotationSource::Encoder,
                            );
                            if let Some(cache) = &span_cache {
                                if result.is_some() {
                                    let key = crate::cache::span_key(doc, &focus);
                                    cache.put(key, corrections.clone());
                                }
                            }
                            result
                        })
                    }
                } else {
                    let corrections = match (fetch)(doc, &focus) {
                        Ok(c) => Some(c),
                        Err(_) => None,
                    };
                    corrections.and_then(|corrections| {
                        let result = adopt_corrections(
                            doc, &base, &focus, &corrections, validator,
                            AnnotationSource::Encoder,
                        );
                        if let Some(cache) = &span_cache {
                            if result.is_some() {
                                let key = crate::cache::span_key(doc, &focus);
                                cache.put(key, corrections.clone());
                            }
                        }
                        result
                    })
                }
            }
            _ => unreachable!("refine phase only has model rungs"),
        };
        if let Some(refined_result) = maybe_refined {
            let (_, refined_interlingua, _) = parse_view(doc, &refined_result, rule, resolver);
            if frame_coverage(&signal, &refined_interlingua) < base_coverage {
                continue;
            }
            return (refined_result, reason);
        }
    }
    (base, reason)
}

/// Collapse a `ResultPoolError<PipelineError>` into a pipeline error.
fn pool_error(e: fluent_concurrency::pool::ResultPoolError<PipelineError>) -> String {
    match e {
        fluent_concurrency::pool::ResultPoolError::Inner(e) => e.to_string(),
        fluent_concurrency::pool::ResultPoolError::Canceled => "worker cancelled".into(),
        fluent_concurrency::pool::ResultPoolError::Pool(p) => p.to_string(),
    }
}

/// Materialize the immutable, shareable [`crate::ArcReadyAnnotation`] from a
/// successful ladder run (OVERLAYS §4.1, M3): an already-attached,
/// sentencized, and — when a resolver is wired — resolved `doc` plus its
/// ladder [`AnnotationResult`] handoff. This is the pipeline hook: it composes
/// [`crate::routing::extract_routing_signals`] over the run's doc and hands
/// the validated output to [`crate::ArcReadyAnnotation::from_doc`].
///
/// Pure and additive: it changes no ladder control flow (`run_ladder_for` /
/// `run_ladder_sync`) and no existing caller's return value, and it is inert
/// for any caller that does not ask for it. The annotation is the validated
/// output — the wire records + signals + token baseline — not the working
/// [`Doc`], so consumers share it behind an `Arc` with no locks at read time.
#[must_use]
pub fn arc_ready(doc: &Doc, result: &AnnotationResult) -> crate::ArcReadyAnnotation {
    let signals = crate::routing::extract_routing_signals(doc);
    crate::ArcReadyAnnotation::from_doc(doc, result, signals)
}

// ─────────────────────────────────────────────────────────────────────────
// Configuration / reflection surface (scaffold — roadmap §5)
// ─────────────────────────────────────────────────────────────────────────

/// Pipeline configuration. Derives `FieldAccess`/`Describable`/`bon::Builder`
/// for the Coral Router control plane (roadmap §5); **compatibility surface**
/// with no in-tree consumer until the router integration lands (Milestone 6).
#[derive(Debug, Clone, Serialize, Deserialize, FieldAccess, Describable, Builder)]
pub struct NlpPipelineConfig {
    /// Dependency labels the validator accepts (comma-joined; parses through
    /// `DepLabelSet::FromStr`).
    #[field(desc = "accepted dependency labels, comma-joined")]
    pub dep_labels: crate::labels::DepLabelSet,
    /// Require projective dependency trees (§10.2 check 8).
    #[field(desc = "require projective dependency trees")]
    pub require_projectivity: bool,
    /// Enable the deterministic rule annotation rung (the terminal ladder
    /// rung — sentencizer + rule lemmatizer + lexeme-flag POS).
    #[field(desc = "enable the deterministic rule annotation rung")]
    pub rule_enabled: bool,
    /// Max concurrent annotations in `annotate_batch_async`.
    #[field(desc = "max concurrent annotations in a batch")]
    pub batch_concurrency: usize,
}

impl Default for NlpPipelineConfig {
    fn default() -> Self {
        Self {
            dep_labels: crate::labels::DepLabelSet::ud_default(),
            require_projectivity: false,
            rule_enabled: true,
            batch_concurrency: 4,
        }
    }
}

impl NlpPipelineConfig {
    /// A validator matching this configuration.
    #[must_use]
    pub fn validator(&self) -> AnnotationValidator {
        let mut v = AnnotationValidator::with_dep_labels(self.dep_labels.clone());
        if self.require_projectivity {
            v = v.require_projectivity(true);
        }
        v
    }
}

#[cfg(test)]
#[path = "../tests/pipeline.rs"]
pub(crate) mod tests;
