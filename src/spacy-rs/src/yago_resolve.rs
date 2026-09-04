//! YagoResolveStage — Alt C, inserted between `attach` and `frame`.
//! For each `dobj/iobj/pobj` head, `store.resolve_yago_iri` + `ancestors_of` → `role.candidate_concept_ids`
//! and `ParseConfidence::semantic_plausibility: Option<f64>` (separate field, never blended into oracle_margins).
//!
//! Tenancy (M5): the stage shell lives here (it is pipeline/`Language`
//! machinery over [`PipelineState`](crate::pipeline::PipelineState)), but the
//! knowledge-half scoring is **injected** from the knowledge owner
//! (`guidance-ontology` `plausibility::score_plausibility`) through the
//! dependency-free [`PlausibilityFetch`](crate::triple::PlausibilityFetch)
//! seam — same pattern as the router injecting `LlmFetch`. The stage is
//! never constructed in-crate with knowledge; `new`/`with_enabled` build
//! the default-off shell (fail-closed `None`), `with_fetch` wires the
//! owner-provided fetch.

#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex};

use fluent_wvr::prelude::*;
use internment::ArcIntern;
use serde_json::json;

use fluent_concept::ConceptStore;
use crate::pipeline::PipelineState;
use crate::triple::PlausibilityFetch;

const STATE_KEY: &str = "spacy.pipeline_state";

/// Stage `yago_resolve`: consults the injected knowledge-half fetch over the
/// text-half triple derivation (ROADMAP M3 — guarded spike, default **off**).
/// When disabled, or when no fetch is wired, the stage leaves
/// `semantic_plausibility` as `None` (the pre-M3 behavior); when enabled with
/// a fetch it fills the field from the owner's score (never blended into
/// `oracle_margins`, E7).
pub struct YagoResolveStage {
    store: Arc<dyn ConceptStore>,
    enabled: bool,
    /// Cached resolver built from `store` + the first doc's vocab strings.
    /// `Mutex` for interior mutability; `None` until first enabled execution.
    cached_resolver: std::sync::Mutex<Option<Arc<crate::interlingua::InterlinguaResolver>>>,
    /// The owner-provided scoring fetch (`None` = knowledge unwired).
    fetch: Option<PlausibilityFetch>,
}

impl Clone for YagoResolveStage {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            enabled: self.enabled,
            cached_resolver: std::sync::Mutex::new(None),
            fetch: self.fetch.clone(),
        }
    }
}

impl YagoResolveStage {
    /// A stage over `store` with the triple spike **disabled** (default) and
    /// no knowledge fetch wired (fail-closed `None`).
    #[must_use]
    pub fn new(store: Arc<dyn ConceptStore>) -> Self {
        Self {
            store,
            enabled: false,
            cached_resolver: std::sync::Mutex::new(None),
            fetch: None,
        }
    }

    /// A stage over `store` with the explicit `enabled` flag for the guarded
    /// M3 spike (still no fetch — enabled without a fetch stays `None`).
    #[must_use]
    pub fn with_enabled(store: Arc<dyn ConceptStore>, enabled: bool) -> Self {
        Self {
            store,
            enabled,
            cached_resolver: std::sync::Mutex::new(None),
            fetch: None,
        }
    }

    /// A stage over `store` with the knowledge-half fetch injected from the
    /// knowledge owner (M5): the owner adapts its kernel (e.g.
    /// `guidance-ontology` `plausibility::score_plausibility`) into the
    /// [`PlausibilityFetch`](crate::triple::PlausibilityFetch) closure at the
    /// wiring site.
    #[must_use]
    pub fn with_fetch(
        store: Arc<dyn ConceptStore>,
        enabled: bool,
        fetch: PlausibilityFetch,
    ) -> Self {
        Self {
            store,
            enabled,
            cached_resolver: std::sync::Mutex::new(None),
            fetch: Some(fetch),
        }
    }

    /// Whether the guarded semantic-plausibility spike is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool { self.enabled }

    /// Whether a knowledge-half fetch is wired.
    #[must_use]
    pub fn has_fetch(&self) -> bool { self.fetch.is_some() }

    fn resolver_for(&self, doc: &crate::doc::Doc) -> Arc<crate::interlingua::InterlinguaResolver> {
        let mut guard = self.cached_resolver.lock().expect("yago resolver lock");
        if let Some(r) = guard.as_ref() {
            return Arc::clone(r);
        }
        let r = Arc::new(crate::interlingua::InterlinguaResolver::new(
            Arc::clone(&self.store),
            Arc::clone(doc.vocab().strings()),
        ));
        *guard = Some(Arc::clone(&r));
        r
    }
}

impl WorkUnit for YagoResolveStage {
    fn name(&self) -> &str { "yago_resolve" }
    fn depends(&self) -> &[ArcIntern<str>] {
        static DEPS: std::sync::LazyLock<[ArcIntern<str>;1]> = std::sync::LazyLock::new(|| [ArcIntern::from("annotated_doc")]);
        &*DEPS
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        static PROVIDES: std::sync::LazyLock<[ArcIntern<str>;1]> = std::sync::LazyLock::new(|| [ArcIntern::from("yago_resolved")]);
        &*PROVIDES
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let state = ctx.get::<Arc<Mutex<PipelineState>>>(STATE_KEY).ok_or_else(|| WorkError::Dependency(STATE_KEY.into()))?;
        let mut state = state.lock().expect("pipeline state lock");
        let doc = state.doc.as_ref().ok_or_else(|| WorkError::Dependency("doc missing".into()))?;
        // Guarded M3 spike + M5 injection: when disabled (default) or when no
        // knowledge fetch is wired, leave plausibility as None (pre-M3
        // behavior). Never touches oracle_margins (E7).
        let plausibility = if !self.enabled {
            None
        } else {
            let resolver = self.resolver_for(doc);
            let triples = crate::triple::extract_triples(doc);
            crate::triple::semantic_plausibility_via_fetch(
                doc,
                &triples,
                &resolver,
                self.fetch.as_ref(),
            )
        };
        state.semantic_plausibility = plausibility;
        if let Some(ann) = state.annotation.as_mut() {
            if let Some(pc) = ann.parse_confidence_mut() {
                pc.semantic_plausibility = plausibility;
            }
        }
        Ok(WorkOutput::ok("yago resolve done"))
    }
}

impl Describable for YagoResolveStage {
    fn describe(&self) -> serde_json::Value { json!({"name":"yago_resolve","depends":["annotated_doc"],"provides":["yago_resolved"]}) }
}
impl FieldAccess for YagoResolveStage {
    fn set_field(&mut self, _n: &str, _v: &str) -> Result<(), FieldError> { Err(FieldError::NotFound(_n.into())) }
    fn get_field(&self, _n: &str) -> Result<String, FieldError> { Err(FieldError::NotFound(_n.into())) }
    fn field_names(&self) -> &'static [&'static str] { &[] }
}
impl_component!(YagoResolveStage);
