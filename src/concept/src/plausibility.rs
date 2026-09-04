//! Text-half / knowledge-half bridge for taxonomy plausibility
//! (ROADMAP_20260903_SPACY_RS_SPLIT M5).
//!
//! [`PlausibilityTriple`] is the neutral input to taxonomy scoring: the
//! text half (`spacy-rs` `triple::build_plausibility_inputs`) fills one per
//! extracted triple — the content-addressed [`InterlinguaId`] plus the raw
//! lemma string each role resolved from — and the knowledge half
//! (`guidance-ontology` `plausibility::score_plausibility`) scores them
//! against the [`ConceptStore`]. Neither half names the other's domain
//! types, so `spacy-rs` keeps no `guidance` edge and the ontology keeps no
//! `spacy-rs` edge (both import-boundary rules hold).
//!
//! The lemma string rides alongside the id because the curie fallback
//! (`"dog"` → `yago:Dog` / `schema:Dog`) needs the surface form when the
//! lemma id itself is unregistered; the id alone cannot rebuild it.

#![forbid(unsafe_code)]

use fluent_types::InterlinguaId;

/// One scorable argument: the content-addressed id plus the surface lemma
/// it resolved from (needed for the curie fallback).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoredLemma {
    /// The `SpacyLemma` content id for [`ScoredLemma::lemma`].
    pub id: InterlinguaId,
    /// The raw (lowercased) lemma string, e.g. `"dog"`.
    pub lemma: String,
}

impl ScoredLemma {
    /// A scorable argument from its id + surface lemma.
    #[must_use]
    pub fn new(id: InterlinguaId, lemma: impl Into<String>) -> Self {
        Self { id, lemma: lemma.into() }
    }
}

/// One deterministic triple in scorable form: the predicate is always
/// present, either argument may be `None` (intransitive etc.).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlausibilityTriple {
    /// The subject lemma, when the sentence has an `nsubj`-role argument.
    pub subject: Option<ScoredLemma>,
    /// The predicate lemma (always present — the sentence root).
    pub predicate: ScoredLemma,
    /// The direct-object lemma, when the sentence has a `dobj` argument.
    pub object: Option<ScoredLemma>,
}

impl PlausibilityTriple {
    /// A scorable triple from its role lemmas.
    #[must_use]
    pub fn new(
        subject: Option<ScoredLemma>,
        predicate: ScoredLemma,
        object: Option<ScoredLemma>,
    ) -> Self {
        Self { subject, predicate, object }
    }
}
