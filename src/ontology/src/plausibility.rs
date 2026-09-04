//! Taxonomy plausibility scoring — the knowledge half of the M3 triple
//! spike (ROADMAP_20260903_SPACY_RS_SPLIT M5).
//!
//! Moved here from `spacy-rs` `triple.rs`: scoring subject/object noun
//! lemmas against the YaGO taxonomy reads the knowledge half of the
//! text/knowledge bridge, so it lives with [`YaGoLoader`](crate::yago_loader).
//! `spacy-rs` keeps only the pure derivation (`extract_triples`) plus the
//! `PlausibilityFetch` seam over the shared [`PlausibilityTriple`] input — no
//! `guidance` edge enters `spacy-rs`, no `spacy-rs` edge enters this crate.
//!
//! The score stays a separate `ParseConfidence` field wherever it is
//! consumed (roadmap E7 — never blended into `oracle_margins`); this module
//! structurally cannot touch margins (it returns a bare `f64` over shared
//! types only).

#![forbid(unsafe_code)]

use fluent_concept::{ConceptStore, ConceptStoreState, PlausibilityTriple, ScoredLemma};

/// Re-export of the shared scoring input for owner-side callers.
pub use fluent_concept::{PlausibilityTriple as PlausibilityInput, ScoredLemma as PlausibilityLemma};

/// Score `triples` against the YaGO taxonomy: for each triple, test whether
/// the subject/object lemmas are known in `store`
/// (`store.contains` on the content id, else the `yago:Capitalized` /
/// `schema:Capitalized` curie fallback). A known argument counts as
/// plausible; the per-triple score is `1.0` (both), `0.5` (one), `0.0`
/// (neither). The sentence-level plausibility is the mean over triples;
/// `None` when there are no triples.
///
/// When `store` is in `Loading` state the call returns `None` (provisional —
/// taxonomy not yet ready, mirrors `FrameKey` provisional gating).
#[must_use]
pub fn score_plausibility(
    triples: &[PlausibilityTriple],
    store: &dyn ConceptStore,
) -> Option<f64> {
    if triples.is_empty() {
        return None;
    }
    if store.state() == ConceptStoreState::Loading {
        return None;
    }
    let mut scores = Vec::with_capacity(triples.len());
    for t in triples {
        let subj_known = t.subject.as_ref().is_some_and(|s| lemma_known(s, store));
        let obj_known = t.object.as_ref().is_some_and(|o| lemma_known(o, store));
        let triple_score = match (subj_known, obj_known, t.subject.is_some(), t.object.is_some()) {
            // Both arguments present and known, or the single structurally
            // present argument resolved — fully plausible.
            (true, true, true, true) | (true, _, true, false) | (_, true, false, true) => 1.0,
            // Both arguments present, exactly one known — half plausible.
            (true, false, true, true) | (false, true, true, true) => 0.5,
            // Present but unknown — implausible.
            (false, false, true, true) | (false, _, true, false) | (_, false, false, true) => 0.0,
            // No arguments (verbless or bare predicate) — predicate-only triple
            // is vacuously plausible when the predicate lemma itself is known.
            (_, _, false, false) => {
                if lemma_known(&t.predicate, store) { 1.0 } else { 0.0 }
            }
        };
        scores.push(triple_score);
    }
    if scores.is_empty() {
        None
    } else {
        Some(scores.iter().sum::<f64>() / scores.len() as f64)
    }
}

fn lemma_known(lemma: &ScoredLemma, store: &dyn ConceptStore) -> bool {
    if lemma.lemma.is_empty() {
        return false;
    }
    if store.contains(lemma.id) {
        return true;
    }
    // YaGO curie fallback: a common noun lemma like "dog" is stored as
    // `yago:Dog` / `schema:Dog` when loaded from the taxonomy. Try the
    // capitalized curie form.
    let curie = format!("yago:{}", capitalize(&lemma.lemma));
    if store.resolve_name(&curie).is_ok() {
        return true;
    }
    let schema = format!("schema:{}", capitalize(&lemma.lemma));
    if store.resolve_name(&schema).is_ok() {
        return true;
    }
    // Hierarchy check: if the lemma id resolves to a class that is a subclass
    // of a known upper class, the store's `is_subclass_of` will surface via
    // `ancestors_of` — but we already checked `contains`; only an
    // is-subclass edge needs no extra lookup here because the store's
    // `contains` covers the node itself and the subclass relationship is
    // transitive through the graph we built at insert time.
    false
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
    }
}

#[cfg(test)]
#[path = "../tests/plausibility.rs"]
mod tests;
