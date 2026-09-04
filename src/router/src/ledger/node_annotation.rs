//! `NodeAnnotation` — the fully-materialized, immutable ledger annotation document.
//!
//! Moved from `spacy-rs` (`arcready.rs::ArcReadyAnnotation`, renamed) per
//! `REVIEW-20260903-SPACY-TO-RUST-COMPARISON.md` §7: a ledger cache entry
//! implementing the ledger's `NodeOverlay` trait must live with the ledger,
//! not in the NLP crate. The rename kills the ArcEager/ArcReady confusion at
//! the source — this type is not a parser stage, it is the shareable product
//! of the annotation ladder.
//!
//! spacy-rs exports only the *inputs* (`AnnotationSet`/`AnnotationResult` via
//! `spacy_rs::llm`, `RoutingSignal` via `spacy_rs::routing`, `TokenRecord` via
//! `spacy_rs::doc`); this module owns the document, the `NodeOverlay` impl,
//! and the `node_annotation` constructor composing
//! `spacy_rs::extract_routing_signals` with `from_doc` (replacing
//! `spacy_rs::pipeline::arc_ready`).
//!
//! Safe to share behind an `Arc` with **no locks at read time**: every field
//! is owned immutable data and nothing in the struct is mutable after
//! construction (`#![forbid(unsafe_code)]`; no `Mutex`/`RefCell`/atomics).

use spacy_rs::{
    AnnotationResult, AnnotationSet, AnnotationSource, Doc, ParseConfidence, RoutingSignal,
    TokenRecord,
};

impl fluent_types::NodeOverlay for NodeAnnotation {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A fully-materialized, immutable annotation document, shareable as an `Arc`.
///
/// Produced by the annotation ladder and validated by the 7-check gate;
/// nothing in it is mutable after construction. The `tokens` detail baseline
/// carries the token orths and byte offsets aligned to the raw request text,
/// so consumers get exact alignment without re-running the tokenizer.
#[derive(Debug, Clone)]
pub struct NodeAnnotation {
    /// The validated per-token annotation records (text/pos/tag/dep/head/lemma).
    pub records: AnnotationSet,
    /// Which rung produced the parse, plus confidence / provenance.
    pub source: AnnotationSource,
    /// Per-token parse confidence (ArcEager/encoder fill it; Llm/Rule set
    /// `None`).
    pub token_confidence: Option<Vec<f64>>,
    /// Parse-level confidence + role coverage.
    pub parse_confidence: Option<ParseConfidence>,
    /// Interlingua collisions surfaced by the resolve step (G9).
    pub collision_count: usize,
    /// Per-sentence routing signals (predicate/subject/object + interlingua).
    pub signals: Vec<RoutingSignal>,
    /// The tokens as the detail baseline (orth + byte offsets), aligned to the
    /// raw request bytes.
    pub tokens: Vec<TokenRecord>,
}

impl NodeAnnotation {
    /// Run the pipeline once and materialize the immutable document from an
    /// attached `doc`, its ladder [`AnnotationResult`], and the extracted
    /// routing `signals`. The caller treats a genuinely empty doc (no tokens,
    /// no signals) as "absent" rather than an annotation.
    #[must_use]
    pub fn from_doc(
        doc: &Doc,
        result: &AnnotationResult,
        signals: Vec<RoutingSignal>,
    ) -> NodeAnnotation {
        NodeAnnotation {
            records: result.records.clone(),
            source: result.source,
            token_confidence: result.token_confidence.clone(),
            parse_confidence: result.parse_confidence.clone(),
            collision_count: result.collision_count,
            signals,
            tokens: doc.tokens().to_vec(),
        }
    }

    /// The primary [`RoutingSignal`] for routing / tree `match_interlingua`
    /// consumption.
    ///
    /// A request is usually a single sentence, in which case the (single)
    /// signal **is** the whole text. For a multi-sentence request the parser's
    /// most-confident sentence is chosen as the routing-relevant one (tie-break
    /// to the earliest), falling back to the first signal when no signal
    /// carries a parse confidence. `None` for an annotation with no signals.
    #[must_use]
    pub fn primary_signal(&self) -> Option<&RoutingSignal> {
        let mut best: Option<&RoutingSignal> = None;
        let mut best_conf = f64::NEG_INFINITY;
        for signal in &self.signals {
            let conf = signal.interlingua.as_ref().and_then(|i| i.confidence);
            let conf = conf.unwrap_or(f64::NEG_INFINITY);
            // `>` (not `>=`) keeps the earliest signal on ties.
            if best.is_none() || conf > best_conf {
                best = Some(signal);
                best_conf = conf;
            }
        }
        best
    }
}

/// Materialize the immutable, shareable [`NodeAnnotation`] from a successful
/// ladder run: an already-attached, sentencized, and — when a resolver is
/// wired — resolved `doc` plus its ladder [`AnnotationResult`] handoff.
/// Composes [`spacy_rs::extract_routing_signals`] over the run's doc and hands
/// the validated output to [`NodeAnnotation::from_doc`].
///
/// Pure and additive: the annotation is the validated output — the wire
/// records + signals + token baseline — not the working [`Doc`], so consumers
/// share it behind an `Arc` with no locks at read time.
#[must_use]
pub fn node_annotation(doc: &Doc, result: &AnnotationResult) -> NodeAnnotation {
    let signals = spacy_rs::extract_routing_signals(doc);
    NodeAnnotation::from_doc(doc, result, signals)
}

#[cfg(test)]
#[path = "../../tests/ledger_node_annotation.rs"]
mod tests;
