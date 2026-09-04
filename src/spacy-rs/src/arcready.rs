//! `ArcReadyAnnotation` — a fully-materialized, immutable annotation document.
//!
//! The shareable product of the annotation ladder (Llm → Encoder → ArcEager →
//! Rule), validated by the 7-check gate, plus the routing signals and the
//! detail-baseline tokens. "ArcReady" means it is completely materialized and
//! safe to share behind an `Arc` with **no locks at read time**: every field is
//! owned immutable data and nothing in the struct is mutable after
//! construction (`#![forbid(unsafe_code)]`; no `Mutex`/`RefCell`/atomics).
//!
//! This is the `spacy-rs`-owned overlay document the router's ledger caches
//! lazily on a `ContentNode` (the `arc_ready` overlays design). It deliberately
//! stores the **validated output** — the wire [`AnnotationSet`], the producing
//! rung + confidence, the surfaced interlingua collisions, the per-sentence
//! routing signals, and the [`TokenRecord`] detail baseline (orth + byte
//! offsets, aligned to the raw request text) — not the mutable working
//! [`crate::doc::Doc`], so consumers read the annotation without re-running the
//! tokenizer or holding the node write lock.

use crate::arc_eager::ParseConfidence;
use crate::doc::{Doc, TokenRecord};
use crate::llm::{AnnotationResult, AnnotationSet, AnnotationSource};
use crate::routing::RoutingSignal;

impl fluent_types::NodeOverlay for ArcReadyAnnotation {
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
pub struct ArcReadyAnnotation {
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

impl ArcReadyAnnotation {
    /// Run the pipeline once and materialize the immutable document from an
    /// attached `doc`, its ladder [`AnnotationResult`], and the extracted
    /// routing `signals`. The caller treats a genuinely empty doc (no tokens,
    /// no signals) as "absent" rather than an annotation.
    #[must_use]
    pub fn from_doc(
        doc: &Doc,
        result: &AnnotationResult,
        signals: Vec<RoutingSignal>,
    ) -> ArcReadyAnnotation {
        ArcReadyAnnotation {
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

#[cfg(test)]
#[path = "../tests/arcready.rs"]
mod tests;
