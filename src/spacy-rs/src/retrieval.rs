//! Pure lemma-grep helpers over a parsed [`Doc`] (ROADMAP_20260903_SPACY_RS_SPLIT
//! M4 — the remainder after the split).
//!
//! This module keeps only the bare pure helpers a caller needs without the
//! tool envelope: byte-offset [`Span`]s, [`LemmaGrepHit`]s, and [`lemma_grep`]
//! itself (fast, confidence-scored, exact). The tool surface —
//! [`RetrievalSource`](https://docs.rs/fluent-router)-style hit tagging,
//! the embedding-driven fuzzy axis, and the cross-check combiner — lives with
//! the router retrieval owner beside `NodeRetrievalService`, which composes
//! [`lemma_grep`] with its own fuzzy + combiner stages.

use fluent_types::InterlinguaId;

use crate::doc::Doc;

/// A byte-offset span into the doc's reconstructed text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the span is empty (zero bytes).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Whether `other` shares any byte with this span.
    #[must_use]
    pub fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// A lemma-grep hit: a span whose token lemma equals the query, **always**
/// carrying its per-token parse confidence (heuristic matches are never
/// presented with tree-sitter's certainty) and its interlingua lemma id when
/// the resolve stage stamped it.
#[derive(Debug, Clone, PartialEq)]
pub struct LemmaGrepHit {
    pub span: Span,
    pub lemma: String,
    pub lemma_id: Option<InterlinguaId>,
    pub parse_confidence: f64,
}

/// Lemma-grep over a parsed doc: every token whose resolved lemma equals
/// `query` (case-insensitive) becomes a hit carrying its span, the lemma, its
/// interlingua id (when the resolve stage stamped it), and its per-token parse
/// confidence. Confidence is mandatory; a token with no resolved lemma is
/// skipped (no lemma → cannot match).
pub fn lemma_grep(doc: &Doc, query: &str) -> Vec<LemmaGrepHit> {
    let q = query.to_lowercase();
    let strings = doc.vocab().strings();
    let mut hits = Vec::new();
    let mut byte = 0usize;
    for (i, token) in doc.tokens().iter().enumerate() {
        let text = doc.token_text(i);
        let span = Span {
            start: byte,
            end: byte + text.len(),
        };
        byte += text.len() + usize::from(token.spacy);
        let Some(lemma) = strings.get(token.lemma) else {
            continue;
        };
        if lemma.to_lowercase() != q {
            continue;
        }
        hits.push(LemmaGrepHit {
            span,
            lemma: lemma.to_string(),
            lemma_id: token.interlingua_lemma_id,
            parse_confidence: token.confidence.unwrap_or(0.0),
        });
    }
    hits
}

#[cfg(test)]
#[path = "../tests/retrieval.rs"]
mod tests;
