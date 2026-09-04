//! Subagent tool retrieval surface (ROADMAP M5): fast, confidence-scored
//! lemma-grep over a parsed [`Doc`], an embedding-driven fuzzy retrieval
//! primitive (the paraphrase coverage the lemma axis cannot — "show" /
//! "display" / "get"), and a cross-check combiner that surfaces **both** axes
//! on a region when they materially disagree — never silently preferring one.
//!
//! The embedding axis is a **substitution point**: [`EmbeddingProvider`] and
//! [`FuzzyRetrieval`] are traits a caller implements (the router's real HNSW
//! index backed by an encoder `ChatBackend`). spacy-rs ships a hermetic
//! in-memory index ([`InMemoryFuzzyIndex`]) and deterministic providers for
//! tests, so nothing here needs a model.

use std::sync::Arc;

use common_core::score::top_k_by_score;
use common_core::vector_math::cosine_similarity_f32;
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

/// The provenance of a retrieval hit. `TreeSitter` / `StringSearch` are
/// produced by sibling tool backends (the compiled-tool world); spacy-rs's own
/// tools produce [`RetrievalSource::LemmaGrep`] and [`RetrievalSource::Fuzzy`].
/// Kept here as pure data so a cross-crate combiner can tag every hit without
/// spacy-rs depending on those backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalSource {
    LemmaGrep,
    Fuzzy,
    TreeSitter,
    StringSearch,
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

/// A fuzzy-retrieval hit: an embedding-similar span (paraphrase coverage).
#[derive(Debug, Clone, PartialEq)]
pub struct FuzzyHit {
    pub span: Span,
    pub score: f32,
    pub embedding_id: u64,
}

/// One surfaced hit, discriminated by [`RetrievalSource`]. A subagent never
/// sees a bare span — it sees the producing axis, the span, and the
/// confidence/score that axis warrants.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalHit {
    pub source: RetrievalSource,
    pub span: Span,
    /// Lemma-grep: the matched lemma. Fuzzy: `None`.
    pub lemma: Option<String>,
    /// Lemma-grep: the lemma's interlingua id, when stamped.
    pub lemma_id: Option<InterlinguaId>,
    /// Lemma-grep: the per-token parse confidence. Always `Some` for lemma hits.
    pub parse_confidence: Option<f64>,
    /// Fuzzy: the similarity score in `0..=1`. `None` otherwise.
    pub fuzzy_score: Option<f32>,
    /// Fuzzy: the matched region's embedding id.
    pub embedding_id: Option<u64>,
}

impl RetrievalHit {
    /// A lemma-grep hit surfaced for a subagent.
    #[must_use]
    pub fn from_lemma(hit: &LemmaGrepHit) -> Self {
        Self {
            source: RetrievalSource::LemmaGrep,
            span: hit.span,
            lemma: Some(hit.lemma.clone()),
            lemma_id: hit.lemma_id,
            parse_confidence: Some(hit.parse_confidence),
            fuzzy_score: None,
            embedding_id: None,
        }
    }

    /// A fuzzy hit surfaced for a subagent.
    #[must_use]
    pub fn from_fuzzy(hit: &FuzzyHit) -> Self {
        Self {
            source: RetrievalSource::Fuzzy,
            span: hit.span,
            lemma: None,
            lemma_id: None,
            parse_confidence: None,
            fuzzy_score: Some(hit.score),
            embedding_id: Some(hit.embedding_id),
        }
    }
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

/// An embedding provider — the substitution point for a real encoder backend.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed `text` into a fixed-dimension vector, or `None` when the provider
    /// is unavailable (fail-open for a missing encoder).
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
}

/// A fuzzy retrieval index over document regions (paraphrase axis).
pub trait FuzzyRetrieval: Send + Sync {
    /// The top-`k` spans closest to `query`.
    fn search(&self, query: &str, k: usize) -> Vec<FuzzyHit>;
}

/// A hermetic in-memory fuzzy index over `(span, text, embedding)` regions
/// ranked by cosine similarity. Deterministic and model-free — the test double
/// a caller replaces with a real HNSW-backed [`FuzzyRetrieval`] implementation.
#[derive(Clone)]
pub struct InMemoryFuzzyIndex {
    provider: Arc<dyn EmbeddingProvider>,
    regions: Vec<(Span, Vec<f32>)>,
}

impl InMemoryFuzzyIndex {
    /// An empty index over `provider`.
    #[must_use]
    pub fn new(provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            provider,
            regions: Vec::new(),
        }
    }

    /// Index a region's text under `span`. Returns `false` (no insert) when the
    /// provider cannot embed it.
    pub fn insert(&mut self, span: Span, text: &str) -> bool {
        match self.provider.embed(text) {
            Some(emb) => {
                self.regions.push((span, emb));
                true
            }
            None => false,
        }
    }
}

impl FuzzyRetrieval for InMemoryFuzzyIndex {
    fn search(&self, query: &str, k: usize) -> Vec<FuzzyHit> {
        let Some(qe) = self.provider.embed(query) else {
            return Vec::new();
        };
        let scored: Vec<(f32, usize)> = self
            .regions
            .iter()
            .enumerate()
            .map(|(i, (_, emb))| (cosine(&qe, emb), i))
            .collect();
        // Shared top-K tail (P2): descending score, take(k). The strict-
        // positive filter stays call-site (fail-open semantics never move).
        top_k_by_score(scored, k, |t| t.0, true)
            .into_iter()
            .filter(|(s, _)| *s > 0.0)
            .map(|(s, i)| FuzzyHit {
                span: self.regions[i].0,
                score: s,
                embedding_id: i as u64,
            })
            .collect()
    }
}

/// Cosine similarity of two same-length vectors (`0.0` on mismatch).
///
/// Thin alias over [`common_core::vector_math::cosine_similarity_f32`] (the P1
/// canonical home). The one known representation delta: on empty-vs-empty the
/// old two-pass body yielded `-0.0` where the canonical yields `0.0`
/// (`-0.0 == 0.0`, and the only caller filters `> 0.0` — false for both —
/// so the delta is filter-invisible; locked in by
/// `cosine_parity_with_canonical`).
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity_f32(a, b)
}

/// A per-region cross-check verdict: whether the lemma and fuzzy axes agreed.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionVerdict {
    pub span: Span,
    pub lemma_confidence: Option<f64>,
    pub fuzzy_score: Option<f32>,
    /// Material disagreement: `|lemma_confidence - fuzzy_score|` exceeded the
    /// tolerance. The subagent should weigh the two axes independently.
    pub disagreed: bool,
}

/// All surfaced hits plus per-region cross-check verdicts. Both axes' hits for
/// a region are **always surfaced when they materially disagree** — the
/// subagent sees the conflict rather than a silent single preference.
#[derive(Debug, Clone)]
pub struct CrossCheckReport {
    pub hits: Vec<RetrievalHit>,
    pub regions: Vec<RegionVerdict>,
}

/// Combine the lemma-grep and fuzzy axes for a query. Every hit from either
/// axis is surfaced; regions covered by both are annotated with a
/// [`RegionVerdict`] so a material disagreement is visible, never collapsed.
pub fn cross_check(
    lemma_hits: &[LemmaGrepHit],
    fuzzy_hits: &[FuzzyHit],
    agree_tolerance: f64,
) -> CrossCheckReport {
    let mut hits: Vec<RetrievalHit> = Vec::new();
    hits.extend(lemma_hits.iter().map(RetrievalHit::from_lemma));
    hits.extend(fuzzy_hits.iter().map(RetrievalHit::from_fuzzy));

    let mut regions: Vec<RegionVerdict> = Vec::new();

    // Regions covered by a fuzzy hit: verdict against every overlapping lemma hit.
    for fh in fuzzy_hits {
        let overlapping: Vec<&LemmaGrepHit> = lemma_hits
            .iter()
            .filter(|lh| lh.span.overlaps(&fh.span))
            .collect();
        if overlapping.is_empty() {
            regions.push(RegionVerdict {
                span: fh.span,
                lemma_confidence: None,
                fuzzy_score: Some(fh.score),
                disagreed: false,
            });
            continue;
        }
        for lh in overlapping {
            let conf = lh.parse_confidence;
            let score = f64::from(fh.score);
            regions.push(RegionVerdict {
                span: fh.span,
                lemma_confidence: Some(conf),
                fuzzy_score: Some(fh.score),
                disagreed: (conf - score).abs() > agree_tolerance,
            });
        }
    }

    // Lemma-only regions (no fuzzy overlap) — surfaced alone, no conflict.
    for lh in lemma_hits {
        if !fuzzy_hits.iter().any(|fh| fh.span.overlaps(&lh.span)) {
            regions.push(RegionVerdict {
                span: lh.span,
                lemma_confidence: Some(lh.parse_confidence),
                fuzzy_score: None,
                disagreed: false,
            });
        }
    }

    CrossCheckReport { hits, regions }
}

#[cfg(test)]
#[path = "../tests/retrieval.rs"]
mod tests;
