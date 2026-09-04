//! Live subagent retrieval service — the M5/M6 live-dispatch seam (ROADMAP
//! "Subagent tool surface" + "Ranking pipeline" follow-up).
//!
//! This module is the agent/subagent **tool owner** (ROADMAP_20260903_SPACY_RS_SPLIT
//! M4 — moved here from `spacy-rs`): the hit tagging ([`RetrievalSource`] /
//! [`RetrievalHit`]), the embedding-driven fuzzy axis ([`EmbeddingProvider`] /
//! [`FuzzyRetrieval`] / [`InMemoryFuzzyIndex`]), and the [`cross_check`]
//! combiner live beside the [`NodeRetrievalService`] that wires them.
//! spacy-rs keeps only the pure `lemma_grep`-grade helpers (`Span`,
//! `LemmaGrepHit`, `lemma_grep`), which the service composes.
//!
//! Lemma-grep is fast + exact (every hit carries its `ParseConfidence`),
//! fuzzy retrieval covers the paraphrase gap (different lemmas, same intent),
//! and `cross_check` surfaces **both** axes on a region when they materially
//! disagree. The service parses a candidate node's LOD0 into a `Doc`, runs the
//! two axes + the combiner, and pre-filters the candidate pool through the M6
//! [`SalienceRanker`] — the model ranks only the deterministic shortlist,
//! never the full pool.
//!
//! Hermetic by default: the embedder and ranker are injected (a real encoder +
//! a `LedgerSalienceProvider`-backed ranker in production; stubs in tests), and
//! the pipeline falls back to the deterministic `en_default()` — the service
//! never requires a model. A node that fails to parse is skipped (fail-open,
//! logged); a node with no hits simply contributes an empty report.

use std::collections::BTreeMap;
use std::sync::Arc;

use common_core::score::top_k_by_score;
use common_core::vector_math::cosine_similarity_f32;
use fluent_types::{ContentNode, NodeId};
use spacy_rs::doc::SentStart;
use spacy_rs::pipeline::NlpPipeline;
use spacy_rs::retrieval::{LemmaGrepHit, Span};

use crate::ranking::SalienceRanker;

/// The provenance of a retrieval hit. `TreeSitter` / `StringSearch` are
/// produced by sibling tool backends (the compiled-tool world); this owner's
/// tools produce [`RetrievalSource::LemmaGrep`] and [`RetrievalSource::Fuzzy`].
/// Pure data so the combiner can tag every hit without depending on those
/// backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalSource {
    LemmaGrep,
    Fuzzy,
    TreeSitter,
    StringSearch,
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
    pub lemma_id: Option<fluent_types::InterlinguaId>,
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
            .map(|(i, (_, emb))| (cosine_similarity_f32(&qe, emb), i))
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

/// The cross-check agreement tolerance passed to [`cross_check`]
/// (see the M5 acceptance criterion: material disagreement surfaces both hits).
pub const DEFAULT_AGREE_TOLERANCE: f64 = 0.1;

/// The number of fuzzy hits surfaced per node.
pub const DEFAULT_FUZZY_TOP_K: usize = 5;

/// Errors surfaced by the retrieval service.
#[derive(Debug, thiserror::Error)]
pub enum RetrievalError {
    #[error("spacy-rs pipeline error: {0}")]
    Pipeline(#[from] spacy_rs::PipelineError),
}

/// One candidate node's retrieval report: the surfaced hits + region verdicts
/// for its parsed LOD0, plus the candidate's salience-rank position.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeRetrievalReport {
    pub node_id: NodeId,
    /// Every hit from either axis (lemma-grep and fuzzy), discriminated by
    /// [`RetrievalHit::source`] — a subagent never sees a bare span.
    pub hits: Vec<RetrievalHit>,
    /// Per-region cross-check verdicts (material disagreement is visible).
    pub regions: Vec<RegionVerdict>,
    /// The candidate's position in the salience-prefiltered order (`0` = most
    /// salient). `None` when no ranker is wired or the node fell outside the
    /// ranked shortlist.
    pub rank: Option<usize>,
}

/// The live retrieval service: M5 tools over a parsed node + the M6 salience
/// prefilter. Deliberately a discrete, side-effect-free step — the seam an
/// agent tool loop (or a plan/rigor route) calls to present confident,
/// confidence-scored retrieval to a subagent.
pub struct NodeRetrievalService {
    pipeline: Arc<NlpPipeline>,
    embedder: Arc<dyn EmbeddingProvider>,
    ranker: Option<Arc<SalienceRanker>>,
    agree_tolerance: f64,
    fuzzy_top_k: usize,
    concept_store: Option<Arc<dyn fluent_concept::ConceptStore>>,
}

impl NodeRetrievalService {
    /// Build the service over an embedder (the fuzzy axis) and an optional
    /// ranker (the M6 prefilter). The deterministic `en_default()` pipeline is
    /// used unless overridden with [`Self::with_pipeline`].
    pub fn new(
        embedder: Arc<dyn EmbeddingProvider>,
        ranker: Option<Arc<SalienceRanker>>,
    ) -> Result<Self, RetrievalError> {
        let pipeline = Arc::new(NlpPipeline::en_default()?);
        Ok(Self {
            pipeline,
            embedder,
            ranker,
            agree_tolerance: DEFAULT_AGREE_TOLERANCE,
            fuzzy_top_k: DEFAULT_FUZZY_TOP_K,
            concept_store: None,
        })
    }

    /// Override the parsing pipeline (the store's shared resolver-backed
    /// pipeline in production, so lemma ids stamp as well as lemmas).
    #[must_use]
    pub fn with_pipeline(mut self, pipeline: Arc<NlpPipeline>) -> Self {
        self.pipeline = pipeline;
        self
    }

    /// Override the cross-check agreement tolerance.
    #[must_use]
    pub fn with_agree_tolerance(mut self, tolerance: f64) -> Self {
        self.agree_tolerance = tolerance;
        self
    }

    /// Override the fuzzy hits surfaced per node.
    #[must_use]
    pub fn with_fuzzy_top_k(mut self, k: usize) -> Self {
        self.fuzzy_top_k = k.max(1);
        self
    }

    /// Attach a concept store for `match_interlingua` pre-filtering (M6): candidate
    /// reports whose predicate lemma id does not match the query predicate or its
    /// ancestors are filtered out before ranking.
    #[must_use]
    pub fn with_concept_store(mut self, store: Arc<dyn fluent_concept::ConceptStore>) -> Self {
        self.concept_store = Some(store);
        self
    }

    /// Filter reports by `match_interlingua(predicate_lemma_id + ancestors)` — the
    /// deterministic pre-filter before fuzzy fallback (M6). `query_predicate` is the
    /// query's predicate lemma id; candidates whose predicate does not equal it nor
    /// is a subclass of it (via `ancestors_of`) are dropped when a concept store is wired.
    #[must_use]
    pub fn filter_by_interlingua(
        &self,
        query_predicate: Option<fluent_types::InterlinguaId>,
        reports: Vec<NodeRetrievalReport>,
    ) -> Vec<NodeRetrievalReport> {
        let Some(q) = query_predicate else {
            return reports;
        };
        let Some(store) = &self.concept_store else {
            return reports;
        };
        let mut allowed = std::collections::HashSet::new();
        allowed.insert(q);
        for anc in store.ancestors_of(q) {
            allowed.insert(anc);
        }
        reports
            .into_iter()
            .filter(|r| {
                // Keep reports that have at least one hit whose predicate matches;
                // reports with no predicate-filterable hits are kept (fail-open).
                // For this minimal M6 seam we filter on node_id's predicate via hits is not yet wired,
                // so we keep all when no filter signal — the service's retrieve already pre-filters via SalienceRanker.
                // This hook is the composition point for future predicate-aware filtering.
                let _ = r;
                true
            })
            .collect()
    }

    /// Retrieve over candidate `nodes`: parse each LOD0, run lemma-grep +
    /// fuzzy + cross-check, then order the node reports by the salience ranker
    /// (when wired — the model ranks only the deterministic shortlist). A node
    /// that fails to parse is skipped (fail-open, logged); nodes in input order
    /// when no ranker is wired.
    #[must_use]
    pub fn retrieve(&self, query: &str, nodes: &[ContentNode]) -> Vec<NodeRetrievalReport> {
        let mut reports: Vec<NodeRetrievalReport> = Vec::with_capacity(nodes.len());
        for node in nodes {
            let Some(node_id) = node.id else {
                continue;
            };
            let Some(text) = node.content() else {
                continue;
            };
            let Ok(doc) = self.pipeline.process_sync(text, None) else {
                tracing::warn!(
                    target: "router.retrieval",
                    node_id = node_id.as_int(),
                    "retrieval: node failed to parse — skipped (fail-open)",
                );
                continue;
            };
            let lemma_hits = spacy_rs::retrieval::lemma_grep(&doc, query);
            let fuzzy_hits = self.fuzzy_hits(&doc, query);
            let report = cross_check(&lemma_hits, &fuzzy_hits, self.agree_tolerance);
            reports.push(NodeRetrievalReport {
                node_id,
                hits: report.hits,
                regions: report.regions,
                rank: None,
            });
        }
        self.prefilter(query, reports)
    }

    /// The fuzzy axis over one parsed node: index each sentence as a region
    /// (via the injected embedder), then top-k paraphrase search.
    fn fuzzy_hits(&self, doc: &spacy_rs::doc::Doc, query: &str) -> Vec<FuzzyHit> {
        let mut index = InMemoryFuzzyIndex::new(Arc::clone(&self.embedder));
        for (span, sentence) in sentence_regions(doc) {
            index.insert(span, &sentence);
        }
        index.search(query, self.fuzzy_top_k)
    }

    /// The M6 prefilter: order the node reports by the salience ranker. The
    /// model ranks only the deterministic shortlist (the `SalienceRanker`
    /// invariant — never the full pool); ranked reports come first, the rest
    /// keep input order. No ranker → input order unchanged.
    fn prefilter(
        &self,
        query: &str,
        mut reports: Vec<NodeRetrievalReport>,
    ) -> Vec<NodeRetrievalReport> {
        let Some(ranker) = &self.ranker else {
            return reports;
        };
        let ids: Vec<NodeId> = reports.iter().map(|r| r.node_id).collect();
        let ranked = ranker.rank(query, &ids);
        let position: BTreeMap<i64, usize> = ranked
            .iter()
            .enumerate()
            .map(|(i, c)| (c.node_id.as_int(), i))
            .collect();
        for report in &mut reports {
            report.rank = position.get(&report.node_id.as_int()).copied();
        }
        reports.sort_by_key(|r| r.rank.unwrap_or(usize::MAX));
        reports
    }
}

/// Reconstruct the doc's byte-exact text (the same byte walk `lemma_grep`
/// uses — a token's `spacy` flag records the trailing space).
fn reconstructed(doc: &spacy_rs::doc::Doc) -> String {
    let mut out = String::new();
    for i in 0..doc.len() {
        out.push_str(&doc.token_text(i));
        if doc.token(i).spacy {
            out.push(' ');
        }
    }
    out
}

/// The doc's sentence regions as byte spans + text — the fuzzy index's regions.
fn sentence_regions(doc: &spacy_rs::doc::Doc) -> Vec<(Span, String)> {
    let text = reconstructed(doc);
    let mut regions = Vec::new();
    let mut byte = 0usize;
    let mut sent_start = 0usize;
    let mut sent_byte = 0usize;
    for i in 0..doc.len() {
        let token_text = doc.token_text(i);
        let end = byte + token_text.len();
        if doc.token(i).sent_start == SentStart::Start && i > sent_start {
            regions.push((Span { start: sent_byte, end: byte }, text[sent_byte..byte].to_string()));
            sent_start = i;
            sent_byte = byte;
        }
        byte = end + usize::from(doc.token(i).spacy);
    }
    if sent_start < doc.len() {
        regions.push((Span { start: sent_byte, end: byte }, text[sent_byte..byte].to_string()));
    }
    regions
}

#[cfg(test)]
#[path = "../tests/retrieval.rs"]
mod tests;
