use super::*;
use crate::ranking::{SalienceSignal, SalienceSource};
use std::collections::HashMap;

/// A deterministic synonym-aware embedder (the M5 paraphrase axis): function
/// words embed by identity, and the synonym table maps paraphrase equivalents
/// onto the same dimension — exactly what lemma-grep cannot cover.
struct SynonymProvider;

impl EmbeddingProvider for SynonymProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let mut v = vec![0.0f32; 16];
        for tok in text.split_whitespace() {
            let dim = match tok.to_lowercase().as_str() {
                "show" | "display" | "get" | "list" => 0usize,
                "me" => 1,
                "the" => 2,
                "report" | "sales" => 3,
                other => (spacy_rs::hash::hash_utf8(other) % 16) as usize,
            };
            v[dim] += 1.0;
        }
        Some(v)
    }
}

/// A fixture salience source (M6): per-node frequency signal.
struct FixtureSource {
    signals: HashMap<i64, SalienceSignal>,
}

impl SalienceSource for FixtureSource {
    fn signals_for(&self, candidates: &[NodeId]) -> Vec<(NodeId, SalienceSignal)> {
        candidates
            .iter()
            .map(|id| {
                (
                    *id,
                    self.signals
                        .get(&id.as_int())
                        .copied()
                        .unwrap_or_default(),
                )
            })
            .collect()
    }
}

fn ranker_for(signals: &[(i64, f64)]) -> Arc<SalienceRanker> {
    let mut map = HashMap::new();
    for (id, freq) in signals {
        map.insert(
            *id,
            SalienceSignal {
                frame_frequency: *freq,
                ..Default::default()
            },
        );
    }
    Arc::new(SalienceRanker::new(
        Arc::new(FixtureSource { signals: map }),
        None,
    ))
}

fn node(id: i64, text: &str) -> ContentNode {
    ContentNode {
        id: Some(NodeId::from_int(id)),
        lod: vec![text.to_string()],
        ..Default::default()
    }
}

fn service() -> NodeRetrievalService {
    NodeRetrievalService::new(Arc::new(SynonymProvider), None).expect("service")
}

#[test]
fn lemma_grep_hits_carry_confidence_and_lemma() {
    let svc = service();
    let report = svc.retrieve("show", &[node(1, "Show me the report")]);
    assert_eq!(report.len(), 1);
    assert_eq!(report[0].node_id, NodeId::from_int(1));
    let lemma: Vec<&RetrievalHit> = report[0]
        .hits
        .iter()
        .filter(|h| h.source == RetrievalSource::LemmaGrep)
        .collect();
    assert_eq!(lemma.len(), 1, "one lemma-grep hit for 'show'");
    assert_eq!(
        lemma[0].lemma.as_deref().map(str::to_lowercase).as_deref(),
        Some("show"),
        "lemma matches the query case-insensitively"
    );
    assert!(
        lemma[0].parse_confidence.is_some(),
        "confidence is mandatory on a lemma hit"
    );
    assert!(lemma[0].span.len() > 0, "a real byte span");
}

#[test]
fn fuzzy_covers_the_paraphrase_gap_in_the_live_service() {
    let svc = service();
    // 'show' never appears in the text; only the fuzzy axis finds it.
    let report = svc.retrieve("display", &[node(1, "show me the report")]);
    assert_eq!(report.len(), 1);
    assert!(
        report[0]
            .hits
            .iter()
            .any(|h| h.source == RetrievalSource::Fuzzy && h.fuzzy_score.is_some()),
        "paraphrase hit surfaced"
    );
}

#[test]
fn cross_check_surfaces_both_axes_on_disagreement() {
    // Low parse confidence on the deterministic hit + high fuzzy similarity
    // on the same region → both axes surfaced (never deduped).
    let report = service().retrieve("show", &[node(1, "show me the report")]);
    let sources: Vec<RetrievalSource> = report[0].hits.iter().map(|h| h.source).collect();
    assert!(sources.contains(&RetrievalSource::LemmaGrep));
    assert!(sources.contains(&RetrievalSource::Fuzzy));
    assert!(
        report[0].regions.iter().any(|r| r.lemma_confidence.is_some() && r.fuzzy_score.is_some()),
        "a region covered by both axes is verdict-annotated"
    );
}

#[test]
fn ranker_prefilters_reports_in_salience_order() {
    let svc = NodeRetrievalService::new(
        Arc::new(SynonymProvider),
        Some(ranker_for(&[(1, 0.3), (2, 1.0), (3, 0.6)])),
    )
    .expect("service");
    // Candidate order is scrambled; the ranker must reorder by salience.
    let reports = svc.retrieve(
        "report",
        &[node(1, "show me the report"), node(2, "the sales report"), node(3, "a report")],
    );
    assert_eq!(reports.len(), 3);
    assert_eq!(reports[0].node_id, NodeId::from_int(2), "highest salience first");
    assert_eq!(reports[0].rank, Some(0));
    assert_eq!(reports[1].node_id, NodeId::from_int(3));
    assert_eq!(reports[2].node_id, NodeId::from_int(1));
}

#[test]
fn without_ranker_reports_keep_input_order() {
    let svc = service();
    let reports = svc.retrieve(
        "report",
        &[node(1, "show me the report"), node(2, "the sales report")],
    );
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0].node_id, NodeId::from_int(1), "input order preserved");
    assert!(reports.iter().all(|r| r.rank.is_none()));
}

#[test]
fn nodes_without_content_are_skipped_fail_open() {
    let svc = service();
    // A node with no LOD0 has no content → skipped; the empty-string node
    // parses to an empty doc and contributes an empty report (never an error).
    let no_content = ContentNode {
        id: Some(NodeId::from_int(2)),
        lod: vec![],
        ..Default::default()
    };
    let reports = svc.retrieve("show", &[node(1, "show me the report"), no_content]);
    assert_eq!(reports.len(), 1, "no-LOD0 node skipped, no error");
    assert_eq!(reports[0].node_id, NodeId::from_int(1));
}

#[test]
fn sentence_regions_split_on_sentence_boundaries() {
    let svc = service();
    // Two sentences → two fuzzy regions.
    let doc = svc
        .pipeline
        .process_sync("Show the report. Get the sales.", None)
        .expect("parse");
    let regions = sentence_regions(&doc);
    assert_eq!(regions.len(), 2);
    assert!(regions[0].0.start < regions[1].0.start);
    assert!(regions[0].1.contains("Show the report"));
    assert!(regions[1].1.contains("Get the sales"));
}

// ── Moved tool surface (M4): spacy-rs produces the inputs, this module ──
// ── consumes them. spacy-rs keeps only `lemma_grep`; the fuzzy axis,   ──
// ── hit tagging, and the combiner live here.                           ──

/// Fixed-embedding provider: query embeds to `q`, regions to listed vectors.
struct FixedProvider {
    q: Option<Vec<f32>>,
    region_embs: Vec<Vec<f32>>,
}
impl EmbeddingProvider for FixedProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        if text == "QUERY" {
            return self.q.clone();
        }
        text.strip_prefix("REGION")
            .and_then(|n| n.parse::<usize>().ok())
            .and_then(|i| self.region_embs.get(i).cloned())
    }
}

#[test]
fn moved_fuzzy_index_covers_the_paraphrase_gap() {
    // Index-level paraphrase: "display the report" finds the "show me the
    // report" region first with high similarity (the gap lemma-grep cannot).
    let mut idx = InMemoryFuzzyIndex::new(Arc::new(SynonymProvider));
    idx.insert(Span { start: 0, end: 18 }, "show me the report");
    idx.insert(Span { start: 20, end: 40 }, "delete the old file");
    let hits = idx.search("display the report", 2);
    assert!(!hits.is_empty(), "paraphrase-matched region found");
    assert_eq!(hits[0].span, Span { start: 0, end: 18 }, "paraphrase region ranks first");
    assert!(hits[0].score > 0.8, "high paraphrase similarity");
    let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "sorted desc");
}

#[test]
fn moved_fuzzy_search_filters_nonpositive() {
    // Strict-positive filter + take(k) order.
    // q=(1,0); regions: r0=(1,0) sim 1.0, r1=(-1,0) sim -1.0, r2=(0,0) sim 0.0.
    let provider = Arc::new(FixedProvider {
        q: Some(vec![1.0, 0.0]),
        region_embs: vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 0.0]],
    });
    let mut idx = InMemoryFuzzyIndex::new(provider);
    for (i, start) in [0usize, 10, 20].iter().enumerate() {
        idx.insert(
            Span { start: *start, end: *start + 5 },
            &format!("REGION{i}"),
        );
    }
    // Negative and zero sims excluded; only the positive region survives.
    let hits = idx.search("QUERY", 10);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].span, Span { start: 0, end: 5 });
    // k == 0 → empty.
    assert!(idx.search("QUERY", 0).is_empty());
    // k larger than region count → capped at the positive survivors.
    assert_eq!(idx.search("QUERY", 100).len(), 1);
    // Unembeddable query → empty.
    let dead = InMemoryFuzzyIndex::new(Arc::new(FixedProvider {
        q: None,
        region_embs: vec![],
    }));
    assert!(dead.search("QUERY", 5).is_empty());
}

#[test]
fn moved_combiner_consumes_spacy_lemma_hits() {
    // Seam round-trip (M4): spacy-rs produces the lemma hits via its kept
    // `lemma_grep` helper; the moved combiner consumes them alongside the
    // moved fuzzy axis. Low parse confidence + high paraphrase similarity on
    // the same region → material disagreement surfaced, never collapsed.
    let svc = service();
    let doc = svc
        .pipeline
        .process_sync("Show me the report", None)
        .expect("parse");
    let lemma_hits = spacy_rs::retrieval::lemma_grep(&doc, "show");
    assert_eq!(lemma_hits.len(), 1, "spacy produces one lemma hit");
    let mut index = InMemoryFuzzyIndex::new(Arc::new(SynonymProvider));
    for (span, text) in sentence_regions(&doc) {
        index.insert(span, &text);
    }
    let fuzzy_hits = index.search("display the report", 2);
    assert!(!fuzzy_hits.is_empty(), "fuzzy axis finds the paraphrase");
    let report = cross_check(&lemma_hits, &fuzzy_hits, DEFAULT_AGREE_TOLERANCE);
    let sources: Vec<RetrievalSource> = report.hits.iter().map(|h| h.source).collect();
    assert!(sources.contains(&RetrievalSource::LemmaGrep));
    assert!(sources.contains(&RetrievalSource::Fuzzy));
    assert!(
        report.regions.iter().any(|r| r.disagreed),
        "material confidence difference is surfaced, not collapsed"
    );
}

#[test]
fn moved_cross_check_lemma_only_regions_have_no_conflict() {
    let svc = service();
    let doc = svc
        .pipeline
        .process_sync("Show me the report", None)
        .expect("parse");
    let lemma_hits = spacy_rs::retrieval::lemma_grep(&doc, "show");
    let report = cross_check(&lemma_hits, &[], DEFAULT_AGREE_TOLERANCE);
    assert_eq!(report.hits.len(), 1);
    let v = &report.regions[0];
    assert_eq!(v.fuzzy_score, None);
    assert!(!v.disagreed);
}
