use super::*;
use crate::ranking::{SalienceSignal, SalienceSource};
use spacy_rs::retrieval::RetrievalSource;
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
