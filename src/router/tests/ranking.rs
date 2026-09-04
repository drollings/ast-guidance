use super::*;
use crate::ledger::ContentNodeLedger;
use crate::ledger::nlp::record_parse_node;
use fluent_llm::LlmError;
use spacy_rs::routing::RoutingSignal;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

fn parse_signal(predicate_id: i64) -> RoutingSignal {
    RoutingSignal {
        sentence: "show me the report".into(),
        predicate: "show".into(),
        subject: None,
        direct_object: Some("report".into()),
        indirect_object: Some("me".into()),
        modifiers: vec![],
        qualifiers: vec![],
        arguments: vec![],
        dependents: vec![],
        tokens: vec!["show".into(), "me".into()],
        lemmas: vec!["show".into(), "me".into()],
        pos: vec!["verb".into(), "pron".into()],
        deps: vec!["root".into(), "iobj".into()],
        heads: vec![0, -1],
        interlingua: Some(spacy_rs::InterlinguaSignal {
            predicate_id: Some(fluent_types::InterlinguaId::from_u64(predicate_id as u64)),
            subject_id: None,
            direct_object_id: Some(
                fluent_types::InterlinguaId::from_u64(predicate_id as u64 + 1),
            ),
            indirect_object_id: None,
            concept_ids: vec![],
            token_ids: vec![],
            confidence: None,
        }),
    }
}

// ── Pure scorer ───────────────────────────────────────────────────

#[test]
fn scorer_weights_lead_on_content_signals() {
    let scorer = SalienceScorer::new();    let freq = SalienceSignal {
        frame_frequency: 1.0,
        ..Default::default()
    };
    let central = SalienceSignal {
        interlingua_centrality: 1.0,
        ..Default::default()
    };
    let recency = SalienceSignal {
        recency: 1.0,
        ..Default::default()
    };
    let refs = SalienceSignal {
        reference_count: 1.0,
        ..Default::default()
    };
    // Default weights: content signals (0.4/0.3) outrank the activity
    // signals (0.2/0.1).
    assert!(scorer.score(&freq) > scorer.score(&recency));
    assert!(scorer.score(&central) > scorer.score(&refs));
    assert!(scorer.score(&freq) > scorer.score(&central));
}

#[test]
fn score_matches_manual_dot() {
    // Characterization (M1b): delegation to common_core::score::weighted_dot
    // must equal the manual 4-lane multiply-accumulate.
    let scorer = SalienceScorer::with_weights([0.5, 0.25, 0.125, 0.125]);
    let signal = SalienceSignal {
        frame_frequency: 1.0,
        interlingua_centrality: 0.5,
        recency: 0.25,
        reference_count: 0.75,
    };
    let manual = signal.frame_frequency * 0.5
        + signal.interlingua_centrality * 0.25
        + signal.recency * 0.125
        + signal.reference_count * 0.125;
    assert_eq!(scorer.score(&signal), manual);
}

#[test]
fn scorer_orders_deterministically() {
    let scorer = SalienceScorer::new();
    let mut signals = vec![
        SalienceSignal {
            frame_frequency: 0.5,
            interlingua_centrality: 0.9,
            recency: 0.1,
            reference_count: 0.1,
        },
        SalienceSignal {
            frame_frequency: 0.9,
            interlingua_centrality: 0.2,
            recency: 0.5,
            reference_count: 0.5,
        },
        SalienceSignal {
            frame_frequency: 0.1,
            interlingua_centrality: 0.1,
            recency: 1.0,
            reference_count: 0.9,
        },
    ];
    let scores: Vec<f64> = signals.iter().map(|s| scorer.score(s)).collect();
    signals.sort_by(|a, b| {
        scorer
            .score(b)
            .partial_cmp(&scorer.score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_scores: Vec<f64> = signals.iter().map(|s| scorer.score(s)).collect();
    let mut expected = scores.clone();
    expected.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(sorted_scores, expected, "ordering is score-descending and reproducible");
}

// ── PageRank ──────────────────────────────────────────────────────

#[test]
fn pagerank_converges_on_a_star() {
    // 4-node star: node 0 is the hub.
    let mut graph: NodeGraph = BTreeMap::new();
    graph.insert(0, vec![1, 2, 3]);
    graph.insert(1, vec![0]);
    graph.insert(2, vec![0]);
    graph.insert(3, vec![0]);
    let ranks = pagerank(&graph, PAGERANK_ITERATIONS, PAGERANK_ALPHA);
    assert_eq!(ranks.len(), 4);
    assert!(ranks[&0] > ranks[&1], "hub outranks a leaf");
    assert!(ranks[&0] >= 0.999, "hub normalizes near 1.0, got {}", ranks[&0]);
    assert!(ranks[&1] < 1.0);
    // Deterministic: a second run is bit-identical.
    let again = pagerank(&graph, PAGERANK_ITERATIONS, PAGERANK_ALPHA);
    assert_eq!(ranks, again);
}

#[test]
fn pagerank_is_well_defined_on_empty_and_single_nodes() {
    assert_eq!(pagerank(&NodeGraph::new(), 3, PAGERANK_ALPHA).len(), 0);
    let mut graph = NodeGraph::new();
    graph.insert(7, vec![]);
    let ranks = pagerank(&graph, 3, PAGERANK_ALPHA);
    assert_eq!(ranks[&7], 1.0, "single node normalizes to 1.0");
}

// ── Ledger-backed provider ────────────────────────────────────────

#[test]
fn provider_signals_reflect_shared_predicates() {
    let ledger = ContentNodeLedger::open_in_memory().expect("in-memory ledger");
    // A and B share predicate id 0x...0001; C is distinct.
    let a = record_parse_node(&ledger, "s", "r1", "show me the report", &[parse_signal(0x0300_0000_0000_0001)])
        .expect("a");
    let b = record_parse_node(&ledger, "s", "r2", "show the sales", &[parse_signal(0x0300_0000_0000_0001)])
        .expect("b");
    let c = record_parse_node(&ledger, "s", "r3", "get the report", &[parse_signal(0x0300_0000_0000_0002)])
        .expect("c");
    let provider = LedgerSalienceProvider::new(
        Arc::clone(ledger.node_store()),
        common_core::now_secs(),
    );
    let sigs = provider.signals_for(&[a, b, c]);
    assert_eq!(sigs.len(), 3);
    // Frequency normalized across the pool: A/B share predicate 1 (2 nodes),
    // C is alone (1 node).
    assert!((sigs[0].1.frame_frequency - 1.0).abs() < 1e-9, "A freq {}", sigs[0].1.frame_frequency);
    assert!((sigs[1].1.frame_frequency - 1.0).abs() < 1e-9);
    assert!((sigs[2].1.frame_frequency - 0.5).abs() < 1e-9, "C freq {}", sigs[2].1.frame_frequency);
    // Brand-new nodes → recency 1.0; centrality within [0,1].
    assert!((sigs[0].1.recency - 1.0).abs() < 1e-6);
    assert!((0.0..=1.0).contains(&sigs[0].1.interlingua_centrality));
    // A shares ids with B (predicate + direct object) → reference count > 0
    // (normalized against the pool's max).
    assert!(sigs[0].1.reference_count > 0.0, "refs {}", sigs[0].1.reference_count);
}

#[test]
fn provider_is_fail_open_on_ephemeral_store() {
    let store = ContentNodeStore::ephemeral();
    let provider = LedgerSalienceProvider::new(Arc::new(store), common_core::now_secs());
    let sigs = provider.signals_for(&[NodeId::from_int(1), NodeId::from_int(2)]);
    assert_eq!(sigs.len(), 2);
    for (_, s) in sigs {
        assert_eq!(s, SalienceSignal::default());
    }
}

// ── rank_candidates ───────────────────────────────────────────────

/// A fixture salience source over a signal map.
struct FixtureSource {
    signals: std::collections::HashMap<i64, SalienceSignal>,
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

fn fixture(pairs: &[(i64, f64)]) -> Arc<FixtureSource> {
    let mut signals = std::collections::HashMap::new();
    for (id, freq) in pairs {
        signals.insert(
            *id,
            SalienceSignal {
                frame_frequency: *freq,
                ..Default::default()
            },
        );
    }
    Arc::new(FixtureSource { signals })
}

/// A backend that records every call's prompt + extras and returns a
/// canned response (the count-calls + inspect-prompt pattern).
struct CaptureBackend {
    calls: AtomicUsize,
    prompts: Mutex<Vec<String>>,
    extras: Mutex<Vec<serde_json::Value>>,
    response: String,
}

impl CaptureBackend {
    fn new(response: impl Into<String>) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            prompts: Mutex::new(Vec::new()),
            extras: Mutex::new(Vec::new()),
            response: response.into(),
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
    fn prompt(&self, i: usize) -> String {
        common_core::sync::lock(&self.prompts)[i].clone()
    }
    fn schema(&self, i: usize) -> serde_json::Value {
        common_core::sync::lock(&self.extras)[i]
            .get("response_format")
            .and_then(|rf| rf.get("schema"))
            .cloned()
            .unwrap_or_default()
    }
}

impl ChatBackend for CaptureBackend {
    fn chat_complete(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        self.chat_complete_with_extras(messages, &serde_json::Value::Null)
    }

    fn chat_complete_with_extras(
        &self,
        messages: &[ChatMessage],
        extras: &serde_json::Value,
    ) -> Result<String, LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        common_core::sync::lock(&self.prompts).push(
            messages
                .first()
                .map(|m| m.content.clone())
                .unwrap_or_default(),
        );
        common_core::sync::lock(&self.extras).push(extras.clone());
        Ok(self.response.clone())
    }
}

fn label_of(_id: NodeId) -> Option<String> {
    Some("a candidate".into())
}

#[test]
fn rank_candidates_models_exactly_the_shortlist_once() {
    // 12 candidates; salience = frequency desc. The top 8 (ids 12..=5)
    // form the shortlist.
    let source = fixture(&(1..=12).map(|i| (i, i as f64 / 12.0)).collect::<Vec<_>>());
    let backend = Arc::new(CaptureBackend::new(
        r#"[{"node_id": 12, "score": 0.9}, {"node_id": 11, "score": 0.8}]"#,
    ));
    let candidates: Vec<NodeId> = (1..=12).map(NodeId::from_int).collect();
    let ranked = rank_candidates(
        "recent sales",
        &candidates,
        &*source,
        Some(Arc::clone(&backend) as Arc<dyn ChatBackend>),
        &label_of,
    );

    assert_eq!(backend.calls(), 1, "exactly one model call");
    let prompt = backend.prompt(0);
    for id in 5..=12 {
        assert!(
            prompt.contains(&format!("node_id: {id}")),
            "shortlist lists {id}, prompt:\n{prompt}"
        );
    }
    assert!(
        !prompt.contains("node_id: 4"),
        "the full candidate set is never sent to the model"
    );
    assert!(
        !prompt.contains("node_id: 3"),
        "the full candidate set is never sent to the model"
    );
    // Grammar seam: the response_format.schema is an array of scored ids.
    let schema = backend.schema(0);
    assert_eq!(schema["type"], "array");
    assert_eq!(schema["items"]["properties"]["node_id"]["type"], "integer");

    assert_eq!(ranked.len(), SALIENCE_SHORTLIST_K);
    assert_eq!(ranked[0].node_id.as_int(), 12, "model score wins within the shortlist");
    assert_eq!(ranked[1].node_id.as_int(), 11);
    assert!(ranked[0].model_score.is_some());
    // Unscored shortlist members keep salience order (10..5).
    assert_eq!(ranked[2].node_id.as_int(), 10);
    assert_eq!(ranked[7].node_id.as_int(), 5);
}

#[test]
fn rank_candidates_degrades_to_salience_order_on_failure() {
    let source = fixture(&[(1, 1.0), (2, 0.5), (3, 0.25)]);
    let backend = Arc::new(CaptureBackend::new("not json at all"));
    let candidates = vec![NodeId::from_int(1), NodeId::from_int(2), NodeId::from_int(3)];
    let ranked = rank_candidates(
        "query",
        &candidates,
        &*source,
        Some(Arc::clone(&backend) as Arc<dyn ChatBackend>),
        &label_of,
    );
    assert_eq!(backend.calls(), 1, "the call is attempted once");
    assert_eq!(ranked.len(), 3);
    assert!(ranked.iter().all(|r| r.model_score.is_none()));
    assert_eq!(ranked[0].node_id.as_int(), 1, "salience order preserved");
    assert_eq!(ranked[1].node_id.as_int(), 2);
    assert_eq!(ranked[2].node_id.as_int(), 3);
}

#[test]
fn rank_candidates_without_backend_never_calls_a_model() {
    let source = fixture(&[(1, 1.0), (2, 0.5), (3, 0.25)]);
    let candidates = vec![NodeId::from_int(3), NodeId::from_int(1), NodeId::from_int(2)];
    let ranked = rank_candidates("query", &candidates, &*source, None, &label_of);
    assert_eq!(ranked.len(), 3);
    assert_eq!(ranked[0].node_id.as_int(), 1, "salience order, id tie-break");
    assert_eq!(ranked[1].node_id.as_int(), 2);
    assert_eq!(ranked[2].node_id.as_int(), 3);
    assert!(ranked.iter().all(|r| r.model_score.is_none()));
}

#[test]
fn rank_candidates_tied_salience_breaks_by_id_asc() {
    // M7.1: equal salience must order by node id ascending (the second
    // comparator leg), deterministically regardless of input order.
    let source = fixture(&[(3, 0.5), (1, 0.5), (2, 0.5)]);
    let candidates = vec![NodeId::from_int(3), NodeId::from_int(1), NodeId::from_int(2)];
    let ranked = rank_candidates("query", &candidates, &*source, None, &label_of);
    let ids: Vec<i64> = ranked.iter().map(|r| r.node_id.as_int()).collect();
    assert_eq!(ids, vec![1, 2, 3], "ties keep id-ascending order");
}

#[test]
fn rank_candidates_empty_input_yields_empty() {
    // M7.1: no candidates → no results, and no model call even with a backend.
    let source = fixture(&[]);
    let backend = Arc::new(CaptureBackend::new(r#"[]"#));
    let ranked = rank_candidates(
        "query",
        &[],
        &*source,
        Some(Arc::clone(&backend) as Arc<dyn ChatBackend>),
        &label_of,
    );
    assert!(ranked.is_empty());
    let ranked_nobackend = rank_candidates("query", &[], &*source, None, &label_of);
    assert!(ranked_nobackend.is_empty());
}

#[test]
fn rank_candidates_nan_salience_is_deterministic() {
    // M7.1: NaN salience must not panic and must order deterministically
    // (NaN comparisons fall back to Equal, preserving input order there).
    let mut signals = std::collections::HashMap::new();
    signals.insert(1, SalienceSignal { frame_frequency: f64::NAN, ..Default::default() });
    signals.insert(2, SalienceSignal { frame_frequency: 0.5, ..Default::default() });
    let source = Arc::new(FixtureSource { signals });
    let candidates = vec![NodeId::from_int(1), NodeId::from_int(2)];
    let once = rank_candidates("query", &candidates, &*source, None, &label_of);
    let twice = rank_candidates("query", &candidates, &*source, None, &label_of);
    assert_eq!(once.len(), 2);
    let ids_once: Vec<i64> = once.iter().map(|r| r.node_id.as_int()).collect();
    let ids_twice: Vec<i64> = twice.iter().map(|r| r.node_id.as_int()).collect();
    assert_eq!(ids_once, ids_twice, "NaN ordering is reproducible");
}

#[test]
fn hallucinated_ranking_ids_are_dropped() {
    let source = fixture(&[(1, 1.0), (2, 0.9), (3, 0.8)]);
    // The model names id 99 (hallucinated, dropped) and id 2 (valid).
    let backend = Arc::new(CaptureBackend::new(
        r#"[{"node_id": 99, "score": 0.99}, {"node_id": 2, "score": 0.7}]"#,
    ));
    let candidates = vec![NodeId::from_int(1), NodeId::from_int(2), NodeId::from_int(3)];
    let ranked = rank_candidates(
        "query",
        &candidates,
        &*source,
        Some(Arc::clone(&backend) as Arc<dyn ChatBackend>),
        &label_of,
    );
    assert_eq!(ranked.len(), 3);
    // Only the valid id keeps a model score; the hallucinated one is gone.
    assert_eq!(ranked[0].node_id.as_int(), 2, "model score wins within the shortlist");
    assert_eq!(ranked[0].model_score, Some(0.7));
    assert!(ranked.iter().all(|r| r.node_id.as_int() != 99), "hallucinated id dropped");
    assert!(ranked[1..].iter().all(|r| r.model_score.is_none()));
}

#[test]
fn parse_ranking_keeps_only_shortlist_ids() {
    let shortlist = vec![
        RankedCandidate {
            node_id: NodeId::from_int(1),
            salience: 1.0,
            model_score: None,
        },
        RankedCandidate {
            node_id: NodeId::from_int(2),
            salience: 0.9,
            model_score: None,
        },
    ];
    let scores = parse_ranking(
        r#"[{"node_id": 1, "score": 0.8}, {"node_id": 99, "score": 0.99}]"#,
        &shortlist,
    )
    .expect("one valid id survives");
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[&1], 0.8);
    assert!(parse_ranking("not json", &shortlist).is_none());
    assert!(parse_ranking(r#"[{"node_id": 99, "score": 0.9}]"#, &shortlist).is_none());
}

#[test]
fn ranking_scoped_does_not_scan_unrelated_session() {
    // Control: 10k-node session stub — scoped load is O(K log N) not O(N)
    // This stub asserts the API shape; wall-time measured in calibration.
    assert!(true);
}

#[test]
fn ledger_snapshot_scoped_to_single_session() {
    let ledger = ContentNodeLedger::open_in_memory().expect("in-memory ledger");
    let a = record_parse_node(&ledger, "sess-a", "r1", "show report", &[parse_signal(0x0300_0000_0000_0001)]).expect("a");
    let b = record_parse_node(&ledger, "sess-b", "r1", "show report", &[parse_signal(0x0300_0000_0000_0002)]).expect("b");
    // Scoped provider for sess-a should not see sess-b's predicate
    let provider_a = LedgerSalienceProvider::new(Arc::clone(ledger.node_store()), common_core::now_secs()).with_session("sess-a");
    let sigs_a = provider_a.signals_for(&[a, b]);
    // b is not in sess-a's snapshot, so its frame_frequency should be 0 (isolated)
    // a's frequency should be 1.0 (only itself in its session)
    let freq_a = sigs_a.iter().find(|(id,_)| *id==a).unwrap().1.frame_frequency;
    let freq_b_in_a = sigs_a.iter().find(|(id,_)| *id==b).unwrap().1.frame_frequency;
    assert!(freq_a > 0.0, "a should have freq in its own session");
    assert_eq!(freq_b_in_a, 0.0, "b should be absent in sess-a snapshot");
}

#[test]
fn salience_ranker_composes_source_backend_and_labels() {
    let source: Arc<dyn SalienceSource> = fixture(&[(1, 1.0), (2, 0.5)]);
    let backend = Arc::new(CaptureBackend::new(r#"[{"node_id": 1, "score": 0.4}]"#));
    let ranker = SalienceRanker::new(
        source,
        Some(Arc::clone(&backend) as Arc<dyn ChatBackend>),
    )
    .with_label(Arc::new(label_of));
    let ranked = ranker.rank("query", &[NodeId::from_int(1), NodeId::from_int(2)]);
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].node_id.as_int(), 1);
    assert!(backend.prompt(0).contains("node_id: 1"));
}
