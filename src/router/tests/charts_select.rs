// Tests compare reordered HNSW/reranker scores against literal
// thresholds — deliberate strict comparisons for exact-delta checks.
#![allow(clippy::float_cmp)]
use super::*;
use crate::charts::store::{chart_from_str, ChartStore};
use fluent_db::hnsw::HnswIndexHandle;
use crate::test_stubs::{HashEmbedder, StubChatBackend};
use std::path::Path;
use tempfile::TempDir;

/// A chart with two targets: `reproduce` (no deps) and `root_cause`
/// (requires the `report` entity). Copy of the Appendix A seed shape.
fn triage_chart_json() -> String {
    r#"{
        "name": "bug_triage",
        "description": "Triage a bug report into reproduction, root cause, and fix plan",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            {
                "name": "reproduce",
                "provides": ["repro_plan"],
                "depends": [],
                "template": "reproduce {{ request }}",
                "essential": true
            },
            {
                "name": "root_cause",
                "provides": ["root_cause"],
                "depends": [
                    { "kind": "capability", "name": "repro_plan" },
                    { "kind": "entity_match", "name": "report",
                      "description": "the bug report",
                      "predicate": {
                        "fields": [
                            { "path": "title", "ty": "string", "required": true }
                        ]
                      },
                      "required": true }
                ],
                "template": "cause {{ request }}",
                "essential": true
            }
        ]
    }"#
    .to_string()
}

fn draft_chart_json() -> String {
    r#"{
        "name": "draft_doc",
        "description": "Draft a technical design document from notes",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            {
                "name": "outline",
                "provides": ["doc_outline"],
                "depends": [],
                "template": "outline {{ request }}",
                "essential": true
            }
        ]
    }"#
    .to_string()
}

fn report_entity() -> Entity {
    Entity {
        id: "issue-42".into(),
        kind: "report".into(),
        value: serde_json::json!({"title": "Segfault on startup"}),
    }
}

/// Build a store (optionally indexed) from chart JSON strings.
fn store_with(charts: &[String], index_path: Option<&Path>) -> Arc<ChartStore> {
    let handle = index_path.map(|p| HnswIndexHandle {
        name: "workflow_library".into(),
        path: p.display().to_string(),
    });
    let store = ChartStore::new(handle);
    for json in charts {
        let chart = chart_from_str(json).unwrap();
        store.upsert(chart).unwrap();
    }
    if index_path.is_some() {
        store
            .build_index(Arc::new(HashEmbedder::new(256)))
            .expect("index builds");
    }
    Arc::new(store)
}

fn selector(
    store: Arc<ChartStore>,
    client: Option<Arc<dyn ChatBackend>>,
    min_score: f64,
) -> ChartSelector {
    ChartSelector::new(
        store,
        client,
        ChartsConfig {
            dir: None,
            index_path: None,
            selector_model: None,
            max_candidates: 5,
            min_score,
            entity_context: true,
        },
    )
}

#[test]
fn deterministic_capability_hit_makes_no_llm_call() {
    let store = store_with(&[triage_chart_json()], None);
    // Empty backend: any LLM call would fail with NoResponse.
    let selector = selector(
        store.clone(),
        Some(Arc::new(StubChatBackend::new(Vec::new()))),
        0.6,
    );
    let m = selector
        .select("please bug_triage this issue", &[report_entity()])
        .expect("deterministic hit must not call the LLM");
    assert_eq!(m.chart, "bug_triage");
    assert!((m.score - 1.0).abs() < f64::EPSILON);
    assert_eq!(m.fit, ChartFit::Exact);
}

#[test]
fn deterministic_hit_names_provides_asset() {
    let store = store_with(&[triage_chart_json(), draft_chart_json()], None);
    let selector = selector(store, Some(Arc::new(StubChatBackend::new(Vec::new()))), 0.6);
    // `repro_plan` is a provides asset of bug_triage.
    let m = selector
        .select("produce the repro_plan for this crash", &[report_entity()])
        .expect("provides-asset hit");
    assert_eq!(m.chart, "bug_triage");
}

#[test]
fn hnsw_top_k_returns_seeded_chart_for_near_duplicate() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    let store = store_with(
        &[triage_chart_json(), draft_chart_json()],
        Some(&index_path),
    );
    assert!(store.is_index_built());
    let selector = selector(store, None, 0.0);
    let m = selector
        .select(
            "Triage a bug report into reproduction, root cause, and fix plan",
            &[report_entity()],
        )
        .expect("hnsw retrieval");
    assert_eq!(
        m.chart, "bug_triage",
        "near-duplicate query retrieves the chart"
    );
    assert!(
        m.score >= 0.9,
        "near-duplicate query should score highly, got {}",
        m.score
    );
}

#[test]
fn min_score_filters_below_threshold_candidates() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    let store = store_with(
        &[triage_chart_json(), draft_chart_json()],
        Some(&index_path),
    );
    // Unrelated request → no candidate clears min_score 0.6.
    let strict = selector(store.clone(), None, 0.6);
    let m = strict
        .select("how do I cook pasta for dinner", &[])
        .expect("selection");
    assert_eq!(m.fit, ChartFit::Mismatch);

    // Same request with a permissive threshold → HNSW still has a top hit.
    let lax = selector(store, None, 0.0);
    let m = lax
        .select("how do I cook pasta for dinner", &[])
        .expect("selection");
    assert_ne!(m.fit, ChartFit::Mismatch);
}

#[test]
fn adjudicator_exact_for_clean_fit() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    let store = store_with(&[triage_chart_json()], Some(&index_path));
    let adjudicator =
        StubChatBackend::always(r#"{"chart": "bug_triage", "fit": "exact", "gaps": []}"#);
    let selector = selector(store.clone(), Some(Arc::new(adjudicator)), 0.0);
    let m = selector
        .select(
            "Triage a bug report into reproduction, root cause, and fix plan",
            &[report_entity()],
        )
        .expect("adjudicated selection");
    assert_eq!(m.chart, "bug_triage");
    assert_eq!(m.fit, ChartFit::Exact);
}

#[test]
fn adjudicator_partial_with_gaps_when_unbound() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    let store = store_with(&[triage_chart_json()], Some(&index_path));
    // The LLM picks the chart; binding derives the fit and the gaps.
    let adjudicator = StubChatBackend::always(
        r#"{"chart": "bug_triage", "fit": "partial", "gaps": ["report"]}"#,
    );
    let selector = selector(store, Some(Arc::new(adjudicator)), 0.0);
    let m = selector
        .select(
            "Triage a bug report into reproduction, root cause, and fix plan",
            &[], // no report entity → root_cause is unbound
        )
        .expect("adjudicated selection");
    assert_eq!(m.chart, "bug_triage");
    match m.fit {
        ChartFit::Partial { gaps } => {
            assert!(
                gaps.iter().any(|g| g == "report"),
                "expected 'report' in gaps, got {gaps:?}"
            );
        }
        other => panic!("expected Partial, got {other:?}"),
    }
}

#[test]
fn entity_only_capability_dep_classifies_partial_not_exact() {
    // A chart whose capability dep has no in-graph provider and no
    // matching entity classifies `Partial { gaps }` (drives the
    // interview) instead of `Exact`-then-`ChartError::Compile`.
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    // Same seed shape, but root_cause depends on a capability nothing
    // provides in-graph (`external_data`) in addition to the report.
    let gapped = r#"{
        "name": "bug_triage",
        "description": "Triage a bug report into reproduction, root cause, and fix plan",
        "schema_version": 1,
        "author_model": "human",
        "targets": [
            {
                "name": "reproduce",
                "provides": ["repro_plan"],
                "depends": [],
                "template": "reproduce {{ request }}",
                "essential": true
            },
            {
                "name": "root_cause",
                "provides": ["root_cause"],
                "depends": [
                    { "kind": "capability", "name": "external_data" },
                    { "kind": "entity_match", "name": "report",
                      "description": "the bug report",
                      "predicate": {
                        "fields": [
                            { "path": "title", "ty": "string", "required": true }
                        ]
                      },
                      "required": true }
                ],
                "template": "cause {{ request }}",
                "essential": true
            }
        ]
    }"#;
    let store = store_with(&[gapped.to_string()], Some(&index_path));
    let selector = selector(store, None, 0.0);
    // No entities: neither `external_data` (no provider) nor `report`
    // (no matching entity) is bound.
    let m = selector
        .select(
            "Triage a bug report into reproduction, root cause, and fix plan",
            &[],
        )
        .expect("selection");
    assert_eq!(m.chart, "bug_triage");
    match m.fit {
        ChartFit::Partial { gaps } => {
            assert!(
                gaps.iter().any(|g| g == "external_data"),
                "expected 'external_data' in gaps, got {gaps:?}"
            );
        }
        other => panic!("expected Partial, got {other:?}"),
    }
}

#[test]
fn adjudicator_mismatch_when_llm_rejects_candidates() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    let store = store_with(&[triage_chart_json()], Some(&index_path));
    let adjudicator = StubChatBackend::always(r#"{"chart": null, "fit": "mismatch"}"#);
    let selector = selector(store, Some(Arc::new(adjudicator)), 0.0);
    let m = selector
        .select("Triage a bug report", &[])
        .expect("adjudicated selection");
    assert_eq!(m.fit, ChartFit::Mismatch);
}

#[test]
fn adjudicator_hallucinated_chart_is_rejected() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    let store = store_with(&[triage_chart_json()], Some(&index_path));
    let adjudicator =
        StubChatBackend::always(r#"{"chart": "not_a_real_chart", "fit": "exact"}"#);
    let selector = selector(store, Some(Arc::new(adjudicator)), 0.0);
    let m = selector
        .select("Triage a bug report", &[])
        .expect("adjudicated selection");
    assert_eq!(m.fit, ChartFit::Mismatch);
}

#[test]
fn parse_adjudicator_output_tolerates_fences() {
    let out = parse_adjudicator_output(
        "```json\n{\"chart\": \"bug_triage\", \"fit\": \"exact\", \"gaps\": []}\n```",
    )
    .unwrap();
    assert_eq!(out.chart.as_deref(), Some("bug_triage"));
    assert_eq!(out.fit, AdjudicatorFit::Exact);
    assert!(out.gaps.is_empty());
}

#[test]
fn parse_adjudicator_output_missing_fit_infers_from_chart() {
    let out = parse_adjudicator_output(r#"{"chart": "bug_triage"}"#).unwrap();
    assert_eq!(out.fit, AdjudicatorFit::Exact);
    let out = parse_adjudicator_output(r#"{"chart": null}"#).unwrap();
    assert_eq!(out.fit, AdjudicatorFit::Mismatch);
}

// ── Step 2.5: reranker ────────────────────────────────────────────

#[test]
fn parse_rerank_output_accepts_array_and_ranking_object() {
    assert_eq!(
        parse_rerank_output(r#"["draft_doc", "bug_triage"]"#).unwrap(),
        vec!["draft_doc".to_string(), "bug_triage".to_string()]
    );
    assert_eq!(
        parse_rerank_output(r#"{"ranking": ["bug_triage"]}"#).unwrap(),
        vec!["bug_triage".to_string()]
    );
    // Fences tolerated; the noise before the JSON array is dropped.
    let fenced = "Sure!\n```json\n[\"draft_doc\", \"bug_triage\"]\n```";
    assert_eq!(
        parse_rerank_output(fenced).unwrap(),
        vec!["draft_doc".to_string(), "bug_triage".to_string()]
    );
}

#[test]
fn parse_rerank_output_rejects_garbage() {
    assert!(parse_rerank_output("not json at all").is_none());
    assert!(parse_rerank_output(r#"{"chart": "bug_triage"}"#).is_none());
    assert!(
        parse_rerank_output("[]").is_none(),
        "empty ranking is unusable"
    );
}

#[test]
fn rerank_reorders_candidates_and_preserves_scores() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    let store = store_with(
        &[triage_chart_json(), draft_chart_json()],
        Some(&index_path),
    );
    // HNSW order puts bug_triage first; the reranker prefers draft_doc.
    let candidates = vec![
        ("bug_triage".to_string(), 0.9),
        ("draft_doc".to_string(), 0.8),
    ];
    let reranker = StubChatBackend::always(r#"["draft_doc", "bug_triage"]"#);
    let selector = selector(store, None, 0.0).with_reranker(Arc::new(reranker));
    let reordered = selector.rerank("Draft a design doc", candidates.clone());
    assert_eq!(reordered[0].0, "draft_doc", "reranker order wins");
    assert_eq!(reordered[1].0, "bug_triage");
    assert_eq!(reordered[0].1, 0.8, "original HNSW score preserved");
    assert_eq!(reordered[1].1, 0.9);
    assert_eq!(reordered.len(), candidates.len());
}

#[test]
fn rerank_degrades_to_hnsw_order_on_failure() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("workflow_library.sqlite");
    let store = store_with(
        &[triage_chart_json(), draft_chart_json()],
        Some(&index_path),
    );
    let candidates = vec![
        ("bug_triage".to_string(), 0.9),
        ("draft_doc".to_string(), 0.8),
    ];
    // NoResponse backend → rerank call fails → HNSW order preserved.
    let sel = selector(store.clone(), None, 0.0)
        .with_reranker(Arc::new(StubChatBackend::new(Vec::new())));
    let reordered = sel.rerank("Draft a design doc", candidates.clone());
    assert_eq!(reordered, candidates, "failure must not reorder");

    // Hallucinated chart names are dropped; unnamed candidates re-appended.
    let reranker = StubChatBackend::always(r#"["not_a_real_chart"]"#);
    let sel = selector(store, None, 0.0).with_reranker(Arc::new(reranker));
    let reordered = sel.rerank("Draft a design doc", candidates.clone());
    assert_eq!(
        reordered, candidates,
        "invalid names fall back to HNSW order"
    );
}

#[test]
fn rerank_missing_backend_keeps_candidates_unchanged() {
    let store = store_with(&[triage_chart_json()], None);
    let sel = selector(store, None, 0.0);
    let candidates = vec![("bug_triage".to_string(), 0.9)];
    assert_eq!(sel.rerank("anything", candidates.clone()), candidates);
}

// ── Ambiguity adjudication ───────────────────────────────────────

/// Two `report` entities both matching the bug_triage `report` predicate
/// → the dep binds ambiguously; adjudication must resolve it.
fn two_report_entities() -> Vec<Entity> {
    vec![
        Entity {
            id: "issue-42".into(),
            kind: "report".into(),
            value: serde_json::json!({"title": "Segfault on startup"}),
        },
        Entity {
            id: "issue-43".into(),
            kind: "report".into(),
            value: serde_json::json!({"title": "Memory leak on shutdown"}),
        },
    ]
}

#[test]
fn ambiguous_dep_resolved_by_llm_adjudicator() {
    let store = store_with(&[triage_chart_json()], None);
    // Deterministic hit → no step-3 adjudicator call; the only LLM call
    // is the ambiguity pick for `report`.
    let selector = selector(
        store,
        Some(Arc::new(StubChatBackend::always(
            r#"{"entity_id": "issue-43"}"#,
        ))),
        0.6,
    );
    let m = selector
        .select("bug_triage this issue", &two_report_entities())
        .expect("selection");
    assert_eq!(m.fit, ChartFit::Exact, "ambiguity must not force a gap");
    let bindings = m.bindings.as_ref().expect("bindings present");
    assert!(
        bindings.ambiguous.is_empty(),
        "ambiguous deps must be adjudicated away: {:?}",
        bindings.ambiguous
    );
    let picked = &bindings.entity_map["report"][0];
    assert_eq!(picked.id, "issue-43", "LLM pick wins");
    assert!(
        bindings.satisfied.contains("entity:report:issue-43"),
        "picked entity is satisfied"
    );
}

#[test]
fn ambiguous_dep_falls_back_to_deterministic_tie_break() {
    let store = store_with(&[triage_chart_json()], None);
    // Unparseable LLM output → deterministic tie-break (min id).
    let selector = selector(
        store,
        Some(Arc::new(StubChatBackend::always("not json at all"))),
        0.6,
    );
    let m = selector
        .select("bug_triage this issue", &two_report_entities())
        .expect("selection");
    assert_eq!(m.fit, ChartFit::Exact);
    let bindings = m.bindings.as_ref().expect("bindings present");
    assert!(bindings.ambiguous.is_empty());
    assert_eq!(
        bindings.entity_map["report"][0].id, "issue-42",
        "lexicographic tie-break picks the smaller id"
    );
}

#[test]
fn ambiguous_dep_resolved_without_llm_backend() {
    let store = store_with(&[triage_chart_json()], None);
    // No backend at all → deterministic tie-break, no LLM call.
    let selector = selector(store, None, 0.6);
    let m = selector
        .select("bug_triage this issue", &two_report_entities())
        .expect("selection");
    assert_eq!(m.fit, ChartFit::Exact);
    let bindings = m.bindings.as_ref().expect("bindings present");
    assert!(bindings.ambiguous.is_empty());
    assert_eq!(bindings.entity_map["report"][0].id, "issue-42");
}

#[test]
fn ambiguous_dep_llm_named_non_candidate_falls_back() {
    let store = store_with(&[triage_chart_json()], None);
    let selector = selector(
        store,
        Some(Arc::new(StubChatBackend::always(
            r#"{"entity_id": "hallucinated-99"}"#,
        ))),
        0.6,
    );
    let m = selector
        .select("bug_triage this issue", &two_report_entities())
        .expect("selection");
    let bindings = m.bindings.as_ref().expect("bindings present");
    assert_eq!(
        bindings.entity_map["report"][0].id, "issue-42",
        "invalid LLM id falls back to the deterministic pick"
    );
}

#[test]
fn parse_ambiguity_output_tolerates_fences_and_noise() {
    let id = parse_ambiguity_output("```json\n{\"entity_id\": \"issue-7\"}\n```").unwrap();
    assert_eq!(id, "issue-7");
    let id = parse_ambiguity_output(
        "considering candidates... {\"entity_id\": \"issue-8\"} hope that helps",
    )
    .unwrap();
    assert_eq!(id, "issue-8");
    assert!(parse_ambiguity_output(r#"{"entity_id": ""}"#).is_none());
    assert!(parse_ambiguity_output("no json").is_none());
}
