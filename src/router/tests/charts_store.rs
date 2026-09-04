use super::*;
use tempfile::TempDir;

fn seed_chart(name: &str, provides: &[&str]) -> String {
    let provides: Vec<String> = provides.iter().map(|p| format!("\"{p}\"")).collect();
    format!(
        r#"{{
            "name": "{name}",
            "description": "seed chart {name}",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                {{
                    "name": "target_a",
                    "provides": [{}],
                    "template": "do {{{{ request }}}}",
                    "essential": true
                }}
            ]
        }}"#,
        provides.join(", ")
    )
}

#[test]
fn load_dir_with_seeded_tempdir() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("alpha.json"),
        seed_chart("alpha", &["a_out"]),
    )
    .unwrap();
    std::fs::write(dir.path().join("beta.json"), seed_chart("beta", &["b_out"])).unwrap();

    let store = ChartStore::new(None);
    store.load_dir(dir.path()).unwrap();
    assert_eq!(store.len(), 2);
    assert!(store.get("alpha").is_some());
    assert!(store.get("beta").is_some());
}

#[test]
fn empty_dir_yields_empty_store() {
    let dir = TempDir::new().unwrap();
    let store = ChartStore::new(None);
    store.load_dir(dir.path()).unwrap();
    assert!(store.is_empty());
    assert!(store.list().is_empty());
}

#[test]
fn missing_dir_yields_empty_store() {
    let missing = std::path::Path::new("/nonexistent/charts/dir");
    let store = ChartStore::new(None);
    store.load_dir(missing).unwrap();
    assert!(store.is_empty());
}

#[test]
fn invalid_file_is_hard_error() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("broken.json"), "not json at all").unwrap();
    let store = ChartStore::new(None);
    let err = store.load_dir(dir.path()).unwrap_err();
    assert!(matches!(err, ChartError::Parse { .. }));
}

#[test]
fn invalid_chart_fails_validation_at_load() {
    let dir = TempDir::new().unwrap();
    // Missing schema_version + a target with no template.
    std::fs::write(
        dir.path().join("bad.json"),
        r#"{
            "name": "bad",
            "description": "bad",
            "schema_version": 1,
            "author_model": "human",
            "targets": [
                { "name": "t", "provides": ["x"], "template": "" }
            ]
        }"#,
    )
    .unwrap();
    let store = ChartStore::new(None);
    let err = store.load_dir(dir.path()).unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}

#[test]
fn upsert_inserts_and_replaces() {
    let store = ChartStore::new(None);
    let chart: ChartDef = chart_from_str(&seed_chart("alpha", &["a_out"])).unwrap();
    store.upsert(chart.clone()).unwrap();
    assert_eq!(store.len(), 1);

    let mut replaced = chart.clone();
    replaced.description = "updated".into();
    store.upsert(replaced.clone()).unwrap();
    assert_eq!(store.len(), 1);
    assert_eq!(store.get("alpha").unwrap().description, "updated");
}

#[test]
fn non_json_files_are_ignored() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("chart.json"),
        seed_chart("alpha", &["a_out"]),
    )
    .unwrap();
    std::fs::write(dir.path().join("README.md"), "not a chart").unwrap();
    let store = ChartStore::new(None);
    store.load_dir(dir.path()).unwrap();
    assert_eq!(store.len(), 1);
}

#[test]
fn golden_loads_real_seed_dir() {
    // Load the real env/workflows/charts seed directory (Appendix A).
    let seed_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../env/workflows/charts");
    let store = ChartStore::new(None);
    store.load_dir(&seed_dir).expect("seed dir loads");
    assert_eq!(store.len(), 2, "expected exactly 2 seed charts");
    let mut names = store.list();
    names.sort_unstable();
    assert_eq!(names, vec!["bug_triage", "draft_doc"]);
}

// ── Idempotent upsert + draft gate ─────────────────────────────

fn indexed_store() -> (ChartStore, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().unwrap();
    let handle = fluent_db::hnsw::HnswIndexHandle {
        name: "workflow_library".into(),
        path: tmp
            .path()
            .join("workflow_library.sqlite")
            .display()
            .to_string(),
    };
    let store = ChartStore::new(Some(handle));
    let chart = chart_from_str(&seed_chart("alpha", &["a_out"])).unwrap();
    store.upsert(chart).unwrap();
    store
        .build_index(Arc::new(crate::test_stubs::HashEmbedder::new(256)))
        .expect("index builds");
    (store, tmp)
}

#[test]
fn upsert_idempotent_inserts_unrelated_chart_as_draft() {
    let (store, _tmp) = indexed_store();
    let new_chart = chart_from_str(&seed_chart("omega", &["o_out"])).unwrap();
    let outcome = store
        .upsert_idempotent(new_chart, CHART_SUBSUME_THRESHOLD)
        .unwrap();
    assert_eq!(outcome, UpsertOutcome::Inserted);
    // The auto-extracted chart is a draft: present but not selectable.
    assert!(store.is_draft("omega"));
    assert!(
        !store.charts_sorted().iter().any(|c| c.name == "omega"),
        "drafts must not be selectable until rubric-validated"
    );
}

#[test]
fn upsert_idempotent_subsumes_near_neighbor() {
    let (store, _tmp) = indexed_store();
    // A near-duplicate of `alpha` (same description/target assets) must be
    // folded into it, not stored twice. Threshold is deliberately below
    // the crude HashEmbedder's measured near-duplicate cosine (~0.61) —
    // the production `CHART_SUBSUME_THRESHOLD` targets real embeddings.
    let mut dup = chart_from_str(&seed_chart("alpha_copy", &["a_out"])).unwrap();
    dup.description = "seed chart alpha".into(); // identical doc text
    let outcome = store.upsert_idempotent(dup, 0.5).unwrap();
    match outcome {
        UpsertOutcome::Subsumed { by } => assert_eq!(by, "alpha"),
        other @ UpsertOutcome::Inserted => panic!("expected subsume, got {other:?}"),
    }
    assert_eq!(store.len(), 1, "near-neighbor dedup must not duplicate");
    assert!(store.get("alpha").is_some());
    assert!(store.is_draft("alpha"), "subsumed chart is a draft");
}

#[test]
fn upsert_idempotent_subsumed_name_keeps_library_selectable() {
    let (store, _tmp) = indexed_store();
    let mut dup = chart_from_str(&seed_chart("alpha_copy", &["a_out"])).unwrap();
    dup.description = "seed chart alpha".into();
    store.upsert_idempotent(dup, 0.5).unwrap();
    // Even as a draft, the original human chart vanished (replaced).
    // After a rubric-validated run it becomes selectable again.
    store.record_rubric_result("alpha", true);
    assert!(store.charts_sorted().iter().any(|c| c.name == "alpha"));
}

// ── Staleness / demotion policy ────────────────────────────────

#[test]
fn record_rubric_result_demotes_after_stale_fails() {
    let (store, _tmp) = indexed_store();
    for i in 0..crate::charts::CHART_STALE_FAILS {
        let demoted = store.record_rubric_result("alpha", false);
        if i + 1 < crate::charts::CHART_STALE_FAILS {
            assert!(demoted.is_none(), "not yet demoted");
            assert!(!store.is_demoted("alpha"));
        } else {
            assert_eq!(
                demoted.as_deref(),
                Some("alpha"),
                "crossing the threshold demotes the chart"
            );
        }
    }
    assert!(store.is_demoted("alpha"));
    assert_eq!(store.demoted_charts(), vec!["alpha".to_string()]);
    assert!(
        !store.charts_sorted().iter().any(|c| c.name == "alpha"),
        "demoted charts are no longer selected"
    );
}

#[test]
fn record_rubric_result_resets_streak_on_success() {
    let (store, _tmp) = indexed_store();
    store.record_rubric_result("alpha", false);
    store.record_rubric_result("alpha", false);
    // A passing run resets the streak before it crosses the threshold.
    store.record_rubric_result("alpha", true);
    assert!(!store.is_demoted("alpha"));
    assert_eq!(store.health("alpha").unwrap().stale_failures, 0);
}

#[test]
fn record_rubric_result_promotes_draft_on_pass() {
    let (store, _tmp) = indexed_store();
    let new_chart = chart_from_str(&seed_chart("omega", &["o_out"])).unwrap();
    store
        .upsert_idempotent(new_chart, CHART_SUBSUME_THRESHOLD)
        .unwrap();
    assert!(store.is_draft("omega"));
    // One rubric-validated run promotes the draft to selectable.
    store.record_rubric_result("omega", true);
    assert!(!store.is_draft("omega"));
    assert!(store.charts_sorted().iter().any(|c| c.name == "omega"));
}

#[test]
fn demoted_chart_is_also_absent_from_hnsw_search() {
    let (store, _tmp) = indexed_store();
    store.record_rubric_result("alpha", false);
    store.record_rubric_result("alpha", false);
    store.record_rubric_result("alpha", false);
    assert!(store.is_demoted("alpha"));
    let hits = store.search("alpha", 5).unwrap();
    assert!(
        hits.iter().all(|(n, _)| n != "alpha"),
        "demoted chart must not surface via HNSW retrieval"
    );
}

#[test]
fn search_edge_cases_unbuilt_and_zero_k() {
    // M5.1: unbuilt index yields no candidates; k=0 yields none.
    let store = ChartStore::new(None);
    assert!(store.search("anything", 5).unwrap().is_empty(), "unbuilt");
    let (built, _tmp) = indexed_store();
    assert!(built.search("alpha", 0).unwrap().is_empty(), "k=0");
}
