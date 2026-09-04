use super::*;
use crate::charts::store::ChartStore;

fn transcript() -> ChartAuditTranscript {
    ChartAuditTranscript {
        query: "Draft a release plan for the v2 API".into(),
        author_model: "claude-4".into(),
        steps: vec![
            ChartAuditStep {
                id: "plan".into(),
                purpose: "outline the release steps".into(),
                prompt: "Draft a release plan for the v2 API: list phases and owners.".into(),
                response: "Phase 1: ...".into(),
                depends_on: vec![],
                provides: vec!["release_plan".into()],
            },
            ChartAuditStep {
                id: "verify".into(),
                purpose: "check the plan is complete".into(),
                prompt: "Given the release plan, verify it covers rollback.".into(),
                response: "Add rollback step.".into(),
                depends_on: vec!["plan".into()],
                provides: vec!["verified_plan".into()],
            },
        ],
    }
}

#[test]
fn extracts_valid_chart_from_transcript() {
    let chart = extract_chart_from_audit(&transcript()).expect("extracts");
    chart.validate().expect("draft validates");
    assert_eq!(chart.author_model, "claude-4");
    assert_eq!(chart.targets.len(), 2);
    assert_eq!(chart.targets[0].name, "plan");
    assert_eq!(chart.targets[1].name, "verify");
    // depends_on edge becomes a Capability dep on the upstream step id.
    match &chart.targets[1].depends[0] {
        DepSpec::Capability { name } => assert_eq!(name, "plan"),
        other @ DepSpec::EntityMatch { .. } => panic!("expected capability dep, got {other:?}"),
    }
    // The query text in the prompt is replaced with {{ request }}.
    assert!(chart.targets[0].template.contains("{{ request }}"));
    assert!(!chart.targets[0].template.contains("v2 API"));
}

#[test]
fn empty_query_is_rejected() {
    let mut t = transcript();
    t.query = "!!".into();
    let err = extract_chart_from_audit(&t).unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}

#[test]
fn no_steps_is_rejected() {
    let mut t = transcript();
    t.steps.clear();
    let err = extract_chart_from_audit(&t).unwrap_err();
    assert!(matches!(err, ChartError::Invalid { .. }));
}

#[test]
fn prompt_without_query_appends_request() {
    let t = template_from_prompt("Produce a checklist.", "write a checklist");
    assert!(t.contains("{{ request }}"));
    assert!(t.contains("Produce a checklist."));
}

#[test]
fn prompt_repeating_the_query_replaces_only_first_occurrence() {
    // Only the *first* occurrence is substituted; a prompt that repeats
    // the query leaves later occurrences literal.
    let t = template_from_prompt(
        "write a checklist for write a checklist",
        "write a checklist",
    );
    assert_eq!(t, "{{ request }} for write a checklist");
}

#[test]
fn slugify_normalizes_and_truncates() {
    assert_eq!(slugify_chart_name("Hello, World!"), "hello_world");
    assert_eq!(
        slugify_chart_name("  multiple   spaces  "),
        "multiple_spaces"
    );
    assert_eq!(slugify_chart_name("already-kebab"), "already-kebab");
    assert!(
        slugify_chart_name(&"a".repeat(200)).len()
            <= super::super::CHART_EXTRACTED_NAME_MAX_CHARS
    );
}

#[test]
fn slugify_chart_name_characterization_table() {
    // Characterization (M4): verbatim outputs of `slugify_chart_name` —
    // the P4 migration must preserve them byte-for-byte.
    let max = super::super::CHART_EXTRACTED_NAME_MAX_CHARS;
    let cases = [
        ("", ""),
        ("___", ""),
        ("---", "---"),
        ("Hello, World!", "hello_world"),
        ("  multiple   spaces  ", "multiple_spaces"),
        ("a  --  b", "a_--_b"),
        ("already-kebab", "already-kebab"),
        ("Café Münchén", "caf_m_nch_n"),
        ("_lead", "_lead"),
        ("trail_", "trail"),
        ("a-", "a-"),
        ("-a", "-a"),
    ];
    for (input, want) in cases {
        assert_eq!(slugify_chart_name(input), want, "chart({input:?})");
    }
    // Over-MAX input truncates to exactly MAX chars (ASCII: byte == char).
    assert_eq!(slugify_chart_name(&"a".repeat(200)).len(), max);
    // Exact-length input passes through.
    assert_eq!(slugify_chart_name(&"a".repeat(max)).len(), max);
    // Parity: the parameterized primitive with chart options.
    for (input, want) in cases {
        assert_eq!(
            common_core::string::slugify_with(
                input,
                &crate::charts::extract::CHART_SLUG_OPTIONS
            ),
            want,
            "slugify_with(chart, {input:?})"
        );
    }
    // Multibyte at the truncation boundary: char-boundary safe, still capped.
    let long_unicode = format!("{}é{}", "b".repeat(max), "c".repeat(10));
    let slugged = slugify_chart_name(&long_unicode);
    assert!(slugged.len() <= max);
    assert!(slugged.chars().count() <= max);
}

#[test]
fn every_target_self_provides_its_id() {
    let mut t = transcript();
    for step in &mut t.steps {
        step.provides.clear();
    }
    let chart = extract_chart_from_audit(&t).expect("extracts");
    // The DependencySession self-provide convention keeps the draft
    // selectable even when the transcript records no explicit provides.
    for target in &chart.targets {
        assert!(
            target.provides.iter().any(|p| p == &target.name),
            "target '{}' must self-provide its id: {:?}",
            target.name,
            target.provides
        );
    }
}

// ── WorkflowExtractor (dispatch post-processing hook) ───────────

#[test]
fn transcript_from_dispatch_produces_single_step() {
    let t = transcript_from_dispatch(
        "write a release plan",
        "system: You are a planner.\nuser: write a release plan",
        "claude-4",
        "Phase 1: ...",
    );
    assert_eq!(t.query, "write a release plan");
    assert_eq!(t.author_model, "claude-4");
    assert_eq!(t.steps.len(), 1);
    assert_eq!(t.steps[0].id, "solve");
    // LOD0 fidelity: the step prompt is the real prompt that was
    // sent to the model — no synthesized "Solve the following request…"
    // wrapper.
    assert_eq!(
        t.steps[0].prompt,
        "system: You are a planner.\nuser: write a release plan"
    );
    assert!(!t.steps[0].prompt.contains("Solve the following request"));
    let chart = extract_chart_from_audit(&t).expect("extracts");
    assert!(
        chart.targets[0].template.contains("{{ request }}"),
        "query text must become a request placeholder"
    );
    // The template captures the real LOD0 prompt shape: the system line
    // is preserved verbatim, only the query is substituted.
    assert_eq!(
        chart.targets[0].template,
        "system: You are a planner.\nuser: {{ request }}"
    );
}

#[test]
fn extractor_disabled_is_a_noop() {
    let store = Arc::new(ChartStore::new(None));
    let extractor = WorkflowExtractor::new(store.clone());
    let outcome = extractor
        .extract_from_transcript(&transcript())
        .expect("disabled extraction never fails");
    assert!(outcome.is_none());
    assert!(store.is_empty(), "disabled extractor must not write charts");
}

#[test]
fn extractor_enabled_writes_draft_chart() {
    let store = Arc::new(ChartStore::new(None));
    let extractor = WorkflowExtractor::new(store.clone()).enabled(true);
    let outcome = extractor
        .extract_from_transcript(&transcript())
        .expect("extracts")
        .expect("enabled extractor writes");
    assert_eq!(outcome, UpsertOutcome::Inserted);
    assert_eq!(store.len(), 1);
    let name = slugify_chart_name(&transcript().query);
    assert!(
        store.get(&name).is_some(),
        "draft chart stored under its slug"
    );
    assert!(store.is_draft(&name), "extracted chart is a draft");
}

#[test]
fn extractor_record_success_swallows_extraction_failure() {
    // A query that slugs to nothing must not panic or propagate — the
    // request already succeeded and the learning loop is best-effort.
    // `is_fallback = true` so the extraction is attempted under the
    // default `Frontier` mode (the swallow path is what we test).
    let store = Arc::new(ChartStore::new(None));
    let extractor = WorkflowExtractor::new(store.clone()).enabled(true);
    extractor.record_success("!!!", "user: !!!", "claude-4", "some answer", true);
    assert!(store.is_empty());
}

// ── Extraction scope (frontier-assisted only by default) ────────

#[test]
fn frontier_mode_skips_primary_dispatch() {
    let store = Arc::new(ChartStore::new(None));
    let extractor = WorkflowExtractor::new(store.clone()).enabled(true);
    extractor.record_success(
        "write a release plan",
        "user: write a release plan",
        "claude-4",
        "Phase 1: ...",
        false,
    );
    assert!(
        store.is_empty(),
        "a primary-target success must not be distilled under Frontier mode"
    );
}

#[test]
fn frontier_mode_extracts_fallback_dispatch() {
    let store = Arc::new(ChartStore::new(None));
    let extractor = WorkflowExtractor::new(store.clone()).enabled(true);
    extractor.record_success(
        "write a release plan",
        "user: write a release plan",
        "claude-4",
        "Phase 1: ...",
        true,
    );
    assert_eq!(store.len(), 1, "a fallback dispatch is distilled");
}

#[test]
fn all_mode_preserves_blanket_extraction() {
    let store = Arc::new(ChartStore::new(None));
    let extractor = WorkflowExtractor::new(store.clone())
        .enabled(true)
        .with_extraction_mode(WorkflowExtractionMode::All);
    extractor.record_success(
        "write a release plan",
        "user: write a release plan",
        "claude-4",
        "Phase 1: ...",
        false,
    );
    assert_eq!(
        store.len(),
        1,
        "mode \"all\" keeps distilling primary-target successes"
    );
}
/// ROADMAP §12.7 (C7): a `parse_review` ledger node becomes a first-class
/// step the chart extractor can replay; non-review nodes are ignored.
#[test]
fn parse_review_step_adapter() {
    use fluent_types::{ContentNode, InterlinguaId, NodeId};
    let node = crate::ledger::nlp::review_node(
        NodeId::from_int(7),
        "s7",
        "r7",
        "the cat sat",
        "review prompt",
        r#"{"corrections":[{"token_index":1,"field":"dep","old_value":"dep","new_value":"nsubj"}]}"#,
        InterlinguaId::from_u64(0x0300_0000_0000_0001),
        None,
        "review-model",
    );
    let mut node = node;
    node.id = Some(NodeId::from_int(99));
    let step = parse_review_step(&node).expect("step");
    assert_eq!(step.id, "review:99");
    assert_eq!(step.purpose, "parse review");
    assert!(step.prompt.contains("review prompt"));
    assert!(step.response.contains("nsubj"));
    assert_eq!(step.depends_on, vec!["parse:7"]);
    assert_eq!(step.provides, vec!["reviewed_parse"]);

    // A non-review node (e.g. a plain request) is ignored.
    let plain = ContentNode {
        id: Some(NodeId::from_int(1)),
        name: "plain".into(),
        source: "x".into(),
        lod: vec![],
        metadata: Some(serde_json::json!({"kind": "request"})),
        ..Default::default()
    };
    assert!(parse_review_step(&plain).is_none());
}
