use super::*;

fn signals() -> Vec<RoutingSignal> {
    vec![RoutingSignal {
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
            predicate_id: Some(InterlinguaId::from_u64(0x0300_0000_0000_0001)),
            subject_id: None,
            direct_object_id: Some(InterlinguaId::from_u64(0x0300_0000_0000_0002)),
            indirect_object_id: Some(InterlinguaId::from_u64(0x0300_0000_0000_0003)),
            concept_ids: vec![InterlinguaId::from_u64(0x0100_0000_0000_0001)],
            token_ids: vec![
                InterlinguaId::from_u64(0x0300_0000_0000_0001),
                InterlinguaId::from_u64(0x0300_0000_0000_0003),
            ],
            confidence: None,
        }),
    }]
}

fn confidence() -> NlpConfidenceSummary {
    NlpConfidenceSummary {
        source: spacy_rs::AnnotationSource::ArcEager,
        overall: 0.82,
        role_coverage: 0.5,
        oracle_tie_count: 1,
        collision_count: 0,
        semantic_plausibility: Some(0.9),
        refine_reason: None,
    }
}

#[test]
fn parse_node_carries_signals_in_metadata() {
    let node = parse_node("s1", "r1", "show me the report", &signals());
    assert!(node.id.is_none(), "the store allocates the id");
    assert!(is_parse_node(&node));
    assert_eq!(node.lod[0], "show me the report"); // LOD0 eager
    let meta = node.metadata.expect("metadata");
    assert_eq!(meta["sentence_count"], 1);
    assert_eq!(meta["signals"][0]["predicate"], "show");
    assert_eq!(meta["signals"][0]["deps"][1], "iobj");
}

#[test]
fn parse_node_metadata_carries_interlingua_frame() {
    let node = parse_node_with_confidence(
        "s1",
        "r1",
        "show me the report",
        &signals(),
        Some(&confidence()),
        Some(&[0.9, 0.8]),
    );
    let meta = node.metadata.expect("metadata");
// interlingua_ids per sentence (§14.1).
    assert_eq!(meta["interlingua_ids"]["sentence_0"]["predicate_id"], 0x0300_0000_0000_0001_i64);
    assert_eq!(meta["interlingua_ids"]["sentence_0"]["direct_object_id"], 0x0300_0000_0000_0002_i64);
    assert_eq!(
        meta["interlingua_ids"]["sentence_0"]["concept_ids"][0],
        0x0100_0000_0000_0001_i64
    );
    // confidence + review_status.
    assert_eq!(meta["confidence"]["overall"], 0.82);
    assert_eq!(meta["confidence"]["source"], "arc_eager");
    assert_eq!(meta["review_status"]["Unreviewed"]["auto_confidence"], 0.82);
    // token_confidence round-trips (L3) — and is absent when not provided.
    assert_eq!(meta["token_confidence"][0], 0.9);
    assert_eq!(meta["token_confidence"][1], 0.8);
    let without = parse_node_with_confidence(
        "s1",
        "r1",
        "show me the report",
        &signals(),
        Some(&confidence()),
        None,
    )
    .metadata
    .expect("metadata");
    assert!(without.get("token_confidence").is_none_or(|v| v.is_null()));
}

#[test]
fn record_round_trips_through_the_ledger() {
    let ledger = ContentNodeLedger::open_in_memory().expect("in-memory ledger");
    let id =
        record_parse_node(&ledger, "s1", "r1", "show me the report", &signals()).expect("record");
    let node = ledger.get_node(id).expect("get node");
    assert!(is_parse_node(&node));
    let meta = node.metadata.expect("metadata");
    assert_eq!(meta["signals"][0]["direct_object"], "report");
    assert_eq!(meta["signals"][0]["lemmas"][0], "show");
}

#[test]
fn record_populates_interlingua_index() {
    let ledger = ContentNodeLedger::open_in_memory().expect("in-memory ledger");
    let id = record_parse_node(&ledger, "s1", "r1", "show me the report", &signals())
        .expect("record");
    let store = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let rows: Vec<(i64, String)> = store
        .query_rows(
            "SELECT interlingua_id, role FROM interlingua_index WHERE node_id = ?1",
            params![id.as_int()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .expect("query");
    // predicate + direct_object + indirect_object + 1 concept = 4 rows.
    assert_eq!(rows.len(), 4);
    assert!(rows.contains(&(0x0300_0000_0000_0001, "predicate".into())));
    assert!(rows.contains(&(0x0300_0000_0000_0002, "direct_object".into())));
    assert!(rows.contains(&(0x0300_0000_0000_0003, "indirect_object".into())));
    assert!(rows.contains(&(0x0100_0000_0000_0001, "concept".into())));
}

/// M1.3 (ROADMAP_20260828_ORT, G2): the consolidated write path
/// (`record_parse_node_with_confidence`) — the helper the live request path
/// (`record_parse_ledger`) now calls — writes one `interlingua_index` row
/// per (sentence × role id) with `review_status='unreviewed'`, exactly like
/// the old test-only `record_parse_node` (G2 closed: the index is no longer
/// reachable only from tests).
#[test]
fn consolidated_write_populates_index_unreviewed() {
    let ledger = ContentNodeLedger::open_in_memory().expect("in-memory ledger");
    let id = record_parse_node_with_confidence(
        &ledger,
        "s1",
        "r1",
        "show me the report",
        &signals(),
        Some(&confidence()),
        Some(&[0.9, 0.8]),
    )
    .expect("record");
    let store = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let rows: Vec<(i64, String, f64, String)> = store
        .query_rows(
            "SELECT interlingua_id, role, confidence, review_status \
             FROM interlingua_index WHERE node_id = ?1 ORDER BY role",
            params![id.as_int()],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .expect("query");
    // predicate + direct_object + indirect_object + 1 concept = 4 rows.
    assert_eq!(rows.len(), 4, "one row per (sentence × role id)");
    for (_id, _role, conf, status) in &rows {
        assert_eq!(status, "unreviewed", "fresh parse rows are unreviewed");
        assert!((0.0..=1.0).contains(conf), "confidence carried onto the index");
    }
    let roles: Vec<&str> = rows.iter().map(|(_, r, _, _)| r.as_str()).collect();
    assert!(roles.contains(&"predicate"));
    assert!(roles.contains(&"direct_object"));
    assert!(roles.contains(&"indirect_object"));
    assert!(roles.contains(&"concept"));
}

#[test]
fn non_parse_node_is_not_a_parse_node() {
    let ledger = ContentNodeLedger::open_in_memory().expect("in-memory ledger");
    let id = ledger.record_request("s1", "r1", "plain request").expect("record");
    let node = ledger.get_node(id).expect("get node");
    assert!(!is_parse_node(&node));
}
#[test]
fn review_node_shapes_and_is_recognized() {
    let node = review_node(
        NodeId::from_int(7),
        "s7",
        "r7",
        "the cat sat",
        "review prompt",
        r#"{"corrections":[]}"#,
        InterlinguaId::from_u64(0x0300_0000_0000_0001),
        Some(InterlinguaId::from_u64(0x0100_0000_0000_0001)),
        "review-model",
    );
    assert!(is_parse_review_node(&node));
    let meta = node.metadata.expect("metadata");
    assert_eq!(meta["kind"], "parse_review");
    assert_eq!(meta["source_node_id"], 7);
    assert_eq!(meta["review_model"], "review-model");
    assert_eq!(node.lod[0], "the cat sat");
    // L4: the node carries its origin (never the "" bucket).
    assert_eq!(node.session_id.as_deref(), Some("s7"));
    assert_eq!(node.request_id.as_deref(), Some("r7"));
}
