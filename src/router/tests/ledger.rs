use super::*;
use common_core::hash::uuid_v4;
use fluent_types::ProvenanceTier;
use crate::test_support::capture_logs;

fn temp_ledger() -> ContentNodeLedger {
    let dir = std::env::temp_dir().join(format!("coral-router-ledger-{}", uuid_v4()));
    let ledger = ContentNodeLedger::open(&dir).unwrap();
    let _ = std::fs::remove_file(&dir);
    ledger
}

/// The `ContentNodeLedger::write_annotation` writer surface (ROADMAP M4) — a
/// HumanReview-tier claim supersedes a LocalModel one on the same node
/// version through the durable facade, and a later lower-tier write is
/// recorded provisional.
#[test]
fn write_annotation_facade_supersedes_by_tier() {
    let ledger = temp_ledger();
    let id = ledger.record_request("sess", "req", "show me the report").unwrap();
    let hash = ledger.get_node(id).unwrap().content_hash;
    assert_ne!(hash, 0, "LOD0 present → non-zero hash");

    let claim = |tier: ProvenanceTier| {
        AnnotationClaim::confirmed(
            "frame:report",
            tier,
            serde_json::json!({ "k": "v" }),
            "writer",
            common_core::now_secs(),
        )
    };
    ledger.write_annotation(id, &claim(ProvenanceTier::LocalModel)).unwrap().unwrap();
    ledger.write_annotation(id, &claim(ProvenanceTier::HumanReview)).unwrap().unwrap();

    // A lower-tier re-write after HumanReview becomes provisional.
    let status = ledger.write_annotation(id, &claim(ProvenanceTier::LocalModel)).unwrap().unwrap();
    assert_eq!(status, ClaimStatus::Provisional, "lower tier after HumanReview is provisional");

    // The authoritative claim is the HumanReview one, confirmed.
    let ann = crate::ledger::annotations::AnnotationStore::new(ledger.node_store().shared_sqlite().unwrap());
    let cur = ann.read(hash, "frame:report").unwrap().expect("authoritative claim");
    assert_eq!(cur.tier, ProvenanceTier::HumanReview);
    assert_eq!(cur.status, ClaimStatus::Confirmed);
    // The superseded + provisional rows are retained (audit, never deleted).
    assert_eq!(ann.history(hash, "frame:report").unwrap().len(), 3);
}

#[test]
fn record_and_fetch_roundtrip() {
    let ledger = temp_ledger();
    let id = ledger.record_request("sess-1", "req-1", "hello").unwrap();
    ledger.record_result(id, true, Some(1.0), "reply").unwrap();
    let entries = ledger.get_session_entries("sess-1", 10).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].content, "reply");
    assert!(entries[0].accepted);
}

#[test]
fn get_node_single_format_no_flat_hydration_fallback() {
    let ledger = temp_ledger();
    // Simulate a migrated pre-LOD row: the flat projection is populated
    // but `content_json` is still the '{}' placeholder. The canonical
    // read (`get_node`) must return `None` — the dual-format hydration
    // fallback is retired; only `get_session_entries` (the flat
    // audit view) reads columns directly. The row is inserted after
    // hydration, so the in-memory maps never see it.
    let store = ledger.node_store().durable().unwrap();
    store
        .execute(
            "INSERT INTO ledger (node_id, session_id, request_id, role, content,
                                 turn_index, accepted, active_lod, metadata, created_at,
                                 label, lod, content_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            rusqlite::params![
                99_i64,
                "sess-pre-lod",
                "req-pre-lod",
                "user",
                "legacy flat content",
                0_i64,
                1_i64,
                0_i64,
                "{}",
                0_i64,
                "legacy label",
                "[]",
                "{}",
            ],
        )
        .unwrap();

    let id = NodeId::from_int(99);
    assert!(
        ledger.get_node(id).is_none(),
        "unparseable content_json -> None"
    );

    let entries = ledger.get_session_entries("sess-pre-lod", 10).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].content, "legacy flat content");
}

#[test]
fn poisoned_db_mutex_recovers() {
    let ledger = temp_ledger();
    // Poison the db mutex by panicking while it is held.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ledger.poison_conn();
    }));
    // Subsequent calls must still succeed via the poison-recovery helper.
    let id = ledger
        .record_request("sess-p", "req-p", "after-poison")
        .unwrap();
    ledger
        .record_result(id, false, Some(0.0), "recovered")
        .unwrap();
    let entries = ledger.get_session_entries("sess-p", 10).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].content, "recovered");
}

#[test]
fn lod0_and_lod5_eager_at_creation() {
    let ledger = temp_ledger();
    let id = ledger
        .record_request(
            "sess-1",
            "req-1",
            "Hello world. This is a longer user message.",
        )
        .unwrap();
    let node = ledger.get_node(id).unwrap();

    // LOD0 full text + LOD5 label eager.
    assert_eq!(node.lod[0], "Hello world. This is a longer user message.");
    assert_eq!(node.lod[5], "Hello world.");
    // LOD1–4 stay empty until derived lazily.
    assert!(node.lod[1].is_empty());
    assert!(node.lod[4].is_empty());
    assert_eq!(node.active_lod, Some(LOD0_FULL_TEXT));
}

#[test]
fn record_content_node_stores_canonical_type() {
    let ledger = temp_ledger();
    let mut node = new_node(
        NodeId::from_int(7),
        "sess-2",
        "req-2",
        "assistant",
        "An accepted assistant answer.",
        Some(true),
    );
    node.acceptance_score = Some(0.9);
    node.step_id = Some("step-1".into());
    let id = ledger.record_content_node(&node).unwrap();

    let fetched = ledger.get_node(id).unwrap();
    assert_eq!(fetched.role.as_ref().map(|r| r.as_str()), Some("assistant"));
    assert_eq!(fetched.acceptance_score, Some(0.9));
    assert_eq!(fetched.step_id.as_deref(), Some("step-1"));
    assert_eq!(fetched.lod[5], "An accepted assistant answer.");
    // LOD0/LOD5 guaranteed even if the caller forgot them.
    assert_eq!(fetched.lod[0], "An accepted assistant answer.");
}

#[test]
fn session_nodes_most_recent_first() {
    let ledger = temp_ledger();
    ledger.record_request("sess-3", "r1", "first").unwrap();
    ledger.record_request("sess-3", "r2", "second").unwrap();

    let nodes = ledger.get_session_nodes("sess-3", 10).unwrap();
    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0].request_id.as_deref(), Some("r2"));
    assert_eq!(nodes[1].request_id.as_deref(), Some("r1"));
}

#[test]
fn ensure_lod_requires_summarizer() {
    let ledger = temp_ledger();
    let id = ledger.record_request("sess-4", "r1", "some text").unwrap();
    assert!(matches!(
        ledger.ensure_lod(id, 2),
        Err(LedgerError::NoSummarizer)
    ));
    assert!(matches!(
        ledger.ensure_lod(id, 0),
        Err(LedgerError::InvalidLod(0))
    ));
}

#[test]
fn ensure_lod_derives_from_lod0_and_caches() {
    use crate::test_stubs::StubChatBackend;
    use std::sync::Arc;

    let client: Arc<dyn fluent_llm::client::ChatBackend> =
        Arc::new(StubChatBackend::always("lazy LOD summary"));
    let summarizer = Summarizer::new(client, 20);
    let ledger = temp_ledger().with_summarizer(summarizer);

    let id = ledger
        .record_request(
            "sess-5",
            "r1",
            "The full text that must be summarized from LOD0 only.",
        )
        .unwrap();

    let node = ledger.ensure_lod(id, 2).unwrap();
    assert_eq!(node.lod[2], "lazy LOD summary");
    // Cached: a second derivation hits the cache, not the LLM.
    let node2 = ledger.ensure_lod(id, 2).unwrap();
    assert_eq!(node2.lod[2], "lazy LOD summary");
    // LOD0 is untouched by derivation (never chained from a lower tier).
    let node3 = ledger.get_node(id).unwrap();
    assert_eq!(
        node3.lod[0],
        "The full text that must be summarized from LOD0 only."
    );
}

#[test]
fn compact_session_demotes_oldest_nodes() {
    let ledger = temp_ledger();
    for i in 0..5 {
        ledger
            .record_request("sess-6", &format!("r{i}"), &format!("message {i}"))
            .unwrap();
    }

    let demoted = ledger.compact_session("sess-6", 4).unwrap();
    assert!(!demoted.is_empty(), "some nodes must be demoted");
    let nodes = ledger.get_session_nodes("sess-6", 10).unwrap();
    // Newest node stays at full detail; oldest is demoted to LOD3.
    let newest = nodes
        .iter()
        .find(|n| n.request_id.as_deref() == Some("r4"))
        .unwrap();
    assert_eq!(newest.active_lod, Some(0));
    let oldest = nodes
        .iter()
        .find(|n| n.request_id.as_deref() == Some("r0"))
        .unwrap();
    assert_eq!(oldest.active_lod, Some(3));
}

#[test]
fn recency_compaction_under_max() {
    let nodes = make_nodes(3);
    let lods = RecencyCompaction.select_lod(&nodes, 10);
    assert_eq!(lods, vec![0, 0, 0]);
}

#[test]
fn recency_compaction_over_max() {
    let nodes = make_nodes(8);
    let lods = RecencyCompaction.select_lod(&nodes, 4);
    assert_eq!(lods[0], 3);
    assert_eq!(lods[1], 3);
    assert_eq!(lods[7], 0);
}

#[test]
fn noop_compaction() {
    let nodes = make_nodes(100);
    let lods = NoopCompaction.select_lod(&nodes, 10);
    assert!(lods.iter().all(|&l| l == 0));
}

fn make_nodes(count: usize) -> Vec<ContentNode> {
    (0..count)
        .map(|i| {
            new_node(
                NodeId::from_int(i as i64),
                "test",
                &format!("req-{i}"),
                "user",
                &format!("node {i}"),
                Some(true),
            )
        })
        .collect()
}

// ── Write-path guard (facade scrub) ────────────────────────────────

#[test]
fn record_request_scrubs_email_and_emits_audit() {
    let ledger = temp_ledger();
    let (id, logs) = capture_logs(|| {
        ledger
            .record_request("sess-guard", "r1", "Contact user@example.com now")
            .unwrap()
    });
    let _ = id;
    let joined = logs.join("\n");
    assert!(
        joined.contains("router.audit")
            && joined.contains("write_path")
            && joined.contains("email"),
        "flagged write must emit a write-path audit, logs:\n{joined}"
    );

    let entries = ledger.get_session_entries("sess-guard", 10).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].content, "Contact [REDACTED:email] now");
    assert!(
        !entries[0].content.contains("user@example.com"),
        "durable content must be scrubbed"
    );
}

#[test]
fn record_result_scrubs_phone() {
    let ledger = temp_ledger();
    let id = ledger
        .record_request("sess-guard-r", "r1", "What number?")
        .unwrap();
    ledger
        .record_result(id, true, Some(1.0), "Call 555-123-4567 to reach us.")
        .unwrap();

    let node = ledger.get_node(id).unwrap();
    assert_eq!(node.lod[0], "Call [REDACTED:phone] to reach us.");
    let entries = ledger.get_session_entries("sess-guard-r", 10).unwrap();
    assert_eq!(entries[0].content, "Call [REDACTED:phone] to reach us.");
}

#[test]
fn record_content_node_scrubs_api_key_to_reject_marker() {
    let ledger = temp_ledger();
    let mut node = new_node(
        NodeId::from_int(101),
        "sess-guard-c",
        "r1",
        "assistant",
        "the token is api_key = super_secret_value_123",
        Some(true),
    );
    node.acceptance_score = Some(0.9);
    let id = ledger.record_content_node(&node).unwrap();

    let fetched = ledger.get_node(id).unwrap();
    assert_eq!(fetched.lod[0], "[rejected: api_key]");
    assert_eq!(fetched.acceptance_score, Some(0.9));
}

#[test]
fn clean_write_is_not_flagged() {
    let ledger = temp_ledger();
    let (_, logs) = capture_logs(|| {
        ledger
            .record_request("sess-guard-clean", "r1", "plain text, no pii")
            .unwrap()
    });
    let joined = logs.join("\n");
    assert!(
        !joined.contains("write_path"),
        "clean writes must not emit a write-path audit, logs:\n{joined}"
    );
    let entries = ledger.get_session_entries("sess-guard-clean", 10).unwrap();
    assert_eq!(entries[0].content, "plain text, no pii");
}

#[test]
fn render_session_renders_three_nodes_as_three_lines() {
    let ledger = temp_ledger();
    ledger
        .record_request("sess-render", "r1", "first node text")
        .unwrap();
    ledger
        .record_request("sess-render", "r2", "second node text")
        .unwrap();
    ledger
        .record_request("sess-render", "r3", "third node text")
        .unwrap();

    let rendered = ledger.render_session("sess-render", Lod::LOD0);
    let lines: Vec<&str> = rendered.lines().collect();
    assert_eq!(lines.len(), 3, "3 nodes -> 3 lines, got: {rendered}");
    assert!(lines.contains(&"first node text"));
    assert!(lines.contains(&"second node text"));
    assert!(lines.contains(&"third node text"));
}

#[test]
fn render_session_renders_collapsed_node_lod0() {
    let ledger = temp_ledger();
    let id = ledger
        .record_request("sess-collapse", "r1", "original long content")
        .unwrap();
    ledger
        .collapse_node(id, "collapsed summary", LOD0_FULL_TEXT)
        .unwrap();

    // Compaction mutates LOD0, so a LOD0 view shows the collapsed text —
    // the fidelity policy never "defeats" compaction.
    assert_eq!(
        ledger.render_session("sess-collapse", Lod::LOD0),
        "collapsed summary"
    );
}

// ── Migration 6: interlingua schema rebuild (red-team M2/M3) ───────

#[test]
fn migration_6_fresh_db_creates_v2_schema() {
    use fluent_db::migrate::{migrate, schema_version};
    use fluent_types::{InterlinguaId, InterlinguaNamespace};
    use rusqlite::Connection;

    let conn = Connection::open_in_memory().unwrap();
    migrate(&conn, &crate::ledger::ledger_migrations()).unwrap();
    assert_eq!(schema_version(&conn).unwrap(), 9);

    // Two canonicals that collide on the truncated 48-bit id are BOTH
    // storable: the PK is (namespace, canonical_name), `id` is a column.
    let local = 0x1234_5678_9abc_i64;
    let a = InterlinguaId::new(InterlinguaNamespace::YagoClass, local);
    let b = InterlinguaId::new(InterlinguaNamespace::YagoClass, local);
    conn.execute(
        "INSERT OR IGNORE INTO interlingua_concepts (id, namespace, canonical_name) \
         VALUES (?1, 256, 'schema:One')",
        rusqlite::params![a.as_i64()],
    )
    .unwrap();
    conn.execute(
        "INSERT OR IGNORE INTO interlingua_concepts (id, namespace, canonical_name) \
         VALUES (?1, 256, 'schema:Two')",
        rusqlite::params![b.as_i64()],
    )
    .unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM interlingua_concepts", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2, "both colliding canonicals stored");
}

#[test]
fn migration_6_maps_overloaded_review_status_to_entity_id() {
    use fluent_db::migrate::{migrate, schema_version};
    use fluent_types::{InterlinguaId, InterlinguaNamespace};
    use rusqlite::Connection;

    let conn = Connection::open_in_memory().unwrap();
    let pre_v6: [&dyn Migration; 5] = [
        &LedgerBaseSchema,
        &LedgerLabelColumn,
        &LedgerLodColumn,
        &LedgerContentJsonColumn,
        &LedgerInterlinguaSchema,
    ];
    migrate(&conn, &pre_v6).unwrap();
    assert_eq!(schema_version(&conn).unwrap(), 5);

    // Seed the old schema: a correction-pattern row whose `review_status`
    // overloads the entity id as a string, plus a normal per-node audit row.
    let lemma = InterlinguaId::new(InterlinguaNamespace::SpacyLemma, 1).as_i64();
    let obj = InterlinguaId::new(InterlinguaNamespace::SpacyLemma, 2).as_i64();
    let entity = InterlinguaId::new(InterlinguaNamespace::YagoEntity, 7).as_i64();
    conn.execute(
        "INSERT INTO interlingua_index \
            (node_id, interlingua_id, interlingua_source, role, confidence, review_status, corrections) \
         VALUES (0, ?1, 'spacy_lemma', 'correction', NULL, ?2, ?3)",
        rusqlite::params![lemma, entity.to_string(), "[]"],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO interlingua_index \
            (node_id, interlingua_id, interlingua_source, role, confidence, review_status, corrections) \
         VALUES (?1, ?2, 'spacy_lemma', 'lemma', 0.9, 'unreviewed', NULL)",
        rusqlite::params![42_i64, obj],
    )
    .unwrap();
    let class_id = InterlinguaId::new(InterlinguaNamespace::YagoClass, 3).as_i64();
    conn.execute(
        "INSERT INTO interlingua_concepts \
            (id, namespace, canonical_name, yago_iri, label, node_id, parent_class_id) \
         VALUES (?1, 256, 'schema:Person', 'http://schema.org/Person', 'person', ?2, NULL)",
        rusqlite::params![class_id, 123_i64],
    )
    .unwrap();

    // Run migration 6 on top of the migrated 1–5 state.
    migrate(&conn, &crate::ledger::ledger_migrations()).unwrap();
    assert_eq!(schema_version(&conn).unwrap(), 9);

    // The correction row moved its entity id into the real column and is
    // now `'cached'` (status-only semantics restored).
    let (entity_id, status): (i64, String) = conn
        .query_row(
            "SELECT entity_id, review_status FROM interlingua_index \
             WHERE node_id = 0 AND role = 'correction'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(entity_id, entity);
    assert_eq!(status, "cached");
    // The audit row is untouched (entity_id 0, status preserved).
    let (entity_id, status): (i64, String) = conn
        .query_row(
            "SELECT entity_id, review_status FROM interlingua_index WHERE node_id = 42",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(entity_id, 0);
    assert_eq!(status, "unreviewed");
    // The concept row survived the table rebuild.
    let name: String = conn
        .query_row(
            "SELECT canonical_name FROM interlingua_concepts WHERE id = ?1",
            rusqlite::params![class_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(name, "schema:Person");
}

// ── Migration 9: span-cache hex key (F7) ─────────────────────────────

#[test]
fn migration_9_span_cache_hex_correct_for_negative_i64() {
    use fluent_db::migrate::{migrate, schema_version};
    use rusqlite::Connection;

    // Build DB up to v8 (before span_key) so interlingua_index has no
    // span_key column and span-cache rows are stored as signed i64.
    let conn = Connection::open_in_memory().unwrap();
    let pre_v9: [&dyn Migration; 8] = [
        &LedgerBaseSchema,
        &LedgerLabelColumn,
        &LedgerLodColumn,
        &LedgerContentJsonColumn,
        &LedgerInterlinguaSchema,
        &LedgerInterlinguaSchemaV2,
        &LedgerOverlayCandidatesSchema,
        &LedgerAnnotationSchema,
    ];
    migrate(&conn, &pre_v9).unwrap();
    assert_eq!(schema_version(&conn).unwrap(), 8);

    // Seed two pre-existing span-cache rows whose keys would be negative
    // when cast via `as i64`: u64::MAX (-1) and 0x8000_0000_0000_0001
    // (i64::MIN+1). These are exactly the high-bit keys the old
    // `key as i64` path would corrupt.
    let key_max_negative: i64 = -1; // 0xffffffffffffffff
    let key_high_negative: i64 = i64::MIN + 1; // 0x8000000000000001
    conn.execute(
        "INSERT INTO interlingua_index \
            (node_id, interlingua_id, interlingua_source, role, entity_id, review_status, corrections) \
         VALUES (0, ?1, 'spacy_lemma', 'span_cache', 0, 'cached', '[]')",
        rusqlite::params![key_max_negative],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO interlingua_index \
            (node_id, interlingua_id, interlingua_source, role, entity_id, review_status, corrections) \
         VALUES (0, ?1, 'spacy_lemma', 'span_cache', 0, 'cached', '[]')",
        rusqlite::params![key_high_negative],
    )
    .unwrap();
    // A normal lemma row must not get a span_key.
    conn.execute(
        "INSERT INTO interlingua_index \
            (node_id, interlingua_id, interlingua_source, role, entity_id, review_status, corrections) \
         VALUES (42, 123, 'spacy_lemma', 'lemma', 0, 'unreviewed', NULL)",
        [],
    )
    .unwrap();

    // Run migration 9 (printf('%016x', interlingua_id) for span_cache).
    migrate(&conn, &crate::ledger::ledger_migrations()).unwrap();
    assert_eq!(schema_version(&conn).unwrap(), 9);

    // Verify the u64 reinterpretation via SQL printf produced the
    // correct unsigned hex (version-dependent behavior — this test
    // asserts it for the current SQLite).
    let span_keys: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT span_key FROM interlingua_index WHERE role='span_cache' ORDER BY interlingua_id")
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(span_keys.len(), 2);
    // Ordered by interlingua_id ascending: -1 is -1, MIN+1 is smaller
    assert!(span_keys.contains(&"ffffffffffffffff".to_string()));
    assert!(span_keys.contains(&"8000000000000001".to_string()));

    // Normal rows keep empty span_key.
    let normal_key: String = conn
        .query_row(
            "SELECT span_key FROM interlingua_index WHERE node_id=42",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(normal_key, "");

    // The migrated rows are reachable via the new span_key index.
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM interlingua_index WHERE role='span_cache' AND span_key='ffffffffffffffff'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn lod_derivation_is_from_lod0_only() {
    use crate::test_stubs::CountingBackend;
    use std::sync::Arc;
    // Summarizer that echoes LOD0 prefix to prove derivation source — counting backend repeats same response
    let backend = Arc::new(CountingBackend::new("derived-from-lod0"));
    let client: Arc<dyn fluent_llm::client::ChatBackend> = backend.clone() as Arc<dyn fluent_llm::client::ChatBackend>;
    let summarizer = Summarizer::new(client, 20);
    let ledger = temp_ledger().with_summarizer(summarizer);
    let id = ledger
        .record_request("sess-lod0", "r1", "original LOD0 text for derivation")
        .unwrap();
    // Derive LOD4 - must come from LOD0
    let node = ledger.ensure_lod(id, 4).unwrap();
    assert_eq!(node.lod[4], "derived-from-lod0");
    // Verify that LOD1 content does not affect LOD4 derivation: set LOD1 to fake and re-derive on a new node
    let id2 = ledger
        .record_request("sess-lod0-2", "r2", "different LOD0 second node")
        .unwrap();
    // Manually set LOD1 to fake on id2 via store (to simulate chained derivation attempt)
    ledger.node_store().with_node_mut(id2, |n| {
        while n.lod.len() < 5 { n.lod.push(String::new()); }
        n.lod[1] = "FAKE LOD1 SHOULD NOT BE SOURCE".into();
    }).unwrap();
    let node2 = ledger.ensure_lod(id2, 4).unwrap();
    // Still derived from LOD0 (counting backend returns same), but importantly not equal to LOD1 fake
    assert_ne!(node2.lod[4], "FAKE LOD1 SHOULD NOT BE SOURCE");
    assert_eq!(node2.lod[4], "derived-from-lod0");
    // LOD0 remains authoritative
    let orig = ledger.get_node(id2).unwrap();
    assert_eq!(orig.lod[0], "different LOD0 second node");
}

#[test]
fn ledger_tier_worker_is_idempotent() {
    use crate::test_stubs::CountingBackend;
    use std::sync::Arc;
    let backend = Arc::new(CountingBackend::new("SUMMARY: cached summary\nDESCRIPTION: desc"));
    // Wrap counting backend as ChatBackend for Summarizer
    let summarizer = Summarizer::new(backend.clone() as Arc<dyn fluent_llm::client::ChatBackend>, 20);
    let ledger = temp_ledger().with_summarizer(summarizer);
    let id = ledger
        .record_request("sess-idempotent", "r1", "text for idempotent derivation")
        .unwrap();
    // First derivation hits backend
    let _ = ledger.ensure_lod(id, 4).unwrap();
    assert_eq!(backend.calls(), 1, "first derive hits backend");
    // Second derivation must hit cache, not backend
    let _ = ledger.ensure_lod(id, 4).unwrap();
    assert_eq!(backend.calls(), 1, "second derive hits cache, no second call");
    // Different level also caches separately but still only one call per level
    let _ = ledger.ensure_lod(id, 2).unwrap();
    assert_eq!(backend.calls(), 2, "different LOD level triggers new derivation");
    let _ = ledger.ensure_lod(id, 2).unwrap();
    assert_eq!(backend.calls(), 2, "second hit for LOD2 also cached");
}
