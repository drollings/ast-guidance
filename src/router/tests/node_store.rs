use super::*;
use crate::test_stubs::{CountingBackend, StubChatBackend};

fn temp_store() -> ContentNodeStore {
    let dir = std::env::temp_dir().join(format!(
        "coral-router-nodestore-{}",
        common_core::hash::uuid_v4()
    ));
    let store = ContentNodeStore::open(&dir).unwrap();
    let _ = std::fs::remove_file(&dir);
    store
}

#[test]
fn same_id_returns_same_arc_identity() {
    let store = temp_store();
    let id = store.record_request("s", "r1", "hello").unwrap();
    let a = store.get_node(id).unwrap();
    let b = store.get_node(id).unwrap();
    assert!(Arc::ptr_eq(&a, &b), "two lookups must share one Arc");
}

#[test]
fn ensure_lod_computed_once_across_views() {
    let client: Arc<dyn fluent_llm::client::ChatBackend> =
        Arc::new(StubChatBackend::always("lazy LOD summary"));
    let summarizer = Summarizer::new(client, 20);
    let store = temp_store().with_summarizer(summarizer);
    let id = store
        .record_request("s", "r1", "The full text that must be summarized once.")
        .unwrap();

    // "Two concurrent views" hold the same Arc — derive once, then both see
    // the cached tier without a second LLM call.
    let v1 = store.get_node(id).unwrap();
    let v2 = store.get_node(id).unwrap();
    let node = store.ensure_lod(id, 2).unwrap();
    assert_eq!(node.lod[2], "lazy LOD summary");
    assert_eq!(lock_read(&v1).lod[2], "lazy LOD summary");
    assert_eq!(lock_read(&v2).lod[2], "lazy LOD summary");
}

#[test]
fn interned_session_and_role_indices_return_correct_sets() {
    let store = temp_store();
    store.record_request("sess-a", "r1", "one").unwrap();
    store.record_request("sess-a", "r2", "two").unwrap();
    store.record_request("sess-b", "r3", "three").unwrap();

    let sess_a = store.get_session_nodes("sess-a", 10).unwrap();
    assert_eq!(sess_a.len(), 2);
    assert_eq!(sess_a[0].request_id.as_deref(), Some("r2"));
    assert_eq!(sess_a[1].request_id.as_deref(), Some("r1"));
    assert!(store.get_session_nodes("sess-b", 10).unwrap().len() == 1);
    assert!(store.get_session_nodes("absent", 10).unwrap().is_empty());

    // role index (interned): all recorded requests carry role "user".
    let user_ids = store.nodes_for_role("user");
    assert_eq!(user_ids.len(), 3);
    assert!(store.nodes_for_role("assistant").is_empty());
}

#[test]
fn hydration_round_trip_preserves_data_and_continues_next_id() {
    let dir = std::env::temp_dir().join(format!(
        "coral-router-nodestore-rt-{}",
        common_core::hash::uuid_v4()
    ));
    let path = dir.clone();
    {
        let store = ContentNodeStore::open(&path).unwrap();
        store.record_request("s", "r1", "first").unwrap();
        store.record_request("s", "r2", "second").unwrap();
    } // drop
    {
        let store = ContentNodeStore::open(&path).unwrap();
        let nodes = store.get_session_nodes("s", 10).unwrap();
        assert_eq!(nodes.len(), 2, "data must survive reopen");
        assert_eq!(nodes[0].request_id.as_deref(), Some("r2"));

        // next_id continues past the hydrated max: the next allocation
        // must not collide with the persisted ids.
        let id = store.record_request("s", "r3", "third").unwrap();
        assert!(id.as_int() > 2, "next id must be past the hydrated max");
        assert!(store.get_node(id).is_some());
        assert_eq!(store.get_session_nodes("s", 10).unwrap().len(), 3);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn knn_search_delegates_to_brute_force_over_embeddings() {
    let store = temp_store();
    let mut node = new_node(
        NodeId::from_int(0),
        "s",
        "r1",
        "assistant",
        "embedding target",
        Some(true),
    );
    node.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
    let id = store.record_content_node(&node).unwrap();

    let hits = store.knn_search(&[1.0, 0.0, 0.0, 0.0], 1);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].node_id, id);

    let no_hits = store.knn_search(&[0.0, 1.0, 0.0, 0.0], 1);
    assert_eq!(no_hits.len(), 1, "orthogonal but still nearest");
    assert_eq!(no_hits[0].node_id, id);
}

#[test]
fn knn_search_k_edges_zero_and_overfetch() {
    // M5.1: k-edge locks for the HNSW→brute-force migration (unbuilt store).
    let store = temp_store();
    let mut node = new_node(
        NodeId::from_int(0),
        "s",
        "r1",
        "assistant",
        "embedding target",
        Some(true),
    );
    node.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
    let id = store.record_content_node(&node).unwrap();

    assert!(store.knn_search(&[1.0, 0.0, 0.0, 0.0], 0).is_empty(), "k=0");
    let all = store.knn_search(&[1.0, 0.0, 0.0, 0.0], 10);
    assert_eq!(all.len(), 1, "k>len returns all");
    assert_eq!(all[0].node_id, id);
}

#[test]
fn ephemeral_store_needs_no_durable() {
    let store = ContentNodeStore::ephemeral();
    let id = store.record_request("s", "r1", "x").unwrap();
    assert!(store.get_node(id).is_some());
    assert!(store.get_session_entries("s", 10).unwrap().is_empty());
}

#[test]
fn lod_text_returns_eager_tiers_directly() {
    let store = temp_store();
    let id = store
        .record_request("s", "r1", "Full text for eager tiers.")
        .unwrap();
    assert_eq!(store.lod_text(id, 0).unwrap(), "Full text for eager tiers.");
    assert_eq!(store.lod_text(id, 5).unwrap(), "Full text for eager tiers.");
}

#[test]
fn lod_text_derives_lazy_tier_exactly_once() {
    let backend = Arc::new(CountingBackend::new("lazy tier text"));
    let summarizer = Summarizer::new(backend.clone(), 20);
    let store = temp_store().with_summarizer(summarizer);
    let id = store
        .record_request("s", "r1", "The full text that must be summarized once.")
        .unwrap();

    let first = store.lod_text(id, 2).unwrap();
    assert_eq!(first, "lazy tier text");
    assert_eq!(backend.calls(), 1, "exactly one derivation");

    let second = store.lod_text(id, 2).unwrap();
    assert_eq!(second, "lazy tier text");
    assert_eq!(backend.calls(), 1, "second read hits the cache");
}

#[test]
fn lod_text_without_summarizer_returns_no_summarizer() {
    let store = temp_store();
    let id = store.record_request("s", "r1", "text").unwrap();
    assert!(matches!(
        store.lod_text(id, 2),
        Err(LedgerError::NoSummarizer)
    ));
    assert!(matches!(
        store.lod_text(id, 9),
        Err(LedgerError::InvalidLod(9))
    ));
}

#[test]
fn session_node_ids_returns_ids_without_node_clones() {
    let store = temp_store();
    let id1 = store.record_request("sess", "r1", "one").unwrap();
    let id2 = store.record_request("sess", "r2", "two").unwrap();
    store.record_request("other", "r3", "three").unwrap();

    let ids = store.session_node_ids("sess");
    assert_eq!(ids, vec![id1, id2], "insertion order, ids only");
    assert!(store.session_node_ids("absent").is_empty());
}

#[test]
fn lod_text_not_found_returns_not_found() {
    let store = temp_store();
    assert!(matches!(
        store.lod_text(NodeId::from_int(9999), 0),
        Err(LedgerError::NotFound(_))
    ));
}

// ─── M12-S3 characterization: LOD0 authority ──────────────────────────
// VISION's LOD0-authority rule: every lazy tier (1–4) derives from LOD0,
// never chained from another cached tier (no summary-of-a-summary). The
// recording backend captures each derivation's input: after LOD1 is cached,
// deriving LOD2 must still be fed the LOD0 full text — not the LOD1 marker.
// Any future extraction of the store behind `db` traits or reconciliation
// with `content-node` slices must keep this table green.
#[test]
fn lazy_tiers_derive_from_lod0_never_from_another_tier() {
    struct RecordingBackend {
        prompts: std::sync::Mutex<Vec<String>>,
    }
    impl fluent_llm::client::ChatBackend for RecordingBackend {
        fn chat_complete(
            &self,
            messages: &[fluent_llm::ChatMessage],
        ) -> Result<String, fluent_llm::LlmError> {
            let user_text = messages
                .iter()
                .find(|m| m.role == "user")
                .map(|m| m.content.clone())
                .unwrap_or_default();
            self.prompts.lock().unwrap().push(user_text);
            Ok("TIER-MARKER".into())
        }
    }

    let backend = Arc::new(RecordingBackend {
        prompts: std::sync::Mutex::new(Vec::new()),
    });
    let store = temp_store().with_summarizer(Summarizer::new(backend.clone(), 20));
    let full = "alpha bravo charlie delta echo full text here";
    let id = store.record_request("s", "r1", full).unwrap();

    assert_eq!(store.lod_text(id, 1).unwrap(), "TIER-MARKER");
    assert_eq!(store.lod_text(id, 2).unwrap(), "TIER-MARKER");
    let prompts = backend.prompts.lock().unwrap();
    assert_eq!(prompts.len(), 2, "one derivation per tier");
    for (i, prompt) in prompts.iter().enumerate() {
        assert_eq!(
            prompt, full,
            "tier derivation {i} must be fed LOD0, never a cached tier"
        );
    }
}

#[test]
fn apply_review_commits_all_three_writes_together() {
    let store = temp_store();
    let parse_id = store.record_request("s1", "r1", "show me the report").unwrap();
    let shared = store.shared_sqlite().expect("durable backing");

    let review_node = new_node(
        NodeId::from_int(0),
        "s1",
        "r1",
        "parse_review",
        "the cat sat",
        None,
    );
    let meta = serde_json::json!({
        "review_status": { "Reviewed": { "source": "human_review" } },
    });
    let rows = [CorrectionRow {
        lemma_id: 0x0300_0000_0000_0001,
        entity_id: 0,
        corrections_json: r#"[{"token_index":0,"field":"dep","old_value":"det","new_value":"nsubj"}]"#
            .into(),
    }];

    let review_id = store
        .apply_review(parse_id, meta.clone(), Some(&review_node), &rows)
        .expect("apply_review");

    // 1. review_status landed on the parse node (in-memory + durable).
    let node = store.snapshot(parse_id).expect("parse node");
    assert_eq!(
        node.metadata.as_ref().and_then(|m| m.get("review_status")),
        Some(&meta["review_status"])
    );
    // 2. the parse_review node is persisted + indexed.
    let review_id = review_id.expect("review node id");
    assert_eq!(
        store.snapshot(review_id).expect("review node").role.as_ref().map(OriginRole::as_str),
        Some("parse_review")
    );
    // 3. the correction pattern is durable in interlingua_index.
    let count: i64 = shared
        .query_row(
            "SELECT COUNT(*) FROM interlingua_index WHERE node_id = 0 AND role = 'correction'",
            &[],
            |r| r.get(0),
        )
        .expect("count")
        .expect("row");
    assert_eq!(count, 1);
}

#[test]
fn apply_review_rolls_back_on_missing_parse_node() {
    let store = temp_store();
    let shared = store.shared_sqlite().expect("durable backing");
    let rows = [CorrectionRow {
        lemma_id: 0x0300_0000_0000_0001,
        entity_id: 0,
        corrections_json: "[]".into(),
    }];
    let meta = serde_json::json!({ "review_status": { "Reviewed": {} } });

    let before: i64 = shared
        .query_row("SELECT COUNT(*) FROM interlingua_index", &[], |r| r.get(0))
        .expect("count")
        .expect("row");
    assert!(matches!(
        store.apply_review(NodeId::from_int(9999), meta, None, &rows),
        Err(LedgerError::NotFound(_))
    ));
    // The transaction rolled back: no partial correction row survived.
    let after: i64 = shared
        .query_row("SELECT COUNT(*) FROM interlingua_index", &[], |r| r.get(0))
        .expect("count")
        .expect("row");
    assert_eq!(before, after, "a failed review must not half-apply");
}

#[test]
fn apply_review_durable_failure_leaves_in_memory_unchanged() {
    let store = temp_store();
    let parse_id = store.record_request("s1", "r1", "show me the report").unwrap();
    let shared = store.shared_sqlite().expect("durable backing");

    // The review node's id will be `next_id` = 2. Pre-occupy that ledger
    // row directly (the store's maps are untouched) so the durable INSERT
    // fails with a PK collision mid-transaction — the failure injection.
    shared
        .execute(
            "INSERT INTO ledger (node_id, session_id, request_id, role, content) \
             VALUES (?1, 's', 'r', 'intruder', 'occupied')",
            rusqlite::params![2],
        )
        .expect("occupy node_id 2");

    let review_node = new_node(
        NodeId::from_int(0),
        "s1",
        "r1",
        "parse_review",
        "the cat sat",
        None,
    );
    let meta = serde_json::json!({
        "review_status": { "Reviewed": { "source": "human_review" } },
    });
    let rows = [CorrectionRow {
        lemma_id: 0x0300_0000_0000_0001,
        entity_id: 0,
        corrections_json: "[]".into(),
    }];

    let before = store.snapshot(parse_id).expect("parse node").metadata;

    let result = store.apply_review(parse_id, meta, Some(&review_node), &rows);
    assert!(
        result.is_err(),
        "the durable insert collision must fail the transaction"
    );

    // 1. The in-memory parse-node metadata is UNCHANGED (durable-first:
    //    no divergence on a failed commit).
    let after = store.snapshot(parse_id).expect("parse node").metadata;
    assert_eq!(
        after, before,
        "a failed durable review must not touch in-memory metadata"
    );
    // 2. No review node is present in the maps.
    assert!(
        store.get_node(NodeId::from_int(2)).is_none(),
        "no in-memory review node on failure"
    );
    // 3. The transaction rolled back: only the injected intruder row
    //    occupies node_id=2, and no correction rows survived.
    let ledger_rows: i64 = shared
        .query_row(
            "SELECT COUNT(*) FROM ledger WHERE node_id = 2",
            &[],
            |r| r.get(0),
        )
        .expect("count")
        .expect("row");
    assert_eq!(ledger_rows, 1, "no partial ledger write");
    let correction_rows: i64 = shared
        .query_row(
            "SELECT COUNT(*) FROM interlingua_index WHERE role = 'correction'",
            &[],
            |r| r.get(0),
        )
        .expect("count")
        .expect("row");
    assert_eq!(correction_rows, 0, "no partial correction rows");
}

// ── Overlay store surface (OVERLAYS M6) ──────────────────────────────

use std::sync::atomic::{AtomicUsize, Ordering};

use fluent_llm::{BatchEmbedding, ChatMessage, EmbeddingError, LlmError};

/// A `HashEmbedder` that counts every `embed` call (the call-counting stub
/// for "derived exactly once").
struct CountingEmbedder {
    inner: crate::test_stubs::HashEmbedder,
    calls: Arc<AtomicUsize>,
}

impl CountingEmbedder {
    fn new(dims: usize) -> Self {
        Self {
            inner: crate::test_stubs::HashEmbedder::new(dims),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl EmbeddingProvider for CountingEmbedder {
    fn name(&self) -> &'static str {
        "counting-test"
    }
    fn dimensions(&self) -> u32 {
        self.inner.dimensions()
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.embed(text)
    }
    fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
        self.calls.fetch_add(texts.len(), Ordering::SeqCst);
        self.inner.embed_batch(texts)
    }
}

/// A `ChatBackend` that counts calls and always fails — the "permanent
/// failure, never retried" stub.
struct FailingChatBackend {
    calls: Arc<AtomicUsize>,
}

impl FailingChatBackend {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl ChatBackend for FailingChatBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(LlmError::NoResponse)
    }
}

fn pipeline_store() -> ContentNodeStore {
    let store = temp_store();
    store.set_overlay_pipeline(Arc::new(
        spacy_rs::NlpPipeline::en_default().expect("en pipeline"),
    ));
    store
}

#[test]
fn annotation_for_derives_once_and_caches_same_arc() {
    let store = pipeline_store();
    let id = store.record_request("s", "r1", "Show me the sales report").unwrap();

    let first = store.annotation_for(id).expect("derive").expect("annotation");
    let second = store.annotation_for(id).expect("derive").expect("annotation");
    assert!(Arc::ptr_eq(&first, &second), "the cache must share one Arc");
    assert_eq!(first.signals[0].predicate, "Show");
    assert_eq!(first.primary_signal().unwrap().sentence, "Show me the sales report");
    // Bookkeeping: ready, from the spacy rung; value rides the shared node.
    let node = store.snapshot(id).unwrap();
    assert_eq!(node.overlay(OverlayKind::Spacy).status, OverlayStatus::Ready);
    assert_eq!(node.overlay(OverlayKind::Spacy).source, "spacy");
    assert_eq!(
        node.annotation_as::<spacy_rs::ArcReadyAnnotation>().unwrap().tokens.len(),
        5
    );
}

#[test]
fn annotation_for_fail_open_without_pipeline() {
    let store = temp_store();
    let id = store.record_request("s", "r1", "text").unwrap();
    assert!(store.annotation_for(id).expect("no pipeline is fail-open").is_none());
}

#[test]
fn embedding_for_derives_once_and_caches() {
    let store = temp_store();
    let embedder = CountingEmbedder::new(8);
    let calls = Arc::clone(&embedder.calls);
    store.set_overlay_embedder(Arc::new(embedder));
    let id = store.record_request("s", "r1", "buy apple stock").unwrap();

    let first = store.embedding_for(id).expect("derive").expect("embedding");
    assert_eq!(first.len(), 8);
    assert_eq!(calls.load(Ordering::SeqCst), 1, "exactly one derivation");
    let second = store.embedding_for(id).expect("derive").expect("embedding");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "second read hits the cache");
    assert_eq!(second, first);
    assert_eq!(store.snapshot(id).unwrap().overlay(OverlayKind::Embedding).status, OverlayStatus::Ready);
}

#[test]
fn llm_overlay_for_derives_scrubbed_and_caches() {
    let store = temp_store();
    let backend = Arc::new(CountingBackend::new("A request to buy apple stock."));
    store.set_overlay_llm(Arc::clone(&backend) as Arc<dyn ChatBackend>);
    let id = store.record_request("s", "r1", "buy apple stock").unwrap();

    let first = store.llm_overlay_for(id).expect("derive").expect("overlay");
    assert_eq!(first, serde_json::json!("A request to buy apple stock."));
    assert_eq!(backend.calls(), 1, "exactly one LLM call");
    store.llm_overlay_for(id).expect("cache hit");
    assert_eq!(backend.calls(), 1, "second read hits the cache");
    let node = store.snapshot(id).unwrap();
    assert_eq!(node.overlay(OverlayKind::Llm).status, OverlayStatus::Ready);
    assert_eq!(
        node.metadata.as_ref().and_then(|m| m.get(LLM_OVERLAY_META_KEY)),
        Some(&serde_json::json!("A request to buy apple stock."))
    );
}

#[test]
fn llm_overlay_failure_fail_open_and_not_retried() {
    let store = temp_store();
    let backend = Arc::new(FailingChatBackend::new());
    let calls = Arc::clone(&backend.calls);
    store.set_overlay_llm(backend);
    let id = store.record_request("s", "r1", "text").unwrap();

    assert!(store.llm_overlay_for(id).expect("fail-open").is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // A permanent failure is never retried: the second call hits the
    // `failed` status and returns without another LLM call.
    assert!(store.llm_overlay_for(id).expect("fail-open").is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1, "no retry of a failed overlay");
    assert_eq!(store.snapshot(id).unwrap().overlay(OverlayKind::Llm).status, OverlayStatus::Failed);
}

#[test]
fn concurrent_annotation_for_installs_at_most_once() {
    let store = Arc::new(pipeline_store());
    let id = store.record_request("s", "r1", "Show me the sales report").unwrap();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                store.annotation_for(id).expect("derive").expect("annotation")
            })
        })
        .collect();
    let mut results: Vec<Arc<spacy_rs::ArcReadyAnnotation>> =
        handles.into_iter().map(|h| h.join().expect("thread")).collect();
    let canonical = results.pop().expect("at least one");
    assert!(
        results.iter().all(|r| Arc::ptr_eq(r, &canonical)),
        "all concurrent callers must share the one installed Arc"
    );
    let node = store.snapshot(id).unwrap();
    assert_eq!(node.overlay(OverlayKind::Spacy).status, OverlayStatus::Ready);
    assert!(node.annotation.is_some());
}

#[test]
fn concurrent_embedding_for_installs_once_under_races() {
    let store = Arc::new(temp_store());
    let embedder = Arc::new(CountingEmbedder::new(4));
    store.set_overlay_embedder(Arc::clone(&embedder) as Arc<dyn EmbeddingProvider>);
    let id = store.record_request("s", "r1", "buy apple stock").unwrap();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            std::thread::spawn(move || store.embedding_for(id).expect("derive"))
        })
        .collect();
    let results: Vec<Option<Vec<f32>>> =
        handles.into_iter().map(|h| h.join().expect("thread")).collect();
    assert!(results.iter().all(Option::is_some));
    let first = results[0].clone().unwrap();
    assert!(
        results.iter().all(|r| r.as_ref().unwrap() == &first),
        "all callers must observe the single installed value"
    );
    let node = store.snapshot(id).unwrap();
    assert_eq!(node.embedding, Some(first));
    assert_eq!(node.overlay(OverlayKind::Embedding).status, OverlayStatus::Ready);
}

#[test]
fn overlay_events_enqueue_nodes_needing_overlays() {
    let store = temp_store();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<NodeId>(16);
    store.set_overlay_events(tx);
    let id = store.record_request("s", "r1", "text").unwrap();
    // The canonical write path enqueues the node missing every overlay.
    assert_eq!(rx.try_recv().expect("enqueued"), id);
    assert!(store.needs_overlay(id));
    assert_eq!(store.node_ids_needing_overlays(), vec![id]);
}

#[test]
fn hnsw_threshold_is_adaptive() {
    let store = ContentNodeStore::ephemeral();
    assert_eq!(store.hnsw_threshold(), 512);
    assert!(!store.is_hnsw_built(), "empty store has no HNSW");
    // Insert 10 nodes with embeddings — still below threshold
    for i in 0..10 {
        let mut node = new_node(
            NodeId::from_int(0),
            "s",
            &format!("r{i}"),
            "user",
            &format!("content {i}"),
            None,
        );
        node.id = None;
        // deterministic embedding: one-hot-ish
        let mut emb = vec![0.0; 8];
        emb[(i as usize) % 8] = 1.0;
        node.embedding = Some(emb);
        store.record_content_node(&node).unwrap();
    }
    assert!(!store.is_hnsw_built(), "10 nodes must not build HNSW");
    // Brute-force path must match HNSW path for top-3 recall
    // Insert 600 nodes to cross threshold
    for i in 10..600 {
        let mut node = new_node(
            NodeId::from_int(0),
            "s",
            &format!("r{i}"),
            "user",
            &format!("content {i}"),
            None,
        );
        node.id = None;
        let mut emb = vec![0.0; 8];
        emb[(i as usize) % 8] = 1.0;
        // vary slightly so nearest is deterministic
        emb[0] += (i as f32) * 1e-4;
        node.embedding = Some(emb);
        store.record_content_node(&node).unwrap();
    }
    assert!(store.is_hnsw_built(), "600 nodes must have HNSW built");
    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let hits = store.knn_search(&query, 3);
    assert_eq!(hits.len(), 3, "top-3 must be returned");
    // All hits should be from the 0-mod-8 group (distance minimal)
    // Check that HNSW did not return wildly off results
    for h in &hits {
        assert!(h.distance < 0.5, "HNSW recall must stay close (distance <0.5)");
    }
}

#[test]
fn hnsw_must_not_fire_control_distance_gt_04() {
    let store = ContentNodeStore::ephemeral();
    for i in 0..600 {
        let mut node = new_node(
            NodeId::from_int(0),
            "s",
            &format!("r{i}"),
            "user",
            &format!("other {i}"),
            None,
        );
        node.id = None;
        // All embeddings point to +X
        node.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
        store.record_content_node(&node).unwrap();
    }
    // Query orthogonal (+Y) → cosine 0 → distance 1.0
    let hits = store.knn_search(&[0.0, 1.0, 0.0, 0.0], 1);
    assert!(!hits.is_empty());
    assert!(hits[0].distance > 0.4, "dissimilar query must stay >0.4 distance");
}

#[test]
fn node_store_scrub_is_non_bypassable() {
    let store = ContentNodeStore::open_in_memory().unwrap();
    let id = store.record_request("sess", "req", "Contact user@example.com now").unwrap();
    let node = store.snapshot(id).unwrap();
    assert_eq!(node.lod[0], "Contact [REDACTED:email] now");
    let mut n = new_node(NodeId::from_int(0), "sess2", "req2", "user", "My ssn is 123-45-6789.", None);
    n.id = None;
    n.lod[0] = "My ssn is 123-45-6789.".into();
    let id2 = store.record_content_node(&n).unwrap();
    let node2 = store.snapshot(id2).unwrap();
    assert!(node2.lod[0].contains("[REDACTED:ssn]"));
    assert!(!node2.lod[0].contains("123-45-6789"));
}

/// M9 characterization: `latest_parse_node_id` — the store-owned spelling of
/// the parse-lookup query migrated from `server/handler.rs`.
#[test]
fn latest_parse_node_id_returns_none_when_empty() {
    let store = ContentNodeStore::open_in_memory().unwrap();
    assert_eq!(store.latest_parse_node_id("nope"), None);
}

#[test]
fn latest_parse_node_id_returns_latest_per_session() {
    use crate::ledger::nlp::parse_node;
    let store = ContentNodeStore::open_in_memory().unwrap();
    // A non-parse node must not match.
    store.record_request("s1", "r0", "hello").unwrap();
    assert_eq!(store.latest_parse_node_id("s1"), None);
    let first = store
        .record_content_node(&parse_node("s1", "r1", "show me the report", &[]))
        .unwrap();
    let second = store
        .record_content_node(&parse_node("s1", "r2", "show me the other report", &[]))
        .unwrap();
    assert_eq!(store.latest_parse_node_id("s1"), Some(second));
    assert_ne!(first, second);
    // Other sessions are unaffected.
    assert_eq!(store.latest_parse_node_id("s2"), None);
}
