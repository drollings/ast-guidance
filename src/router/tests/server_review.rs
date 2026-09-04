use super::*;
use std::time::Duration;

use crate::ledger::correction_index::SqliteCorrectionIndex;
use fluent_concurrency::tokio_runtime;
use fluent_types::InterlinguaNamespace;
use spacy_rs::concept_store_mem::InMemoryConceptStore;
use spacy_rs::{AnnotationRecord, AnnotationSet, AnnotationSource};

fn parse() -> AnnotationResult {
    AnnotationResult::new(
        AnnotationSet(vec![AnnotationRecord {
            text: "the".into(),
            pos: "det".into(),
            tag: String::new(),
            dep: "det".into(),
            head: 1,
            lemma: "the".into(),
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        }]),
        AnnotationSource::ArcEager,
    )
}

fn lemma() -> InterlinguaId {
    InterlinguaId::new(InterlinguaNamespace::SpacyLemma, 1)
}

const CORRECTIONS_REPLY: &str = r#"{"corrections":[{"token_index":0,"field":"dep","old_value":"det","new_value":"nsubj"}],"linked_entities":[],"note":null}"#;

fn job(node_id: NodeId) -> ReviewJob {
    ReviewJob {
        node_id,
        session_id: "s1".into(),
        request_id: "r1".into(),
        text: "show me the report".into(),
        parse: parse(),
        patterns: vec![(lemma(), None)],
        review_model: "review-model".into(),
        pii_spans: Vec::new(),
    }
}

/// A pre-filter stub that flags the text when asked.
#[derive(Default)]
struct StubPrefilter {
    /// When `Some`, every job whose text contains the needle yields a
    /// span; `None` (the default) yields nothing.
    needle: Option<String>,
}

impl PiiSpanDetector for StubPrefilter {
    fn detect(&self, text: &str) -> Result<Vec<PiiSpan>, fluent_onnx::pii::PiiError> {
        let Some(needle) = &self.needle else {
            return Ok(Vec::new());
        };
        let Some(start) = text.find(needle.as_str()) else {
            return Ok(Vec::new());
        };
        Ok(vec![PiiSpan::new(
            start,
            start + needle.len(),
            "credential.password",
            1.0,
        )])
    }
}

/// An always-failing pre-filter (fail-open contract probe).
struct FailingPrefilter;

impl PiiSpanDetector for FailingPrefilter {
    fn detect(&self, _text: &str) -> Result<Vec<PiiSpan>, fluent_onnx::pii::PiiError> {
        Err(fluent_onnx::pii::PiiError::Inference("boom".into()))
    }
}

/// Poll the parse node until the atomic `review_status` write lands.
async fn wait_for_reviewed(ledger: &Arc<ContentNodeLedger>, node_id: NodeId) {
    for _ in 0..100 {
        let reviewed = ledger
            .get_node(node_id)
            .and_then(|n| n.metadata)
            .is_some_and(|m| m.get("review_status").is_some());
        if reviewed {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("review never landed");
}

#[tokio::test]
async fn miss_writes_correction_and_parse_review_node_atomically() {
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "show me the report")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");

    // The real router CorrectionIndex over the shared connection — the
    // production shape (F4: spacy-rs defines the trait, the router
    // implements it over its own table).
    let index = Arc::new(SqliteCorrectionIndex::new(shared.clone()));
    let concepts = Arc::new(InMemoryConceptStore::new());
    let fetched = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let f = Arc::clone(&fetched);
    let fetch: ReviewFetch = Arc::new(move |_prompt| {
        f.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(CORRECTIONS_REPLY.into())
    });

    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        None,
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));

    worker.enqueue(job(node_id)).await.expect("enqueue");
    wait_for_reviewed(&ledger, node_id).await;
    worker.drain().await;

    // H1 regression: patterns present + uncached → the review model was
    // actually consulted (never a silent no-op).
    assert_eq!(
        fetched.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the miss path must call the review fetch"
    );

    // 1. The correction pattern is durable in `interlingua_index` (the
    //    same SQLite connection as the ledger — §12.6 atomicity).
    let cached = index
        .query_previous_corrections(lemma(), None)
        .expect("cached pattern");
    assert!(!cached.is_empty(), "correction recorded on miss");

    // 2. The C7 handoff: a `parse_review` node was persisted.
    let review_count: i64 = shared
        .query_row(
            "SELECT COUNT(*) FROM ledger WHERE json_extract(metadata, '$.kind') = 'parse_review'",
            &[],
            |r| r.get(0),
        )
        .expect("count")
        .expect("row");
    assert_eq!(review_count, 1, "parse_review node written on miss");

    // 3. The `review_status` overlay landed on the parse node.
    let node = ledger.get_node(node_id).expect("parse node");
    let status = node
        .metadata
        .as_ref()
        .and_then(|m| m.get("status_note"))
        .and_then(serde_json::Value::as_str);
    assert_eq!(status, Some("reviewed"));
}

#[test]
fn candidate_concepts_resolves_parse_lemmas_and_falls_back() {
    use fluent_types::{ConceptMetadata, NodeId};
    use spacy_rs::llm::{AnnotationRecord, AnnotationSet};

    let store = InMemoryConceptStore::new();
    let report = fluent_types::InterlinguaId::new(
        InterlinguaNamespace::YagoClass,
        0x1111,
    );
    store
        .insert(ConceptMetadata {
            id: report,
            canonical_name: "report".into(),
            namespace: InterlinguaNamespace::YagoClass,
            yago_iri: Some("http://schema.org/report".into()),
            yago_class_iri: None,
            label: Some("report".into()),
            node_id: Some(NodeId::from_int(42)),
            parent_class_id: None,
        })
        .expect("insert report");
    let person = fluent_types::InterlinguaId::new(
        InterlinguaNamespace::YagoClass,
        0x2222,
    );
    store
        .insert(ConceptMetadata {
            id: person,
            canonical_name: "schema:Person".into(),
            namespace: InterlinguaNamespace::YagoClass,
            yago_iri: Some("http://schema.org/Person".into()),
            yago_class_iri: None,
            label: Some("person".into()),
            node_id: Some(NodeId::from_int(43)),
            parent_class_id: None,
        })
        .expect("insert person");

    let store_arc: Arc<dyn ConceptStore> = Arc::new(store);

    // A parse whose NOUN lemma is store-known → resolved candidates.
    let parse = AnnotationResult::new(
        AnnotationSet(vec![
            AnnotationRecord {
                text: "show".into(),
                pos: "verb".into(),
                tag: String::new(),
                dep: "root".into(),
                head: 0,
                lemma: "show".into(),
                morph: String::new(),
                ent_iob: String::new(),
                ent_type: String::new(),
            },
            AnnotationRecord {
                text: "report".into(),
                pos: "noun".into(),
                tag: String::new(),
                dep: "dobj".into(),
                head: -1,
                lemma: "report".into(),
                morph: String::new(),
                ent_iob: String::new(),
                ent_type: String::new(),
            },
        ]),
        AnnotationSource::ArcEager,
    );
    let candidates = candidate_concepts(&store_arc, &parse);
    assert_eq!(candidates.len(), 1, "the NOUN lemma resolves");
    assert_eq!(candidates[0].canonical_name, "report");

    // A parse whose NOUN lemma is unknown → bounded iter_ids fallback.
    let unknown = AnnotationResult::new(
        AnnotationSet(vec![AnnotationRecord {
            text: "gibberish".into(),
            pos: "noun".into(),
            tag: String::new(),
            dep: "dep".into(),
            head: 0,
            lemma: "gibberish".into(),
            morph: String::new(),
            ent_iob: String::new(),
            ent_type: String::new(),
        }]),
        AnnotationSource::RuleRung,
    );
    let fallback = candidate_concepts(&store_arc, &unknown);
    assert!(
        !fallback.is_empty(),
        "falls back to the bounded registered-id scan"
    );
    assert!(
        fallback.iter().any(|c| c.canonical_name == "schema:Person"),
        "the fallback surfaces registered concepts"
    );
}

#[tokio::test]
async fn reuse_skips_the_llm() {
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "show me the report")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let index = Arc::new(SqliteCorrectionIndex::new(shared));
    let concepts = Arc::new(InMemoryConceptStore::new());

    // Seed the pattern cache through the atomic miss path (a real review).
    let fetch: ReviewFetch = Arc::new(|_prompt| Ok(CORRECTIONS_REPLY.into()));
    let seed = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        None,
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));
    seed.enqueue(job(node_id)).await.expect("enqueue");
    wait_for_reviewed(&ledger, node_id).await;
    seed.drain().await;

    // A second review of the same pattern is served from the index —
    // zero LLM cost (the fetch must never fire).
    let fetched = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let f = Arc::clone(&fetched);
    let fetch2: ReviewFetch = Arc::new(move |_p| {
        f.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Err("should not be called".into())
    });
    let worker2 = Arc::new(ReviewWorker::new(
        &ledger,
        &(index as Arc<dyn CorrectionIndex>),
        &(concepts as Arc<dyn ConceptStore>),
        &fetch2,
        None,
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));
    worker2.enqueue(job(node_id)).await.expect("enqueue");
    // Give the worker time to process the (instant, reuse) job.
    tokio::time::sleep(Duration::from_millis(150)).await;
    worker2.drain().await;
    assert_eq!(
        fetched.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "reuse skips the LLM entirely"
    );
}

// ── M3: PII pre-filter seam + auto-enqueue (fail-open contract) ──

/// The enqueue path with a flagging pre-filter must still land the job
/// (never a drop), and the recorded `review_status` must carry the spans.
#[tokio::test]
async fn prefilter_adds_candidates_never_drops_a_manual_job() {
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "show me hunter2 the report")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let index = Arc::new(SqliteCorrectionIndex::new(shared));
    let concepts = Arc::new(InMemoryConceptStore::new());
    let fetch: ReviewFetch = Arc::new(|_prompt| Ok(CORRECTIONS_REPLY.into()));
    let mut flagged = job(node_id);
    flagged.text = "show me hunter2 the report".into();
    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        Some(Arc::new(StubPrefilter {
            needle: Some("hunter2".into()),
        })),
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));

    // Manual enqueue always proceeds with a pre-filter present.
    worker.enqueue(flagged).await.expect("manual enqueue proceeds");
    wait_for_reviewed(&ledger, node_id).await;
    worker.drain().await;

    let node = ledger.get_node(node_id).expect("parse node");
    let status = node
        .metadata
        .as_ref()
        .and_then(|m| m.get("review_status"))
        .expect("review_status");
    let spans = status
        .get("Reviewed")
        .and_then(|r| r.get("pii_spans"))
        .and_then(|v| v.as_array())
        .expect("pii_spans recorded");
    assert_eq!(spans.len(), 1, "pre-filter candidate recorded");
    assert_eq!(spans[0]["label"], "credential.password");
    assert_eq!(spans[0]["start"], 8);
    assert_eq!(spans[0]["end"], 15);
}

#[tokio::test]
async fn failing_prefilter_never_blocks_manual_enqueue() {
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "show me the report")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let index = Arc::new(SqliteCorrectionIndex::new(shared));
    let concepts = Arc::new(InMemoryConceptStore::new());
    let fetch: ReviewFetch = Arc::new(|_prompt| Ok(CORRECTIONS_REPLY.into()));
    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        Some(Arc::new(FailingPrefilter)),
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));
    worker.enqueue(job(node_id)).await.expect("fail-open: still enqueues");
    wait_for_reviewed(&ledger, node_id).await;
    worker.drain().await;
    let node = ledger.get_node(node_id).expect("parse node");
    assert!(
        node.metadata
            .as_ref()
            .and_then(|m| m.get("review_status"))
            .is_some(),
        "a failing pre-filter degrades to 'no spans', never a drop"
    );
}

#[tokio::test]
async fn auto_enqueue_flags_spans_and_enqueues_through_the_credit_gate() {
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "hunter2 is a secret")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let index = Arc::new(SqliteCorrectionIndex::new(shared));
    let concepts = Arc::new(InMemoryConceptStore::new());
    let fetch: ReviewFetch = Arc::new(|_prompt| Ok(CORRECTIONS_REPLY.into()));
    let mut flagged = job(node_id);
    flagged.text = "hunter2 is a secret".into();
    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        Some(Arc::new(StubPrefilter {
            needle: Some("hunter2".into()),
        })),
        true, // auto_enqueue
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));

    let enqueued = worker.maybe_auto_enqueue(flagged).await;
    assert!(enqueued, "flagged content auto-enqueues");
    wait_for_reviewed(&ledger, node_id).await;
    worker.drain().await;
    let node = ledger.get_node(node_id).expect("parse node");
    let status = node.metadata.as_ref().and_then(|m| m.get("review_status"));
    assert!(status.is_some(), "the auto-enqueued job reviewed the parse");
}

#[tokio::test]
async fn auto_enqueue_does_not_fire_without_spans_or_when_disabled() {
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "nothing sensitive")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let index = Arc::new(SqliteCorrectionIndex::new(shared));
    let concepts = Arc::new(InMemoryConceptStore::new());
    let fetch: ReviewFetch = Arc::new(|_prompt| Ok(CORRECTIONS_REPLY.into()));

    // Enabled pre-filter but no match → nothing enqueued.
    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        Some(Arc::new(StubPrefilter {
            needle: Some("hunter2".into()),
        })),
        true,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));
    let plain = job(node_id);
    assert!(
        !worker.maybe_auto_enqueue(plain).await,
        "no spans → no auto-enqueue"
    );

    // A flagging pre-filter but auto_enqueue disabled → nothing enqueued.
    let worker2 = Arc::new(ReviewWorker::new(
        &ledger,
        &(index as Arc<dyn CorrectionIndex>),
        &(concepts as Arc<dyn ConceptStore>),
        &fetch,
        Some(Arc::new(StubPrefilter {
            needle: Some("hunter2".into()),
        })),
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));
    let mut flagged = job(node_id);
    flagged.text = "hunter2 here".into();
    assert!(
        !worker2.maybe_auto_enqueue(flagged).await,
        "auto_enqueue off → no auto-enqueue"
    );

    worker.drain().await;
    worker2.drain().await;
}

// ── M6.5: candidate promotion via the linked-entities handoff ──────

#[tokio::test]
async fn review_promotes_matching_entity_link_candidates() {
    use crate::ledger::overlay::{OverlayCandidateStore, CandidateStatus};

    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "who is Ada Lovelace?")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");
    let index = Arc::new(SqliteCorrectionIndex::new(shared.clone()));
    let concepts = Arc::new(InMemoryConceptStore::new());

    // Seed an entity-link candidate that the review will link.
    let linked_id = InterlinguaId::from_u64(0x0200_0000_0000_0001);
    let other_id = InterlinguaId::from_u64(0x0200_0000_0000_0002);
    let candidates = OverlayCandidateStore::new(shared.clone());
    candidates
        .write_candidate(&crate::ledger::overlay::OverlayCandidate::entity_link(
            node_id,
            0,
            4,
            linked_id,
            0.9,
            "entity_link",
        ))
        .expect("write linked candidate");
    candidates
        .write_candidate(&crate::ledger::overlay::OverlayCandidate::entity_link(
            node_id,
            8,
            12,
            other_id,
            0.8,
            "entity_link",
        ))
        .expect("write other candidate");

    // The review model links the entity (a non-empty linked_entities reply).
    let reply = r#"{"corrections":[],"linked_entities":[
        {"token_start":0,"token_end":1,"entity_type":"person","interlingua_id":131072,"confidence":0.95}
    ],"note":null}"#;
    // interlingua_id is the JSON as_i64 of 0x0200_0000_0000_0001.
    let reply = reply.replace("131072", &(linked_id.as_i64()).to_string());
    let fetch: ReviewFetch = Arc::new(move |_p| Ok(reply.clone()));
    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        None,
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));
    worker.enqueue(job(node_id)).await.expect("enqueue");
    wait_for_reviewed(&ledger, node_id).await;
    worker.drain().await;

    let rows = candidates.for_node(node_id).expect("query");
    let linked = rows.iter().find(|r| r.entity_id == Some(linked_id)).unwrap();
    assert_eq!(
        linked.status,
        CandidateStatus::Promoted,
        "the review's linked entity promotes the matching candidate"
    );
    let other = rows.iter().find(|r| r.entity_id == Some(other_id)).unwrap();
    assert_eq!(other.status, CandidateStatus::Pending);
}

#[tokio::test]
async fn fenced_reply_recovers_corrections() {
    // M2.7: the intended widening — a fence-wrapped LLM reply is recovered
    // by the tolerant codec (previously fail-open with no corrections).
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "show me the report")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");

    let index = Arc::new(SqliteCorrectionIndex::new(shared.clone()));
    let concepts = Arc::new(InMemoryConceptStore::new());
    let fenced = format!("Sure, here is the review:\n```json\n{CORRECTIONS_REPLY}\n```");
    let fetch: ReviewFetch = Arc::new(move |_prompt| Ok(fenced.clone()));

    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        None,
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));

    worker.enqueue(job(node_id)).await.expect("enqueue");
    wait_for_reviewed(&ledger, node_id).await;
    worker.drain().await;

    let cached = index
        .query_previous_corrections(lemma(), None)
        .unwrap_or_default();
    assert!(!cached.is_empty(), "fenced reply recovers the correction");
}

#[tokio::test]
async fn garbage_reply_fails_open_with_no_corrections() {
    // Characterization (M2.1): an unparseable LLM reply is fail-open today —
    // no corrections, no parse_review node, but the parse is still marked
    // reviewed. Must pass unchanged after the tolerant-codec migration
    // (pure garbage has no JSON value for the codec to recover).
    let ledger = Arc::new(ContentNodeLedger::open_in_memory().expect("ledger"));
    let node_id = ledger
        .record_request("s1", "r1", "show me the report")
        .expect("parse node");
    let shared = ledger.node_store().shared_sqlite().expect("shared sqlite");

    let index = Arc::new(SqliteCorrectionIndex::new(shared.clone()));
    let concepts = Arc::new(InMemoryConceptStore::new());
    let fetch: ReviewFetch = Arc::new(|_prompt| Ok("definitely not json {{{".into()));

    let worker = Arc::new(ReviewWorker::new(
        &ledger,
        &(Arc::clone(&index) as Arc<dyn CorrectionIndex>),
        &(Arc::clone(&concepts) as Arc<dyn ConceptStore>),
        &fetch,
        None,
        false,
        "review-model".into(),
        8,
        4,
        tokio_runtime(),
    ));

    worker.enqueue(job(node_id)).await.expect("enqueue");
    wait_for_reviewed(&ledger, node_id).await;
    worker.drain().await;

    let cached = index
        .query_previous_corrections(lemma(), None)
        .unwrap_or_default();
    assert!(cached.is_empty(), "garbage reply records no corrections");

    let review_count: i64 = shared
        .query_row(
            "SELECT COUNT(*) FROM ledger WHERE json_extract(metadata, '$.kind') = 'parse_review'",
            &[],
            |r| r.get(0),
        )
        .expect("count")
        .expect("row");
    assert_eq!(review_count, 0, "no parse_review node for empty review");

    let node = ledger.get_node(node_id).expect("parse node");
    let status = node
        .metadata
        .as_ref()
        .and_then(|m| m.get("status_note"))
        .and_then(serde_json::Value::as_str);
    assert_eq!(status, Some("reviewed"), "fail-open still marks reviewed");
}
