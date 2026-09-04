use super::*;
use crate::dag_session::SessionRegistry;
use crate::kv_cache::{ColdSnapshotIndex, HotSnapshotIndex};
use crate::ledger::tiering::TierConfig;
use crate::test_stubs::StubChatBackend;

fn test_registry() -> SessionRegistry {
    SessionRegistry::with_kv_cache(fork_kv())
}

fn temp_store() -> Arc<ContentNodeStore> {
    let dir = std::env::temp_dir().join(format!(
        "coral-router-orch-{}",
        common_core::hash::uuid_v4()
    ));
    let store = Arc::new(ContentNodeStore::open(&dir).unwrap());
    let _ = std::fs::remove_file(&dir);
    store
}

/// A fork-enabled `SnapshotStore` (records snapshot metadata via a
/// `StubServer` fork handle) so the coordinator's snapshot/restore round-trip
/// is real.
fn fork_kv() -> SnapshotStore {
    use crate::instances::stub::StubServer;
    use crate::instances::InstanceClient;

    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |method, path, _body| {
            if method == "POST" && path.ends_with("/snapshot") {
                (200, "{}".into())
            } else {
                (200, "[]".into())
            }
        },
    );
    let stub = StubServer::start(handler);
    let fork = Arc::new(InstanceClient::new(
        reqwest::Client::new(),
        stub.base_url(),
        None,
    ));
    let dir = tempfile::tempdir().unwrap();
    let hot = Arc::new(HotSnapshotIndex::new(10, 1024));
    SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            dir.path(),
            1024,
            86400,
        )),
    )
    .with_fork_io(fork)
}

/// A coordinator whose tier worker is started and whose sessions share a
/// fork-enabled KV cache (so snapshot metadata is recorded and restore
/// works).
fn coordinator(
    store: Arc<ContentNodeStore>,
    sessions: Arc<SessionRegistry>,
    backend: Arc<dyn ChatBackend>,
    kv_policy: KvSnapshotPolicy,
) -> (LedgerAgentCoordinator, Arc<LedgerTierWorker>) {
    let tiers = LedgerTierWorker::new(
        Arc::clone(&store),
        Arc::new(StubChatBackend::always("SUMMARY: s\nDESCRIPTION: d")),
        vec![4, 5],
        TierConfig {
            poll_interval_ms: 5,
            ..Default::default()
        },
        fluent_concurrency::tokio_runtime(),
    );
    let kv = sessions.kv_cache().clone();
    let coordinator = LedgerAgentCoordinator::new(
        store,
        sessions,
        kv,
        Arc::clone(&tiers),
        LedgerPromptAssembler,
        backend,
        OrchestratorConfig {
            kv_policy,
            ..Default::default()
        },
    );
    (coordinator, tiers)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_model_handoff_restores_same_model_only() {
    let store = temp_store();
    let sessions = Arc::new(test_registry());
    let backend: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::new(vec!["A1".to_string(), "B1".to_string(), "A2".to_string()]));
    let (coord, tiers) = coordinator(
        Arc::clone(&store),
        Arc::clone(&sessions),
        backend,
        KvSnapshotPolicy::RestoreIfSameModel,
    );
    store.set_tier_events(tiers.sender());
    let handle = tiers.start();
    let worker = WorkerContext::new("assistant", "help");

    // Model A runs (re-prefill), records a node, snapshots KV under (A).
    let a1 = coord
        .run_agent("sess", "model-a", &worker, "task 1")
        .await
        .unwrap();
    assert!(!a1.kv_restored, "first turn always re-prefills");

    // Model B runs (different model -> re-prefill, kv_restored = false),
    // records a node, snapshots KV under (B).
    let b1 = coord
        .run_agent("sess", "model-b", &worker, "task 2")
        .await
        .unwrap();
    assert!(!b1.kv_restored, "different model re-prefills");

    // Model A runs again (same model -> restores A snapshot).
    let a2 = coord
        .run_agent("sess", "model-a", &worker, "task 3")
        .await
        .unwrap();
    assert!(
        a2.kv_restored,
        "same model re-entry restores its own KV snapshot"
    );
    assert_eq!(a2.budget_used, 0, "restore sends no assembled prompt");

    // Both per-model snapshots coexist under their keys; the session's
    // pending snapshot is the most recent (model-a, after A ran last).
    let session = sessions.get_or_create("sess");
    let guard = session.lock().unwrap();
    assert!(guard.pending_snapshot().is_some());
    assert_eq!(
        guard.pending_snapshot().unwrap().model,
        "model-a",
        "pending snapshot is the most recent model's"
    );

    // The ledger records: 3 agent nodes + 3 checkpoint nodes (one per turn).
    let agent_ids = store.nodes_for_role("agent");
    let checkpoint_ids = store.nodes_for_role("checkpoint");
    assert_eq!(agent_ids.len(), 3, "one agent node per turn");
    assert_eq!(checkpoint_ids.len(), 3, "one checkpoint node per turn");

    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn never_restore_always_rep_prefills() {
    let store = temp_store();
    let sessions = Arc::new(test_registry());
    let backend: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::new(vec!["x".into(), "y".into(), "z".into()]));
    let (coord, tiers) = coordinator(
        Arc::clone(&store),
        Arc::clone(&sessions),
        backend,
        KvSnapshotPolicy::NeverRestore,
    );
    store.set_tier_events(tiers.sender());
    let handle = tiers.start();
    let worker = WorkerContext::new("assistant", "");

    let _ = coord.run_agent("sess", "model-a", &worker, "1").await.unwrap();
    let again = coord.run_agent("sess", "model-a", &worker, "2").await.unwrap();
    assert!(
        !again.kv_restored,
        "NeverRestore ignores the pending snapshot and re-prefills"
    );

    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn records_nodes_and_enqueues_tiers() {
    let store = temp_store();
    let sessions = Arc::new(test_registry());
    let backend: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::new(vec!["out-1".into(), "out-2".into()]));
    let (coord, tiers) = coordinator(
        Arc::clone(&store),
        Arc::clone(&sessions),
        backend,
        KvSnapshotPolicy::RestoreIfSameModel,
    );
    store.set_tier_events(tiers.sender());
    let handle = tiers.start();
    let worker = WorkerContext::new("assistant", "");

    let o1 = coord.run_agent("sess", "model-a", &worker, "hi").await.unwrap();
    let node = store.snapshot(o1.node_id).unwrap();
    assert_eq!(node.lod[0], "out-1");
    assert_eq!(node.role.as_ref().map(|r| r.as_str()), Some("agent"));
    assert_eq!(node.step_id.as_deref(), Some("model-a-1"));

    // LOD4/LOD5 are enqueued and filled in the background (proves the
    // coordinator enqueued every recorded node).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        let n = store.snapshot(o1.node_id).unwrap();
        if !n.lod[4].is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(
        !store.snapshot(o1.node_id).unwrap().lod[4].is_empty(),
        "recorded node enqueued + background-filled"
    );

    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembles_prompt_from_ledger_context() {
    // The coordinator's re-prefill path must assemble the ledger context
    // (not just the input) — observable via the assembled body reaching the
    // backend. Use a RecordingBackend to capture the user message.
    let store = temp_store();
    store
        .record_request("sess", "r1", "Prior ledger context node.")
        .unwrap();
    let sessions = Arc::new(test_registry());

    let captured = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let capture = Arc::clone(&captured);
    let backend: Arc<dyn ChatBackend> = Arc::new(RecordingBackend { captured: capture });
    let (coord, tiers) = coordinator(
        Arc::clone(&store),
        sessions,
        backend,
        KvSnapshotPolicy::RestoreIfSameModel,
    );
    store.set_tier_events(tiers.sender());
    let handle = tiers.start();
    let worker = WorkerContext::new("assistant", "");

    coord
        .run_agent("sess", "model-a", &worker, "New request text")
        .await
        .unwrap();

    let msgs = captured.lock().unwrap().clone();
    let joined = msgs.join("\n");
    assert!(
        joined.contains("Prior ledger context node."),
        "assembled prompt includes prior ledger context, got: {joined}"
    );
    assert!(
        joined.contains("New request text"),
        "assembled prompt includes the new request"
    );

    handle.abort();
}

/// Records every user message it receives.
struct RecordingBackend {
    captured: Arc<std::sync::Mutex<Vec<String>>>,
}

impl ChatBackend for RecordingBackend {
    fn chat_complete(
        &self,
        messages: &[fluent_llm::ChatMessage],
    ) -> Result<String, LlmError> {
        self.captured.lock().unwrap().extend(
            messages
                .iter()
                .filter(|m| m.role == "user")
                .map(|m| m.content.clone()),
        );
        Ok("recorded output".into())
    }
}

#[test]
fn lod_spec_and_role_defaults() {
    let cfg = OrchestratorConfig::default();
    assert_eq!(cfg.role, "agent");
    assert_eq!(cfg.kv_policy, KvSnapshotPolicy::RestoreIfSameModel);
    assert!(cfg.budget.max_chars > 0);
    assert_eq!(cfg.lod_spec, LodSpec::full());
}

/// A deterministic embedder for the retrieval-seam test (identity-dim
/// embeddings: paraphrase equivalents share a dimension).
struct TestEmbedder;

impl spacy_rs::retrieval::EmbeddingProvider for TestEmbedder {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let mut v = vec![0.0f32; 8];
        for tok in text.split_whitespace() {
            let dim = match tok.to_lowercase().as_str() {
                "show" | "display" | "get" => 0usize,
                "report" | "sales" => 1,
                other => (spacy_rs::hash::hash_utf8(other) % 8) as usize,
            };
            v[dim] += 1.0;
        }
        Some(v)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retrieve_nodes_composes_the_live_retrieval_seam() {
    let store = temp_store();
    let sessions = Arc::new(test_registry());
    let backend: Arc<dyn ChatBackend> = Arc::new(StubChatBackend::new(vec![]));
    let (coord, tiers) = coordinator(
        Arc::clone(&store),
        sessions,
        backend,
        KvSnapshotPolicy::RestoreIfSameModel,
    );
    store.set_tier_events(tiers.sender());

    // Uncomposed → fail-open empty.
    let no_service = coord.retrieve_nodes("show", &[]);
    assert!(no_service.is_empty(), "no service → empty, not an error");

    // Composed → the M5 tools run over the candidate nodes.
    let svc = Arc::new(
        crate::retrieval::NodeRetrievalService::new(
            Arc::new(TestEmbedder),
            None,
        )
        .expect("retrieval service"),
    );
    let coord = coord.with_retrieval(svc);
    let nodes = vec![ContentNode {
        id: Some(NodeId::from_int(1)),
        lod: vec!["Show me the report".to_string()],
        ..Default::default()
    }];
    let reports = coord.retrieve_nodes("show", &nodes);
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].node_id, NodeId::from_int(1));
    assert!(
        reports[0]
            .hits
            .iter()
            .any(|h| h.source == spacy_rs::retrieval::RetrievalSource::LemmaGrep
                && h.parse_confidence.is_some()),
        "lemma-grep hits carry their parse confidence"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn agent_node_records_node_plan_for_learning_replay() {
    // On a re-prefill turn, the recorded agent node's metadata embeds
    // the assembled node_plan (node→Lod) so a future workflow-extraction
    // pass can replay the same decomposition.
    let store = temp_store();
    let prior = store
        .record_request("sess", "r0", "Prior context node text")
        .unwrap();
    let sessions = Arc::new(test_registry());
    let backend: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::new(vec!["out".to_string()]));
    let (coord, tiers) = coordinator(
        Arc::clone(&store),
        sessions,
        backend,
        KvSnapshotPolicy::RestoreIfSameModel,
    );
    store.set_tier_events(tiers.sender());
    let handle = tiers.start();
    let worker = WorkerContext::new("assistant", "");

    let outcome = coord
        .run_agent("sess", "model-a", &worker, "hi")
        .await
        .unwrap();
    assert!(!outcome.kv_restored, "re-prefill on first turn");

    let node = store.snapshot(outcome.node_id).unwrap();
    let meta = node.metadata.expect("metadata present");
    let plan = meta["node_plan"].as_array().expect("node_plan array");
    assert!(
        plan.iter().any(|pair| pair[0].as_i64() == Some(prior.as_int())),
        "node_plan must include the prior ledger node (anchored at LOD0), got: {plan:?}"
    );
    assert!(
        plan.iter().all(|pair| pair[1].as_u64() == Some(0)),
        "single prior node is the LOD0 anchor, got: {plan:?}"
    );

    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn affinity_tracks_current_session_through_run_agent() {
    // Composing an `AffinityScheduler` must mark the active session
    // as the currently-affine one and submit its turn identity through the
    // shared scheduler (minimize context switches), without changing the
    // restore decision or the transport.
    let store = temp_store();
    let sessions = Arc::new(test_registry());
    let backend: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::new(vec!["a".into(), "b".into(), "c".into()]));
    let (coord, tiers) = coordinator(
        Arc::clone(&store),
        Arc::clone(&sessions),
        backend,
        KvSnapshotPolicy::RestoreIfSameModel,
    );
    let scheduler = LedgerAgentCoordinator::build_affinity_scheduler(2);
    let coord = coord.with_affinity(scheduler);
    store.set_tier_events(tiers.sender());
    let handle = tiers.start();
    let worker = WorkerContext::new("assistant", "");

    let _ = coord.run_agent("sess", "model-a", &worker, "1").await.unwrap();
    let _ = coord.run_agent("sess", "model-a", &worker, "2").await.unwrap();

    // The composed scheduler marks the active session as affine (minimize
    // context switches) across interleaved sessions.
    assert_eq!(
        coord.affinity_session().as_deref(),
        Some("sess"),
        "run_agent marks the session as currently KV-affine"
    );

    // A same-model re-entry still restores KV (affinity bookkeeping must
    // not alter the restore decision or the transport).
    let outcome = coord.run_agent("sess", "model-a", &worker, "3").await.unwrap();
    assert!(outcome.kv_restored);

    handle.abort();
}

/// The config-driven boot path: a coordinator built by
/// `RouterConfig::build_ledger_coordinator` with
/// `ledger.orchestrator.affinity_cap` set attaches the KV-affinity
/// scheduler, so `affinity_session()` reflects the session after a
/// `run_agent` (and stays `None` without the opt-in).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn affinity_cap_config_attaches_scheduler() {
    let store = temp_store();
    let sessions = Arc::new(test_registry());
    let backend: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::new(vec!["a".into(), "b".into()]));
    let tiers = Arc::new(LedgerTierWorker::new(
        Arc::clone(&store),
        Arc::new(StubChatBackend::always("SUMMARY: s\nDESCRIPTION: d")),
        vec![4, 5],
        TierConfig {
            poll_interval_ms: 5,
            ..Default::default()
        },
        fluent_concurrency::tokio_runtime(),
    ));
    let config: crate::config::RouterConfig = serde_json::from_str(
        r#"{
            "ledger": { "orchestrator": { "enabled": true, "affinity_cap": 2 } }
        }"#,
    )
    .expect("valid config with affinity_cap");
    let coord = config
        .build_ledger_coordinator(
            Arc::clone(&store),
            Arc::clone(&sessions),
            sessions.kv_cache().clone(),
            Arc::clone(&tiers),
            backend,
        )
        .expect("coordinator built from the config section");
    store.set_tier_events(tiers.sender());
    let handle = tiers.start();
    let worker = WorkerContext::new("assistant", "");

    assert_eq!(coord.affinity_session(), None, "no run yet -> no affine session");
    let _ = coord.run_agent("sess", "model-a", &worker, "1").await.unwrap();
    assert_eq!(
        coord.affinity_session().as_deref(),
        Some("sess"),
        "affinity_cap-attached scheduler marks the run_agent session"
    );

    handle.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preferred_model_prefers_last_resident_and_falls_back() {
    // The coordinator derives the last-resident (KV-affine) model
    // from the durable `checkpoint` ledger nodes and prefers it, falling
    // back to re-prefill (None) only on a true capability gap.
    let store = temp_store();
    let sessions = Arc::new(test_registry());
    let backend: Arc<dyn ChatBackend> =
        Arc::new(StubChatBackend::new(vec!["a".into(), "b".into()]));
    let (coord, tiers) = coordinator(
        Arc::clone(&store),
        Arc::clone(&sessions),
        backend,
        KvSnapshotPolicy::RestoreIfSameModel,
    );
    store.set_tier_events(tiers.sender());
    let handle = tiers.start();
    let worker = WorkerContext::new("assistant", "");

    // Empty session: no resident instance -> no preference (re-prefill).
    assert_eq!(coord.last_resident_model("sess"), None);
    assert_eq!(coord.preferred_model("sess", &["model-a"]), None);

    // Model A runs (records a checkpoint node), then Model B runs (the most
    // recent checkpoint node is now Model B's).
    let _ = coord.run_agent("sess", "model-a", &worker, "1").await.unwrap();
    let _ = coord.run_agent("sess", "model-b", &worker, "2").await.unwrap();

    assert_eq!(
        coord.last_resident_model("sess").as_deref(),
        Some("model-b"),
        "last-resident model is the most recent checkpoint"
    );
    assert_eq!(
        coord.preferred_model("sess", &["model-a", "model-b"]).as_deref(),
        Some("model-b"),
        "prefers the resident candidate (KV affinity)"
    );
    assert_eq!(
        coord.preferred_model("sess", &["model-a"]),
        None,
        "no resident candidate -> re-prefill (capability gap)"
    );

    // Model B re-enters and wins affinity again (still resident).
    assert_eq!(
        coord.preferred_model("sess", &["model-b", "model-a"]).as_deref(),
        Some("model-b")
    );

    handle.abort();
}
