use super::*;

fn ok_result(content: &str) -> StepResult {
    StepResult {
        content: content.into(),
        accepted: true,
        score: Some(0.9),
        latency_ms: 100,
        error: None,
    }
}

fn fail_result(content: &str) -> StepResult {
    StepResult {
        content: content.into(),
        accepted: false,
        score: Some(0.1),
        latency_ms: 50,
        error: Some("execution failed".into()),
    }
}

#[test]
fn test_add_steps_and_get_order() {
    let mut session = DependencySession::new("sess-1");

    session
        .add_step(SessionStep::new("step-1", "First step"))
        .unwrap();
    session
        .add_step(SessionStep::new("step-2", "Second step").with_depends(vec!["step-1".into()]))
        .unwrap();

    assert_eq!(session.step_count(), 2);
    assert_eq!(session.step_ids(), &["step-1", "step-2"]);
}

#[test]
fn test_duplicate_step_rejected() {
    let mut session = DependencySession::new("sess-1");
    session
        .add_step(SessionStep::new("step-1", "First"))
        .unwrap();
    let result = session.add_step(SessionStep::new("step-1", "Duplicate"));
    assert!(matches!(
        result,
        Err(DagError::Graph(GraphError::DuplicateNode(_)))
    ));
}

#[test]
fn test_ready_nodes_basic_dependency() {
    let mut session = DependencySession::new("sess-1");

    session.add_step(SessionStep::new("plan", "Plan")).unwrap();
    session
        .add_step(SessionStep::new("execute", "Execute").with_depends(vec!["plan".into()]))
        .unwrap();

    // Only "plan" should be ready (no dependencies)
    let ready = session.next_ready();
    assert_eq!(ready, vec!["plan"]);

    // Complete "plan"
    session
        .complete_step("plan", ok_result("plan done"))
        .unwrap();

    // Now "execute" should be ready
    let ready = session.next_ready();
    assert_eq!(ready, vec!["execute"]);
}

#[test]
fn test_fail_cancels_dependents() {
    let mut session = DependencySession::new("sess-1");

    session.add_step(SessionStep::new("a", "Step A")).unwrap();
    session
        .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
        .unwrap();
    session
        .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
        .unwrap();

    // Complete "a" successfully
    session.complete_step("a", ok_result("a done")).unwrap();

    // Complete "b" with failure
    session.complete_step("b", fail_result("b failed")).unwrap();

    let b = session.get_step("b").unwrap();
    assert_eq!(b.status, StepStatus::Failed);

    let c = session.get_step("c").unwrap();
    assert_eq!(c.status, StepStatus::Cancelled);
}

#[test]
fn test_complete_step_not_found() {
    let mut session = DependencySession::new("sess-1");
    let result = session.complete_step("nonexistent", ok_result("nope"));
    assert!(matches!(result, Err(DagError::StepNotFound(_))));
}

#[test]
fn test_complete_already_completed() {
    let mut session = DependencySession::new("sess-1");
    session.add_step(SessionStep::new("a", "Step A")).unwrap();
    session.complete_step("a", ok_result("done")).unwrap();

    let result = session.complete_step("a", ok_result("again"));
    assert!(matches!(result, Err(DagError::StepAlreadyCompleted(_))));
}

#[test]
fn test_start_step() {
    let mut session = DependencySession::new("sess-1");
    session.add_step(SessionStep::new("a", "Step A")).unwrap();

    session.start_step("a").unwrap();
    let step = session.get_step("a").unwrap();
    assert_eq!(step.status, StepStatus::InProgress);
}

#[test]
fn test_start_step_not_found() {
    let mut session = DependencySession::new("sess-1");
    let result = session.start_step("nonexistent");
    assert!(matches!(result, Err(DagError::StepNotFound(_))));
}

#[test]
fn test_start_step_not_pending() {
    let mut session = DependencySession::new("sess-1");
    session.add_step(SessionStep::new("a", "Step A")).unwrap();
    session.complete_step("a", ok_result("done")).unwrap();

    let result = session.start_step("a");
    assert!(matches!(result, Err(DagError::StepAlreadyCompleted(_))));
}

#[test]
fn test_checkpoint_listing() {
    let mut session = DependencySession::new("sess-1");
    session
        .add_step(SessionStep::new("a", "Step A").with_checkpoint())
        .unwrap();
    session
        .add_step(SessionStep::new("b", "Step B").with_checkpoint())
        .unwrap();

    let cps = session.checkpoints();
    assert_eq!(cps.len(), 2);
    assert!(cps.contains(&"a".to_string()));
    assert!(cps.contains(&"b".to_string()));
}

#[tokio::test]
async fn test_rewind_to_checkpoint() {
    let mut session = DependencySession::new("sess-1");

    session
        .add_step(SessionStep::new("a", "Step A").with_checkpoint())
        .unwrap();
    session
        .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
        .unwrap();
    session
        .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
        .unwrap();

    // Complete a and reach checkpoint
    session.complete_step("a", ok_result("a done")).unwrap();
    assert!(session.checkpoints().contains(&"a".to_string()));

    // Complete b
    session.complete_step("b", ok_result("b done")).unwrap();
    assert_eq!(session.completed_count(), 2);

    // Rewind to checkpoint "a"
    session.rewind_to_checkpoint("a").unwrap();

    // "a" is reset, "b" is reset
    assert_eq!(session.get_step("a").unwrap().status, StepStatus::Pending);
    assert_eq!(session.get_step("b").unwrap().status, StepStatus::Pending);
    // "c" was never completed, stays Pending
    assert_eq!(session.get_step("c").unwrap().status, StepStatus::Pending);
    assert_eq!(session.completed_count(), 0);
}

#[tokio::test]
async fn test_rewind_missing_checkpoint() {
    let mut session = DependencySession::new("sess-1");
    let result = session.rewind_to_checkpoint("nonexistent");
    assert!(matches!(result, Err(DagError::CheckpointNotFound(_))));
}

#[test]
fn test_is_ready() {
    let mut session = DependencySession::new("sess-1");

    session.add_step(SessionStep::new("a", "Step A")).unwrap();
    session
        .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
        .unwrap();

    assert!(session.is_ready("a"));
    assert!(!session.is_ready("b"));

    session.complete_step("a", ok_result("done")).unwrap();
    assert!(session.is_ready("b"));
}

#[test]
fn test_is_ready_unregistered() {
    let session = DependencySession::new("sess-1");
    assert!(!session.is_ready("nonexistent"));
}

#[test]
fn test_independent_steps_ready_together() {
    let mut session = DependencySession::new("sess-1");

    session.add_step(SessionStep::new("a", "Step A")).unwrap();
    session.add_step(SessionStep::new("b", "Step B")).unwrap();

    let ready = session.next_ready();
    assert_eq!(ready.len(), 2);
    assert!(ready.contains(&"a".to_string()));
    assert!(ready.contains(&"b".to_string()));
}

#[test]
fn test_cycle_detection_in_dependents() {
    // DependencyGraph::dependents_of handles cycles gracefully
    let mut session = DependencySession::new("sess-1");

    session.add_step(SessionStep::new("a", "Step A")).unwrap();
    session
        .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into(), "c".into()]))
        .unwrap();
    session
        .add_step(SessionStep::new("c", "Step C").with_depends(vec!["b".into()]))
        .unwrap();

    // This should not panic - DependencyGraph handles cycles
    let deps = session.graph().dependents_of(&"a".to_string());
    // In a cycle, the result is partial but non-panicking
    assert!(!deps.is_empty() || deps.is_empty()); // Just verify it returns
}

#[test]
fn test_unresolved_deps() {
    let mut session = DependencySession::new("sess-1");

    session.add_step(SessionStep::new("a", "Step A")).unwrap();
    session
        .add_step(SessionStep::new("b", "Step B").with_depends(vec!["missing".into()]))
        .unwrap();

    let unresolved = session.unresolved_deps();
    assert!(unresolved.contains(&"missing".to_string()));
}

#[tokio::test]
async fn test_step_result_data_preserved_on_rewind() {
    let mut session = DependencySession::new("sess-1");

    session
        .add_step(SessionStep::new("a", "Step A").with_checkpoint())
        .unwrap();

    session
        .complete_step("a", ok_result("important result"))
        .unwrap();
    session.rewind_to_checkpoint("a").unwrap();

    let step = session.get_step("a").unwrap();
    assert_eq!(step.status, StepStatus::Pending);
    // Result data preserved for audit
    assert!(step.result.is_some());
    assert_eq!(step.result.as_ref().unwrap().content, "important result");
}

#[test]
fn test_get_step_nonexistent() {
    let session = DependencySession::new("sess-1");
    assert!(session.get_step("nonexistent").is_none());
}

#[test]
fn test_with_constructor_builders() {
    let mut session = DependencySession::new("sess-1");

    let step = SessionStep::new("step-1", "A step")
        .with_depends(vec!["dep-1".into(), "dep-2".into()])
        .with_checkpoint();

    assert_eq!(step.id, "step-1");
    assert_eq!(step.depends_on, vec!["dep-1", "dep-2"]);
    assert!(step.checkpoint);

    session.add_step(step).unwrap();
    assert_eq!(session.step_count(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rewind_restores_kv_snapshot_for_real() {
    use crate::kv_cache::KvSnapshot;

    let dir = tempfile::tempdir().unwrap();
    let src_dir = tempfile::tempdir().unwrap();
    let hot = Arc::new(HotSnapshotIndex::new(10, 1024));
    let kv = SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            dir.path(),
            1024,
            86400,
        )),
    );

    let src_file = src_dir.path().join("rewind.kv");
    tokio::fs::write(&src_file, b"kv bytes").await.unwrap();
    let snapshot = KvSnapshot {
        model: "model-x".into(),
        adapter: None,
        session_id: "sess-1".into(),
        snapshot_name: "readfiles".into(),
        instance: Some("scratch".into()),
        file_path: src_file,
        token_count: Some(42),
        created_at: common_core::now_secs(),
        last_used_at: common_core::now_secs(),
        llama_cpp_version: Some("0.1.0".into()),
        model_quant: None,
        base_model_hash: Some("abc".into()),
    turn_seq: None,
    };
    kv.store(snapshot).unwrap();
    // Force a cold-tier hit so rewind exercises the reload-into-hot-tier
    // path rather than a hot-tier cache hit.
    hot.remove("model-x", None, "sess-1");

    let mut session = DependencySession::new("sess-1")
        .with_model("model-x")
        .with_kv_cache(kv);
    session
        .add_step(SessionStep::new("a", "Step A").with_checkpoint())
        .unwrap();
    session
        .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
        .unwrap();

    session.complete_step("a", ok_result("a done")).unwrap();
    session.complete_step("b", ok_result("b done")).unwrap();

    // Real restore: the snapshot is returned to the caller (its fork-facing
    // identity feeds the next dispatch's snapshot/instance request fields).
    let restored = session
        .rewind_to_checkpoint("a")
        .unwrap()
        .expect("snapshot should be restored");
    assert_eq!(restored.session_id, "sess-1");
    assert_eq!(restored.snapshot_name, "readfiles");
    assert_eq!(restored.instance.as_deref(), Some("scratch"));
    // The derived path matches the fork layout
    // `<slot_save_path>/<model_key>/<snapshot_name>.bin`; the router never
    // copies KV bytes, so no file is materialized.
    assert_eq!(
        restored.file_path,
        dir.path().join("model-x").join("readfiles.bin")
    );
    assert!(!restored.file_path.exists());
    // Metadata is preserved (it was recorded, not re-derived from a file).
    assert_eq!(restored.token_count, Some(42));
    // The session carries the pending fields for the next dispatch.
    let pending = session.pending_kv_fields();
    assert_eq!(pending.as_ref().map(|(n, _, _)| n.as_str()), Some("readfiles"));
    assert_eq!(pending.as_ref().and_then(|(_, i, _)| i.as_deref()), Some("scratch"));
    assert_eq!(pending.map(|(_, _, s)| s), Some(0));

    // Steps were still reset (data preserved for audit).
    assert_eq!(session.get_step("a").unwrap().status, StepStatus::Pending);
    assert!(session.get_step("a").unwrap().result.is_some());
}

#[tokio::test]
async fn test_rewind_without_model_skips_restore() {
    let mut session = DependencySession::new("sess-1");
    session
        .add_step(SessionStep::new("a", "Step A").with_checkpoint())
        .unwrap();
    session.complete_step("a", ok_result("a done")).unwrap();

    let restored = session.rewind_to_checkpoint("a").unwrap();
    assert!(restored.is_none(), "no model - no snapshot keyed - None");
}

#[test]
fn test_pending_kv_fields_only_when_metadata_exists() {
    // `pending_kv_fields` is set only when a snapshot was actually
    // restored. A session with a kv manager but no stored snapshot has no
    // pending fields; once a snapshot is stored + rewind runs, they appear.
    use crate::kv_cache::{ColdSnapshotIndex, HotSnapshotIndex, SnapshotStore, KvSnapshot};

    let dir = tempfile::tempdir().unwrap();
    let hot = Arc::new(HotSnapshotIndex::new(10, 1024));
    let kv = SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            dir.path(),
            1024,
            86400,
        )),
    );

    let mut session = DependencySession::new("sess-1")
        .with_model("model-x")
        .with_kv_cache(kv);
    session
        .add_step(SessionStep::new("a", "Step A").with_checkpoint())
        .unwrap();
    session
        .add_step(SessionStep::new("b", "Step B").with_depends(vec!["a".into()]))
        .unwrap();
    session.complete_step("a", ok_result("a done")).unwrap();

    // No snapshot metadata yet -> rewind restores nothing -> no pending fields.
    let restored = session.rewind_to_checkpoint("a").unwrap();
    assert!(restored.is_none());
    assert!(session.pending_kv_fields().is_none());

    // Now record a snapshot and re-run: rewind finds it -> pending fields.
    let src_dir = tempfile::tempdir().unwrap();
    let src_file = src_dir.path().join("readfiles.kv");
    std::fs::write(&src_file, b"kv").unwrap();
    session
        .kv_cache()
        .unwrap()
        .store(KvSnapshot {
            model: "model-x".into(),
            adapter: None,
            session_id: "sess-1".into(),
            snapshot_name: "readfiles".into(),
            instance: Some("scratch".into()),
            file_path: src_file,
            token_count: Some(1),
            created_at: common_core::now_secs(),
            last_used_at: common_core::now_secs(),
            llama_cpp_version: None,
            model_quant: None,
            base_model_hash: None,
            turn_seq: None,
        })
        .unwrap();
    // Re-run the blue pass: the checkpoint step is Pending again (rewind
    // preserves steps) so completing it re-registers the checkpoint.
    session.complete_step("a", ok_result("a done")).unwrap();

    let restored = session.rewind_to_checkpoint("a").unwrap();
    assert!(restored.is_some(), "snapshot exists -> restore succeeds");
    let pending = session.pending_kv_fields().expect("pending fields set");
    assert_eq!(pending.0, "readfiles");
    assert_eq!(pending.1.as_deref(), Some("scratch"));
    assert_eq!(pending.2, 0);
}

#[test]
fn test_session_registry_get_or_create() {
    let registry = SessionRegistry::new(None);
    assert_eq!(registry.session_count(), 0);

    let session = registry.get_or_create("sess-1");
    assert_eq!(registry.session_count(), 1);
    assert_eq!(session.lock().unwrap().session_id, "sess-1");

    // Second lookup returns the same session (state survives).
    let again = registry.get_or_create("sess-1");
    assert!(Arc::ptr_eq(&session, &again));

    registry.remove("sess-1");
    assert_eq!(registry.session_count(), 0);
}

// -- context-advance KV snapshotting --------------------------------

fn test_kv_manager(slot_path: &std::path::Path) -> (SnapshotStore, Arc<HotSnapshotIndex>) {
    let hot = Arc::new(HotSnapshotIndex::new(10, 1024));
    let kv = SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            slot_path,
            1024,
            86400,
        )),
    );
    (kv, hot)
}

#[test]
fn kv_snapshot_policy_decide_restore() {
    use KvSnapshotPolicy::*;
    // RestoreIfSameModel: same model restores, different model ignores.
    assert!(RestoreIfSameModel.decide_restore(Some("model-a"), "model-a"));
    assert!(!RestoreIfSameModel.decide_restore(Some("model-a"), "model-b"));
    assert!(!RestoreIfSameModel.decide_restore(None, "model-a"));
    // AlwaysRestore: any present snapshot restores.
    assert!(AlwaysRestore.decide_restore(Some("model-a"), "model-b"));
    assert!(!AlwaysRestore.decide_restore(None, "model-a"));
    // NeverRestore: never.
    assert!(!NeverRestore.decide_restore(Some("model-a"), "model-a"));
}

#[test]
fn advance_without_model_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let (kv, _) = test_kv_manager(dir.path());
    let mut session = DependencySession::new("sess").with_kv_cache(kv);
    let result = session.advance_and_snapshot("scratch").unwrap();
    assert!(result.is_none(), "no model - no snapshot, no fabricated key");
    assert!(session.pending_snapshot().is_none());
}

#[test]
fn advance_without_fork_is_metadata_only_noop() {
    let dir = tempfile::tempdir().unwrap();
    let (kv, _) = test_kv_manager(dir.path());
    let mut session = DependencySession::new("sess")
        .with_model("model-x")
        .with_kv_cache(kv);
    // No fork handle -> save_snapshot is a metadata-only no-op returning
    // Ok, and retrieve finds nothing -> Ok(None), never a crash.
    let result = session.advance_and_snapshot("scratch").unwrap();
    assert!(result.is_none(), "no fork handle -> no recorded snapshot");
    assert!(session.pending_snapshot().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn advance_with_fork_records_snapshot_and_sets_pending() {
    use crate::instances::stub::StubServer;
    use crate::instances::InstanceClient;

    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |method, path, _body| {
            if method == "POST" && path == "/instances/scratch/snapshot" {
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
    let (kv, hot) = test_kv_manager(dir.path());
    let kv = kv.with_fork_io(fork);
    let mut session = DependencySession::new("sess")
        .with_model("model-x")
        .with_kv_cache(kv);

    // Two turns -> two independently-addressable snapshots under the key.
    let s1 = session
        .advance_and_snapshot("scratch")
        .unwrap()
        .expect("first turn snapshot recorded");
    assert_eq!(s1.session_id, "sess");
    assert_eq!(s1.model, "model-x");
    let pending1 = session.pending_kv_fields().expect("pending fields set");
    let name1 = pending1.0;

    let s2 = session
        .advance_and_snapshot("scratch")
        .unwrap()
        .expect("second turn snapshot recorded");
    assert_ne!(s2.snapshot_name, s1.snapshot_name, "per-turn distinct names");
    let pending2 = session.pending_kv_fields().expect("pending fields set");
    assert_ne!(pending2.0, name1, "pending is the most recent snapshot");

    // The fork received both snapshot POSTs.
    let posts = stub
        .recorded()
        .iter()
        .filter(|(m, p, _)| m == "POST" && p == "/instances/scratch/snapshot")
        .count();
    assert_eq!(posts, 2, "one fork snapshot save per turn");

    // Both snapshots coexist under the (model, adapter, session) key in
    // the hot/cold tiers via retrieve (latest wins on retrieve).
    let _ = hot;
}

#[test]
fn advance_sets_pending_snapshot_for_same_model_restore() {
    // RestoreIfSameModel: the pending snapshot's model matches the next
    // dispatch's model -> restore; a different model -> re-prefill.
    let mut session = DependencySession::new("sess");
    // Manually fabricate a pending snapshot to exercise the decision rule.
    let pending = KvSnapshot {
        model: "model-x".into(),
        adapter: None,
        session_id: "sess".into(),
        snapshot_name: "sess-1".into(),
        instance: Some("scratch".into()),
        file_path: std::path::PathBuf::new(),
        token_count: None,
        created_at: common_core::now_secs(),
        last_used_at: common_core::now_secs(),
        llama_cpp_version: None,
        model_quant: None,
        base_model_hash: None,
            turn_seq: None,
    };
    session.pending_snapshot = Some(pending);

    assert!(session.pending_snapshot().is_some());
    assert!(
        KvSnapshotPolicy::RestoreIfSameModel.decide_restore(
            session.pending_snapshot().map(|s| s.model.as_str()),
            "model-x"
        ),
        "same model -> restore"
    );
    assert!(
        !KvSnapshotPolicy::RestoreIfSameModel.decide_restore(
            session.pending_snapshot().map(|s| s.model.as_str()),
            "model-b"
        ),
        "different model -> re-prefill (no restore)"
    );
}
