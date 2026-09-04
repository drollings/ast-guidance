use super::*;

fn test_snapshot(session_id: &str) -> KvSnapshot {
    KvSnapshot {
        model: "test-model".into(),
        adapter: None,
        session_id: session_id.into(),
        snapshot_name: "default".into(),
        instance: None,
        file_path: PathBuf::new(),
        token_count: Some(100),
        created_at: now_secs(),
        last_used_at: now_secs(),
        llama_cpp_version: Some("0.1.0".into()),
        model_quant: None,
        base_model_hash: Some("abc123".into()),
        turn_seq: None,
    }
}

#[test]
fn test_hot_cache_put_get() {
    let cache = HotSnapshotIndex::new(10, 1024);
    let snap = test_snapshot("sess-1");
    cache.put(snap.clone());

    let retrieved = cache.get("test-model", None, "sess-1").unwrap();
    assert_eq!(retrieved.session_id, "sess-1");
    assert_eq!(retrieved.token_count, Some(100));
}

#[test]
fn test_hot_cache_miss() {
    let cache = HotSnapshotIndex::new(10, 1024);
    assert!(cache.get("nonexistent", None, "sess-x").is_none());
}

#[test]
fn test_hot_cache_remove() {
    let cache = HotSnapshotIndex::new(10, 1024);
    cache.put(test_snapshot("sess-1"));
    assert_eq!(cache.len(), 1);

    cache.remove("test-model", None, "sess-1");
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_hot_cache_lru_eviction() {
    let cache = HotSnapshotIndex::new(3, 1024);

    for i in 0..5 {
        cache.put(KvSnapshot {
            model: "m".into(),
            adapter: None,
            session_id: format!("sess-{i}"),
            snapshot_name: "default".into(),
            instance: None,
            file_path: PathBuf::new(),
            token_count: Some(1),
            created_at: now_secs(),
            last_used_at: now_secs(),
            llama_cpp_version: Some("0.1".into()),
            model_quant: None,
            base_model_hash: Some("hash".into()),
            turn_seq: None,
        });
    }

    assert_eq!(cache.len(), 3);
    assert!(cache.get("m", None, "sess-0").is_none());
    assert!(cache.get("m", None, "sess-1").is_none());
    assert!(cache.get("m", None, "sess-2").is_some());
}

#[test]
fn model_key_sanitizes_slashes_and_colons() {
    assert_eq!(model_key("abiray/lfm2.5"), "abiray_lfm2.5");
    assert_eq!(model_key("org/model:q4"), "org_model_q4");
}

#[test]
fn kv_snapshot_path_matches_fork_layout() {
    let p = kv_snapshot_path(Path::new("/srv/slots"), "abiray/lfm2.5", "readfiles");
    assert_eq!(p, PathBuf::from("/srv/slots/abiray_lfm2.5/readfiles.bin"));
}

#[tokio::test]
async fn test_cold_cache_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let cold = ColdSnapshotIndex::new(dir.path(), 1024, 86400);

    let mut snap = test_snapshot("sess-cold");
    snap.snapshot_name = "readfiles".into();
    snap.instance = Some("scratch".into());
    cold.save(&snap).unwrap();

    let loaded = cold.load("test-model", None, "sess-cold").unwrap();
    assert_eq!(loaded.model, "test-model");
    assert_eq!(loaded.session_id, "sess-cold");
    assert_eq!(loaded.snapshot_name, "readfiles");
    assert_eq!(loaded.instance.as_deref(), Some("scratch"));
    // The derived path matches the fork layout: <slot_save_path>/<model_key>/<name>.bin
    assert_eq!(
        loaded.file_path,
        dir.path().join("test-model").join("readfiles.bin")
    );
}

#[tokio::test]
async fn test_cold_cache_load_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let cold = ColdSnapshotIndex::new(dir.path(), 1024, 86400);

    let result = cold.load("test-model", None, "no-such-session");
    assert!(result.is_err());
}

#[tokio::test]
async fn test_kv_cache_manager_two_tier() {
    let dir = tempfile::tempdir().unwrap();
    let hot = Arc::new(HotSnapshotIndex::new(10, 1024));
    let cold = Arc::new(ColdSnapshotIndex::new(
        dir.path(),
        1024,
        86400,
    ));
    let mgr = SnapshotStore::new(Arc::clone(&hot), Arc::clone(&cold));

    mgr.store(test_snapshot("sess-tier")).unwrap();

    // Should be in hot tier
    let retrieved = mgr.retrieve("test-model", None, "sess-tier").unwrap();
    assert_eq!(retrieved.session_id, "sess-tier");

    // Remove from hot, should fall back to cold
    hot.remove("test-model", None, "sess-tier");
    let retrieved2 = mgr.retrieve("test-model", None, "sess-tier").unwrap();
    assert_eq!(retrieved2.session_id, "sess-tier");
}

#[tokio::test]
async fn test_cold_cache_evict_by_ttl() {
    let dir = tempfile::tempdir().unwrap();
    let cold = ColdSnapshotIndex::new(
        dir.path(),
        1024,
        0, // immediate TTL
    );

    cold.save(&test_snapshot("sess-evict")).unwrap();

    let evicted = cold.evict().unwrap();
    assert_eq!(evicted, 1);
}

#[tokio::test]
async fn test_list_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let cold = ColdSnapshotIndex::new(dir.path(), 1024, 86400);

    cold.save(&test_snapshot("sess-list")).unwrap();

    let snapshots = cold.list_snapshots("sess-list");
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].session_id, "sess-list");
}

#[tokio::test]
async fn metadata_only_cold_tier_degrades_gracefully() {
    let cold = ColdSnapshotIndex::metadata_only(86400);
    let mut snap = test_snapshot("sess-meta");
    snap.snapshot_name = "x".into();
    cold.save(&snap).unwrap();

    let loaded = cold.load("test-model", None, "sess-meta").unwrap();
    // No server-owned store: the derived path is empty, never a crash.
    assert!(loaded.file_path.as_os_str().is_empty());
}

// -- Fork round-trip via the optional InstanceClient handle --------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn save_snapshot_with_fork_records_metadata_and_posts() {
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
    let hot = Arc::new(HotSnapshotIndex::new(10, 1024));
    let kv = SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            dir.path(),
            1024,
            86400,
            )),
    )
    .with_fork_io(fork);

    kv.save_snapshot("model-x", None, "sess-1", "readfiles", "scratch")
        .expect("save succeeds");

    // The fork received the snapshot POST.
    assert!(
        stub.recorded()
            .iter()
            .any(|(m, p, _)| m == "POST" && p == "/instances/scratch/snapshot"),
        "stub fork must receive the snapshot save"
    );

    // Metadata was recorded under the session key: a rewind can find it.
    let retrieved = kv.retrieve("model-x", None, "sess-1").expect("retrieve");
    assert_eq!(retrieved.snapshot_name, "readfiles");
    assert_eq!(retrieved.instance.as_deref(), Some("scratch"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn save_snapshot_without_fork_is_noop() {
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

    // No fork handle -> metadata-only no-op returning Ok, nothing recorded.
    kv.save_snapshot("model-x", None, "sess-1", "readfiles", "scratch")
        .expect("no-op save returns Ok");
    assert!(
        kv.retrieve("model-x", None, "sess-1").is_err(),
        "no metadata recorded without a fork handle"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_and_delete_delegate_to_fork_when_present() {
    use crate::instances::stub::StubServer;
    use crate::instances::InstanceClient;

    let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
        |method, path, _body| {
            if method == "GET" && path == "/instances/scratch/snapshots" {
                (
                    200,
                    serde_json::json!([
                        { "name": "readfiles", "size": 512, "mtime": "x" }
                    ])
                    .to_string(),
                )
            } else {
                (200, "{}".into())
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
    let kv = SnapshotStore::new(
        Arc::clone(&hot),
        Arc::new(ColdSnapshotIndex::new(
            dir.path(),
            1024,
            86400,
            )),
    )
    .with_fork_io(fork);

    let listed = kv.list_snapshots("scratch");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].snapshot_name, "readfiles");

    kv.delete_snapshot("scratch", "readfiles").expect("delete");
    assert!(
        stub.recorded()
            .iter()
            .any(|(m, p, _)| m == "DELETE" && p == "/instances/scratch/snapshot/readfiles"),
        "fork must receive the snapshot delete"
    );
}

// ─── M10.1 characterization: cold-tier TTL edges + empty store ─────────

#[tokio::test]
async fn cold_evict_empty_store_returns_zero() {
    let cold = ColdSnapshotIndex::metadata_only(86400);
    assert_eq!(cold.evict().unwrap(), 0, "empty cold tier evicts nothing");
}

#[tokio::test]
async fn cold_evict_ttl_boundary_age_equals_ttl_is_evicted() {
    // Locks the cold tier's boundary: entries are kept while
    // `age < ttl_secs`, so `age == ttl_secs` is evicted. (This differs by
    // one second from `db::cache::TtlCache::get`'s strict `now > ts + ttl`
    // keep — a deliberate, documented divergence preserved by M10, not a
    // bug to "fix" by unification.)
    let ttl = 100u64;
    let dir = tempfile::tempdir().unwrap();
    let cold = ColdSnapshotIndex::new(dir.path(), 1024, ttl);
    let now = now_secs();

    let mut fresh = test_snapshot("sess-fresh");
    fresh.last_used_at = now;
    cold.save(&fresh).unwrap();

    let mut boundary = test_snapshot("sess-boundary");
    boundary.last_used_at = now.saturating_sub(ttl);
    cold.save(&boundary).unwrap();

    assert_eq!(cold.evict().unwrap(), 1, "only the age == ttl entry is evicted");
    assert!(cold.load("test-model", None, "sess-fresh").is_ok());
    assert!(cold.load("test-model", None, "sess-boundary").is_err());
}

#[tokio::test]
async fn cold_new_takes_no_eviction_policy() {
    // M10.2 lock: the former `_eviction: EvictionPolicy` parameter was
    // removed (it was accepted and dropped at every call site, always
    // `Lru`). The constructor is now `(slot_save_path, max_mb, ttl_secs)`;
    // eviction is always the TTL predicate sweep in `ColdSnapshotIndex::evict`.
    let dir = tempfile::tempdir().unwrap();
    let cold = ColdSnapshotIndex::new(dir.path(), 1024, 86400);
    cold.save(&test_snapshot("sess-pol")).unwrap();
    assert!(cold.load("test-model", None, "sess-pol").is_ok());
    assert_eq!(cold.evict().unwrap(), 0, "fresh entry survives the TTL sweep");
}
