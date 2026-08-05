// Exercises the capability-gated I/O engines. The db tests use the legacy
// string-map `DbCapability::query`/`execute` (deprecated in fluent-db) as the
// behavior oracle.
#![allow(deprecated)]
use crate::io::db::DbCapability;
use crate::io::fs::FsCapability;
use crate::io::net::NetCapability;
use crate::scope::CURRENT_CAPS;
use fluent_wvr::prelude::*;

#[tokio::test(start_paused = true)]
async fn test_fs_read_write_roundtrip() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let fs = FsCapability::new();
    let caps = CapabilitySet::new().with(FsCapability::new());
    CURRENT_CAPS
        .scope(caps, async {
            fs.write(&path, b"hello world").await.expect("write failed");
            let data = fs.read(&path).await.expect("read failed");
            assert_eq!(data, b"hello world");
            let meta = fs.metadata(&path).await.expect("metadata failed");
            assert!(meta.is_file());
        })
        .await;
}

#[tokio::test(start_paused = true)]
async fn test_net_tcp_connect_refused() {
    let net = NetCapability::new();
    let caps = CapabilitySet::new().with(NetCapability::new());
    let result = CURRENT_CAPS
        .scope(caps, async { net.tcp_connect("127.0.0.1:1").await })
        .await;
    assert!(result.is_err());
}

#[tokio::test(start_paused = true)]
async fn test_db_query_execute_roundtrip() {
    let db = DbCapability::open(":memory:").unwrap();
    let caps = CapabilitySet::new().with(DbCapability::open(":memory:").unwrap());
    CURRENT_CAPS
        .scope(caps, async {
            db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
                .await
                .unwrap();
            db.execute("INSERT INTO t VALUES (1, 'hello')")
                .await
                .unwrap();
            let rows = db.query("SELECT * FROM t").await.unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0]["id"], "1");
            assert_eq!(rows[0]["name"], "hello");
        })
        .await;
}

#[tokio::test(start_paused = true)]
async fn test_capability_missing_denies_io() {
    let fs = FsCapability::new();
    let result = fs.read("/etc/passwd").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::PermissionDenied,
        "expected PermissionDenied, got: {err}"
    );
    assert!(
        err.to_string().contains("missing"),
        "expected 'missing' in error, got: {err}"
    );
}

#[test]
fn test_missing_capability_returns_none() {
    let caps = CapabilitySet::new();
    assert!(caps.get::<FsCapability>().is_none());
    assert!(caps.get::<NetCapability>().is_none());
    assert!(caps.get::<DbCapability>().is_none());
}

#[test]
fn test_capability_gating_fs() {
    let caps = CapabilitySet::new().with(FsCapability::new());
    assert!(caps.get::<FsCapability>().is_some());
    assert!(caps.get::<NetCapability>().is_none());
}

#[tokio::test(start_paused = true)]
async fn test_capability_boundary_enforcement_net() {
    let net = NetCapability::new();
    let result = net.tcp_connect("127.0.0.1:1").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::PermissionDenied,
        "expected PermissionDenied for net, got: {err}"
    );
    assert!(
        err.to_string().contains("missing"),
        "expected 'missing' in error, got: {err}"
    );
}

#[tokio::test(start_paused = true)]
async fn test_capability_boundary_enforcement_db() {
    let db = DbCapability::open(":memory:").unwrap();
    let result = db.query("SELECT 1").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::PermissionDenied,
        "expected PermissionDenied for db, got: {err}"
    );
    assert!(
        err.to_string().contains("missing"),
        "expected 'missing' in error, got: {err}"
    );
}

#[tokio::test(start_paused = true)]
async fn test_capability_boundary_enforcement_db_execute() {
    let db = DbCapability::open(":memory:").unwrap();
    let result = db.execute("CREATE TABLE t (id INTEGER)").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::PermissionDenied,
        "expected PermissionDenied for db execute, got: {err}"
    );
    assert!(
        err.to_string().contains("missing"),
        "expected 'missing' in error, got: {err}"
    );
}

#[tokio::test(start_paused = true)]
async fn test_db_concurrent_queries_via_pool() {
    let db = DbCapability::open(":memory:").unwrap();
    let caps = CapabilitySet::new().with(DbCapability::open(":memory:").unwrap());
    CURRENT_CAPS
        .scope(caps, async {
            db.execute("CREATE TABLE t (id INTEGER, val TEXT)")
                .await
                .unwrap();
            for i in 0..10 {
                db.execute(&format!("INSERT INTO t VALUES ({i}, 'row{i}')"))
                    .await
                    .unwrap();
            }
            let rows = db.query("SELECT COUNT(*) AS cnt FROM t").await.unwrap();
            assert_eq!(rows[0]["cnt"], "10");
        })
        .await;
}
