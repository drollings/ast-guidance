#[cfg(feature = "sqlite")]
use common_core::sqlite::*;
#[cfg(feature = "sqlite")]


#[cfg(feature = "sqlite")]
#[test]
fn open_wal_sets_pragmas() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let conn = open_wal(&path).unwrap();
        let journal: String = conn
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        assert_eq!(journal, "wal");
}

#[cfg(feature = "sqlite")]
#[test]
fn open_in_memory_works() {
        let conn = open_in_memory().unwrap();
        // In-memory databases return "memory" for journal_mode — that's expected.
        // The important thing is that the connection works.
        conn.execute_batch("CREATE TABLE t (id INTEGER)").unwrap();
}

#[cfg(feature = "sqlite")]
#[test]
fn open_shared_in_memory_is_shared_across_connections() {
        // Two connections to the same named shared-cache memory DB must see the
        // same database: a write through one is visible through the other.
        let a = open_shared_in_memory("memdb_shared_test_a").unwrap();
        let b = open_shared_in_memory("memdb_shared_test_a").unwrap();
        a.execute_batch("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (7)")
            .unwrap();
        let n: i64 = b
            .query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1, "shared-cache memory connections share the same DB");
}

#[cfg(feature = "sqlite")]
#[test]
fn open_shared_in_memory_names_are_isolated() {
        // Distinct names never share a cache: a write to one is invisible to
        // the other.
        let a = open_shared_in_memory("memdb_isolated_a").unwrap();
        let b = open_shared_in_memory("memdb_isolated_b").unwrap();
        a.execute_batch("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1)")
            .unwrap();
        let err = b
            .query_row("SELECT COUNT(*) FROM t", [], |r| r.get::<_, i64>(0))
            .unwrap_err();
        assert!(
            err.to_string().contains("no such table"),
            "different shared-cache names must be isolated, got {err}"
        );
}

#[cfg(feature = "sqlite")]
#[test]
fn open_shared_in_memory_sets_busy_timeout_not_wal() {
        let conn = open_shared_in_memory("memdb_timeout_test").unwrap();
        // The memory journal (not WAL) is the correct mode for a shared cache.
        let journal: String = conn
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        assert_eq!(journal, "memory", "no WAL on shared-cache memory DBs");
        // busy_timeout must be applied (5000ms default).
        let timeout: i64 = conn
            .pragma_query_value(None, "busy_timeout", |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
}

#[cfg(feature = "sqlite")]
#[test]
fn run_batch_executes_ddl() {
        let conn = open_in_memory().unwrap();
        run_batch(&conn, "CREATE TABLE t (id INTEGER PRIMARY KEY)").unwrap();
        conn.execute_batch("INSERT INTO t (id) VALUES (1)").unwrap();
}

// NOTE (ROADMAP_20260903_LLM M11): the `embedding_cache` DDL goldens moved to
// `fluent-llm --test embeddings_cache` (canonical owner
// `fluent_llm::embeddings_cache`) in M5, and M11 deleted the
// `common_core::sqlite` shims (with the shim-lock test that pinned them).
// The generic suites below (`open_*`, `run_batch`, `is_unique_violation`,
// `in_clause`) stay.

#[cfg(feature = "sqlite")]
#[test]
fn is_unique_violation_detects_unique_and_primary_key() {
        let conn = open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
            .unwrap();

        let err = conn
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, ?2)",
                rusqlite::params![1, "x"],
            )
            .and_then(|_| {
                conn.execute(
                    "INSERT INTO t (id, name) VALUES (?1, ?2)",
                    rusqlite::params![1, "y"],
                )
            })
            .expect_err("duplicate primary key must fail");
        assert!(is_unique_violation(&err), "PK: {err}");

        let err = conn
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, ?2)",
                rusqlite::params![2, "x"],
            )
            .expect_err("duplicate unique name must fail");
        assert!(is_unique_violation(&err), "UNIQUE: {err}");
}

#[cfg(feature = "sqlite")]
#[test]
fn is_unique_violation_rejects_other_errors() {
        let conn = open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
            .unwrap();
        // NOT NULL violation is not a unique/primary-key violation.
        let not_null = conn
            .execute(
                "INSERT INTO t (id, name) VALUES (?1, NULL)",
                rusqlite::params![3],
            )
            .expect_err("NOT NULL violation");
        assert!(!is_unique_violation(&not_null), "NOT NULL: {not_null}");
}

#[cfg(feature = "sqlite")]
#[test]
fn in_clause_builds_placeholders() {
        assert_eq!(in_clause(1), "?");
        assert_eq!(in_clause(3), "?,?,?");
        assert_eq!(in_clause(0), "");
}
