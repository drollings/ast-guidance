use super::*;

#[test]
fn test_version_recorded_and_retrieved() {
    let mut reg = VersionRegistry::new();
    reg.record(OntologyVersion {
        version: "4.5".to_string(),
        loaded_at: 1_700_000_000.0,
        source_url: "data/yago-4.5.0.2-tiny/yago-tiny.ttl".to_string(),
        triple_count: 1234,
    });

    let latest = reg.latest().unwrap();
    assert_eq!(latest.version, "4.5");
    assert_eq!(latest.triple_count, 1234);
}

#[test]
fn test_migration_stub_is_noop() {
    (MIGRATIONS[0].migrate_fn)();
}

#[test]
fn test_latest_returns_none_when_empty() {
    let reg = VersionRegistry::new();
    assert!(reg.latest().is_none());
}
