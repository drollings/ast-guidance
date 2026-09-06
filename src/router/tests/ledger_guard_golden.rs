use crate::test_support::capture_logs;

#[test]
fn emit_write_audit_on_flagged() {
    let (_, logs) = capture_logs(|| {
        let ledger = crate::ledger::ContentNodeLedger::open(&std::env::temp_dir().join(format!("golden-{}", common_core::hash::uuid_v4()))).unwrap();
        ledger.record_request("sess", "req", "Contact user@example.com now").unwrap();
    });
    let joined = logs.join("\n");
    assert!(joined.contains("write_path") || joined.contains("router.audit"), "audit emitted {joined}");
}
