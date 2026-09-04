use super::*;

#[test]
fn audit_record_constructs() {
    let r = AuditRecord::new("filter", serde_json::json!({"pattern": "pii"}));
    assert_eq!(r.kind, "filter");
    assert_eq!(r.detail["pattern"], "pii");
}

#[test]
fn emit_renders_kind_and_json_detail() {
    // The audit shape contract: every emitted record carries the `kind`
    // field plus a JSON `detail` payload, rendered into the single
    // `router.audit` tracing target. Capture the formatted line and assert
    // both invariants hold so the durable-audit consumer's expectations
    // cannot silently drift.
    let (_, lines) = crate::test_support::capture_logs(|| {
        emit("filter", serde_json::json!({"pattern": "ssn", "scope": "any"}));
    });
    let joined = lines.join("\n");
    assert!(
        joined.contains("router.audit"),
        "audit record must land on the router.audit target, got:\n{joined}"
    );
    assert!(
        joined.contains("\"kind\":") || joined.contains("kind="),
        "audit record must carry a kind field, got:\n{joined}"
    );
    assert!(
        joined.contains("\"pattern\":\"ssn\"") && joined.contains("\"scope\":\"any\""),
        "audit detail must be JSON, got:\n{joined}"
    );
}

#[test]
fn audit_target_constant_is_stable() {
    // The durable-audit subscription is keyed on this exact target; a
    // change here silently drops every record from the audit file.
    assert_eq!(AUDIT_TARGET, "router.audit");
}

#[test]
fn audit_kind_as_str_is_stable() {
    assert_eq!(AuditKind::Route.as_str(), "route");
    assert_eq!(AuditKind::Filter.as_str(), "filter");
    assert_eq!(AuditKind::Escalation.as_str(), "escalation");
    assert_eq!(AuditKind::Stream.as_str(), "stream");
    assert_eq!(AuditKind::Instances.as_str(), "instances");
}

#[test]
fn audit_record_serializes_to_expected_json() {
    let r = AuditRecord::route(
        crate::pipeline_types::PipelineStage::Classifier,
        crate::pipeline_types::StageVerdict::Passed,
        None,
        Some(42),
        None,
    );
    assert_eq!(r.kind, "route");
    assert_eq!(r.detail["stage"], "Classifier");
    assert_eq!(r.detail["verdict"], "Passed");
    assert_eq!(r.detail["response_len"], 42);

    let f = AuditRecord::filter(
        crate::pipeline_types::PipelineStage::DeterministicPreFilter,
        "command_dispatched",
        Some("my_cmd"),
    );
    assert_eq!(f.kind, "filter");
    assert_eq!(f.detail["stage"], "DeterministicPreFilter");
    assert_eq!(f.detail["verdict"], "command_dispatched");

    let e = AuditRecord::escalation("filter", true, "payload", "response", "trigger");
    assert_eq!(e.kind, "escalation");
    assert_eq!(e.detail["mode"], "filter");
    assert_eq!(e.detail["accepted"], true);
    assert_eq!(e.detail["trigger"], "trigger");

    let i = AuditRecord::instances("allocate_on_miss", serde_json::json!({"group":"swarm"}));
    assert_eq!(i.kind, "instances");
    assert_eq!(i.detail["action"], "allocate_on_miss");
    assert_eq!(i.detail["group"], "swarm");
}

#[test]
fn audit_emit_is_durable_under_test_subscriber() {
    let (_, lines) = crate::test_support::capture_logs(|| {
        AuditRecord::route(
            crate::pipeline_types::PipelineStage::Classifier,
            crate::pipeline_types::StageVerdict::Passed,
            None,
            Some(10),
            None,
        )
        .emit();
        AuditRecord::filter(
            crate::pipeline_types::PipelineStage::DeterministicPreFilter,
            "pii_block",
            Some("ssn"),
        )
        .emit();
        AuditRecord::escalation("team", false, "prompt", "", "judge rejected").emit();
    });
    let joined = lines.join("\n");
    assert!(joined.contains("router.audit"), "typed emit must land on router.audit, got:\n{joined}");
    assert!(joined.contains("route") || joined.contains("filter") || joined.contains("escalation"));
}
