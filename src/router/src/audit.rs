//! Canonical durable-audit surface for the router
//!
//! The audit layer (`logging::audit_layer`) subscribes to a single `tracing`
//! target, [`AUDIT_TARGET`], and every audit producer — chart target runs,
//! filter verdicts, route decisions, dispatch attempts, and
//! escalation-ladder interactions — emits through [`emit`] into it. Audit
//! *kinds* are distinguished by the `kind` structured field, never by a second
//! dot-namespace (the `router.charts.audit` target predates this module and is
//! being retired).

/// The single `tracing` target the durable audit layer subscribes to.
pub const AUDIT_TARGET: &str = "router.audit";

/// One durable audit record: a `kind` plus a JSON payload of detail fields.
///
/// `detail` carries the event-specific fields (e.g. `chart`/`target`/`score`
/// for chart runs, `pattern` for filter verdicts, `model`/`url` for dispatch
/// attempts). It is rendered as a single JSON field so the audit file stays
/// flat and machine-readable.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditRecord {
    pub kind: &'static str,
    pub detail: serde_json::Value,
}

impl AuditRecord {
    pub fn new(kind: &'static str, detail: serde_json::Value) -> Self {
        Self { kind, detail }
    }
}

/// Emit a durable audit record to [`AUDIT_TARGET`].
///
/// `fields` is the event's JSON payload. The `kind` field distinguishes the
/// record class (e.g. `"chart_target"`, `"filter"`, `"route"`, `"escalation"`).
#[allow(clippy::needless_pass_by_value)]
pub fn emit(kind: &'static str, fields: serde_json::Value) {
    tracing::info!(
        target: AUDIT_TARGET,
        kind = kind,
        detail = %fields,
        "audit record",
    );
}

#[cfg(test)]
mod tests {
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
}
