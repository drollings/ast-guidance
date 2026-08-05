//! Canonical durable-audit surface for the router (ROADMAP_20260805_REVIEW M1).
//!
//! The audit layer (`logging::audit_layer`) subscribes to a single `tracing`
//! target, [`AUDIT_TARGET`], and every audit producer — chart target runs,
//! filter verdicts, route decisions, dispatch attempts, and (M3)
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
// Signature is prescribed by ROADMAP_20260805_REVIEW M1.2; the payload is
// deliberately owned so callers can build it inline with `serde_json::json!`.
#[allow(clippy::needless_pass_by_value)]
pub fn emit(kind: &'static str, fields: serde_json::Value) {
    tracing::info!(
        target: AUDIT_TARGET,
        kind = kind,
        detail = %fields,
        "audit record",
    );
}
