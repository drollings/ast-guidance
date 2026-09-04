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

/// Typed audit kinds — the single source of truth for the `kind` field.
///
/// Every `emit` used to hand-type `"route"` / `"filter"` etc. — a typo drifts
/// silently. `AuditKind` makes the kind compiler-checked and derives the
/// string via `as_str()` so the wire shape stays byte-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditKind {
    Route,
    Filter,
    Escalation,
    Stream,
    Instances,
    // Back-compat for kinds still emitted via generic pipeline (chart, tree, etc.)
    Chart,
    Generic,
}

impl AuditKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Route => "route",
            Self::Filter => "filter",
            Self::Escalation => "escalation",
            Self::Stream => "stream",
            Self::Instances => "instances",
            Self::Chart => "chart_target",
            Self::Generic => "generic",
        }
    }
}

impl From<AuditKind> for &'static str {
    fn from(k: AuditKind) -> Self {
        k.as_str()
    }
}

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

    pub fn kind_enum(&self) -> AuditKind {
        match self.kind {
            "route" => AuditKind::Route,
            "filter" => AuditKind::Filter,
            "escalation" => AuditKind::Escalation,
            "stream" => AuditKind::Stream,
            "instances" => AuditKind::Instances,
            "chart_target" | "chart_summary" => AuditKind::Chart,
            _ => AuditKind::Generic,
        }
    }

    /// Typed constructor for `kind="route"` — replaces `emit("route", json!({stage,verdict,...}))`.
    #[allow(clippy::needless_pass_by_value)]
    pub fn route(
        stage: crate::pipeline_types::PipelineStage,
        verdict: crate::pipeline_types::StageVerdict,
        target: Option<&crate::pipeline::RoutingTarget>,
        response_len: Option<usize>,
        reason: Option<&str>,
    ) -> Self {
        let mut detail = serde_json::json!({
            "stage": format!("{stage:?}"),
            "verdict": format!("{verdict:?}"),
        });
        if let Some(t) = target {
            detail["target_route"] = serde_json::json!(t.target_name);
            detail["target_model"] = serde_json::json!(t.model);
            detail["target_url"] = serde_json::json!(t.url);
        }
        if let Some(l) = response_len {
            detail["response_len"] = serde_json::json!(l);
        }
        if let Some(r) = reason {
            detail["reason"] = serde_json::json!(r);
        }
        Self {
            kind: AuditKind::Route.as_str(),
            detail,
        }
    }

    /// Typed constructor for `kind="filter"` — replaces `emit("filter", json!({stage,verdict,reason}))`.
    pub fn filter(
        stage: crate::pipeline_types::PipelineStage,
        verdict: &str,
        reason: Option<&str>,
    ) -> Self {
        let mut detail = serde_json::json!({
            "stage": format!("{stage:?}"),
            "verdict": verdict,
        });
        if let Some(r) = reason {
            detail["reason"] = serde_json::json!(r);
        }
        Self {
            kind: AuditKind::Filter.as_str(),
            detail,
        }
    }

    /// Typed constructor for `kind="escalation"`.
    pub fn escalation(
        mode: &str,
        accepted: bool,
        payload: &str,
        raw_response: &str,
        trigger: &str,
    ) -> Self {
        Self {
            kind: AuditKind::Escalation.as_str(),
            detail: serde_json::json!({
                "mode": mode,
                "accepted": accepted,
                "payload": payload,
                "raw_response": raw_response,
                "trigger": trigger,
                "timestamp": common_core::now_secs(),
            }),
        }
    }

    /// Typed constructor for `kind="instances"`.
    pub fn instances(action: &str, detail: serde_json::Value) -> Self {
        let mut d = detail;
        if d.is_null() {
            d = serde_json::json!({});
        }
        d["action"] = serde_json::json!(action);
        Self {
            kind: AuditKind::Instances.as_str(),
            detail: d,
        }
    }

    /// Emit this record to [`AUDIT_TARGET`] — typed path replaces `audit::emit("route", json!(...))`.
    pub fn emit(self) {
        emit(self.kind, self.detail);
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
#[path = "../tests/audit.rs"]
mod tests;
