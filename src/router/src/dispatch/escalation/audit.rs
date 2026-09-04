//! Escalation audit-record building: one `kind = "escalation"` record per mode
//! interaction, carrying `mode`/`accepted`/`payload`/`raw_response`/`trigger`/
//! `timestamp` plus any mode-specific `extra`.

/// Emit a `kind = "escalation"` audit record for a ladder interaction.
pub(super) fn emit_audit(
    mode: &str,
    accepted: bool,
    payload: &str,
    raw_response: &str,
    trigger: &str,
    extra: &serde_json::Value,
) {
    let mut base = crate::audit::AuditRecord::escalation(mode, accepted, payload, raw_response, trigger);
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            base.detail[k] = v.clone();
        }
    }
    base.emit();
}
