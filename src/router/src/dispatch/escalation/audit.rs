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
    let mut record = serde_json::json!({
        "mode": mode,
        "accepted": accepted,
        "payload": payload,
        "raw_response": raw_response,
        "trigger": trigger,
        "timestamp": common_core::now_secs(),
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            record[k] = v.clone();
        }
    }
    crate::audit::emit("escalation", record);
}
