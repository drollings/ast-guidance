use super::*;

#[test]
fn escalation_mode_serde_round_trips_snake_case() {
    for (mode, json) in [
        (EscalationMode::Filter, "\"filter\""),
        (EscalationMode::Question, "\"question\""),
        (EscalationMode::Team, "\"team\""),
        (EscalationMode::Turnover, "\"turnover\""),
    ] {
        let serialized = serde_json::to_string(&mode).unwrap();
        assert_eq!(serialized, json);
        let back: EscalationMode = serde_json::from_str(json).unwrap();
        assert_eq!(back, mode);
    }
}

#[test]
fn frontier_result_carries_the_stage_that_produced_it() {
    let result = FrontierResult {
        mode: EscalationMode::Team,
        response: "answer".into(),
        audit_entry: AuditEntry {
            payload: "prompt".into(),
            raw_response: "raw".into(),
            trigger: "low judge confidence".into(),
            timestamp: 1,
        },
    };
    assert_eq!(result.mode, EscalationMode::Team);
    assert_eq!(result.response, "answer");
    assert_eq!(result.audit_entry.trigger, "low judge confidence");
}
