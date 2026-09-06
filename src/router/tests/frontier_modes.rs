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
