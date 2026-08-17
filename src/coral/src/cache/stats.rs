#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoralStats {
    pub tier_count: usize,
    pub total_count: u64,
    pub total_sum_ms: u64,
    pub p50_ms: u64,
    pub p99_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_serde_round_trip() {
        let stats = CoralStats {
            tier_count: 3,
            total_count: 120,
            total_sum_ms: 9000,
            p50_ms: 40,
            p99_ms: 250,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let back: CoralStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.tier_count, 3);
        assert_eq!(back.total_count, 120);
        assert_eq!(back.total_sum_ms, 9000);
        assert_eq!(back.p50_ms, 40);
        assert_eq!(back.p99_ms, 250);
    }

    #[test]
    fn stats_field_names_match_wire_schema() {
        // The wire shape is a stable part of the management contract; assert
        // the exact field names so a rename is a visible, deliberate change.
        let json = serde_json::json!({
            "tier_count": 1, "total_count": 1, "total_sum_ms": 1,
            "p50_ms": 1, "p99_ms": 1,
        });
        let stats: CoralStats = serde_json::from_value(json).expect("parse");
        assert_eq!(stats.tier_count, 1);
    }

    #[test]
    fn stats_zero_values_are_valid() {
        let stats = CoralStats {
            tier_count: 0,
            total_count: 0,
            total_sum_ms: 0,
            p50_ms: 0,
            p99_ms: 0,
        };
        let back: CoralStats =
            serde_json::from_str(&serde_json::to_string(&stats).expect("serialize"))
                .expect("round trip");
        assert_eq!(back.total_count, 0);
        assert_eq!(back.p99_ms, 0);
    }
}