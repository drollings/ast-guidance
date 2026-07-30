#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoralStats {
    pub tier_count: usize,
    pub total_count: u64,
    pub total_sum_ms: u64,
    pub p50_ms: u64,
    pub p99_ms: u64,
}
