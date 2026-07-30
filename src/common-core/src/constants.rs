//! Cross-crate magic numbers (size caps, dimension limits, HNSW defaults).

pub const MAX_VALUE_LEN: usize = 128;
pub const MAX_FILE_SIZE: usize = 100 * 1024 * 1024;
pub const MAX_JSON_DEPTH: usize = 100;
pub const MAX_EMBEDDING_DIMENSIONS: usize = 4_096;
/// Max characters of a log message body to include in tracing/error records.
pub const MAX_LOG_MESSAGE_LEN: usize = 120;

/// Serde default helper that returns `true`.
pub const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    pub max_nb_connection: usize,
    pub max_layer: usize,
    pub ef_construction: usize,
    pub initial_capacity: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            max_layer: 16,
            ef_construction: 200,
            initial_capacity: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_expected() {
        assert_eq!(MAX_VALUE_LEN, 128);
        assert_eq!(MAX_FILE_SIZE, 100 * 1024 * 1024);
        assert_eq!(MAX_JSON_DEPTH, 100);
        assert_eq!(MAX_EMBEDDING_DIMENSIONS, 4_096);
    }

    #[test]
    fn hnsw_params_default_matches_previous_inline_values() {
        let p = HnswParams::default();
        assert_eq!(p.max_nb_connection, 16);
        assert_eq!(p.max_layer, 16);
        assert_eq!(p.ef_construction, 200);
        assert_eq!(p.initial_capacity, 1024);
    }
}
