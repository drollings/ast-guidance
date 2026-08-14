//! Cross-crate limit moved to `common-core::constants`
//! Re-exported here for backward compatibility — any consumer may
//! switch to `common_core::MAX_EMBEDDING_DIMENSIONS` directly.
pub use common_core::MAX_EMBEDDING_DIMENSIONS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_expected() {
        assert_eq!(MAX_EMBEDDING_DIMENSIONS, 4_096);
    }
}
