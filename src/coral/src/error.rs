use thiserror::Error;

#[derive(Error, Debug)]
pub enum CacheError {
    #[error("cache miss")]
    Miss,
    #[error("frontier error: {0}")]
    FrontierError(String),
    #[error("persist failed: {0}")]
    PersistFailed(String),
    #[error("invalid cache capacity: {0}")]
    InvalidCapacity(usize),
    #[error("embedding error: {0}")]
    Embedding(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_error_frontier() {
        let err = CacheError::FrontierError("timeout".into());
        assert_eq!(format!("{err}"), "frontier error: timeout");
    }

    #[test]
    fn cache_error_persist_failed() {
        let err = CacheError::PersistFailed("disk full".into());
        assert_eq!(format!("{err}"), "persist failed: disk full");
    }

    #[test]
    fn cache_error_miss() {
        assert_eq!(format!("{}", CacheError::Miss), "cache miss");
    }

    #[test]
    fn cache_error_invalid_capacity() {
        let err = CacheError::InvalidCapacity(0);
        assert_eq!(format!("{err}"), "invalid cache capacity: 0");
    }

    #[test]
    fn cache_error_embedding() {
        let err = CacheError::Embedding("no dims".into());
        assert_eq!(format!("{err}"), "embedding error: no dims");
    }
}
