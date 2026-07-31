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
    #[error("database error: {0}")]
    Database(#[from] common_core::error::SqliteError),
    #[error("embedding error: {0}")]
    Embedding(String),
}

impl From<rusqlite::Error> for CacheError {
    fn from(e: rusqlite::Error) -> Self {
        CacheError::Database(common_core::error::SqliteError(e))
    }
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
}
