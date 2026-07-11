//! Shared leaf error types: `IoError`, `ResolverError`, `SqliteError` (feature-gated).

use thiserror::Error;

/// I/O error wrapper.
///
/// Single-variant tuple struct that wraps `std::io::Error`. Consumers can use
/// the `IoError(e)` constructor directly (or, equivalently, `e.into()` thanks
/// to the `#[from]` derive). `kind()` mirrors `std::io::Error::kind()` so
/// callers do not need to unwrap an `Option` to inspect the I/O error kind.
///
/// The older `FileTooLarge` / `PathNotFound` / `InvalidPath` variants were
/// dead and have been removed; the `MAX_FILE_SIZE` guard in
/// `crate::io::read_to_string_err` emits a plain `io::Error` with
/// `ErrorKind::InvalidData` instead.
#[derive(Error, Debug)]
#[error("I/O error: {0}")]
pub struct IoError(#[from] pub std::io::Error);

impl IoError {
    /// Returns the inner `std::io::Error::kind()`.
    #[must_use]
    pub fn kind(&self) -> std::io::ErrorKind {
        self.0.kind()
    }

    /// Borrow the wrapped `std::io::Error`.
    #[must_use]
    pub fn as_inner(&self) -> &std::io::Error {
        &self.0
    }
}

#[derive(Error, Debug)]
pub enum ResolverError {
    #[error("circular dependency detected")]
    CircularDependency,
    #[error("target not found: {0}")]
    TargetNotFound(String),
    #[error("missing dependency: {0}")]
    MissingDependency(String),
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
}

/// Shared SQLite error wrapper. Feature-gated on `sqlite` so the crate stays
/// zero-domain by default — `rusqlite` is a generic storage dep, not a
/// domain concern.
#[cfg(feature = "sqlite")]
#[derive(Error, Debug)]
#[error("sqlite error: {0}")]
pub struct SqliteError(#[from] pub rusqlite::Error);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn io_error_from_std() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = IoError(io_err);
        assert!(format!("{err}").contains("file not found"));
    }

    #[test]
    fn io_error_kind_returns_inner_kind() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err = IoError(io_err);
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn io_error_from_via_into() {
        let io_err = std::io::Error::other("boom");
        let err: IoError = io_err.into();
        assert_eq!(err.kind(), std::io::ErrorKind::Other);
    }

    #[test]
    fn io_error_as_inner_borrows_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short");
        let err = IoError(io_err);
        assert_eq!(err.as_inner().kind(), std::io::ErrorKind::UnexpectedEof);
    }
}
