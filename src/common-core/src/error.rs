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
    #[error("ambiguous dependency: '{name}' could be provided by {}", candidates.join(", "))]
    AmbiguousDependency {
        name: String,
        candidates: Vec<String>,
    },
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

/// Generate the standard `impl From<std::io::Error>` that wraps the source
/// error in `common_core::error::IoError` and stores it in the `Io` variant
/// of `$ErrorType`.
///
/// Consumer enums that carry an `Io(#[from] common_core::error::IoError)`
/// variant still need an explicit `std::io::Error` conversion (the
/// `#[from]` on the variant only covers the `IoError` hop; `From` is not
/// transitive). This macro is that one-line shape.
///
/// # Usage
///
/// ```ignore
/// use common_core::error::impl_from_io_error;
///
/// #[derive(thiserror::Error, Debug)]
/// enum MyError {
///     #[error("I/O error: {0}")]
///     Io(#[from] common_core::error::IoError),
/// }
///
/// impl_from_io_error!(MyError);
/// ```
///
/// Equivalent to:
///
/// ```ignore
/// impl From<std::io::Error> for MyError {
///     fn from(e: std::io::Error) -> Self {
///         MyError::Io(common_core::error::IoError(e))
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_from_io_error {
    ($error_type:ty) => {
        impl From<std::io::Error> for $error_type {
            fn from(e: std::io::Error) -> Self {
                <$error_type>::Io($crate::error::IoError(e))
            }
        }
    };
}

