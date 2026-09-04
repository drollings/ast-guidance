use common_core::error::*;
use common_core::impl_from_io_error;
use thiserror::Error;


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

#[derive(Error, Debug)]
    enum MacroTestError {
        #[error("I/O error: {0}")]
        Io(#[from] IoError),
}

    impl_from_io_error!(MacroTestError);

#[test]
fn impl_from_io_error_generates_wrapping_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err: MacroTestError = io_err.into();
        let msg = err.to_string();
        assert!(msg.contains("I/O error"), "got: {msg}");
        assert!(msg.contains("missing"), "got: {msg}");
}
