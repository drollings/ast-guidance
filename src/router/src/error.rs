//! Server-level error type — the single typed error for the router's
//! startup/bind/dispatch surfaces. Router-internal (single consumer crate),
//! so it lives here rather than in `common-core::error` (per the
//! consolidation contract's single-consumer rule).

use thiserror::Error;

use crate::dispatch::frontier::DispatchError;

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("bind failed on {addr}: {source}")]
    Bind {
        addr: String,
        source: std::io::Error,
    },
    #[error("http error: {0}")]
    Http(String),
    #[error("invalid address: {0}")]
    Addr(String),
    /// A more precise dispatch error, preserved rather than stringified.
    #[error(transparent)]
    Dispatch(#[from] DispatchError),
}
