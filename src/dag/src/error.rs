use thiserror::Error;

#[derive(Error, Debug)]
pub enum RegistryError {
    #[error("target '{name}' already exists")]
    DuplicateTarget { name: String },
    #[error("target not found: {0}")]
    TargetNotFound(String),
    #[error("invalid capability reference: {0}")]
    InvalidCapability(String),
    #[error("bit index {0} out of range")]
    BitIndexOutOfRange(usize),
    #[error("database error: {0}")]
    Database(#[from] common_core::error::SqliteError),
}

#[cfg(test)]
#[path = "../tests/error.rs"]
mod tests;
