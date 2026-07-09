use thiserror::Error;

#[derive(Error, Debug)]
pub enum CopilotError {
    #[error("I/O error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("LLM error: {0}")]
    Llm(#[from] guidance_llm::LlmError),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("native messaging framing error: {0}")]
    NativeMessaging(String),
    #[error("loopback bind failed: {0}")]
    LoopbackBind(String),
    #[error("auth rejected: {0}")]
    Auth(String),
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("profile not loaded: {0}")]
    ProfileNotLoaded(String),
    #[error("audit error: {0}")]
    Audit(String),
    #[error("JSON-RPC method not found: {0}")]
    MethodNotFound(String),
    #[error("JSON-RPC invalid params: {0}")]
    InvalidParams(String),
    #[error("internal error: {0}")]
    Internal(String),
    #[error("database error: {0}")]
    Sqlite(#[from] common_core::error::SqliteError),
}

pub type Result<T> = std::result::Result<T, CopilotError>;

impl From<rusqlite::Error> for CopilotError {
    fn from(e: rusqlite::Error) -> Self {
        CopilotError::Sqlite(common_core::error::SqliteError(e))
    }
}
