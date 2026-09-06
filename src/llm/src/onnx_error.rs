//! Error type for `fluent-onnx`.

/// Errors surfaced by `fluent-onnx`: config validation, session load/run, and
/// tokenization. The runtime providers (`OrtEncoder` etc.) map these onto the
/// consumer traits' error types at the trait boundary.
#[derive(Debug, thiserror::Error)]
pub enum OrtError {
    #[error("onnx model file not found: {model}")]
    ModelFileNotFound { model: String },
    #[error("model config.json unreadable ({path}): {detail}")]
    ConfigRead { path: String, detail: String },
    #[error("model config.json unparseable ({path}): {detail}")]
    ConfigParse { path: String, detail: String },
    #[error(
        "onnx task mismatch: declared task {task} but config.json architectures {declared:?} \
         do not match any expected family {expected:?}"
    )]
    TaskMismatch {
        task: String,
        declared: Vec<String>,
        expected: Vec<&'static str>,
    },
    #[error(
        "onnx task mismatch: task {task} requires output \"{missing}\" but config.json \
         declares outputs {declared:?}"
    )]
    OutputMismatch {
        task: String,
        missing: String,
        declared: Vec<String>,
    },
    #[error("onnx session load failed for model \"{model}\": {detail}")]
    SessionLoad { model: String, detail: String },
    #[error("onnx session run failed for model \"{model}\": {detail}")]
    SessionRun { model: String, detail: String },
    #[error("LFM tokenizer error: {message}")]
    Tokenization { message: String },
    #[error("onnx error: {0}")]
    Other(String),
}

impl OrtError {
    pub fn tokenization(message: String) -> Self {
        Self::Tokenization { message }
    }
}