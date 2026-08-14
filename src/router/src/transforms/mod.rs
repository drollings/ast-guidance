pub mod codeword_anonymize;
pub mod decompose_hypothetical;
pub mod decompose_subtasks;
pub mod none;
pub mod pii_anonymize;
pub mod sanitize;
pub mod secret_mask;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::{RouterMessageContent, RouterRequest};

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum TransformError {
    #[error("transform failed: {0}")]
    Failed(String),
    #[error("transform not applicable: {0}")]
    NotApplicable(String),
}

pub trait TransformStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError>;
}

pub type TransformStrategyRef = Arc<dyn TransformStrategy>;

/// Shared clone-messages/iterate/match boilerplate for transforms that rewrite
/// each `Text` message in place (the six-copy clone/iterate/match
/// skeleton is consolidated here). `Parts` messages are left untouched.
///
/// Each transform becomes a thin closure: only the per-message rewrite
/// (and any side-channel it wants, e.g. a captured anonymize map) lives in the
/// caller.
pub fn rewrite_text_messages(
    request: &RouterRequest,
    mut rewrite: impl FnMut(&str) -> Result<String, TransformError>,
) -> Result<RouterRequest, TransformError> {
    let mut transformed = request.clone();
    for message in &mut transformed.messages {
        let text = match &message.content {
            RouterMessageContent::Text(s) => s.clone(),
            RouterMessageContent::Parts(_) => continue,
        };
        message.content = RouterMessageContent::Text(rewrite(&text)?);
    }
    Ok(transformed)
}
