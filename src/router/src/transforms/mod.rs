pub mod codeword_anonymize;
pub mod none;
pub mod pii_anonymize;
pub mod decompose_hypothetical;
pub mod decompose_subtasks;
pub mod sanitize;
pub mod secret_mask;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::RouterRequest;

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum TransformError {
    #[error("transform failed: {0}")]
    Failed(String),
    #[error("transform not applicable: {0}")]
    NotApplicable(String),
}

pub trait TransformStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn transform(
        &self,
        request: &RouterRequest,
        pii_classes: &[String],
    ) -> Result<RouterRequest, TransformError>;
}

pub type TransformStrategyRef = Arc<dyn TransformStrategy>;
