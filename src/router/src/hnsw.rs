use std::sync::Arc;

use thiserror::Error;

/// Three separate HNSW indices per MOA_ROUTER_SPEC §9.
pub struct HnswIndices {
    /// Vague prompts → prior WorkflowConfig JSON.
    pub workflow_library: HnswIndexHandle,
    /// Extracted frontier hypotheticals → validated Q&A pairs.
    pub rubric_cache: HnswIndexHandle,
    /// Content embeddings → known-bad category embeddings.
    pub blacklist_similarity: HnswIndexHandle,
}

#[derive(Debug, Clone)]
pub struct HnswIndexHandle {
    pub name: String,
    pub path: String,
}

#[derive(Error, Debug)]
pub enum HnswError {
    #[error("index '{0}' not initialized: {1}")]
    NotInitialized(String, String),
    #[error("embedding error: {0}")]
    Embedding(String),
}

impl HnswIndices {
    pub fn new(
        workflow_path: &str,
        rubric_path: &str,
        blacklist_path: &str,
    ) -> Result<Self, HnswError> {
        Ok(Self {
            workflow_library: HnswIndexHandle {
                name: "workflow_library".into(),
                path: workflow_path.into(),
            },
            rubric_cache: HnswIndexHandle {
                name: "rubric_cache".into(),
                path: rubric_path.into(),
            },
            blacklist_similarity: HnswIndexHandle {
                name: "blacklist_similarity".into(),
                path: blacklist_path.into(),
            },
        })
    }
}

impl HnswIndexHandle {
    pub fn is_initialized(&self) -> bool {
        std::path::Path::new(&self.path).exists()
    }
}

pub type HnswIndicesRef = Arc<HnswIndices>;
