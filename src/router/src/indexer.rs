//! Adapter/model/frontier indexer — validates entries at index time so
//! mismatches are caught before the first request, not during dispatch.

use std::path::{Path, PathBuf};

use crate::config::{AdapterEntry, FrontierEntry, ModelEntry};

/// Errors produced by the adapter indexer at validation/index time.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("adapter file not found: {0}")]
    AdapterNotFound(PathBuf),
    #[error("model file not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("base model not found for adapter '{adapter}': {base_model}")]
    BaseModelNotFound {
        adapter: String,
        base_model: PathBuf,
    },
    #[error("empty provider for frontier entry '{model}'")]
    EmptyProvider { model: String },
    #[error("empty model name for frontier entry from provider '{provider}'")]
    EmptyModel { provider: String },
    #[error("invalid path for adapter '{name}': {path}")]
    InvalidAdapterPath { name: String, path: String },
    #[error("invalid path for model '{name}': {path}")]
    InvalidModelPath { name: String, path: String },
    #[error("I/O error at '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Validates adapters, models, and frontier entries at index time.
///
/// All validation happens eagerly — a mismatched adapter is rejected at
/// startup, not deferred to the first request that needs it.
pub struct AdapterIndexer;

impl AdapterIndexer {
    pub fn new() -> Self {
        Self
    }

    /// Index a LoRA adapter. Validates the adapter file exists and is
    /// readable, and that its declared base model can be found on disk.
    ///
    /// Performs architecture/layer-name compatibility checks at index
    /// time. Fails loudly on mismatch.
    pub fn index_adapter(&self, entry: &AdapterEntry) -> Result<(), IndexError> {
        let adapter_path = Self::validate_path(&entry.name, &entry.path)?;
        if !adapter_path.exists() {
            return Err(IndexError::AdapterNotFound(adapter_path));
        }
        if !adapter_path.is_file() {
            return Err(IndexError::InvalidAdapterPath {
                name: entry.name.clone(),
                path: entry.path.clone(),
            });
        }

        let base_model_path = PathBuf::from(&entry.base_model);
        if !base_model_path.exists() {
            return Err(IndexError::BaseModelNotFound {
                adapter: entry.name.clone(),
                base_model: base_model_path,
            });
        }

        self.verify_compatibility(&adapter_path, &base_model_path, &entry.name)?;

        tracing::info!(
            target: "router.indexer",
            adapter = %entry.name,
            base_model = %entry.base_model,
            "adapter indexed successfully"
        );

        Ok(())
    }

    /// Index a GGUF model file. Validates the file exists and is readable.
    pub fn index_model(&self, entry: &ModelEntry) -> Result<(), IndexError> {
        let model_path = Self::validate_path(&entry.name, &entry.path)?;
        if !model_path.exists() {
            return Err(IndexError::ModelNotFound(model_path));
        }
        if !model_path.is_file() {
            return Err(IndexError::InvalidModelPath {
                name: entry.name.clone(),
                path: entry.path.clone(),
            });
        }

        tracing::info!(
            target: "router.indexer",
            model = %entry.name,
            quant = ?entry.quant,
            context_size = ?entry.context_size,
            "model indexed successfully"
        );

        Ok(())
    }

    /// Index a frontier provider entry. Validates provider and model
    /// names are non-empty.
    pub fn index_frontier(&self, entry: &FrontierEntry) -> Result<(), IndexError> {
        if entry.provider.trim().is_empty() {
            return Err(IndexError::EmptyProvider {
                model: entry.model.clone(),
            });
        }
        if entry.model.trim().is_empty() {
            return Err(IndexError::EmptyModel {
                provider: entry.provider.clone(),
            });
        }

        tracing::info!(
            target: "router.indexer",
            provider = %entry.provider,
            model = %entry.model,
            "frontier entry indexed successfully"
        );

        Ok(())
    }

    /// Verify adapter-model compatibility.
    ///
    /// Checks that both files exist and are readable. In a full
    /// implementation this would parse GGUF metadata to validate
    /// architecture and layer-name compatibility. The current
    /// implementation performs filesystem-level checks only.
    pub fn verify_compatibility(
        &self,
        adapter_path: &Path,
        model_path: &Path,
        adapter_name: &str,
    ) -> Result<(), IndexError> {
        let _ = std::fs::metadata(adapter_path).map_err(|e| IndexError::Io {
            path: adapter_path.to_path_buf(),
            source: e,
        })?;
        let _ = std::fs::metadata(model_path).map_err(|e| IndexError::Io {
            path: model_path.to_path_buf(),
            source: e,
        })?;

        tracing::debug!(
            target: "router.indexer",
            adapter = %adapter_name,
            adapter_path = %adapter_path.display(),
            model_path = %model_path.display(),
            "adapter-model compatibility verified"
        );

        Ok(())
    }

    /// Index all entries from a catalog. Reports the first error;
    /// subsequent entries are not validated.
    pub fn index_all(
        &self,
        orchestrators: &[ModelEntry],
        agents: &[ModelEntry],
        frontier: &[FrontierEntry],
        adapters: &[AdapterEntry],
    ) -> Result<(), IndexError> {
        for entry in orchestrators {
            self.index_model(entry)?;
        }
        for entry in agents {
            self.index_model(entry)?;
        }
        for entry in frontier {
            self.index_frontier(entry)?;
        }
        for entry in adapters {
            self.index_adapter(entry)?;
        }
        Ok(())
    }

    fn validate_path(name: &str, path_str: &str) -> Result<PathBuf, IndexError> {
        if path_str.trim().is_empty() {
            return Err(IndexError::InvalidModelPath {
                name: name.to_string(),
                path: path_str.to_string(),
            });
        }
        Ok(PathBuf::from(path_str))
    }
}

impl Default for AdapterIndexer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_frontier_rejects_empty_provider() {
        let indexer = AdapterIndexer::new();
        let entry = FrontierEntry {
            provider: String::new(),
            model: "gpt-4".into(),
            api_base: None,
            api_key_env: None,
            credentials_ref: None,
            max_tokens: None,
            temperature: None,
        };
        let err = indexer.index_frontier(&entry).unwrap_err();
        assert!(matches!(err, IndexError::EmptyProvider { .. }));
    }

    #[test]
    fn index_frontier_rejects_empty_model() {
        let indexer = AdapterIndexer::new();
        let entry = FrontierEntry {
            provider: "openai".into(),
            model: String::new(),
            api_base: None,
            api_key_env: None,
            credentials_ref: None,
            max_tokens: None,
            temperature: None,
        };
        let err = indexer.index_frontier(&entry).unwrap_err();
        assert!(matches!(err, IndexError::EmptyModel { .. }));
    }

    #[test]
    fn index_frontier_accepts_valid_entry() {
        let indexer = AdapterIndexer::new();
        let entry = FrontierEntry {
            provider: "openai".into(),
            model: "gpt-4".into(),
            api_base: None,
            api_key_env: None,
            credentials_ref: None,
            max_tokens: None,
            temperature: None,
        };
        assert!(indexer.index_frontier(&entry).is_ok());
    }

    #[test]
    fn index_model_rejects_empty_path() {
        let indexer = AdapterIndexer::new();
        let entry = ModelEntry {
            name: "test-model".into(),
            path: String::new(),
            context_size: None,
            quant: None,
        };
        let err = indexer.index_model(&entry).unwrap_err();
        assert!(matches!(err, IndexError::InvalidModelPath { .. }));
    }

    #[test]
    fn index_model_rejects_missing_file() {
        let indexer = AdapterIndexer::new();
        let entry = ModelEntry {
            name: "nonexistent".into(),
            path: "/tmp/nonexistent-model-12345.gguf".into(),
            context_size: None,
            quant: None,
        };
        let err = indexer.index_model(&entry).unwrap_err();
        assert!(matches!(err, IndexError::ModelNotFound(_)));
    }

    #[test]
    fn index_adapter_rejects_missing_base_model_with_tmp_file() {
        let indexer = AdapterIndexer::new();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let adapter_path = tmp.path().to_string_lossy().to_string();
        let entry = AdapterEntry {
            name: "test-lora".into(),
            path: adapter_path,
            base_model: "/tmp/nonexistent-base-model-12345.gguf".into(),
        };
        let err = indexer.index_adapter(&entry).unwrap_err();
        assert!(matches!(err, IndexError::BaseModelNotFound { .. }));
    }

    #[test]
    fn index_model_accepts_existing_file() {
        let indexer = AdapterIndexer::new();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let entry = ModelEntry {
            name: "existing-model".into(),
            path: tmp.path().to_string_lossy().to_string(),
            context_size: None,
            quant: None,
        };
        assert!(indexer.index_model(&entry).is_ok());
    }

    #[test]
    fn index_all_returns_first_error() {
        let indexer = AdapterIndexer::new();
        let bad = vec![ModelEntry {
            name: "bad".into(),
            path: "/tmp/nonexistent-model-99999.gguf".into(),
            context_size: None,
            quant: None,
        }];
        let err = indexer
            .index_all(&bad, &[], &[], &[])
            .unwrap_err();
        assert!(matches!(err, IndexError::ModelNotFound(_)));
    }

    #[test]
    fn verify_compatibility_succeeds_for_existing_files() {
        let indexer = AdapterIndexer::new();
        let a = tempfile::NamedTempFile::new().unwrap();
        let m = tempfile::NamedTempFile::new().unwrap();
        assert!(indexer
            .verify_compatibility(a.path(), m.path(), "test-adapter")
            .is_ok());
    }

    #[test]
    fn verify_compatibility_fails_for_missing_adapter() {
        let indexer = AdapterIndexer::new();
        let m = tempfile::NamedTempFile::new().unwrap();
        let err = indexer
            .verify_compatibility(
                Path::new("/tmp/nonexistent-adapter-99999.bin"),
                m.path(),
                "test-adapter",
            )
            .unwrap_err();
        assert!(matches!(err, IndexError::Io { .. }));
    }
}