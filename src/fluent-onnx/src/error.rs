//! Error type for `fluent-onnx` — re-exported from `fluent_llm`.
//!
//! The single definition lives in [`fluent_llm::onnx_error`]; this module
//! re-exports it so the crate's workers and existing `fluent_onnx::…` paths
//! keep resolving to the same type.

pub use fluent_llm::onnx_error::OrtError;
