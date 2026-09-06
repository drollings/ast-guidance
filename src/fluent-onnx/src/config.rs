//! ONNX model configuration schema — re-exported from `fluent_llm`.
//!
//! The single definition lives in [`fluent_llm::onnx_config`]; this module
//! re-exports it so the crate's workers and existing `fluent_onnx::config::…`
//! paths keep resolving to the same types.

pub use fluent_llm::onnx_config::{
    AnnotationHeads, AnnotationLabels, LlmIo, OnnxConfig, OnnxFleetConfig, OnnxInstanceProfile,
    OnnxRole, OnnxRoleConfig, OnnxTask, Quant, ResidencyPolicy,
};
