//! ONNX session registry — re-exported from `fluent_llm`.
//!
//! The single definition lives in [`fluent_llm::onnx_session`]; this module
//! re-exports it so the crate's workers and existing `fluent_onnx::session::…`
//! paths keep resolving to the same types. The `ort`-bound loader lives in
//! [`crate::ort_loader`] behind the `onnx` feature.

pub use fluent_llm::onnx_session::{
    OrtSessionRegistry, ResidencyReportEntry, SessionHandle, SessionLoader,
};
