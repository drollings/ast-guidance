//! Real session loader backed by ONNX Runtime (feature `onnx`).
//!
//! Extracted from the former `session.rs`: the pure registry now lives in
//! `fluent_llm::onnx_session` and this module keeps only the `ort`-bound
//! loader behind the `onnx` feature.

use super::{SessionHandle, SessionLoader};
use crate::config::OnnxConfig;
use crate::error::OrtError;
use ort::ep::ExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;

/// Real session loader backed by ONNX Runtime.
#[derive(Default)]
pub struct OrtSessionLoader;

fn optimization_level(name: &str) -> GraphOptimizationLevel {
    match name {
        "disable" => GraphOptimizationLevel::Disable,
        "basic" => GraphOptimizationLevel::Level1,
        "extended" => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::All,
    }
}

/// Whether a config's `execution_provider` requests the AMD ROCm GPU
/// provider. Case-insensitive. The GPU provider is `MIGraphX` — AMD's
/// supported ONNX Runtime EP for ROCm. (The `ROCMExecutionProvider` was
/// removed from upstream ORT in 1.23; MIGraphX is its successor.)
pub(crate) fn is_gpu_provider(name: &str) -> bool {
    name.eq_ignore_ascii_case("gpu") || name.eq_ignore_ascii_case("migraphx")
}

impl SessionLoader for OrtSessionLoader {
    fn load(&self, config: &OnnxConfig, model_key: &str) -> Result<SessionHandle, OrtError> {
        let model_file = config.resolve_model_file()?;
        let mut builder = ort::session::Session::builder()
            .map_err(|e| session_load_error(model_key, &e))?
            .with_intra_threads(config.intra_threads)
            .map_err(|e| session_load_error(model_key, &e))?
            .with_optimization_level(optimization_level(&config.optimization_level))
            .map_err(|e| session_load_error(model_key, &e))?;

        // Execution-provider selection. Only the CPU and the AMD ROCm GPU
        // providers are wired:
        //
        //   "cpu"        → `CPUExecutionProvider` (deterministic, the
        //                  hermetic default).
        //   "gpu" | "migraphx"
        //                → `MIGraphXExecutionProvider` — AMD's supported
        //                  GPU EP for ROCm (successor to the removed
        //                  `ROCMExecutionProvider`). The linked runtime is
        //                  probed first: a build without MIGraphX support
        //                  (e.g. the CPU prebuilt binary) fails open to the
        //                  CPU with a loud, actionable warning — a `"gpu"`
        //                  request is never silently served.
        //
        // Anything else fails open to the CPU with a loud warning (a
        // mistyped or unsupported provider must never silently pretend to
        // accelerate).
        if config.execution_provider.eq_ignore_ascii_case("cpu") {
            builder = builder
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(|e| session_load_error(model_key, &e))?;
        } else if is_gpu_provider(&config.execution_provider) {
            let gpu = ort::ep::MIGraphX::default();
            match gpu.is_available() {
                Ok(true) => {
                    tracing::info!(
                        target: "fluent-onnx",
                        model = model_key,
                        execution_provider = %config.execution_provider,
                        "using MIGraphX (AMD ROCm GPU) execution provider",
                    );
                    builder = builder
                        .with_execution_providers([gpu.build()])
                        .map_err(|e| session_load_error(model_key, &e))?;
                }
                Ok(false) => {
                    tracing::warn!(
                        target: "fluent-onnx",
                        model = model_key,
                        execution_provider = %config.execution_provider,
                        "GPU execution requested but the linked ONNX Runtime has no \
                         MIGraphX (AMD ROCm) support; falling back to the CPU. Link a \
                         MIGraphX-enabled onnxruntime (an AMD ROCm build, e.g. via \
                         ORT_LIB_PATH) to actually accelerate",
                    );
                    builder = builder
                        .with_execution_providers([ort::ep::CPU::default().build()])
                        .map_err(|e| session_load_error(model_key, &e))?;
                }
                Err(e) => {
                    tracing::warn!(
                        target: "fluent-onnx",
                        model = model_key,
                        execution_provider = %config.execution_provider,
                        error = %e,
                        "GPU execution probe failed; falling back to the CPU",
                    );
                    builder = builder
                        .with_execution_providers([ort::ep::CPU::default().build()])
                        .map_err(|e| session_load_error(model_key, &e))?;
                }
            }
        } else {
            tracing::warn!(
                target: "fluent-onnx",
                model = model_key,
                execution_provider = %config.execution_provider,
                "requested execution provider is not wired; falling back to the default (CPU)",
            );
            builder = builder
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(|e| session_load_error(model_key, &e))?;
        }

        let session = builder
            .commit_from_file(&model_file)
            .map_err(|e| session_load_error(model_key, &e))?;
        // Session::run needs `&mut self`; the handle stores a Mutex so
        // every worker (encoder, two-tower, PII) can serialize runs on the
        // shared session.
        Ok(SessionHandle::new(std::sync::Mutex::new(session)))
    }
}

fn session_load_error(model_key: &str, source: &impl ToString) -> OrtError {
    OrtError::SessionLoad {
        model: model_key.to_string(),
        detail: source.to_string(),
    }
}
