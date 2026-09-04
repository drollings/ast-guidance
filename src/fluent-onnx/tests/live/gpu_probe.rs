//! Live-AI probe for the ONNX Runtime execution-provider link.
//!
//! Reports whether the *linked* onnxruntime exposes the AMD ROCm GPU
//! (MIGraphX) execution provider — the single fact that decides whether
//! `execution_provider: "gpu"` in `env/coral-router.json` engages the GPU or
//! fails open to the CPU (with a loud warning). Requires no model: it probes
//! the runtime's available-providers table. Run via `make onnx-gpu-check`
//! (the full `make ort-test-live` also runs it). The ort prebuilt binaries
//! ship no AMD GPU EP (the ROCm EP was removed upstream in ORT 1.23; MIGraphX
//! is its successor), so a CPU-only report is expected unless a
//! MIGraphX-enabled onnxruntime (an AMD ROCm build) is linked via
//! `ORT_LIB_PATH` (static) or `ORT_DYLIB_PATH` (load-dynamic) and the crate
//! rebuilt. See `doc/fluent-onnx/ARCHITECTURE.md` §Execution-provider selection.

use ort::ep::ExecutionProvider;

#[test]
#[ignore = "live-AI: probes the linked onnxruntime for the AMD ROCm GPU (MIGraphX) EP"]
fn gpu_provider_available() {
    let gpu = ort::ep::MIGraphX::default()
        .is_available()
        .unwrap_or(false);
    let cpu = ort::ep::CPU::default().is_available().unwrap_or(false);

    eprintln!("onnxruntime providers: CPU={cpu}, MIGraphX (AMD ROCm GPU)={gpu}");
    eprintln!(
        "execution_provider \"gpu\" will {}",
        if gpu {
            "engage the GPU"
        } else {
            "fail open to CPU (link a MIGraphX-enabled onnxruntime via ORT_LIB_PATH to accelerate)"
        }
    );

    // A diagnostic, not a gate: with only the CPU prebuilt linked, GPU is
    // expected to be absent and the loader logs a loud warning + CPU fallback.
    assert!(cpu, "CPU execution provider must be available in the linked onnxruntime");
}