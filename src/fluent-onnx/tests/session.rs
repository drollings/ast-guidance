#[cfg(feature = "onnx")]
use crate::ort_loader::is_gpu_provider;

    /// The GPU-provider classification the loader dispatches on — the
    /// `"gpu"`/`"migraphx"` config values must map to the AMD ROCm path
    /// (case-insensitive), and CPU must never be misclassified.
    /// Loader-gated: `ort_loader` (and `is_gpu_provider`) need `ort`.
    #[cfg(feature = "onnx")]
    #[test]
    fn gpu_provider_classification() {
        for gpu in ["gpu", "GPU", "migraphx", "MIGraphX", "Migraphx"] {
            assert!(
                is_gpu_provider(gpu),
                "{gpu:?} should classify as the AMD ROCm GPU provider"
            );
        }
        for cpu in ["cpu", "CPU", "Cpu"] {
            assert!(
                !is_gpu_provider(cpu),
                "{cpu:?} must not classify as the GPU provider"
            );
        }
        assert!(!is_gpu_provider("tensorrt"));
        assert!(!is_gpu_provider(""));
    }
