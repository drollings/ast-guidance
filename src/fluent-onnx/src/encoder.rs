//! `OrtEncoder` — the base Encoder session as an `EmbeddingProvider`
//! (ROADMAP_20260827_ORT §1).
//!
//! Mean-pools `last_hidden_state` over non-pad tokens (mask-aware). The
//! pooling math is a pure host-side function (`mean_pool`), so its correctness
//! is unit-tested without a model; the session-facing glue is behind the
//! `onnx` feature.
//!
//! ## Determinism
//!
//! onnxruntime's reduction kernels are nondeterministic across threads, so
//! sessions run with `intra_op_threads=1` (the config default) — a fixed
//! input then produces bit-identical output run after run, which the live
//! test (`tests/live/encoder_live.rs`) asserts. Parallelism comes from
//! batching (`embed_batch`), never from intra-op threads.
//!
//! ## Quantization acceptance
//!
//! The crate does NOT license a quantization globally. The live test records
//! the q8-vs-reference cosine drift band on a fixed sample; each consumer
//! (chart/ledger retrieval vs entity-similarity) gates its own q8 decision in
//! its own acceptance test (ROADMAP_20260827_ORT §1.4).

/// Mask-aware mean-pool of `last_hidden_state` over non-pad tokens.
///
/// `hidden` is `seq * dims` floats, `mask` is `seq` i64s (`0` = pad). A
/// fully-masked row yields a zero vector (never NaN — matching
/// `NoopEmbedding`'s empty-vector convention).
pub fn mean_pool(hidden: &[f32], mask: &[i64], seq: usize, dims: usize) -> Vec<f32> {
    debug_assert_eq!(hidden.len(), seq * dims, "hidden length must be seq*dims");
    debug_assert_eq!(mask.len(), seq, "mask length must equal seq");
    let mut pooled = vec![0.0f32; dims];
    let mut count = 0usize;
    for i in 0..seq {
        if mask[i] == 0 {
            continue;
        }
        let row = &hidden[i * dims..(i + 1) * dims];
        for (p, v) in pooled.iter_mut().zip(row.iter()) {
            *p += v;
        }
        count += 1;
    }
    if count > 0 {
        let inv = 1.0 / count as f32;
        for p in &mut pooled {
            *p *= inv;
        }
    }
    pooled
}

#[cfg(feature = "onnx")]
mod ort_encoder {
    use std::sync::{Arc, Mutex};

    use fluent_llm::embeddings::{BatchEmbedding, EmbeddingError, EmbeddingProvider};

    use crate::config::OnnxConfig;
    use crate::error::OrtError;
    use crate::session::SessionHandle;
    use crate::tokenizer::{LfmEncoding, LfmTokenizer};

    /// Base Encoder session: LFM-tokenize → `session.run` → mask-aware
    /// mean-pool. `name()` returns `"onnx:<key>"` so a
    /// `CachedEmbeddingProvider` cache key is model-scoped.
    pub struct OrtEncoder {
        session: Arc<Mutex<ort::session::Session>>,
        tokenizer: Arc<LfmTokenizer>,
        dims: u32,
        name: &'static str,
    }

    impl OrtEncoder {
        /// Build an encoder over an already-loaded registry session handle.
        pub fn from_handle(
            handle: &SessionHandle,
            config: &OnnxConfig,
            model_key: &str,
        ) -> Result<Self, OrtError> {
            let session = handle
                .downcast_arc::<Mutex<ort::session::Session>>()
                .ok_or_else(|| {
                    OrtError::Other("session handle does not hold an ort session".to_string())
                })?;
            let tokenizer_path = config.tokenizer_path.as_ref().ok_or_else(|| {
                OrtError::Other(format!("encoder role '{model_key}' missing tokenizer_path"))
            })?;
            let tokenizer = LfmTokenizer::from_file(tokenizer_path, config.max_seq_len)?;
            let dims = config.dims.unwrap_or(1024);
            // The trait returns `&'static str`; the key is dynamic. One leak
            // per encoder (bounded by the registry's model count) is
            // acceptable and keeps `name()` model-scoped for cache keys.
            let name: &'static str = Box::leak(format!("onnx:{model_key}").into_boxed_str());
            Ok(Self {
                session,
                tokenizer,
                dims,
                name,
            })
        }

        fn run_encoding(&self, encoding: &LfmEncoding) -> Result<Vec<f32>, EmbeddingError> {
            let seq = encoding.len().max(1);
            let ids: Vec<i64> = encoding.ids.iter().map(|&i| i64::from(i)).collect();
            let mask: Vec<i64> = encoding.attention_mask.iter().map(|&m| i64::from(m)).collect();
            let shape = [1usize, seq];
            let input = ort::value::Tensor::from_array((shape, ids))
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            let mask_t = ort::value::Tensor::from_array((shape, mask.clone()))
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            let mut guard = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = guard
                .run(ort::inputs!["input_ids" => input, "attention_mask" => mask_t])
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            let (_, hidden) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            let dims = self.dims as usize;
            let full_mask = full_mask(seq, &mask);
            Ok(super::mean_pool(hidden, &full_mask, seq, dims))
        }
    }

    /// Pad a truncated mask out to the padded sequence length (`0` past the
    /// real tokens — the model pads with zero attention there).
    fn full_mask(seq: usize, mask: &[i64]) -> Vec<i64> {
        let mut out = vec![0i64; seq];
        out[..mask.len()].copy_from_slice(mask);
        out
    }

    impl EmbeddingProvider for OrtEncoder {
        fn name(&self) -> &'static str {
            self.name
        }

        fn dimensions(&self) -> u32 {
            self.dims
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            if text.is_empty() {
                return Ok(Vec::new());
            }
            let encoding = self
                .tokenizer
                .encode(text)
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            self.run_encoding(&encoding)
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbedding, EmbeddingError> {
            let dims = self.dims as usize;
            if texts.is_empty() {
                return Ok(BatchEmbedding {
                    flat: vec![],
                    count: 0,
                    dims,
                });
            }
            let encodings: Vec<LfmEncoding> = texts
                .iter()
                .map(|t| self.tokenizer.encode(t))
                .collect::<Result<_, OrtError>>()
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            let batch = encodings.len();
            let max_len = encodings
                .iter()
                .map(LfmEncoding::len)
                .max()
                .unwrap_or(0)
                .max(1);

            let mut ids = vec![0i64; batch * max_len];
            let mut mask = vec![0i64; batch * max_len];
            for (bi, enc) in encodings.iter().enumerate() {
                for (si, &id) in enc.ids.iter().enumerate() {
                    ids[bi * max_len + si] = i64::from(id);
                }
                for (si, &m) in enc.attention_mask.iter().enumerate() {
                    mask[bi * max_len + si] = i64::from(m);
                }
            }

            let shape = [batch, max_len];
            let input = ort::value::Tensor::from_array((shape, ids))
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            let mask_t = ort::value::Tensor::from_array((shape, mask.clone()))
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            let mut guard = self
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let outputs = guard
                .run(ort::inputs!["input_ids" => input, "attention_mask" => mask_t])
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;
            let (_, hidden) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;

            let mut flat = Vec::with_capacity(batch * dims);
            for bi in 0..batch {
                let row_start = bi * max_len * dims;
                let row = &hidden[row_start..row_start + max_len * dims];
                let row_mask = &mask[bi * max_len..(bi + 1) * max_len];
                flat.extend(super::mean_pool(row, row_mask, max_len, dims));
            }
            Ok(BatchEmbedding {
                flat,
                count: batch,
                dims,
            })
        }
    }

    /// Build the raw (uncached) encoder over a loaded registry session handle.
    /// The router wraps the result in `CachedEmbeddingProvider` — caching is
    /// deliberately NOT done here. Returns the concrete encoder so the router
    /// can wrap it generically.
    pub fn build_encoder(
        config: &OnnxConfig,
        model_key: &str,
        handle: &SessionHandle,
    ) -> Result<OrtEncoder, OrtError> {
        OrtEncoder::from_handle(handle, config, model_key)
    }

    /// Build the encoder from the registry's session for `model_key`, if the
    /// model is registered and its task is `FillMask`.
    pub fn build_encoder_from_registry(
        registry: &crate::session::OrtSessionRegistry,
        model_key: &str,
    ) -> Result<Option<OrtEncoder>, OrtError> {
        let Some(config) = registry.config(model_key) else {
            return Ok(None);
        };
        if config.task != crate::config::OnnxTask::FillMask {
            return Ok(None);
        }
        let Some(handle) = registry.ensure_loaded(model_key)? else {
            return Ok(None);
        };
        build_encoder(&config, model_key, &handle).map(Some)
    }
}

#[cfg(feature = "onnx")]
pub use ort_encoder::{build_encoder, build_encoder_from_registry, OrtEncoder};
/// Factory + registry-selection behavior (M1.3), hermetic — no real session.
#[cfg(all(test, feature = "onnx"))]
mod factory_tests {
    use std::sync::Arc;

    use super::*;
    use crate::config::{OnnxConfig, OnnxTask, Quant};
    use crate::error::OrtError;
    use crate::session::{OrtSessionRegistry, SessionHandle, SessionLoader};

    /// A loader that returns a non-ort handle (e.g. a test stub session).
    #[derive(Default)]
    struct StubLoader;

    impl SessionLoader for StubLoader {
        fn load(&self, _config: &OnnxConfig, _model_key: &str) -> Result<SessionHandle, OrtError> {
            Ok(SessionHandle::new("not an ort session"))
        }
    }

    fn config_for(task: OnnxTask) -> OnnxConfig {
        OnnxConfig::new()
            .model_path("/models/test.onnx")
            .tokenizer_path("/models/tokenizer.json")
            .task(task)
            .quantization(Quant::Q8)
            .build()
    }

    #[test]
    fn unregistered_model_yields_none() {
        let registry = OrtSessionRegistry::new(Arc::new(StubLoader));
        let encoder = build_encoder_from_registry(&registry, "missing").expect("no error");
        assert!(encoder.is_none());
    }

    #[test]
    fn non_encoder_task_yields_none() {
        let registry = OrtSessionRegistry::new(Arc::new(StubLoader));
        registry
            .register("router", config_for(OnnxTask::ZeroShotRouting))
            .expect("register");
        let encoder = build_encoder_from_registry(&registry, "router").expect("no error");
        assert!(encoder.is_none());
    }

    #[test]
    fn wrong_handle_type_is_a_loud_error() {
        let registry = OrtSessionRegistry::new(Arc::new(StubLoader));
        registry
            .register("encoder", config_for(OnnxTask::FillMask))
            .expect("register");
        // The stub loader returns a non-ort handle: building the encoder must
        // surface a loud error, never a silent None (a broken registry entry
        // must not masquerade as "no encoder configured").
        let result = build_encoder_from_registry(&registry, "encoder");
        assert!(result.is_err());
    }
}

#[cfg(test)]
#[path = "../tests/encoder.rs"]
mod tests;
