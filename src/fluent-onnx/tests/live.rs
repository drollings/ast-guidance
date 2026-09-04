//! Live-AI integration test crate for fluent-onnx.
//!
//! Compiled ONLY when the `live-ai` feature is enabled. Tests perform real
//! ONNX inference and are `#[ignore]`d; they run exclusively via
//! `make test-live` / `make ort-test-live`. See `tests/live/README.md` for
//! the env contract and skip-not-fail policy.

#![cfg(feature = "live-ai")]

#[path = "live/encoder_annotate_live.rs"]
mod encoder_annotate_live;
#[path = "live/encoder_live.rs"]
mod encoder_live;
#[path = "live/gpu_probe.rs"]
mod gpu_probe;
#[path = "live/llm_live.rs"]
mod llm_live;
#[path = "live/pii_live.rs"]
mod pii_live;
#[path = "live/policy_linter_live.rs"]
mod policy_linter_live;
#[path = "live/two_tower_live.rs"]
mod two_tower_live;