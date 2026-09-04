//! Live-AI integration test crate for spacy-rs.
//!
//! Compiled ONLY when the `live-ai` feature is enabled. Tests perform real
//! model calls and are `#[ignore]`d; they run exclusively via
//! `make test-live` / `make spacy-test-live`. See `tests/live/README.md` for
//! the env contract and skip-not-fail policy.

#![cfg(feature = "live-ai")]

#[path = "live/smoke_live.rs"]
mod smoke_live;

#[path = "live/refine_live.rs"]
mod refine_live;
