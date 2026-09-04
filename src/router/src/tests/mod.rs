//! Router test modules.
//!
//! Test modules are conditionally compiled with `#[cfg(test)]` and declared
//! in `lib.rs`. This module provides:
//! - `rubric_fixtures` — rubric-based scoring tests
//! - `golden` — golden test set (labeled corpus)
//! - `e2e_tests` — end-to-end pipeline tests (mock mode)

pub mod common;
pub mod e2e_tests;
pub mod golden;
pub mod liveness_calibration;
pub mod overlay_calibration;
pub mod rubric_fixtures;
pub mod threshold_calibration;
