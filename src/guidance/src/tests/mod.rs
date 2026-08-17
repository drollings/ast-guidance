//! Tier-1 test suites for guidance-core, wired from `lib.rs` via
//! `#[cfg(test)] mod tests;`.
//!
//! Suites stay inline in their owning source modules (Tier 0); this module
//! only hosts the crate-typed fixtures they share (`common`).

pub mod common;