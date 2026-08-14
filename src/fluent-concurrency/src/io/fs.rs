//! Capability-gated filesystem I/O (read, write, metadata).
//!
//! The `FsCapability` token and its async operations now live in the
//! capability model's canonical home, `fluent-wvr::capability`. This module
//! re-exports it so the existing `fluent_concurrency::io::fs::FsCapability`
//! path (and its `read`/`write`/`metadata` methods) keeps working unchanged,
//! and the sync serving-path gate
//! (`fluent_wvr::capability::capability_aware_fs`) shares the same single
//! token type. One `FsCapability` type, two gated surfaces.

pub use fluent_wvr::capability::FsCapability;
