//! SQLite-backed database capability (pooled, async).
//!
//! The pooled connection machinery and the `DbCapability` token now live in
//! `fluent-db` (the canonical database-access crate, D5). This module keeps
//! the historical `fluent_concurrency::io::db::DbCapability` path alive for
//! existing callers.

pub use fluent_db::capability::DbCapability;
