//! Embedding vector math — re-export of `fluent-db::vector`.
//!
//! The canonical home for embedding math moved to `fluent-db` so the
//! dependency direction stays acyclic: `search-vector` depends on `fluent-db`,
//! never the reverse. This module is a pure re-export so the
//! `search_vector::math::*` paths keep working for `coral` and `guidance`
//! call sites.

pub use fluent_db::vector::*;
