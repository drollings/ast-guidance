//! Embedding vector math — re-export of `fluent-db::vector` (M4).
//!
//! The canonical home for embedding math moved to `fluent-db` (D8) so the
//! dependency direction stays acyclic: `search-vector` depends on `fluent-db`,
//! never the reverse. This module is a pure re-export so the
//! `search_vector::math::*` paths keep working for `coral` and `guidance`
//! call sites.

pub use fluent_db::vector::*;
