//! search-vector: SQLite hybrid search engine — KNN vector search,
//! keyword search, RRF merge, quantized embeddings, and semantic aliases.

pub mod aliases;
pub mod db;
pub mod error;
pub mod math;

pub use aliases::SemanticAliases;
pub use db::GuidanceDb;
pub use math::QuantizedEmbedding;
