//! fluent-knowledge: Word/trigram index, CSR graph, frequency
//! tables, and tokenizer for project-level knowledge representation.

pub mod csr_graph;
pub mod freq_table;
pub mod index_header;
pub mod query_cache;
pub mod tokenizer;
pub mod trigram_index;
pub mod word_index;

pub use csr_graph::CsrGraph;
pub use freq_table::{build_frequency_table, default_frequency_table, pair_weight};
pub use index_header::Header as IndexHeader;
pub use query_cache::QueryCache;
pub use tokenizer::WordTokenizer;
pub use word_index::{DocRegistry, WordHit, WordIndex};
