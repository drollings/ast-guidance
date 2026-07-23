//! content-node: Level-of-detail text slicing and file content
//! annotation. Produces annotated nodes from source files.

pub mod doc_node;
pub mod file_node;
pub mod lod;
pub mod node;
pub mod source_node;

pub use doc_node::DocumentContentNode;
pub use file_node::FileContentNode;
pub use lod::generate_lod_slices;
pub use node::{ContentNode, LodLevel, NodeType, NodeTypeInfo};
pub use source_node::SourceCodeContentNode;
