pub mod schema;
pub mod nodes;
pub mod edges;
pub mod hnsw;
pub mod embeddings;
pub mod kv_cache;

use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use std::sync::{Mutex, RwLock};

use anndists::dist::DistCosine;
use bitvec::vec::BitVec;
use fluent_types::{ContentNode, NodeId};
use guidance_llm::EmbeddingProvider;
use hnsw_rs::hnsw::Hnsw;
use search_vector::error::DbError;
use thiserror::Error;

pub const MAX_KNN_CANDIDATES: usize = 100_000;

#[derive(Error, Debug)]
pub enum LibraryError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] common_core::error::SqliteError),
    #[error("database error: {0}")]
    Db(#[from] DbError),
    #[error("node not found: {0}")]
    NodeNotFound(String),
    #[error("duplicate node: {0}")]
    DuplicateNode(String),
}

impl From<rusqlite::Error> for LibraryError {
    fn from(e: rusqlite::Error) -> Self {
        LibraryError::Sqlite(common_core::error::SqliteError(e))
    }
}

pub struct Library {
    pub(crate) conn: Mutex<rusqlite::Connection>,
    pub(crate) hnsw: RwLock<Option<Hnsw<'static, f32, DistCosine>>>,
    pub(crate) hnsw_id_map: Mutex<Vec<i64>>,
}

impl Library {
    pub fn open(path: &Path) -> Result<Self, LibraryError> {
        let conn = common_core::sqlite::open_wal(path)?;
        let lib = Self {
            conn: Mutex::new(conn),
            hnsw: RwLock::new(None),
            hnsw_id_map: Mutex::new(Vec::new()),
        };
        lib.init_schema()?;
        Ok(lib)
    }

    pub fn open_in_memory() -> Result<Self, LibraryError> {
        let conn = common_core::sqlite::open_in_memory()?;
        let lib = Self {
            conn: Mutex::new(conn),
            hnsw: RwLock::new(None),
            hnsw_id_map: Mutex::new(Vec::new()),
        };
        lib.init_schema()?;
        Ok(lib)
    }
}

pub(crate) fn blob_to_bitvec(b: &[u8]) -> BitVec {
    let words: Vec<usize> = b
        .chunks(size_of::<usize>())
        .map(|chunk| {
            let mut arr = [0u8; size_of::<usize>()];
            let len = chunk.len().min(size_of::<usize>());
            arr[..len].copy_from_slice(chunk);
            usize::from_le_bytes(arr)
        })
        .collect();
    BitVec::from_slice(&words)
}

pub struct HydrationPipeline<'a> {
    library: &'a Library,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl<'a> HydrationPipeline<'a> {
    pub fn new(library: &'a Library, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self { library, embedder }
    }

    pub fn insert_and_hydrate(&self, node: &mut ContentNode) -> Result<NodeId, LibraryError> {
        let text = node.lod.first().map_or("", String::as_str);
        if !text.is_empty() {
            if let Ok(emb) = self.embedder.embed(text) {
                node.embedding = Some(emb);
            }
        }
        let node_id = self.library.insert_node(node)?;
        if let Some(ref emb) = node.embedding {
            if let Ok(hits) = self.library.knn_search(emb, 10, None) {
                for hit in hits {
                    if hit.node_id != node_id && hit.distance < 0.3 {
                        let _ = self.library.insert_edge(
                            node_id,
                            hit.node_id,
                            "neighbor_of",
                            f64::from(hit.distance),
                        );
                    }
                }
            }
        }
        Ok(node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_types::{ContentNode, WasmTool};
    use rusqlite::params;

    fn make_empty_embeddings(count: usize, dims: usize) -> Vec<(String, Vec<f32>)> {
        (0..count)
            .map(|i| {
                let mut v = Vec::with_capacity(dims);
                for j in 0..dims {
                    v.push(((i * dims + j) as f32) / (count * dims) as f32);
                }
                (format!("node_{i}"), v)
            })
            .collect()
    }

    #[test]
    fn test_init_schema() {
        let lib = Library::open_in_memory().expect("in-memory db");
        assert!(lib.node_count().is_ok());
    }

    #[test]
    fn test_insert_and_get_node() {
        let lib = Library::open_in_memory().expect("in-memory db");
        let node = ContentNode {
            id: None,
            name: "test_node".into(),
            source: "full_source_text".into(),
            lod: vec!["summary".into(), "brief".into()],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let node_id = lib.insert_node(&node).expect("insert");
        assert!(node_id.as_int() > 0);

        let found = lib
            .find_node_by_name("test_node")
            .expect("find")
            .expect("should exist");
        assert_eq!(found.as_int(), node_id.as_int());
    }

    #[test]
    fn test_get_node_roundtrip() {
        let lib = Library::open_in_memory().expect("in-memory db");
        let emb: Vec<f32> = vec![0.1, 0.2, 0.3];
        let node = ContentNode {
            id: None,
            name: "roundtrip_node".into(),
            source: "source_text".into(),
            lod: vec!["full".into(), "summary".into(), "brief".into()],
            embedding: Some(emb.clone()),
            capabilities: None,
            ..Default::default()
        };
        let node_id = lib.insert_node(&node).expect("insert");

        let gotten = lib.get_node(node_id).expect("get").expect("should exist");
        assert_eq!(gotten.name.as_str(), "roundtrip_node");
        assert_eq!(gotten.source, "source_text");
        assert_eq!(gotten.lod.len(), 3);
        if let Some(got_emb) = &gotten.embedding {
            assert!((got_emb[0] - 0.1).abs() < 1e-6);
        } else {
            panic!("embedding should exist");
        }
    }

    #[test]
    fn test_knn_search() {
        let lib = Library::open_in_memory().expect("in-memory db");
        let items = make_empty_embeddings(10, 4);

        for (name, emb) in &items {
            let node = ContentNode {
                id: None,
                name: name.as_str().into(),
                source: "source".into(),
                lod: vec![],
                embedding: Some(emb.clone()),
                capabilities: None,
                ..Default::default()
            };
            lib.insert_node(&node).expect("insert");
        }

        let query: Vec<f32> = vec![0.0, 0.1, 0.2, 0.3];
        let hits = lib.knn_search(&query, 3, None).expect("knn search");
        assert_eq!(hits.len(), 3);
        assert!(hits[0].distance <= hits[1].distance);
    }

    #[test]
    fn test_traverse_from() {
        let lib = Library::open_in_memory().expect("in-memory db");

        let root = ContentNode {
            id: None,
            name: "root".into(),
            source: "root".into(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let root_id = lib.insert_node(&root).expect("insert");

        let child = ContentNode {
            id: None,
            name: "child".into(),
            source: "child".into(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let child_id = lib.insert_node(&child).expect("insert");

        let grandchild = ContentNode {
            id: None,
            name: "grandchild".into(),
            source: "grandchild".into(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let grandchild_id = lib.insert_node(&grandchild).expect("insert");

        lib.insert_edge(root_id, child_id, "depends", 1.0)
            .expect("edge");
        lib.insert_edge(child_id, grandchild_id, "depends", 1.0)
            .expect("edge");

        let nodes = lib.traverse_from(root_id, 2).expect("traverse");
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn test_insert_wasm_tool() {
        let lib = Library::open_in_memory().expect("in-memory db");
        let tool = WasmTool {
            name: "tokenizer".into(),
            path: "/bin/tokenizer.wasm".into(),
            capabilities: vec!["tokenize".into(), "split".into()],
        };
        lib.insert_wasm_tool(&tool).expect("insert");

        let found = lib.find_wasm_tools_by_capability("tokenize").expect("find");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name.as_str(), "tokenizer");

        let not_found = lib.find_wasm_tools_by_capability("embed").expect("find");
        assert!(not_found.is_empty());
    }

    #[test]
    fn test_embedding_cache() {
        let lib = Library::open_in_memory().expect("in-memory db");
        let emb: Vec<f32> = vec![0.5, 0.5, 0.5];
        lib.cache_embedding("hash123", "test query", &emb)
            .expect("cache");

        let cached = lib.get_cached_embedding("hash123").expect("get cached");
        assert!(cached.is_some());
        let cached = cached.unwrap();
        assert!((cached[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_knn_search_with_capability_filter() {
        let lib = Library::open_in_memory().expect("in-memory db");

        let node_a = ContentNode {
            id: None,
            name: "node_a".into(),
            source: "a".into(),
            lod: vec![],
            embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            capabilities: Some(vec![0b0001]),
            ..Default::default()
        };
        lib.insert_node(&node_a).expect("insert");

        let node_b = ContentNode {
            id: None,
            name: "node_b".into(),
            source: "b".into(),
            lod: vec![],
            embedding: Some(vec![0.5, 0.6, 0.7, 0.8]),
            capabilities: Some(vec![0b0010]),
            ..Default::default()
        };
        lib.insert_node(&node_b).expect("insert");

        let node_c = ContentNode {
            id: None,
            name: "node_c".into(),
            source: "c".into(),
            lod: vec![],
            embedding: Some(vec![0.9, 1.0, 1.1, 1.2]),
            capabilities: Some(vec![0b0100]),
            ..Default::default()
        };
        lib.insert_node(&node_c).expect("insert");

        let query = vec![0.0, 0.1, 0.2, 0.3];

        let mut filter_cap0 = BitVec::new();
        filter_cap0.resize(4, false);
        filter_cap0.set(0, true);
        let hits_cap0 = lib.knn_search(&query, 10, Some(&filter_cap0)).expect("knn");
        assert_eq!(hits_cap0.len(), 1);
        assert_eq!(hits_cap0[0].name.as_str(), "node_a");

        let mut filter_cap1 = BitVec::new();
        filter_cap1.resize(4, false);
        filter_cap1.set(1, true);
        let hits_cap1 = lib.knn_search(&query, 10, Some(&filter_cap1)).expect("knn");
        assert_eq!(hits_cap1.len(), 1);
        assert_eq!(hits_cap1[0].name.as_str(), "node_b");

        let mut filter_all = BitVec::new();
        filter_all.resize(4, false);
        filter_all.set(0, true);
        filter_all.set(1, true);
        filter_all.set(2, true);
        let hits_all = lib.knn_search(&query, 10, Some(&filter_all)).expect("knn");
        assert_eq!(hits_all.len(), 3);
    }

    #[test]
    fn test_is_a_duck_typing() {
        let lib = Library::open_in_memory().expect("db");
        let node = ContentNode {
            id: None,
            name: "alice".into(),
            source: "source".into(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let node_id = lib.insert_node(&node).expect("insert");

        lib.insert_entity_type(node_id, "http://schema.org/Person")
            .expect("insert type");
        lib.insert_entity_hierarchy("http://schema.org/Person", "http://schema.org/Thing")
            .expect("insert hierarchy");

        let is_person = lib.is_a(node_id, "http://schema.org/Person").expect("is_a");
        assert!(is_person, "node should be a Person");

        let is_thing = lib.is_a(node_id, "http://schema.org/Thing").expect("is_a");
        assert!(is_thing, "node should be a Thing via hierarchy");

        let is_place = lib.is_a(node_id, "http://schema.org/Place").expect("is_a");
        assert!(!is_place, "node should NOT be a Place");
    }

    #[test]
    fn test_hydration_pipeline() {
        use guidance_llm::NoopEmbedding;

        let lib = Library::open_in_memory().expect("db");
        let embedder = Arc::new(NoopEmbedding::new(4));
        let pipeline = HydrationPipeline::new(&lib, embedder);

        let mut node = ContentNode {
            id: None,
            name: "hydrate_test".into(),
            source: "test source".into(),
            lod: vec!["some text to embed".into()],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let node_id = pipeline.insert_and_hydrate(&mut node).expect("hydrate");
        assert!(node_id.as_int() > 0, "should get a valid node ID");

        let stored = lib.get_node(node_id).expect("get").expect("should exist");
        assert_eq!(stored.name.as_str(), "hydrate_test");
    }

    #[test]
    fn test_insert_entity_hierarchy() {
        let lib = Library::open_in_memory().expect("db");
        lib.insert_entity_hierarchy("sub", "super").expect("insert");
        let conn = lib.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM entity_hierarchy WHERE subclass_iri = ?1 AND superclass_iri = ?2",
            params!["sub", "super"],
            |row| row.get(0),
        ).expect("query");
        assert_eq!(count, 1);
    }
}
