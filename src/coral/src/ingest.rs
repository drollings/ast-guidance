use std::sync::Arc;

use std::collections::HashMap;

use fluent_types::ContentNode;
use fluent_types::NodeId;
use guidance_ontology::mapper::PendingNode;
use guidance_ontology::yago;
use search_vector::error::DbError;
use thiserror::Error;

use crate::db::Library;

#[derive(Error, Debug)]
pub enum IngestError {
    #[error("IO error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("database error: {0}")]
    Db(#[from] DbError),
    #[error("library error: {0}")]
    Library(#[from] crate::db::LibraryError),
    #[error("parse error: {0}")]
    Parse(String),
}

impl From<std::io::Error> for IngestError {
    fn from(e: std::io::Error) -> Self {
        IngestError::Io(common_core::error::IoError(e))
    }
}

#[derive(Debug, Clone, Default)]
pub struct IngestStats {
    pub triples_processed: usize,
    pub nodes_created: usize,
    pub edges_created: usize,
    pub errors_skipped: usize,
    pub batches_flushed: usize,
    pub triples_filtered: usize,
}

#[derive(Debug, Clone)]
pub struct IngestionConfig {
    pub yago_whitelist_only: bool,
    pub batch_size: usize,
    pub preferred_lang: String,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            yago_whitelist_only: false,
            batch_size: 10000,
            preferred_lang: "en".to_string(),
        }
    }
}

pub struct BatchIngestor {
    library: Arc<Library>,
    batch: Vec<ContentNode>,
    batch_size: usize,
    config: IngestionConfig,
    stats: IngestStats,
}

impl BatchIngestor {
    pub fn new(library: Arc<Library>, batch_size: usize) -> Self {
        Self {
            library,
            batch: Vec::with_capacity(batch_size),
            batch_size,
            config: IngestionConfig::default(),
            stats: IngestStats::default(),
        }
    }

    pub fn with_config(library: Arc<Library>, config: IngestionConfig) -> Self {
        let batch_size = config.batch_size;
        Self {
            library,
            batch: Vec::with_capacity(batch_size),
            batch_size,
            config,
            stats: IngestStats::default(),
        }
    }

    pub fn add(&mut self, node: ContentNode) -> Result<Option<NodeId>, IngestError> {
        let has_embedding = node.embedding.is_some();
        self.batch.push(node);

        if has_embedding || self.batch.len() >= self.batch_size {
            self.flush()?;
        }

        Ok(None)
    }

    pub fn add_pending_nodes(
        &mut self,
        pending_nodes: Vec<PendingNode>,
    ) -> Result<usize, IngestError> {
        let mut added = 0;
        for pn in pending_nodes {
            if self.config.yago_whitelist_only {
                let has_whitelisted = pn
                    .types
                    .iter()
                    .any(|&type_id| yago::is_whitelisted_hash(type_id));
                if !has_whitelisted {
                    self.stats.triples_filtered += 1;
                    continue;
                }
            }
            let cn = pn.to_content_node();
            added += 1;
            let has_embedding = cn.embedding.is_some();
            self.batch.push(cn);
            if has_embedding || self.batch.len() >= self.batch_size {
                self.flush()?;
            }
        }
        Ok(added)
    }

    pub fn flush(&mut self) -> Result<(), IngestError> {
        let batch = std::mem::take(&mut self.batch);
        if batch.is_empty() {
            return Ok(());
        }
        self.library.insert_nodes_batch(&batch)?;
        self.stats.nodes_created += batch.len();
        self.stats.batches_flushed += 1;
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.batch.len()
    }

    pub fn stats(&self) -> &IngestStats {
        &self.stats
    }

    /// Ingest triples from an RDF file (Turtle or N-Quads).
    /// Streams through the RDF parser, maps via TripleMapper, and flushes in batches.
    pub fn ingest_file(&mut self, path: &std::path::Path) -> Result<IngestStats, IngestError> {
        let source = common_core::io::read_to_string_err(path).map_err(IngestError::Io)?;

        let mut mapper = guidance_ontology::mapper::TripleMapper::new(
            guidance_ontology::mapper::MappingConfig {
                preferred_lang: self.config.preferred_lang.clone(),
                scope: path.to_string_lossy().to_string(),
            },
        );

        // Try Turtle parser first
        let parser = guidance_rdf::parser::Parser::new(&source);
        let mut triples_processed = 0;

        for result in parser {
            if let Ok(triple) = result {
                if mapper.process_triple(&triple).is_err() {
                    self.stats.errors_skipped += 1;
                    continue;
                }
                triples_processed += 1;
                self.stats.triples_processed += 1;
            } else {
                self.stats.errors_skipped += 1;
            }
        }

        // If no triples from Turtle, try N-Quads
        if triples_processed == 0 {
            for line in source.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Ok(Some(quad)) = guidance_rdf::nquads::NQuadsParser::parse_line(line) {
                    let triple = guidance_rdf::parser::Triple {
                        subject: quad.subject,
                        predicate: quad.predicate,
                        object: quad.object,
                    };
                    if mapper.process_triple(&triple).is_err() {
                        self.stats.errors_skipped += 1;
                        continue;
                    }
                    self.stats.triples_processed += 1;
                }
            }
        }

        // Drain mapper and add pending nodes
        let pending_nodes = mapper.drain_nodes();
        let pending_edges = mapper.drain_edges();

        // Build hash_id → node name map for edge resolution
        let mut id_to_name: HashMap<i64, String> = HashMap::new();
        for pn in &pending_nodes {
            let cn = pn.to_content_node();
            let name = cn.name.to_string();
            if !name.is_empty() {
                id_to_name.insert(pn.id, name);
            }
        }

        let added = self.add_pending_nodes(pending_nodes)?;
        self.stats.nodes_created += added;

        self.flush()?;

        // Insert edges: resolve hash-based IDs to SQLite NodeIds via node name lookup
        for edge in &pending_edges {
            let (Some(from_name), Some(to_name)) =
                (id_to_name.get(&edge.from_id), id_to_name.get(&edge.to_id))
            else {
                self.stats.errors_skipped += 1;
                continue;
            };
            let (Ok(Some(from_sql_id)), Ok(Some(to_sql_id))) = (
                self.library.find_node_by_name(from_name),
                self.library.find_node_by_name(to_name),
            ) else {
                self.stats.errors_skipped += 1;
                continue;
            };
            match self
                .library
                .insert_edge(from_sql_id, to_sql_id, &edge.predicate, 1.0)
            {
                Ok(()) => self.stats.edges_created += 1,
                Err(_) => self.stats.errors_skipped += 1,
            }
        }

        Ok(self.stats.clone())
    }
}

// TODO(M6): remove — superseded by guidance_ontology::mapper::TripleMapper
pub struct TripleMapper;

impl TripleMapper {
    pub fn new() -> Self {
        Self
    }

    pub fn map_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> (ContentNode, ContentNode) {
        let sub = ContentNode {
            id: None,
            name: subject.into(),
            source: format!("{subject} {predicate} {object}"),
            lod: vec![
                format!("{subject} {predicate} {object}"),
                format!("{subject} -> {object}"),
                subject.into(),
            ],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };

        let obj = ContentNode {
            id: None,
            name: object.into(),
            source: object.into(),
            lod: vec![object.into()],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };

        (sub, obj)
    }
}

impl Default for TripleMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_ingestor_buffers_then_flushes() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let mut ingestor = BatchIngestor::new(Arc::clone(&lib), 100);

        let node1 = ContentNode {
            id: None,
            name: "batched_1".into(),
            source: "source".into(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let id = ingestor.add(node1).expect("add");
        assert!(id.is_none());
        assert_eq!(ingestor.pending_count(), 1);

        let node2 = ContentNode {
            id: None,
            name: "batched_2".into(),
            source: "source".into(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        let id = ingestor.add(node2).expect("add");
        assert!(id.is_none());
        assert_eq!(ingestor.pending_count(), 2);

        ingestor.flush().expect("flush");
        assert_eq!(ingestor.pending_count(), 0);

        assert!(lib.find_node_by_name("batched_1").unwrap().is_some());
        assert!(lib.find_node_by_name("batched_2").unwrap().is_some());
    }

    #[test]
    fn test_batch_ingestor_flushes_on_embedding() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let mut ingestor = BatchIngestor::new(Arc::clone(&lib), 100);

        let node = ContentNode {
            id: None,
            name: "embedded_node".into(),
            source: "source".into(),
            lod: vec![],
            embedding: Some(vec![0.1, 0.2, 0.3]),
            capabilities: None,
            ..Default::default()
        };
        let id = ingestor.add(node).expect("add");
        assert!(id.is_none());
        assert_eq!(ingestor.pending_count(), 0);

        assert!(lib.find_node_by_name("embedded_node").unwrap().is_some());
    }

    #[test]
    fn test_batch_ingestor_flushes_when_full() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let mut ingestor = BatchIngestor::new(Arc::clone(&lib), 2);

        let node1 = ContentNode {
            id: None,
            name: "full_1".into(),
            source: "s".into(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        ingestor.add(node1).expect("add");
        assert_eq!(ingestor.pending_count(), 1);

        let node2 = ContentNode {
            id: None,
            name: "full_2".into(),
            source: "s".into(),
            lod: vec![],
            embedding: None,
            capabilities: None,
            ..Default::default()
        };
        ingestor.add(node2).expect("add");
        assert_eq!(ingestor.pending_count(), 0);

        assert!(lib.find_node_by_name("full_1").unwrap().is_some());
        assert!(lib.find_node_by_name("full_2").unwrap().is_some());
    }

    #[test]
    fn test_triple_mapper() {
        let mapper = TripleMapper::new();
        let (sub, obj) = mapper.map_triple("Zig", "is_a", "language");
        assert_eq!(sub.name.as_str(), "Zig");
        assert_eq!(obj.name.as_str(), "language");
        assert_eq!(sub.lod.len(), 3);
    }
}
