use std::sync::Arc;

use fluent_db::error::DbError;
use fluent_types::ContentNode;
use fluent_types::NodeId;
use guidance_ontology::mapper::PendingNode;
use guidance_ontology::yago;
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

common_core::impl_from_io_error!(IngestError);

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

        // `flush` counts batch.len() against stats.nodes_created, so the
        // returned `added` count is intentionally not added here again.
        self.add_pending_nodes(pending_nodes)?;

        self.flush()?;

        // The hash_iri values ARE the NodeIds now: `PendingNode.id` /
        // `PendingEdge.from_id|to_id` are content addresses, and
        // `insert_nodes_batch` preserved them as the primary key (§4.2), so
        // edges reference the hash ids directly — no id→name round-trip.
        for edge in &pending_edges {
            let from = NodeId::from_int(edge.from_id);
            let to = NodeId::from_int(edge.to_id);
            match self.library.insert_edge(from, to, &edge.predicate, 1.0) {
                Ok(()) => self.stats.edges_created += 1,
                Err(_) => self.stats.errors_skipped += 1,
            }
        }

        Ok(self.stats.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::make_node;

    /// A YaGO-style fragment: two classes joined by `rdfs:subClassOf`, each
    /// with a label. Content addresses come from `hash_iri`.
    const FRAGMENT: &str = "\
<http://yago-knowledge.org/resource/Person> <http://www.w3.org/2000/01/rdf-schema#label> \"Person\" .
<http://yago-knowledge.org/resource/Artist> <http://www.w3.org/2000/01/rdf-schema#label> \"Artist\" .
<http://yago-knowledge.org/resource/Artist> <http://www.w3.org/2000/01/rdf-schema#subClassOf> <http://yago-knowledge.org/resource/Person> .
";

    fn write_fragment(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("fragment.ttl");
        std::fs::write(&path, FRAGMENT).expect("write fragment");
        path
    }

    #[test]
    fn ingest_fragment_is_content_addressed_and_idempotent() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let mut ingestor = BatchIngestor::new(Arc::clone(&lib), 100);
        let stats = ingestor
            .ingest_file(&write_fragment(dir.path()))
            .expect("first ingest");
        assert_eq!(stats.nodes_created, 2);
        assert_eq!(stats.edges_created, 1);

        // Re-ingesting the same file is idempotent: no new rows, no duplicate
        // edges (the content-addressed INSERT OR IGNORE + unique edge index).
        let mut ingestor = BatchIngestor::new(Arc::clone(&lib), 100);
        let stats = ingestor
            .ingest_file(&write_fragment(dir.path()))
            .expect("second ingest");
        assert_eq!(stats.nodes_created, 2);
        assert_eq!(stats.edges_created, 1);
        assert_eq!(lib.node_count().expect("count"), 2);
        assert_eq!(lib.edge_count().expect("count"), 1);
    }

    #[test]
    fn ingest_edges_reference_hash_ids_and_traverse() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let mut ingestor = BatchIngestor::new(Arc::clone(&lib), 100);
        ingestor
            .ingest_file(&write_fragment(dir.path()))
            .expect("ingest");

        // The stored node ids ARE the content addresses (hash_iri).
        let artist = NodeId::from_int(guidance_rdf::normalize::hash_iri(
            "http://yago-knowledge.org/resource/Artist",
        ));
        let person = NodeId::from_int(guidance_rdf::normalize::hash_iri(
            "http://yago-knowledge.org/resource/Person",
        ));
        assert_eq!(lib.get_node(artist).expect("artist").expect("exists").id, Some(artist));
        assert_eq!(lib.get_node(person).expect("person").expect("exists").id, Some(person));

        // traverse_from follows the subClassOf edge: Artist → Person.
        let hops = lib.traverse_from(artist, 1).expect("traverse");
        let names: Vec<&str> = hops.iter().map(|g| g.name.as_str()).collect();
        assert!(names.contains(&"Artist") && names.contains(&"Person"));

        // The `is_a` CTE traverses `entity_types` × `entity_hierarchy`, both
        // keyed by the content-addressed node ids (roadmap 7.5).
        lib.insert_entity_type(artist, "http://yago-knowledge.org/resource/Artist")
            .expect("type");
        lib.insert_entity_hierarchy(
            "http://yago-knowledge.org/resource/Artist",
            "http://yago-knowledge.org/resource/Person",
        )
        .expect("hierarchy");
        assert!(lib.is_a(artist, "http://yago-knowledge.org/resource/Person").expect("is_a"));
    }

    #[test]
    fn test_batch_ingestor_buffers_then_flushes() {
        let lib = Arc::new(Library::open_in_memory().expect("db"));
        let mut ingestor = BatchIngestor::new(Arc::clone(&lib), 100);

        let node1 = make_node("batched_1", "source");
        let id = ingestor.add(node1).expect("add");
        assert!(id.is_none());
        assert_eq!(ingestor.pending_count(), 1);

        let node2 = make_node("batched_2", "source");
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
            embedding: Some(vec![0.1, 0.2, 0.3]),
            ..make_node("embedded_node", "source")
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

        let node1 = make_node("full_1", "s");
        ingestor.add(node1).expect("add");
        assert_eq!(ingestor.pending_count(), 1);

        let node2 = make_node("full_2", "s");
        ingestor.add(node2).expect("add");
        assert_eq!(ingestor.pending_count(), 0);

        assert!(lib.find_node_by_name("full_1").unwrap().is_some());
        assert!(lib.find_node_by_name("full_2").unwrap().is_some());
    }
}
