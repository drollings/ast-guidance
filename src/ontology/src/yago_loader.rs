//! The YaGO taxonomy loader (ROADMAP §13.3, M11).
//!
//! Parses the class registry (`ontology/data/yago_classes.json`, embedded at
//! build time or loaded from a downloaded file) into [`ConceptMetadata`]
//! with **deterministic content-addressed ids**:
//!
//! - `local = local_id_of(hash_iri(iri))` (48-bit truncation),
//! - `id    = InterlinguaId::new(YagoClass, local)`,
//! - `node_id = Some(NodeId::from_int(hash_iri(iri)))` — the **full 64-bit**
//!   hash (F5: stored, never derived; the 16 truncated-away bits are
//!   unrecoverable).
//!
//! The loader also exposes the `rdfs:subClassOf` edges so the boot sequence
//! can build the [`TaxonomyHierarchy`](spacy_rs::TaxonomyHierarchy) (C5).
//!
//! **One loader, two consumers (C3):** the same `Vec<ConceptMetadata>` feeds
//! both coral's durable content-addressed graph and the router's
//! `SqliteConceptStore`; a boot reconciliation locks them equal (§13.8).

use std::collections::HashMap;
use std::path::Path;

use fluent_types::{yago_class_id_for_iri, ConceptMetadata, InterlinguaId, InterlinguaNamespace, NodeId};
use guidance_rdf::normalize::hash_iri;
use serde::Deserialize;

use crate::OntologyError;

/// One registry entry (the `yago_classes.json` line format).
#[derive(Debug, Clone, Deserialize)]
pub struct ClassEntry {
    pub iri: String,
    pub label: String,
    #[serde(default)]
    pub superclass: Option<String>,
}

/// The embedded default registry — the curated sample shipped in the crate.
/// The full 130k taxonomy is produced by `tools/download_yago_taxonomy.sh` +
/// `tools/gen_yago_classes.py`.
pub const EMBEDDED_YAGO_CLASSES: &str = include_str!("../data/yago_classes.json");

/// Load statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadStats {
    pub classes: usize,
    pub labels: usize,
    pub edges: usize,
}

/// The compact canonical name for a class IRI: `schema:Person` for
/// schema.org, `yago:Person` for the YaGO resource namespace, else the IRI.
pub fn canonical_class_name(iri: &str) -> String {
    const SCHEMA: &str = "http://schema.org/";
    const YAGO: &str = "http://yago-knowledge.org/resource/";
    if let Some(local) = iri.strip_prefix(SCHEMA) {
        format!("schema:{local}")
    } else if let Some(local) = iri.strip_prefix(YAGO) {
        format!("yago:{local}")
    } else {
        iri.to_string()
    }
}

/// The deterministic YagoClass id for a class IRI.
pub fn yago_class_id(iri: &str) -> InterlinguaId {
    yago_class_id_for_iri(iri)
}

/// The `ConceptMetadata` for a registry entry (F5 discipline: `node_id` is
/// the full 64-bit hash, never derived from the truncated `local_id`).
fn to_metadata(entry: &ClassEntry) -> ConceptMetadata {
    let id = yago_class_id(&entry.iri);
    ConceptMetadata {
        id,
        canonical_name: canonical_class_name(&entry.iri),
        namespace: InterlinguaNamespace::YagoClass,
        yago_iri: Some(entry.iri.clone()),
        yago_class_iri: Some(entry.iri.clone()),
        label: Some(entry.label.clone()),
        node_id: Some(NodeId::from_int(hash_iri(&entry.iri))),
        // The single source of the subclass edge (DRY): the same `superclass`
        // field that drives `subclass_edges` below.
        parent_class_id: entry.superclass.as_ref().map(|iri| yago_class_id(iri)),
    }
}

/// The YaGO taxonomy loader: registry entries → concepts + subclass edges.
#[derive(Debug, Default)]
pub struct YaGoLoader {
    class_iri_to_id: HashMap<String, InterlinguaId>,
    concepts: Vec<ConceptMetadata>,
    edges: Vec<(InterlinguaId, InterlinguaId)>,
    stats: LoadStats,
}

impl YaGoLoader {
    /// An empty loader.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a registry JSON file (`[{iri, label, superclass}, ...]`) into
    /// concepts with deterministic ids and the subclass edges.
    pub fn load_taxonomy(&mut self, path: &Path) -> Result<LoadStats, OntologyError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| OntologyError::Mapping(format!("read {}: {e}", path.display())))?;
        self.load_taxonomy_json(&raw)
    }

    /// Parse the embedded default registry.
    pub fn load_embedded(&mut self) -> Result<LoadStats, OntologyError> {
        self.load_taxonomy_json(EMBEDDED_YAGO_CLASSES)
    }

    /// Parse a registry JSON string into concepts + edges.
    pub fn load_taxonomy_json(&mut self, raw: &str) -> Result<LoadStats, OntologyError> {
        let entries: Vec<ClassEntry> = serde_json::from_str(raw)
            .map_err(|e| OntologyError::Mapping(format!("parse yago_classes: {e}")))?;
        for entry in &entries {
            let meta = to_metadata(entry);
            self.class_iri_to_id.entry(entry.iri.clone()).or_insert(meta.id);
            // First-wins: a repeated IRI does not duplicate the concept.
            if !self.concepts.iter().any(|c| c.id == meta.id) {
                self.concepts.push(meta.clone());
            }
            if let Some(superclass) = &entry.superclass {
                self.edges.push((meta.id, yago_class_id(superclass)));
            }
        }
        self.stats.classes = self.concepts.len();
        self.stats.edges = self.edges.len();
        Ok(self.stats)
    }

    /// Merge label overrides from a `[{iri, label}]` JSON file into the
    /// already-loaded concepts (used for a separate labels export). Returns
    /// the number of labels applied.
    pub fn load_class_labels(&mut self, path: &Path) -> Result<usize, OntologyError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| OntologyError::Mapping(format!("read {}: {e}", path.display())))?;
        #[derive(Deserialize)]
        struct LabelEntry {
            iri: String,
            label: String,
        }
        let entries: Vec<LabelEntry> = serde_json::from_str(&raw)
            .map_err(|e| OntologyError::Mapping(format!("parse labels: {e}")))?;
        let mut applied = 0;
        for e in &entries {
            if let Some(id) = self.class_iri_to_id.get(&e.iri) {
                if let Some(meta) = self.concepts.iter_mut().find(|c| &c.id == id) {
                    meta.label = Some(e.label.clone());
                    applied += 1;
                }
            }
        }
        self.stats.labels = applied;
        Ok(applied)
    }

    /// The loaded concepts (the single write path feeds both consumers, C3).
    #[must_use]
    pub fn into_concepts(self) -> Vec<ConceptMetadata> {
        self.concepts
    }

    /// The `(child ← parent)` `rdfs:subClassOf` edges for building the
    /// [`TaxonomyHierarchy`](spacy_rs::TaxonomyHierarchy) at boot.
    #[must_use]
    pub fn subclass_edges(&self) -> &[(InterlinguaId, InterlinguaId)] {
        &self.edges
    }

    /// Load statistics.
    #[must_use]
    pub const fn stats(&self) -> LoadStats {
        self.stats
    }
}

#[cfg(test)]
#[path = "../tests/yago_loader.rs"]
mod tests;
