//! The router's durable [`ConceptStore`] over the shared ledger SQLite
//! connection (ROADMAP §14.3, C3).
//!
//! `interlingua_concepts` is the materialized runtime index of the YaGO
//! taxonomy — the **second home** of the one-loader/two-consumers design: the
//! same `Vec<ConceptMetadata>` that fed coral's durable content-addressed
//! graph is inserted here, keyed by the same ids, and boot reconciliation
//! (11.10/13.10) locks the two homes equal.
//!
//! Ancestor/subclass queries delegate to a boot-built
//! [`TaxonomyHierarchy`](fluent_concept::TaxonomyHierarchy) over the
//! `parent_class_id` edges (C5 — the DAG primitive, not a hand-rolled walk).

use std::sync::Arc;
use std::sync::RwLock;

use fluent_concept::{ConceptStore, ConceptStoreError, TaxonomyHierarchy};
use fluent_db::store::SqliteStore;
use fluent_types::{ConceptMetadata, InterlinguaId, InterlinguaNamespace, NodeId, local_id_of};
use rusqlite::params;

/// `ConceptStore` over the `interlingua_concepts` table, sharing the ledger's
/// SQLite connection.
pub struct SqliteConceptStore {
    store: Arc<SqliteStore>,
    /// Boot-built from the `parent_class_id` edges (C5).
    hierarchy: RwLock<TaxonomyHierarchy>,
}

impl std::fmt::Debug for SqliteConceptStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteConceptStore").finish_non_exhaustive()
    }
}

impl SqliteConceptStore {
    /// A store over the shared connection.
    pub fn new(store: Arc<SqliteStore>) -> Self {
        Self {
            store,
            hierarchy: RwLock::new(TaxonomyHierarchy::default()),
        }
    }

    /// Attach the subclass hierarchy backing `ancestors_of`/`is_subclass_of`.
    /// Built once at boot from the `rdfs:subClassOf` edges (§11.7/§13.9).
    pub fn set_hierarchy(&self, hierarchy: TaxonomyHierarchy) {
        *self
            .hierarchy
            .write()
            .expect("concept store write lock poisoned") = hierarchy;
    }

    /// Load every `parent_class_id` edge from the table and rebuild the
    /// hierarchy — the lazy alternative to an explicit boot-time
    /// `set_hierarchy` (idempotent; cheap for the ~130k-class registry).
    pub fn hydrate_hierarchy(&self) -> Result<(), ConceptStoreError> {
        let rows = self
            .store
            .query_rows(
                "SELECT id, parent_class_id FROM interlingua_concepts \
                 WHERE parent_class_id IS NOT NULL",
                &[],
                |row| {
                    let id: i64 = row.get(0)?;
                    let parent: i64 = row.get(1)?;
                    Ok((id, parent))
                },
            )
            .map_err(|e| ConceptStoreError::Storage(e.to_string()))?;
        let edges: Vec<(InterlinguaId, InterlinguaId)> = rows
            .into_iter()
            .filter_map(|(id, parent)| {
                Some((
                    InterlinguaId::from_i64(id)?,
                    InterlinguaId::from_i64(parent)?,
                ))
            })
            .collect();
        let hierarchy = TaxonomyHierarchy::from_edges(&edges)?;
        self.set_hierarchy(hierarchy);
        Ok(())
    }

    /// The class count (used by the boot reconciliation with coral, §13.8).
    pub fn yago_class_count(&self) -> Result<i64, ConceptStoreError> {
        self.store
            .query_row(
                "SELECT COUNT(*) FROM interlingua_concepts WHERE namespace = ?1",
                params![InterlinguaNamespace::YagoClass as u16],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| ConceptStoreError::Storage(e.to_string()))
            .map(|c| c.unwrap_or(0))
    }
}

fn to_sql(id: InterlinguaId) -> (i64, i64) {
    (id.as_i64(), i64::from(id.namespace() as u16))
}

impl ConceptStore for SqliteConceptStore {
    fn get(&self, id: InterlinguaId) -> Result<ConceptMetadata, ConceptStoreError> {
        let (sql_id, _ns) = to_sql(id);
        self.store
            .query_row(
                "SELECT id, namespace, canonical_name, yago_iri, yago_class_iri, \
                        label, node_id, parent_class_id \
                 FROM interlingua_concepts WHERE id = ?1 ORDER BY rowid LIMIT 1",
                params![sql_id],
                |row| {
                    let id_i64: i64 = row.get(0)?;
                    let ns: i64 = row.get(1)?;
                    let canonical_name: String = row.get(2)?;
                    let yago_iri: Option<String> = row.get(3)?;
                    let yago_class_iri: Option<String> = row.get(4)?;
                    let label: Option<String> = row.get(5)?;
                    let node_id: Option<i64> = row.get(6)?;
                    let namespace = InterlinguaNamespace::from_u16(ns as u16);
                    Ok(ConceptMetadata {
                        id: InterlinguaId::from_sql(ns as u16, local_id_of(id_i64)),
                        canonical_name,
                        namespace,
                        yago_iri,
                        yago_class_iri,
                        label,
                        node_id: node_id.map(NodeId::from_int),
                        parent_class_id: row
                            .get::<_, Option<i64>>(7)?
                            .and_then(InterlinguaId::from_i64),
                    })
                },
            )
            .map_err(|e| ConceptStoreError::Storage(e.to_string()))?
            .ok_or_else(|| ConceptStoreError::NotFound(format!("{id}")))
    }

    fn resolve_name(&self, name: &str) -> Result<InterlinguaId, ConceptStoreError> {
        self.store
            .query_row(
                "SELECT id FROM interlingua_concepts WHERE canonical_name = ?1",
                params![name],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| ConceptStoreError::Storage(e.to_string()))?
            .and_then(InterlinguaId::from_i64)
            .ok_or_else(|| ConceptStoreError::NotFound(name.to_string()))
    }

    fn resolve_yago_iri(&self, iri: &str) -> Result<InterlinguaId, ConceptStoreError> {
        self.store
            .query_row(
                "SELECT id FROM interlingua_concepts WHERE yago_iri = ?1",
                params![iri],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| ConceptStoreError::Storage(e.to_string()))?
            .and_then(InterlinguaId::from_i64)
            .ok_or_else(|| ConceptStoreError::NotFound(iri.to_string()))
    }

    fn insert(&self, meta: ConceptMetadata) -> Result<(), ConceptStoreError> {
        self.store
            .execute(
                "INSERT OR IGNORE INTO interlingua_concepts \
                 (id, namespace, canonical_name, yago_iri, yago_class_iri, label, node_id, parent_class_id) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    meta.id.as_i64(),
                    i64::from(meta.namespace as u16),
                    meta.canonical_name,
                    meta.yago_iri,
                    meta.yago_class_iri,
                    meta.label,
                    meta.node_id.map(NodeId::as_int),
                    meta.parent_class_id.map(InterlinguaId::as_i64),
                ],
            )
            .map_err(|e| ConceptStoreError::Storage(e.to_string()))?;
        Ok(())
    }

    fn contains(&self, id: InterlinguaId) -> bool {
        self.get(id).is_ok()
    }

    fn iter_ids(&self) -> Box<dyn Iterator<Item = InterlinguaId> + '_> {
        let ids = self
            .store
            .query_rows(
                "SELECT DISTINCT id FROM interlingua_concepts",
                &[],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or_default()
            .into_iter()
            .filter_map(InterlinguaId::from_i64)
            .collect::<Vec<_>>();
        Box::new(ids.into_iter())
    }

    fn ancestors_of(&self, id: InterlinguaId) -> Vec<InterlinguaId> {
        self.hierarchy
            .read()
            .expect("concept store read lock poisoned")
            .ancestors(id)
    }

    fn is_subclass_of(&self, child: InterlinguaId, parent: InterlinguaId) -> bool {
        self.hierarchy
            .read()
            .expect("concept store read lock poisoned")
            .is_subclass(child, parent)
    }
}

#[cfg(test)]
#[path = "../tests/concept_store_sqlite.rs"]
mod tests;
