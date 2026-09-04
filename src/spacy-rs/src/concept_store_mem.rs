//! The **hermetic in-memory** [`ConceptStore`] — the test double of ROADMAP
//! §6.4 (C3). Never the production store: the router's `SqliteConceptStore`
//! and coral's durable graph are the real homes, and boot reconciliation locks
//! them equal. This store exists so resolver/pipeline unit tests run with no
//! SQLite and no disk.

use std::collections::HashMap;
use std::sync::RwLock;

use fluent_types::{ConceptMetadata, InterlinguaId};

use crate::concept_store::{ConceptStore, ConceptStoreError, TaxonomyHierarchy};

/// An in-memory concept registry: id → metadata, plus name/IRI indexes and a
/// [`TaxonomyHierarchy`] backing the subclass queries. The hierarchy is built
/// from the same `ConceptMetadata.parent_class_id` field the loader fills
/// (C5 — one source of edges), so both the router's SQLite store and this
/// test double share the construction path. [`Self::set_hierarchy`] remains an
/// explicit override for tests that hand-build a hierarchy directly.
pub struct InMemoryConceptStore {
    /// Bucket id → every canonical registered under it (first entry is the
    /// incumbent, first-wins). Keeping the full list lets [`Self::candidates`]
    /// surface predicate polysemy while `get` still returns the incumbent.
    meta: RwLock<HashMap<InterlinguaId, Vec<ConceptMetadata>>>,
    names: RwLock<HashMap<String, InterlinguaId>>,
    yago_iris: RwLock<HashMap<String, InterlinguaId>>,
    hierarchy: RwLock<TaxonomyHierarchy>,
    /// Accumulated `(child ← parent)` edges collected from `insert` metadata.
    edges: RwLock<Vec<(InterlinguaId, InterlinguaId)>>,
    state: RwLock<crate::concept_store::ConceptStoreState>,
}

impl std::fmt::Debug for InMemoryConceptStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryConceptStore")
            .field("meta", &self.meta)
            .field("names", &self.names)
            .field("yago_iris", &self.yago_iris)
            .finish_non_exhaustive()
    }
}

impl Default for InMemoryConceptStore {
    fn default() -> Self {
        Self {
            meta: RwLock::new(HashMap::new()),
            names: RwLock::new(HashMap::new()),
            yago_iris: RwLock::new(HashMap::new()),
            hierarchy: RwLock::new(TaxonomyHierarchy::default()),
            edges: RwLock::new(Vec::new()),
            state: RwLock::new(crate::concept_store::ConceptStoreState::Ready),
        }
    }
}

impl InMemoryConceptStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the subclass hierarchy backing `ancestors_of`/`is_subclass_of`.
    /// Built once at boot from the `rdfs:subClassOf` edges (§6.3, §11.7). This
    /// is an explicit override — the default construction path derives the
    /// hierarchy from each `insert`'s `parent_class_id` metadata (the same
    /// field the router store hydrates from).
    pub fn set_hierarchy(&self, hierarchy: TaxonomyHierarchy) {
        *self.hierarchy.write().expect("concept store write lock poisoned") = hierarchy;
    }

    pub fn set_state(&self, state: crate::concept_store::ConceptStoreState) {
        *self.state.write().expect("concept store write lock poisoned") = state;
    }

    pub fn from_n2_fixture(path: &std::path::Path) -> Result<Self, crate::error::SpacyError> {
        let view = crate::yago_view::YagoView::load(path)?;
        let store = Self::new();
        // Seed fallback concepts from n2 — classes become minimal ConceptMetadata
        for (curie, id) in view.classes_iter() {
            let meta = fluent_types::ConceptMetadata {
                id,
                canonical_name: curie.clone(),
                namespace: id.namespace(),
                yago_iri: Some(format!("http://yago-knowledge.org/resource/{}", curie.trim_start_matches("yago:").trim_start_matches("schema:"))),
                yago_class_iri: None,
                label: Some(curie.clone()),
                node_id: None,
                parent_class_id: None,
            };
            let _ = store.insert(meta);
        }
        // Rebuild hierarchy from view edges via is_subclass checks not needed for stub
        Ok(store)
    }
}

impl ConceptStore for InMemoryConceptStore {
    fn get(&self, id: InterlinguaId) -> Result<ConceptMetadata, ConceptStoreError> {
        self.meta
            .read()
            .expect("concept store read lock poisoned")
            .get(&id)
            .and_then(|v| v.first().cloned())
            .ok_or_else(|| ConceptStoreError::NotFound(format!("{id}")))
    }

    fn resolve_name(&self, name: &str) -> Result<InterlinguaId, ConceptStoreError> {
        self.names
            .read()
            .expect("concept store read lock poisoned")
            .get(name)
            .copied()
            .ok_or_else(|| ConceptStoreError::NotFound(name.to_string()))
    }

    fn resolve_yago_iri(&self, iri: &str) -> Result<InterlinguaId, ConceptStoreError> {
        self.yago_iris
            .read()
            .expect("concept store read lock poisoned")
            .get(iri)
            .copied()
            .ok_or_else(|| ConceptStoreError::NotFound(iri.to_string()))
    }

    fn insert(&self, meta: ConceptMetadata) -> Result<(), ConceptStoreError> {
        {
            let mut map = self.meta.write().expect("concept store write lock poisoned");
            // First-wins: a colliding canonical is kept alongside under the
            // shared id, never replacing the incumbent (spaCy StringStore
            // semantics, §2.3). A re-insertion of the same canonical is a no-op
            // (boot reconciliation is idempotent).
            let bucket = map.entry(meta.id).or_default();
            if !bucket.iter().any(|m| m.canonical_name == meta.canonical_name) {
                bucket.push(meta.clone());
            }
        }
        {
            let mut names = self.names.write().expect("concept store write lock poisoned");
            names.entry(meta.canonical_name.clone()).or_insert(meta.id);
        }
        if let Some(iri) = &meta.yago_iri {
            let mut yago = self
                .yago_iris
                .write()
                .expect("concept store write lock poisoned");
            yago.entry(iri.clone()).or_insert(meta.id);
        }
        // Derive the hierarchy from the same metadata field the loader fills
        // (C5 — the router store hydrates `parent_class_id` from `insert`;
        // this store rebuilds the `TaxonomyHierarchy` from the accumulated
        // edges so both share the construction path).
        if let Some(parent) = meta.parent_class_id {
            {
                let mut edges = self.edges.write().expect("concept store write lock poisoned");
                if !edges.contains(&(meta.id, parent)) {
                    edges.push((meta.id, parent));
                }
            }
            let edges = self
                .edges
                .read()
                .expect("concept store read lock poisoned")
                .clone();
            let built = TaxonomyHierarchy::from_edges(&edges)
                .map_err(|e| ConceptStoreError::Storage(e.to_string()))?;
            *self
                .hierarchy
                .write()
                .expect("concept store write lock poisoned") = built;
        }
        Ok(())
    }

    fn contains(&self, id: InterlinguaId) -> bool {
        self.meta
            .read()
            .expect("concept store read lock poisoned")
            .contains_key(&id)
    }

    fn iter_ids(&self) -> Box<dyn Iterator<Item = InterlinguaId> + '_> {
        Box::new(
            self.meta
                .read()
                .expect("concept store read lock poisoned")
                .keys()
                .copied()
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }

    fn candidates(&self, id: InterlinguaId) -> Vec<ConceptMetadata> {
        self.meta
            .read()
            .expect("concept store read lock poisoned")
            .get(&id)
            .cloned()
            .unwrap_or_default()
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

    fn state(&self) -> crate::concept_store::ConceptStoreState {
        *self.state.read().expect("concept store read lock poisoned")
    }
}

#[cfg(test)]
#[path = "../tests/concept_store_mem.rs"]
mod tests;
