//! The single source of truth for concept lookup (ROADMAP §6 — pulled forward
//! by F9; neutral home `fluent-concept` per ROADMAP_20260903_SPACY_RS_SPLIT
//! M3 — moved here from `spacy-rs`).
//!
//! [`InterlinguaResolver`](https://docs.rs/spacy-rs) (in `spacy-rs`) consumes
//! it; `YaGoLoader` (in `guidance-ontology`) feeds it; the router implements
//! the SQLite backend; [`InMemoryConceptStore`](crate::concept_store_mem::InMemoryConceptStore)
//! is the hermetic test double. **No second registry anywhere.**
//!
//! All IDs are content-addressed (`trunc48(hash(content))`); a store only
//! *records* which canonicals claim which ids (first-wins, §2.3) and resolves
//! names/IRIs back to ids.

use std::collections::HashMap;

use fluent_dag::dep_graph::DependencyGraph;
use fluent_types::{ConceptMetadata, InterlinguaId};

/// Store readiness — `Loading` until batched WAL completes, `Ready` thereafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConceptStoreState {
    #[default]
    Ready,
    Loading,
}

/// Errors surfaced by a [`ConceptStore`].
#[derive(Debug, thiserror::Error)]
pub enum ConceptStoreError {
    #[error("concept not found: {0}")]
    NotFound(String),
    #[error("storage error: {0}")]
    Storage(String),
}

/// A registry of concepts keyed by content-addressed [`InterlinguaId`]s. The
/// trait is the seam between the pure resolver and whatever backend serves it
/// (in-memory test double here; SQLite in the router; coral's durable graph).
pub trait ConceptStore: Send + Sync {
    /// The metadata for `id`, or `NotFound`.
    fn get(&self, id: InterlinguaId) -> Result<ConceptMetadata, ConceptStoreError>;

    /// Resolve a canonical name (e.g. `"schema:Person"`) to its id.
    fn resolve_name(&self, name: &str) -> Result<InterlinguaId, ConceptStoreError>;

    /// Resolve a YaGO IRI (e.g. `"http://yago-knowledge.org/resource/Person"`)
    /// to its id.
    fn resolve_yago_iri(&self, iri: &str) -> Result<InterlinguaId, ConceptStoreError>;

    /// Register `meta`. First-wins: a colliding canonical for an already-taken
    /// id is kept alongside, not rejected (and never mutates the id).
    fn insert(&self, meta: ConceptMetadata) -> Result<(), ConceptStoreError>;

    /// Whether `id` is registered.
    fn contains(&self, id: InterlinguaId) -> bool;

    /// Iterate registered ids (used to pre-warm resolver caches and to audit).
    fn iter_ids(&self) -> Box<dyn Iterator<Item = InterlinguaId> + '_>;

    /// Every registered canonical under `id` (a bucket may hold several
    /// canonicals that truncated to the same local id — first-wins keeps them
    /// alongside, §2.3). The first entry is the incumbent canonical. Defaults
    /// to empty for stores that do not track colliding canonicals; a frame
    /// extractor reads it to detect predicate polysemy (a lemma id resolving
    /// to >1 candidate).
    fn candidates(&self, id: InterlinguaId) -> Vec<ConceptMetadata> {
        let _ = id;
        Vec::new()
    }

    /// All ancestors of `id` up the subclass chain, nearest first.
    fn ancestors_of(&self, id: InterlinguaId) -> Vec<InterlinguaId>;

    /// Transitive `rdfs:subClassOf` check (identity returns true).
    fn is_subclass_of(&self, child: InterlinguaId, parent: InterlinguaId) -> bool;

    /// Store readiness — default `Ready`; `SqliteConceptStore` overrides to `Loading` while batched.
    fn state(&self) -> ConceptStoreState {
        ConceptStoreState::Ready
    }
}

/// A subclass hierarchy keyed by [`InterlinguaId`], built once at boot from
/// the `rdfs:subClassOf` edges. Composes `fluent_dag::dep_graph::DependencyGraph`
/// (dag/SKILL prime directive — never hand-roll graph algorithms); it backs
/// [`ConceptStore::ancestors_of`]/[`ConceptStore::is_subclass_of`] on every
/// store.
///
/// The coral `is_a()` CTE remains the authoritative *durable* traversal over
/// `context_nodes`; this is the runtime mirror built from the same edges.
pub struct TaxonomyHierarchy {
    graph: DependencyGraph<InterlinguaId>,
}

impl Default for TaxonomyHierarchy {
    fn default() -> Self {
        Self {
            graph: DependencyGraph::new(),
        }
    }
}

impl TaxonomyHierarchy {
    /// Register `(child ← parent)` subclass edges so that
    /// [`ancestors`](Self::ancestors) (a `DependencyGraph::dependents_of`
    /// walk) yields the transitive superclass chain, nearest first.
    ///
    /// Edge encoding: each **parent** is registered as depending on its
    /// **child**, and every node **self-provides** (the concrete-target
    /// semantic — `dependents_of` only expands nodes that provide the asset
    /// being walked). `dependents_of(child)` therefore walks child → parent →
    /// grandparent — the ancestor closure. Repeated parents (multiple
    /// subclasses) aggregate their children into one registration.
    pub fn from_edges(edges: &[(InterlinguaId, InterlinguaId)]) -> Result<Self, ConceptStoreError> {
        let mut graph = DependencyGraph::new();
        let mut by_parent: HashMap<InterlinguaId, Vec<InterlinguaId>> = HashMap::new();
        for (child, parent) in edges {
            by_parent.entry(*parent).or_default().push(*child);
        }
        for (parent, children) in &by_parent {
            graph
                .register(parent, children, &[*parent])
                .map_err(|e| ConceptStoreError::Storage(e.to_string()))?;
        }
        // Register any child that appears only as a dependency (a leaf in the
        // depends-on view) so it is itself a queryable node.
        for (child, _) in edges {
            if !graph.contains(child) {
                graph
                    .register(child, &[], &[*child])
                    .map_err(|e| ConceptStoreError::Storage(e.to_string()))?;
            }
        }
        Ok(Self { graph })
    }

    /// All transitive superclasses of `id`, nearest first, cycle-resilient.
    /// Empty when `id` is unregistered or has no ancestors.
    pub fn ancestors(&self, id: InterlinguaId) -> Vec<InterlinguaId> {
        self.graph.dependents_of(&id)
    }

    /// Transitive `rdfs:subClassOf` check; identity returns true.
    pub fn is_subclass(&self, child: InterlinguaId, parent: InterlinguaId) -> bool {
        child == parent || self.ancestors(child).contains(&parent)
    }

    /// Whether `id` is a registered node in the hierarchy.
    pub fn contains(&self, id: InterlinguaId) -> bool {
        self.graph.contains(&id)
    }
}

#[cfg(test)]
#[path = "../tests/concept_store.rs"]
mod tests;
