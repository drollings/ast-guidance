//! The pure hash→ID bridge (ROADMAP §7 — F3, F9).
//!
//! [`InterlinguaResolver`] is **stateless in the common path**: every ID is a
//! pure function of content, so there is no `Mutex`, no mutable probe state,
//! and no registry to serialize parallel workers. It consumes an
//! [`Arc<dyn ConceptStore>`](crate::concept_store::ConceptStore) (the only
//! concept state — F9) and the vocabulary's [`StringStore`] for hash→canonical
//! lookups.
//!
//! Because it is `Send + Sync` with no interior write mutation, it can be
//! shared by every `ResultPool` worker and the pipeline without a lock (F3).
//!
//! Collisions are surfaced, never masked (§2.3): when a second canonical
//! claims an already-taken id, [`CollisionNote`] records it for audit and
//! downstream "needs disambiguation" routing (F7).

use std::sync::Arc;

use crate::concept_store::ConceptStore;
use crate::doc::Doc;
use crate::hash::hash_utf8;
use crate::strings::StringStore;
use fluent_types::{lemma_id_for_str, ConceptMetadata, InterlinguaId};

/// Recorded when a second canonical claims an already-taken id (first-wins).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollisionNote {
    None,
    Collision {
        id: InterlinguaId,
        prior_canonical: String,
    },
}

impl std::fmt::Display for CollisionNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollisionNote::None => write!(f, "no collision"),
            CollisionNote::Collision { id, prior_canonical } => {
                write!(f, "interlingua collision on {id}: prior canonical {prior_canonical:?}")
            }
        }
    }
}

/// The pure hash→ID bridge over a shared [`ConceptStore`] + [`StringStore`].
pub struct InterlinguaResolver {
    concepts: Arc<dyn ConceptStore>,
    strings: Arc<StringStore>,
}

impl std::fmt::Debug for InterlinguaResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterlinguaResolver").finish_non_exhaustive()
    }
}

impl InterlinguaResolver {
    /// A resolver over `concepts` (the single concept registry) and `strings`
    /// (the vocab's reverse hash map).
    #[must_use]
    pub fn new(concepts: Arc<dyn ConceptStore>, strings: Arc<StringStore>) -> Self {
        Self { concepts, strings }
    }

    /// Pure: the SpacyLemma id for a canonical lemma string. Never consults
    /// registry state. Collisions are handled by first-wins (§2.3).
    pub fn lemma_id(&self, canonical: &str) -> InterlinguaId {
        lemma_id_for_str(canonical)
    }

    /// Resolve a spaCy `StringStore` hash (the lemma/dep u64 on a TokenRecord)
    /// to an InterlinguaId. The canonical comes from `strings`; the id is
    /// deterministic. Flags a collision when a different canonical already
    /// claims the id.
    pub fn resolve_hash(&self, hash: u64, canonical: &str) -> (InterlinguaId, CollisionNote) {
        // The content-hash invariant: the caller's hash and the canonical
        // string must agree (both are MurmurHash64A of the same content).
        debug_assert_eq!(hash_utf8(canonical), hash, "canonical/hash mismatch");
        let id = self.lemma_id(canonical);
        let note = match self.concepts.get(id) {
            Ok(existing) if existing.canonical_name != canonical => CollisionNote::Collision {
                id,
                prior_canonical: existing.canonical_name,
            },
            _ => CollisionNote::None,
        };
        (id, note)
    }

    /// The SpacyLemma id for a canonical string (pure, no registry consult).
    pub fn resolve_string(&self, canonical: &str) -> InterlinguaId {
        self.lemma_id(canonical)
    }

    /// The canonical string for an id, when the concept is registered.
    pub fn canonical(&self, id: InterlinguaId) -> Option<String> {
        self.concepts.get(id).ok().map(|c| c.canonical_name)
    }

    /// The full metadata for an id, when registered.
    pub fn metadata(&self, id: InterlinguaId) -> Option<ConceptMetadata> {
        self.concepts.get(id).ok()
    }

    /// The shared concept store (the single registry).
    pub fn concepts(&self) -> &Arc<dyn ConceptStore> {
        &self.concepts
    }

    /// The shared string store.
    pub fn strings(&self) -> &Arc<StringStore> {
        &self.strings
    }

    /// Stamp `interlingua_lemma_id`/`confidence` on every token of an
    /// already-attached doc. **Pure and read-only** (C2): no store writes, no
    /// locks beyond the store's own read path — registration is boot-only and
    /// corrections happen in the review worker. Returns the collision notes
    /// seen (for audit metadata).
    pub fn resolve_doc(
        &self,
        doc: &mut Doc,
        token_confidence: Option<&[f64]>,
    ) -> Vec<CollisionNote> {
        // Clone the shared Arc so the store lookups do not borrow `doc` while
        // the mutation loop below takes `&mut` access to its tokens.
        let strings = Arc::clone(doc.vocab().strings());
        let mut notes = Vec::new();
        for i in 0..doc.len() {
            let canonical = strings
                .get(doc.token(i).lemma)
                .map(|s| s.to_string())
                .unwrap_or_default();
            if canonical.is_empty() {
                continue;
            }
            let (id, note) = self.resolve_hash(doc.token(i).lemma, &canonical);
            if let CollisionNote::Collision { .. } = note {
                notes.push(note);
            }
            doc.token_mut(i).interlingua_lemma_id = Some(id);
            // L5: guard the confidence slice — a caller-bug shorter vector
            // must leave the token's confidence unset, never panic the hot
            // path.
            doc.token_mut(i).confidence = token_confidence.and_then(|c| c.get(i)).copied();
        }
        notes
    }
}

#[cfg(test)]
#[path = "../tests/interlingua.rs"]
mod tests;
