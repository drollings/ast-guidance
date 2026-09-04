//! The neutral shared home for concept lookup (ROADMAP_20260903_SPACY_RS_SPLIT
//! M3 — moved here from `spacy-rs`).
//!
//! [`ConceptStore`](crate::concept_store::ConceptStore) is the seam between the
//! pure resolver (`spacy-rs` `InterlinguaResolver`) and whatever backend serves
//! it. All three backends are **homes, not owners**, of the trait:
//!
//! - [`InMemoryConceptStore`](crate::concept_store_mem::InMemoryConceptStore)
//!   (here) — the hermetic test double, never production;
//! - the router's `SqliteConceptStore` (`interlingua_concepts`) — the durable
//!   materialized index;
//! - coral's content-addressed graph (`context_nodes`, keyed by the full
//!   64-bit `hash_iri`) — the durable graph.
//!
//! Boot reconciliation (one loader, two durable homes) locks the homes equal.
//!
//! This crate depends only on `fluent-types` + `fluent-dag`, so `spacy-rs`,
//! the router, coral, and the ontology all depend on it without a cycle
//! (neither `fluent-types` nor `common-core` depends on `dag`).

#![forbid(unsafe_code)]
#![allow(clippy::pedantic, clippy::all)]

pub mod concept_store;
pub mod concept_store_mem;
pub mod plausibility;

pub use concept_store::{ConceptStore, ConceptStoreError, ConceptStoreState, TaxonomyHierarchy};
pub use concept_store_mem::InMemoryConceptStore;
pub use plausibility::{PlausibilityTriple, ScoredLemma};
