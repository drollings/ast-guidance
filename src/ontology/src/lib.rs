//! guidance-ontology: Entity extraction, capability inference, and YAGO
//! taxonomy integration for semantic knowledge representation.

pub mod entity;
pub mod inference;
pub mod mapper;
pub mod migration;
pub mod yago;
pub mod yago_loader;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum OntologyError {
    #[error("mapping error: {0}")]
    Mapping(String),
    #[error("inference error: {0}")]
    Inference(String),
}

pub use yago_loader::{canonical_class_name, yago_class_id, LoadStats, YaGoLoader};
