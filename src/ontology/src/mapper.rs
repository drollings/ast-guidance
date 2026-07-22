use std::collections::HashMap;

use guidance_rdf::normalize::hash_blank_node;
use guidance_rdf::normalize::hash_iri;
use guidance_rdf::parser::Term;
use guidance_rdf::parser::Triple;
use fluent_types::ContextNode;
use fluent_types::NodeId;
use fluent_types::LOD_COUNT;
use thiserror::Error;

use crate::yago;

#[derive(Debug, Clone)]
pub struct PendingNode {
    pub id: i64,
    pub lod: [Vec<u8>; LOD_COUNT],
    pub types: Vec<i64>,
}

impl PendingNode {
    pub fn new(id: i64) -> Self {
        Self {
            id,
            lod: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            types: Vec::new(),
        }
    }
}

impl PendingNode {
    pub fn to_context_node(&self) -> ContextNode {
        let lod: Vec<String> = self
            .lod
            .iter()
            .map(|v| String::from_utf8_lossy(v).to_string())
            .collect();
        ContextNode {
            id: Some(NodeId::from_int(self.id)),
            name: lod.get(4).cloned().unwrap_or_default().into(),
            source: lod
                .iter()
                .filter(|s| !s.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(" "),
            lod,
            embedding: None,
            capabilities: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PendingEdge {
    pub from_id: i64,
    pub to_id: i64,
    pub predicate: String,
}

#[derive(Debug, Clone)]
pub struct PendingContradiction {
    pub subject_id: i64,
    pub predicate: String,
    pub value_a: String,
    pub value_b: String,
}

#[derive(Debug, Clone)]
pub struct MappingConfig {
    pub preferred_lang: String,
    pub scope: String,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self {
            preferred_lang: "en".to_string(),
            scope: "default".to_string(),
        }
    }
}

#[derive(Error, Debug)]
pub enum MappingError {
    #[error("literal cannot be a subject")]
    LiteralAsSubject,
    #[error("predicate must be an IRI")]
    PredicateNotIRI,
    #[error("duplicate node key: {0}")]
    DuplicateKey(String),
}

#[derive(Debug, Clone, Default)]
pub struct FlushResult {
    pub nodes_created: usize,
    pub edges_created: usize,
    pub contradictions_detected: usize,
}

pub struct TripleMapper {
    nodes: HashMap<String, PendingNode>,
    edges: Vec<PendingEdge>,
    contradictions: Vec<PendingContradiction>,
    config: MappingConfig,
}

impl TripleMapper {
    pub fn new(config: MappingConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            contradictions: Vec::new(),
            config,
        }
    }

    pub fn process_triple(&mut self, triple: &Triple) -> Result<(), MappingError> {
        let subj_id = self.term_to_id(&triple.subject);
        let pred_iri = Self::term_to_iri(&triple.predicate)?;

        self.get_or_create_node(&triple.subject, subj_id)?;

        let rdf_type = yago::NS_RDF.to_owned() + "type";
        let rdfs_label = yago::NS_RDFS.to_owned() + "label";
        let skos_pref = yago::NS_SKOS.to_owned() + "prefLabel";
        let rdfs_comment = yago::NS_RDFS.to_owned() + "comment";
        let schema_desc = yago::NS_SCHEMA.to_owned() + "description";

        if pred_iri == rdf_type {
            if let Term::Iri(_) = &triple.object {
                let type_id = self.term_to_id(&triple.object);
                let key = Self::term_key(&triple.subject)?;
                if let Some(node) = self.nodes.get_mut(&key) {
                    node.types.push(type_id);
                }
            }
        } else if pred_iri == rdfs_label || pred_iri == skos_pref {
            if let Term::Literal(lit) = &triple.object {
                if should_use_lang(lit.lang.as_deref(), &self.config.preferred_lang) {
                    let key = Self::term_key(&triple.subject)?;
                    if let Some(node) = self.nodes.get_mut(&key) {
                        if let Some(c) =
                            check_contradiction(node, subj_id, &pred_iri, 4, &lit.value)
                        {
                            self.contradictions.push(c);
                        }
                        node.lod[4] = lit.value.as_bytes().to_vec();
                    }
                }
            }
        } else if pred_iri == rdfs_comment {
            if let Term::Literal(lit) = &triple.object {
                if should_use_lang(lit.lang.as_deref(), &self.config.preferred_lang) {
                    let key = Self::term_key(&triple.subject)?;
                    if let Some(node) = self.nodes.get_mut(&key) {
                        if let Some(c) =
                            check_contradiction(node, subj_id, &pred_iri, 0, &lit.value)
                        {
                            self.contradictions.push(c);
                        }
                        node.lod[0] = lit.value.as_bytes().to_vec();
                    }
                }
            }
        } else if pred_iri == schema_desc {
            if let Term::Literal(lit) = &triple.object {
                if should_use_lang(lit.lang.as_deref(), &self.config.preferred_lang) {
                    let key = Self::term_key(&triple.subject)?;
                    if let Some(node) = self.nodes.get_mut(&key) {
                        if let Some(c) =
                            check_contradiction(node, subj_id, &pred_iri, 1, &lit.value)
                        {
                            self.contradictions.push(c);
                        }
                        node.lod[1] = lit.value.as_bytes().to_vec();
                    }
                }
            }
        } else if matches!(triple.object, Term::Iri(_) | Term::BlankNode(_)) {
            let obj_id = self.term_to_id(&triple.object);
            self.get_or_create_node(&triple.object, obj_id)?;
            self.edges.push(PendingEdge {
                from_id: subj_id,
                to_id: obj_id,
                predicate: pred_iri,
            });
        }

        Ok(())
    }

    pub fn drain_nodes(&mut self) -> Vec<PendingNode> {
        std::mem::take(&mut self.nodes).into_values().collect()
    }

    pub fn drain_edges(&mut self) -> Vec<PendingEdge> {
        std::mem::take(&mut self.edges)
    }

    pub fn drain_contradictions(&mut self) -> Vec<PendingContradiction> {
        std::mem::take(&mut self.contradictions)
    }

    pub fn pending_node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn pending_edge_count(&self) -> usize {
        self.edges.len()
    }

    fn term_key(term: &Term) -> Result<String, MappingError> {
        match term {
            Term::Iri(s) => Ok(s.clone()),
            Term::BlankNode(s) => Ok(format!("bnode:{s}")),
            Term::Literal(_) => Err(MappingError::LiteralAsSubject),
        }
    }

    fn term_to_id(&self, term: &Term) -> i64 {
        match term {
            Term::Iri(s) => hash_iri(s),
            Term::BlankNode(s) => hash_blank_node(&self.config.scope, s),
            Term::Literal(_) => hash_iri("_literal_"),
        }
    }

    fn term_to_iri(term: &Term) -> Result<String, MappingError> {
        match term {
            Term::Iri(s) => Ok(s.clone()),
            _ => Err(MappingError::PredicateNotIRI),
        }
    }

    fn get_or_create_node(&mut self, term: &Term, id: i64) -> Result<(), MappingError> {
        let key = match term {
            Term::Iri(s) => s.clone(),
            Term::BlankNode(s) => format!("bnode:{s}"),
            Term::Literal(_) => return Err(MappingError::LiteralAsSubject),
        };
        self.nodes
            .entry(key)
            .or_insert_with(|| PendingNode::new(id));
        Ok(())
    }
}
fn check_contradiction(
    node: &PendingNode,
    subject_id: i64,
    predicate: &str,
    lod_index: usize,
    new_value: &str,
) -> Option<PendingContradiction> {
    let existing = &node.lod[lod_index];
    if existing.is_empty() {
        return None;
    }
    if existing == new_value.as_bytes() {
        return None;
    }
    Some(PendingContradiction {
        subject_id,
        predicate: predicate.to_string(),
        value_a: String::from_utf8_lossy(existing).to_string(),
        value_b: new_value.to_string(),
    })
}

fn should_use_lang(actual: Option<&str>, preferred: &str) -> bool {
    match actual {
        None => true,
        Some(lang) => lang == preferred,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use guidance_rdf::parser::Parser;

    fn parse_one(src: &str) -> Triple {
        let mut p = Parser::new(src);
        p.next().unwrap().unwrap()
    }

    fn parse_all(src: &str) -> Vec<Triple> {
        let mut p = Parser::new(src);
        let mut triples = Vec::new();
        while let Some(Ok(t)) = p.next() {
            triples.push(t);
        }
        triples
    }

    #[test]
    fn test_entity_from_type_triple() {
        let mut mapper = TripleMapper::new(MappingConfig::default());
        let triple = parse_one(
            "<http://example.org/alice> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/Person> .",
        );
        mapper.process_triple(&triple).unwrap();
        assert!(mapper.pending_node_count() >= 1);
    }

    #[test]
    fn test_label_routes_to_lod4() {
        let mut mapper = TripleMapper::new(MappingConfig::default());
        let triple = parse_one(
            "<http://example.org/alice> <http://www.w3.org/2000/01/rdf-schema#label> \"Alice\" .",
        );
        mapper.process_triple(&triple).unwrap();
        let nodes = mapper.drain_nodes();
        let node = nodes
            .iter()
            .find(|n| !n.lod[4].is_empty())
            .expect("node with lod[4]");
        assert_eq!(String::from_utf8_lossy(&node.lod[4]), "Alice");
    }

    #[test]
    fn test_comment_routes_to_lod0() {
        let mut mapper = TripleMapper::new(MappingConfig::default());
        let triple = parse_one(
            "<http://example.org/alice> <http://www.w3.org/2000/01/rdf-schema#comment> \"A person named Alice\" .",
        );
        mapper.process_triple(&triple).unwrap();
        let nodes = mapper.drain_nodes();
        let node = nodes.iter().find(|n| !n.lod[0].is_empty()).unwrap();
        assert_eq!(
            String::from_utf8_lossy(&node.lod[0]),
            "A person named Alice"
        );
    }

    #[test]
    fn test_object_property_creates_edge() {
        let mut mapper = TripleMapper::new(MappingConfig::default());
        let triple = parse_one(
            "<http://example.org/alice> <http://yago-knowledge.org/resource/bornIn> <http://example.org/Paris> .",
        );
        mapper.process_triple(&triple).unwrap();
        assert_eq!(mapper.pending_edge_count(), 1);
        assert_eq!(
            mapper.drain_edges()[0].predicate,
            "http://yago-knowledge.org/resource/bornIn"
        );
    }

    #[test]
    fn test_multiple_triples_same_entity() {
        let mut mapper = TripleMapper::new(MappingConfig::default());
        let src = "\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            @prefix schema: <http://schema.org/> .\n\
            <http://example.org/alice> rdfs:label \"Alice\" ; rdfs:comment \"A person\" .\n\
        ";
        let triples = parse_all(src);
        for t in &triples {
            mapper.process_triple(t).unwrap();
        }
        let nodes = mapper.drain_nodes();
        let node = nodes.iter().find(|n| !n.lod[4].is_empty()).unwrap();
        assert_eq!(String::from_utf8_lossy(&node.lod[4]), "Alice");
        assert_eq!(String::from_utf8_lossy(&node.lod[0]), "A person");
    }

    #[test]
    fn test_contradiction_detection() {
        let mut mapper = TripleMapper::new(MappingConfig::default());
        let t1 = parse_one(
            "<http://example.org/alice> <http://www.w3.org/2000/01/rdf-schema#label> \"Alice\" .",
        );
        mapper.process_triple(&t1).unwrap();
        assert_eq!(mapper.drain_contradictions().len(), 0);

        let t2 = parse_one(
            "<http://example.org/alice> <http://www.w3.org/2000/01/rdf-schema#label> \"Alicia\" .",
        );
        mapper.process_triple(&t2).unwrap();
        let contradictions = mapper.drain_contradictions();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(contradictions[0].value_a, "Alice");
        assert_eq!(contradictions[0].value_b, "Alicia");
    }

    #[test]
    fn test_no_contradiction_for_identical_label() {
        let mut mapper = TripleMapper::new(MappingConfig::default());
        let src = "\
            <http://example.org/bob> <http://www.w3.org/2000/01/rdf-schema#label> \"Bob\" .\n\
            <http://example.org/bob> <http://www.w3.org/2000/01/rdf-schema#label> \"Bob\" .\n\
        ";
        let triples = parse_all(src);
        for t in &triples {
            mapper.process_triple(t).unwrap();
        }
        assert_eq!(mapper.drain_contradictions().len(), 0);
    }

    #[test]
    fn test_desc_routes_to_lod1() {
        let mut mapper = TripleMapper::new(MappingConfig::default());
        let triple = parse_one(
            "<http://example.org/alice> <http://schema.org/description> \"An example person\" .",
        );
        mapper.process_triple(&triple).unwrap();
        let nodes = mapper.drain_nodes();
        let node = nodes.iter().find(|n| !n.lod[1].is_empty()).unwrap();
        assert_eq!(String::from_utf8_lossy(&node.lod[1]), "An example person");
    }
}
