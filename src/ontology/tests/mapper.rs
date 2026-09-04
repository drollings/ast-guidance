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
