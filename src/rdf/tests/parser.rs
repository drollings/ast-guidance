use super::*;

fn parse_triples(src: &str) -> Vec<Triple> {
    let p = Parser::new(src);
    p.collect::<Result<Vec<_>, _>>().unwrap()
}

#[test]
fn test_parse_simple_triple() {
    let trips = parse_triples("<http://s> <http://p> <http://o> .");
    assert_eq!(trips.len(), 1);
    assert_eq!(trips[0].subject, Term::Iri("http://s".into()));
    assert_eq!(trips[0].predicate, Term::Iri("http://p".into()));
    assert_eq!(trips[0].object, Term::Iri("http://o".into()));
}

#[test]
fn test_parse_predicate_object_list() {
    let trips = parse_triples("<http://s> <http://p1> <http://o1> ; <http://p2> <http://o2> .");
    assert_eq!(trips.len(), 2);
    assert_eq!(trips[0].predicate, Term::Iri("http://p1".into()));
    assert_eq!(trips[1].predicate, Term::Iri("http://p2".into()));
}

#[test]
fn test_parse_object_list() {
    let trips = parse_triples("<http://s> <http://p> <http://o1> , <http://o2> .");
    assert_eq!(trips.len(), 2);
    assert_eq!(trips[0].object, Term::Iri("http://o1".into()));
    assert_eq!(trips[1].object, Term::Iri("http://o2".into()));
}

#[test]
fn test_parse_a_shorthand() {
    let trips = parse_triples("<http://s> a <http://Class> .");
    assert_eq!(trips[0].predicate, Term::Iri(RDF_TYPE.into()));
}

#[test]
fn test_parse_prefix_expansion() {
    let trips = parse_triples("@prefix ex: <http://example.org/> .\nex:foo a ex:Thing .");
    assert_eq!(trips[0].subject, Term::Iri("http://example.org/foo".into()));
    assert_eq!(
        trips[0].object,
        Term::Iri("http://example.org/Thing".into())
    );
}

#[test]
fn test_parse_blank_node_subject() {
    let trips = parse_triples("_:b1 <http://p> <http://o> .");
    assert_eq!(trips[0].subject, Term::BlankNode("b1".into()));
}

#[test]
fn test_parse_literal_object() {
    let trips = parse_triples("<http://s> <http://p> \"hello\" .");
    match &trips[0].object {
        Term::Literal(lit) => assert_eq!(lit.value, "hello"),
        _ => panic!("expected literal"),
    }
}

#[test]
fn test_parse_inline_blank_node() {
    let trips = parse_triples("<http://s> <http://p> [ <http://p2> <http://o2> ] .");
    assert!(matches!(trips[0].object, Term::BlankNode(_)));
    assert_eq!(trips[1].predicate, Term::Iri("http://p2".into()));
}

#[test]
fn test_parse_multiple_subjects() {
    let trips =
        parse_triples("<http://a> <http://p> <http://x> .\n<http://b> <http://p> <http://y> .");
    assert_eq!(trips.len(), 2);
    assert_eq!(trips[0].subject, Term::Iri("http://a".into()));
    assert_eq!(trips[1].subject, Term::Iri("http://b".into()));
}

#[test]
fn test_parse_literal_with_lang_tag() {
    let trips = parse_triples("<http://s> <http://p> \"bonjour\"@fr .");
    match &trips[0].object {
        Term::Literal(lit) => {
            assert_eq!(lit.lang, Some("fr".into()));
            assert_eq!(lit.value, "bonjour");
        }
        _ => panic!("expected literal"),
    }
}

#[test]
fn test_parse_literal_with_datatype() {
    let trips = parse_triples(
        "<http://s> <http://p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
    );
    match &trips[0].object {
        Term::Literal(lit) => {
            assert_eq!(
                lit.datatype,
                Some("http://www.w3.org/2001/XMLSchema#integer".into())
            );
            assert_eq!(lit.value, "42");
        }
        _ => panic!("expected literal"),
    }
}

#[test]
fn test_parse_numeric_literal() {
    let trips = parse_triples("<http://s> <http://p> 42 .");
    match &trips[0].object {
        Term::Literal(lit) => {
            assert_eq!(lit.value, "42");
            assert_eq!(lit.datatype, Some(format!("{}integer", crate::XSD_NS)));
        }
        _ => panic!("expected literal"),
    }
}

#[test]
fn test_parse_triple_quoted_literal() {
    let trips = parse_triples("<http://s> <http://p> \"\"\"hello\nworld\"\"\" .");
    match &trips[0].object {
        Term::Literal(lit) => {
            assert_eq!(lit.value, "hello\nworld");
        }
        _ => panic!("expected literal"),
    }
}

#[test]
fn test_parse_true_false_literals() {
    let trips = parse_triples("<http://s> <http://p> true .");
    match &trips[0].object {
        Term::Literal(lit) => {
            assert_eq!(lit.value, "true");
            assert_eq!(lit.datatype, Some(format!("{}boolean", crate::XSD_NS)));
        }
        _ => panic!("expected literal"),
    }
}

#[test]
fn test_parse_prefixed_name_object() {
    let trips = parse_triples("@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .");
    assert_eq!(trips[0].subject, Term::Iri("http://example.org/s".into()));
}

#[test]
fn test_parse_empty_prefix_map() {
    let trips = parse_triples(":foo <http://p> <http://o> .");
    assert_eq!(trips[0].subject, Term::Iri(":foo".into()));
}

#[test]
fn test_parse_empty_blank_node_brackets() {
    let trips = parse_triples("<http://s> <http://p> [ ] .");
    assert!(matches!(trips[0].object, Term::BlankNode(_)));
    assert_eq!(trips.len(), 1);
}
