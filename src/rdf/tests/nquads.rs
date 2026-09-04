use super::*;

#[test]
fn test_parse_line_simple() {
    let line = "<http://s> <http://p> <http://o> .";
    let quad = NQuadsParser::parse_line(line).unwrap().unwrap();
    assert_eq!(quad.subject, Term::Iri("http://s".into()));
    assert_eq!(quad.predicate, Term::Iri("http://p".into()));
    assert_eq!(quad.object, Term::Iri("http://o".into()));
    assert!(quad.graph.is_none());
}

#[test]
fn test_parse_line_with_graph() {
    let line = "<http://s> <http://p> <http://o> <http://g> .";
    let quad = NQuadsParser::parse_line(line).unwrap().unwrap();
    assert_eq!(quad.graph, Some(Term::Iri("http://g".into())));
}

#[test]
fn test_parse_line_blank_node() {
    let line = "_:s <http://p> _:o .";
    let quad = NQuadsParser::parse_line(line).unwrap().unwrap();
    assert_eq!(quad.subject, Term::BlankNode("s".into()));
    assert_eq!(quad.object, Term::BlankNode("o".into()));
}

#[test]
fn test_parse_line_literal() {
    let line = "<http://s> <http://p> \"hello\" .";
    let quad = NQuadsParser::parse_line(line).unwrap().unwrap();
    match quad.object {
        Term::Literal(lit) => {
            assert_eq!(lit.value, "hello");
            assert!(lit.lang.is_none());
            assert!(lit.datatype.is_none());
        }
        _ => panic!("expected literal"),
    }
}

#[test]
fn test_parse_line_skip_comment() {
    let result = NQuadsParser::parse_line("# this is a comment").unwrap();
    assert!(result.is_none());
}

#[test]
fn test_parse_line_skip_empty() {
    let result = NQuadsParser::parse_line("").unwrap();
    assert!(result.is_none());
}

#[test]
fn test_parse_line_literal_with_lang() {
    let line = "<http://s> <http://p> \"bonjour\"@fr .";
    let quad = NQuadsParser::parse_line(line).unwrap().unwrap();
    match quad.object {
        Term::Literal(lit) => {
            assert_eq!(lit.value, "bonjour");
            assert_eq!(lit.lang, Some("fr".into()));
        }
        _ => panic!("expected literal"),
    }
}

#[test]
fn test_parse_line_literal_with_datatype() {
    let line = "<http://s> <http://p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .";
    let quad = NQuadsParser::parse_line(line).unwrap().unwrap();
    match quad.object {
        Term::Literal(lit) => {
            assert_eq!(lit.value, "42");
            assert_eq!(
                lit.datatype,
                Some("http://www.w3.org/2001/XMLSchema#integer".into())
            );
        }
        _ => panic!("expected literal"),
    }
}
