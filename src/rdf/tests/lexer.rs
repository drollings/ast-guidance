use super::*;

#[test]
fn test_lex_iri_basic() {
    let src = "<http://example.org/foo>";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::Iri);
    assert_eq!(tok.value, "<http://example.org/foo>");
    let eof = lex.next_token().unwrap();
    assert_eq!(eof.kind, TokenKind::Eof);
}

#[test]
fn test_lex_prefixed_name() {
    let src = "ex:foo";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::PrefixedName);
    assert_eq!(tok.value, "ex:foo");
}

#[test]
fn test_lex_literal_basic() {
    let src = "\"hello world\"";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::Literal);
    assert_eq!(tok.value, "\"hello world\"");
}

#[test]
fn test_lex_literal_with_lang_tag() {
    let src = "\"bonjour\"@fr";
    let mut lex = Lexer::new(src);
    let lit = lex.next_token().unwrap();
    assert_eq!(lit.kind, TokenKind::Literal);
    let lang = lex.next_token().unwrap();
    assert_eq!(lang.kind, TokenKind::LangTag);
    assert_eq!(lang.value, "@fr");
}

#[test]
fn test_lex_literal_with_datatype() {
    let src = "\"42\"^^xsd:integer";
    let mut lex = Lexer::new(src);
    let lit = lex.next_token().unwrap();
    assert_eq!(lit.kind, TokenKind::Literal);
    let marker = lex.next_token().unwrap();
    assert_eq!(marker.kind, TokenKind::DatatypeMarker);
    let dt = lex.next_token().unwrap();
    assert_eq!(dt.kind, TokenKind::PrefixedName);
    assert_eq!(dt.value, "xsd:integer");
}

#[test]
fn test_lex_blank_node() {
    let src = "_:node1";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::BlankNode);
    assert_eq!(tok.value, "_:node1");
}

#[test]
fn test_lex_comment_skipping() {
    let src = "# this is a comment\n<http://foo>";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::Iri);
}

#[test]
fn test_lex_at_prefix_keyword() {
    let src = "@prefix ex: <http://example.org/> .";
    let mut lex = Lexer::new(src);
    let kw = lex.next_token().unwrap();
    assert_eq!(kw.kind, TokenKind::Keyword);
    assert_eq!(kw.value, "@prefix");
}

#[test]
fn test_lex_keyword_a() {
    let src = "<s> a <Class> .";
    let mut lex = Lexer::new(src);
    let _ = lex.next_token().unwrap();
    let a = lex.next_token().unwrap();
    assert_eq!(a.kind, TokenKind::Keyword);
    assert_eq!(a.value, "a");
}

#[test]
fn test_lex_punctuation() {
    let src = ". ; ,";
    let mut lex = Lexer::new(src);
    let dot = lex.next_token().unwrap();
    assert_eq!(dot.kind, TokenKind::Dot);
    let semi = lex.next_token().unwrap();
    assert_eq!(semi.kind, TokenKind::Semicolon);
    let comma = lex.next_token().unwrap();
    assert_eq!(comma.kind, TokenKind::Comma);
}

#[test]
fn test_lex_unterminated_iri() {
    let src = "<http://example.org/foo";
    let mut lex = Lexer::new(src);
    let result = lex.next_token();
    assert!(result.is_err());
    matches!(result.unwrap_err(), RdfError::UnterminatedIRI);
}

#[test]
fn test_lex_unterminated_literal() {
    let src = "\"unterminated";
    let mut lex = Lexer::new(src);
    let result = lex.next_token();
    assert!(result.is_err());
    matches!(result.unwrap_err(), RdfError::UnterminatedLiteral);
}

#[test]
fn test_lex_literal_with_escape() {
    let src = "\"line1\\nline2\"";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::Literal);
}

#[test]
fn test_lex_blank_node_brackets() {
    let src = "[ ]";
    let mut lex = Lexer::new(src);
    let open = lex.next_token().unwrap();
    assert_eq!(open.kind, TokenKind::BlankNodeOpen);
    let close = lex.next_token().unwrap();
    assert_eq!(close.kind, TokenKind::BlankNodeClose);
}

#[test]
fn test_lex_multiple_tokens() {
    let src = "<http://s> <http://p> <http://o> .";
    let mut lex = Lexer::new(src);
    assert_eq!(lex.next_token().unwrap().kind, TokenKind::Iri);
    assert_eq!(lex.next_token().unwrap().kind, TokenKind::Iri);
    assert_eq!(lex.next_token().unwrap().kind, TokenKind::Iri);
    assert_eq!(lex.next_token().unwrap().kind, TokenKind::Dot);
    assert_eq!(lex.next_token().unwrap().kind, TokenKind::Eof);
}

#[test]
fn test_lex_numeric_integer() {
    let src = "42";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::Literal);
    assert_eq!(tok.value, "42");
}

#[test]
fn test_lex_numeric_decimal() {
    let src = "3.14";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::Literal);
    assert_eq!(tok.value, "3.14");
}

#[test]
fn test_lex_numeric_negative() {
    let src = "-5";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::Literal);
    assert_eq!(tok.value, "-5");
}

#[test]
fn test_lex_triple_quoted_literal() {
    let src = "\"\"\"hello\nworld\"\"\"";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.kind, TokenKind::Literal);
    assert!(tok.value.starts_with("\"\"\""));
}

#[test]
fn test_lex_keyword_true_false() {
    let src = "true false";
    let mut lex = Lexer::new(src);
    let t = lex.next_token().unwrap();
    assert_eq!(t.kind, TokenKind::Keyword);
    assert_eq!(t.value, "true");
    let f = lex.next_token().unwrap();
    assert_eq!(f.kind, TokenKind::Keyword);
    assert_eq!(f.value, "false");
}

#[test]
fn test_lex_line_tracking() {
    let src = "\n<http://foo>";
    let mut lex = Lexer::new(src);
    let tok = lex.next_token().unwrap();
    assert_eq!(tok.line, 2);
    assert_eq!(tok.col, 1);
}
