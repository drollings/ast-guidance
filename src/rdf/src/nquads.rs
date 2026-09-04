use crate::lexer::{Lexer, TokenKind};
use crate::parser::{Literal, Term};
use crate::RdfError;

#[derive(Debug, Clone, PartialEq)]
pub struct Quad {
    pub subject: Term,
    pub predicate: Term,
    pub object: Term,
    pub graph: Option<Term>,
}

pub struct NQuadsParser;

impl NQuadsParser {
    pub fn parse_line(line: &str) -> Result<Option<Quad>, RdfError> {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            return Ok(None);
        }

        let mut lex = Lexer::new(line);
        let mut terms: Vec<Term> = Vec::new();
        let mut graph: Option<Term> = None;

        loop {
            let tok = lex.next_token()?;
            match tok.kind {
                TokenKind::Eof | TokenKind::Dot => break,
                TokenKind::Iri => {
                    let iri = Self::extract_iri(tok.value);
                    terms.push(Term::Iri(iri));
                }
                TokenKind::PrefixedName => {
                    let iri = tok.value.to_string();
                    terms.push(Term::Iri(iri));
                }
                TokenKind::BlankNode => {
                    let id = tok.value[2..].to_string();
                    terms.push(Term::BlankNode(id));
                }
                TokenKind::Literal => {
                    let lit = Self::parse_literal(&mut lex, tok.value)?;
                    terms.push(Term::Literal(lit));
                }
                _ => {
                    return Err(RdfError::UnexpectedToken {
                        line: tok.line,
                        col: tok.col,
                        expected: "term".into(),
                        got: format!("{:?}", tok.kind),
                    });
                }
            }
        }

        if terms.len() < 3 {
            return Ok(None);
        }

        if terms.len() > 3 {
            graph = Some(terms.remove(3));
        }

        if terms.len() >= 3 {
            Ok(Some(Quad {
                subject: terms.remove(0),
                predicate: terms.remove(0),
                object: terms.remove(0),
                graph,
            }))
        } else {
            Ok(None)
        }
    }

    fn extract_iri(raw: &str) -> String {
        if raw.len() >= 2 && raw.starts_with('<') && raw.ends_with('>') {
            raw[1..raw.len() - 1].to_string()
        } else {
            raw.to_string()
        }
    }

    fn parse_literal(lex: &mut Lexer, value: &str) -> Result<Literal, RdfError> {
        let content = Self::extract_literal_content(value);
        let pk = lex.next_token()?;
        if pk.kind == TokenKind::LangTag {
            let lang = pk.value[1..].to_string();
            return Ok(Literal {
                value: content,
                lang: Some(lang),
                datatype: None,
            });
        }
        if pk.kind == TokenKind::DatatypeMarker {
            let dt_tok = lex.next_token()?;
            let dt_iri = match dt_tok.kind {
                TokenKind::Iri => Self::extract_iri(dt_tok.value),
                TokenKind::PrefixedName => dt_tok.value.to_string(),
                _ => {
                    return Err(RdfError::UnexpectedToken {
                        line: dt_tok.line,
                        col: dt_tok.col,
                        expected: "datatype IRI".into(),
                        got: format!("{:?}", dt_tok.kind),
                    });
                }
            };
            return Ok(Literal {
                value: content,
                lang: None,
                datatype: Some(dt_iri),
            });
        }
        Ok(Literal {
            value: content,
            lang: None,
            datatype: None,
        })
    }

    fn extract_literal_content(raw: &str) -> String {
        if raw.len() >= 6 && raw.starts_with("\"\"\"") && raw.ends_with("\"\"\"") {
            raw[3..raw.len() - 3].to_string()
        } else if raw.len() >= 2 && raw.starts_with('"') && raw.ends_with('"') {
            raw[1..raw.len() - 1].to_string()
        } else {
            raw.to_string()
        }
    }
}

#[cfg(test)]
#[path = "../tests/nquads.rs"]
mod tests;
