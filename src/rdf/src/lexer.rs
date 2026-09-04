use crate::RdfError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Iri,
    PrefixedName,
    BlankNode,
    BlankNodeOpen,
    BlankNodeClose,
    Literal,
    LangTag,
    DatatypeMarker,
    Keyword,
    Dot,
    Semicolon,
    Comma,
    OpenParen,
    CloseParen,
    Eof,
}

#[derive(Debug, Clone, Copy)]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub value: &'a str,
    pub line: u32,
    pub col: u32,
}

pub struct Lexer<'a> {
    src: &'a str,
    pos: usize,
    line: u32,
    col: u32,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.src.len() {
            let c = self.src.as_bytes()[self.pos];
            if c == b'#' {
                while self.pos < self.src.len() && self.src.as_bytes()[self.pos] != b'\n' {
                    self.pos += 1;
                    self.col += 1;
                }
            } else if c == b'\n' {
                self.pos += 1;
                self.line += 1;
                self.col = 1;
            } else if c == b'\r' {
                self.pos += 1;
            } else if c == b' ' || c == b'\t' {
                self.pos += 1;
                self.col += 1;
            } else {
                break;
            }
        }
    }

    fn advance(&mut self) {
        if self.pos < self.src.len() {
            if self.src.as_bytes()[self.pos] == b'\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    pub fn next_token(&mut self) -> Result<Token<'a>, RdfError> {
        self.skip_whitespace_and_comments();

        if self.pos >= self.src.len() {
            return Ok(Token {
                kind: TokenKind::Eof,
                value: "",
                line: self.line,
                col: self.col,
            });
        }

        let start_line = self.line;
        let start_col = self.col;
        let c = self.peek().unwrap();

        match c {
            b'<' => self.lex_iri(start_line, start_col),
            b'"' => self.lex_literal(start_line, start_col),
            b'_' if self.src.as_bytes().get(self.pos + 1) == Some(&b':') => {
                Ok(self.lex_blank_node(start_line, start_col))
            }
            b'[' => {
                let start = self.pos;
                self.advance();
                Ok(Token {
                    kind: TokenKind::BlankNodeOpen,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b']' => {
                let start = self.pos;
                self.advance();
                Ok(Token {
                    kind: TokenKind::BlankNodeClose,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b'@' => Ok(self.lex_at_directive(start_line, start_col)),
            b':' => Ok(self.lex_prefixed_name(start_line, start_col)),
            b'^' if self.src.as_bytes().get(self.pos + 1) == Some(&b'^') => {
                let start = self.pos;
                self.advance();
                self.advance();
                Ok(Token {
                    kind: TokenKind::DatatypeMarker,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b'.' if self.pos + 1 < self.src.len()
                && self.src.as_bytes()[self.pos + 1].is_ascii_digit() =>
            {
                Ok(self.lex_numeric_literal(start_line, start_col))
            }
            b'.' => {
                let start = self.pos;
                self.advance();
                Ok(Token {
                    kind: TokenKind::Dot,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b';' => {
                let start = self.pos;
                self.advance();
                Ok(Token {
                    kind: TokenKind::Semicolon,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b',' => {
                let start = self.pos;
                self.advance();
                Ok(Token {
                    kind: TokenKind::Comma,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b'(' => {
                let start = self.pos;
                self.advance();
                Ok(Token {
                    kind: TokenKind::OpenParen,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b')' => {
                let start = self.pos;
                self.advance();
                Ok(Token {
                    kind: TokenKind::CloseParen,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b'a' if self.pos + 1 >= self.src.len()
                || is_name_end_char(self.src.as_bytes()[self.pos + 1]) =>
            {
                let start = self.pos;
                self.advance();
                Ok(Token {
                    kind: TokenKind::Keyword,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                })
            }
            b'+' | b'-' | b'0'..=b'9' => Ok(self.lex_numeric_literal(start_line, start_col)),
            _ if is_prefix_start_char(c) => Ok(self.lex_prefixed_name(start_line, start_col)),
            _ => Err(RdfError::UnexpectedChar {
                line: start_line,
                col: start_col,
            }),
        }
    }

    fn lex_iri(&mut self, start_line: u32, start_col: u32) -> Result<Token<'a>, RdfError> {
        let start = self.pos;
        self.advance();
        while self.pos < self.src.len() {
            let ch = self.src.as_bytes()[self.pos];
            if ch == b'>' {
                self.advance();
                return Ok(Token {
                    kind: TokenKind::Iri,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                });
            }
            if ch == b'\\' {
                self.advance();
                if self.pos >= self.src.len() {
                    return Err(RdfError::InvalidEscape);
                }
                let esc = self.src.as_bytes()[self.pos];
                if esc != b'u' && esc != b'U' {
                    return Err(RdfError::InvalidEscape);
                }
                self.advance();
            } else if ch == b'\n' || ch == b'\r' {
                return Err(RdfError::UnterminatedIRI);
            } else {
                self.advance();
            }
        }
        Err(RdfError::UnterminatedIRI)
    }

    fn lex_literal(&mut self, start_line: u32, start_col: u32) -> Result<Token<'a>, RdfError> {
        let start = self.pos;
        self.advance();

        let triple = self.pos + 1 < self.src.len()
            && self.src.as_bytes()[self.pos] == b'"'
            && self.src.as_bytes()[self.pos + 1] == b'"';

        if triple {
            self.advance();
            self.advance();
            while self.pos + 2 < self.src.len() {
                if self.src.as_bytes()[self.pos] == b'"'
                    && self.src.as_bytes()[self.pos + 1] == b'"'
                    && self.src.as_bytes()[self.pos + 2] == b'"'
                {
                    self.advance();
                    self.advance();
                    self.advance();
                    return Ok(Token {
                        kind: TokenKind::Literal,
                        value: &self.src[start..self.pos],
                        line: start_line,
                        col: start_col,
                    });
                }
                if self.src.as_bytes()[self.pos] == b'\\' {
                    self.advance();
                    if self.pos >= self.src.len() {
                        return Err(RdfError::InvalidEscape);
                    }
                }
                self.advance();
            }
            return Err(RdfError::UnterminatedLiteral);
        }

        while self.pos < self.src.len() {
            let ch = self.src.as_bytes()[self.pos];
            if ch == b'"' {
                self.advance();
                return Ok(Token {
                    kind: TokenKind::Literal,
                    value: &self.src[start..self.pos],
                    line: start_line,
                    col: start_col,
                });
            }
            if ch == b'\\' {
                self.advance();
                if self.pos >= self.src.len() {
                    return Err(RdfError::InvalidEscape);
                }
                let esc = self.src.as_bytes()[self.pos];
                let valid = matches!(esc, b't' | b'n' | b'r' | b'"' | b'\'' | b'\\' | b'u' | b'U');
                if !valid {
                    return Err(RdfError::InvalidEscape);
                }
                self.advance();
            } else if ch == b'\n' || ch == b'\r' {
                return Err(RdfError::UnterminatedLiteral);
            } else {
                self.advance();
            }
        }
        Err(RdfError::UnterminatedLiteral)
    }

    fn lex_numeric_literal(&mut self, start_line: u32, start_col: u32) -> Token<'a> {
        let start = self.pos;
        if let Some(c) = self.peek() {
            if c == b'+' || c == b'-' {
                self.advance();
            }
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        if let Some(c) = self.peek() {
            if c == b'.'
                && self.pos + 1 < self.src.len()
                && self.src.as_bytes()[self.pos + 1].is_ascii_digit()
            {
                self.advance();
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }
        if let Some(c) = self.peek() {
            if c == b'e' || c == b'E' {
                self.advance();
                if let Some(c) = self.peek() {
                    if c == b'+' || c == b'-' {
                        self.advance();
                    }
                }
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }
        Token {
            kind: TokenKind::Literal,
            value: &self.src[start..self.pos],
            line: start_line,
            col: start_col,
        }
    }

    fn lex_blank_node(&mut self, start_line: u32, start_col: u32) -> Token<'a> {
        let start = self.pos;
        self.advance();
        self.advance();
        while self.pos < self.src.len() && is_name_char(self.src.as_bytes()[self.pos]) {
            self.advance();
        }
        Token {
            kind: TokenKind::BlankNode,
            value: &self.src[start..self.pos],
            line: start_line,
            col: start_col,
        }
    }

    fn lex_at_directive(&mut self, start_line: u32, start_col: u32) -> Token<'a> {
        let start = self.pos;
        self.advance();
        while self.pos < self.src.len() && is_lang_char(self.src.as_bytes()[self.pos]) {
            self.advance();
        }
        let word = &self.src[start..self.pos];
        let is_keyword = matches!(word, "@prefix" | "@base" | "@PREFIX" | "@BASE");
        Token {
            kind: if is_keyword {
                TokenKind::Keyword
            } else {
                TokenKind::LangTag
            },
            value: word,
            line: start_line,
            col: start_col,
        }
    }

    fn lex_prefixed_name(&mut self, start_line: u32, start_col: u32) -> Token<'a> {
        let start = self.pos;
        while self.pos < self.src.len() && is_prefix_char(self.src.as_bytes()[self.pos]) {
            self.advance();
        }
        if self.pos < self.src.len() && self.src.as_bytes()[self.pos] == b':' {
            self.advance();
            while self.pos < self.src.len() && is_local_name_char(self.src.as_bytes()[self.pos]) {
                self.advance();
            }
        }
        let word = &self.src[start..self.pos];
        let is_keyword = matches!(word, "PREFIX" | "BASE" | "true" | "false");
        Token {
            kind: if is_keyword {
                TokenKind::Keyword
            } else {
                TokenKind::PrefixedName
            },
            value: word,
            line: start_line,
            col: start_col,
        }
    }
}

fn is_name_end_char(ch: u8) -> bool {
    matches!(
        ch,
        b' ' | b'\t' | b'\n' | b'\r' | b'.' | b';' | b',' | b')' | b']'
    )
}

fn is_prefix_start_char(ch: u8) -> bool {
    ch.is_ascii_alphabetic() || ch > 127
}

fn is_prefix_char(ch: u8) -> bool {
    is_prefix_start_char(ch) || ch.is_ascii_digit() || ch == b'-' || ch == b'_'
}

fn is_local_name_char(ch: u8) -> bool {
    is_prefix_char(ch) || ch == b'.' || ch == b'%' || ch == b':'
}

fn is_name_char(ch: u8) -> bool {
    ch.is_ascii_alphanumeric() || ch == b'_' || ch == b'-' || ch == b'.'
}

fn is_lang_char(ch: u8) -> bool {
    ch.is_ascii_alphabetic() || ch == b'-'
}

#[cfg(test)]
#[path = "../tests/lexer.rs"]
mod tests;
