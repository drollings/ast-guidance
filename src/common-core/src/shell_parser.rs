//! Safe shell parser: whitespace+quote tokenizer that refuses metacharacters.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShellParseError {
    #[error("shell metacharacter detected")]
    ShellMetacharacter,
    #[error("unterminated quote")]
    UnterminatedQuote,
    #[error("empty command")]
    EmptyCommand,
    #[error("out of memory")]
    OutOfMemory,
}

const METACHARACTERS: &[u8] = b"|&;<>`$(){}";

fn is_metachar(c: u8) -> bool {
    METACHARACTERS.contains(&c) || c == b'\n' || c == b'\r'
}

enum State {
    Idle,
    Token,
    SingleQuote,
    DoubleQuote,
}

pub fn parse_command(cmd: &str) -> Result<Vec<String>, ShellParseError> {
    let bytes = cmd.as_bytes();
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut state = State::Idle;
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];
        match state {
            State::Idle => {
                if c.is_ascii_whitespace() {
                    i += 1;
                    continue;
                }
                if c == b'\'' {
                    state = State::SingleQuote;
                    i += 1;
                    continue;
                }
                if c == b'"' {
                    state = State::DoubleQuote;
                    i += 1;
                    continue;
                }
                if is_metachar(c) {
                    return Err(ShellParseError::ShellMetacharacter);
                }
                if c == b'\\' && i + 1 < bytes.len() {
                    current.push(bytes[i + 1] as char);
                    i += 2;
                    continue;
                }
                state = State::Token;
                current.push(c as char);
                i += 1;
            }
            State::Token => {
                if c.is_ascii_whitespace() {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                    state = State::Idle;
                    i += 1;
                    continue;
                }
                if is_metachar(c) {
                    return Err(ShellParseError::ShellMetacharacter);
                }
                if c == b'\'' {
                    state = State::SingleQuote;
                    i += 1;
                    continue;
                }
                if c == b'"' {
                    state = State::DoubleQuote;
                    i += 1;
                    continue;
                }
                if c == b'\\' && i + 1 < bytes.len() {
                    current.push(bytes[i + 1] as char);
                    i += 2;
                    continue;
                }
                current.push(c as char);
                i += 1;
            }
            State::SingleQuote => {
                if c == b'\'' {
                    state = State::Token;
                    i += 1;
                    continue;
                }
                current.push(c as char);
                i += 1;
            }
            State::DoubleQuote => {
                if c == b'"' {
                    state = State::Token;
                    i += 1;
                    continue;
                }
                if c == b'\\' && i + 1 < bytes.len() {
                    current.push(bytes[i + 1] as char);
                    i += 2;
                    continue;
                }
                if is_metachar(c) {
                    return Err(ShellParseError::ShellMetacharacter);
                }
                current.push(c as char);
                i += 1;
            }
        }
    }

    match state {
        State::SingleQuote | State::DoubleQuote => {
            return Err(ShellParseError::UnterminatedQuote);
        }
        State::Token => {
            if !current.is_empty() {
                tokens.push(current);
            }
        }
        State::Idle => {}
    }

    if tokens.is_empty() {
        return Err(ShellParseError::EmptyCommand);
    }

    Ok(tokens)
}

