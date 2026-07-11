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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_command() {
        let tokens = parse_command("echo hello").unwrap();
        assert_eq!(tokens, vec!["echo", "hello"]);
    }

    #[test]
    fn three_tokens() {
        let tokens = parse_command("ls -la /tmp").unwrap();
        assert_eq!(tokens, vec!["ls", "-la", "/tmp"]);
    }

    #[test]
    fn single_quoted() {
        let tokens = parse_command("echo 'hello world'").unwrap();
        assert_eq!(tokens, vec!["echo", "hello world"]);
    }

    #[test]
    fn double_quoted() {
        let tokens = parse_command("echo \"hello world\"").unwrap();
        assert_eq!(tokens, vec!["echo", "hello world"]);
    }

    #[test]
    fn quoted_concatenation() {
        let tokens = parse_command("echo a'b'c").unwrap();
        assert_eq!(tokens, vec!["echo", "abc"]);
    }

    #[test]
    fn backslash_escape() {
        let tokens = parse_command("echo hello\\ world").unwrap();
        assert_eq!(tokens, vec!["echo", "hello world"]);
    }

    #[test]
    fn rejects_pipe() {
        assert!(parse_command("echo | cat").is_err());
    }

    #[test]
    fn rejects_redirect() {
        assert!(parse_command("echo > file").is_err());
    }

    #[test]
    fn rejects_backtick() {
        assert!(parse_command("echo `ls`").is_err());
    }

    #[test]
    fn rejects_dollar_sign() {
        assert!(parse_command("echo $HOME").is_err());
    }

    #[test]
    fn rejects_double_ampersand() {
        assert!(parse_command("make && make install").is_err());
    }

    #[test]
    fn rejects_metachar_in_double_quotes() {
        assert!(parse_command("echo \"$(pwd)\"").is_err());
    }

    #[test]
    fn empty_string() {
        assert!(parse_command("").is_err());
    }

    #[test]
    fn whitespace_only() {
        assert!(parse_command("   ").is_err());
    }

    #[test]
    fn unterminated_single_quote() {
        assert!(parse_command("echo 'hello").is_err());
    }

    #[test]
    fn unterminated_double_quote() {
        assert!(parse_command("echo \"hello").is_err());
    }
}
