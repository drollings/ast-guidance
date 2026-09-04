use common_core::shell_parser::*;


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
