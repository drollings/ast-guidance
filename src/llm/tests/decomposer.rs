use super::*;

// Hermetic by construction: `127.0.0.1:1` is a refused (unreachable)
// loopback port, so the fallback test below can never fire real inference
// even if an Ollama server happens to listen on the default 11434.
const UNREACHABLE_API_URL: &str = "http://127.0.0.1:1/v1";

fn make_decomposer() -> LocalDecomposer {
    let config = DecomposerConfig::builder()
        .llm(
            LlmConfig::new()
                .api_url(UNREACHABLE_API_URL.into())
                .model("llama3".into())
                .build(),
        )
        .max_subtasks(5)
        .max_depth(2)
        .build();
    LocalDecomposer::new(config)
}

#[test]
fn test_decomposer_creation() {
    let d = make_decomposer();
    assert_eq!(d.config.max_subtasks, 5);
    assert_eq!(d.config.max_depth, 2);
}

#[test]
fn test_decomposer_config_builder() {
    let config = DecomposerConfig::builder()
        .llm(
            LlmConfig::new()
                .api_url("http://localhost:11434/v1".into())
                .model("llama3".into())
                .build(),
        )
        .build();
    assert_eq!(config.max_subtasks, 5);
    assert_eq!(config.max_depth, 2);
}

#[test]
fn test_is_malformed_json_array() {
    assert!(is_malformed_json_array(""));
    assert!(is_malformed_json_array("hello"));
    assert!(!is_malformed_json_array("[\"a\"]"));
    assert!(!is_malformed_json_array("[ \"a\", \"b\" ]"));
}

#[test]
fn test_parse_json_array_valid() {
    let result = parse_json_array("[\"a\", \"b\", \"c\"]", 5).unwrap();
    assert_eq!(result, vec!["a", "b", "c"]);
}

#[test]
fn test_parse_json_array_with_limit() {
    let result = parse_json_array("[\"a\", \"b\", \"c\"]", 2).unwrap();
    assert_eq!(result, vec!["a", "b"]);
}

#[test]
fn test_parse_json_array_invalid() {
    assert!(parse_json_array("not json", 5).is_err());
    assert!(parse_json_array("{}", 5).is_err());
}

#[test]
fn test_parse_json_array_empty() {
    assert!(parse_json_array("[]", 5).is_err());
}

#[test]
fn test_decomposer_fallback_on_malformed_response() {
    // The decomposer points at the refused loopback port 127.0.0.1:1, so
    // `chat_complete` fails immediately (no real inference) and `decompose`
    // must fall back gracefully to the whole task as a single subtask.
    let d = make_decomposer();
    let tasks = d.decompose("test task");
    assert_eq!(tasks, vec!["test task"]);
}
