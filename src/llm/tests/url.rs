use super::*;

#[test]
fn is_local_host_variants() {
    assert!(is_local_host("localhost"));
    assert!(is_local_host("127.0.0.1"));
    assert!(is_local_host("::1"));
    assert!(!is_local_host("example.com"));
}

#[test]
fn local_host_case_whitespace_and_loopback_range() {
    // Characterization (M5): the llm equivalence class — case-folded,
    // whitespace-trimmed, full 127/8 range. The 127.0.0.2 case DIVERGES
    // from the router's hosts_equivalent (locked on both sides).
    assert!(is_local_host("LocalHost"));
    assert!(is_local_host("  LOCALHOST  "));
    assert!(is_local_host("127.0.0.2"));
    assert!(is_local_host("127.255.0.1"));
    assert!(!is_local_host("128.0.0.1"));
    assert!(!is_local_host("localhost.example.com"));
    assert!(!is_local_host(""));
}

#[test]
fn extract_host_shapes() {
    // Characterization (M5): verbatim host-extraction behavior.
    assert_eq!(extract_host("http://localhost:11434/v1"), "localhost");
    assert_eq!(extract_host("https://api.example.com/v1"), "api.example.com");
    assert_eq!(extract_host("http://127.0.0.1:8079/x"), "127.0.0.1");
    assert_eq!(extract_host("bare-host"), "bare-host");
    // IPv6-bracket naive: cuts at the first ':' inside "[::1]". Locked
    // (not "fixed") — this is why the router's host:port parsing keeps its
    // own bracket-aware shape instead of composing this helper.
    assert_eq!(extract_host("http://[::1]:8079/x"), "[");
}

#[test]
fn hosts_equivalent_with_preserves_both_classes() {
    // The parameterized primitive reproduces each side's class exactly.
    let llm = HostEquivalence { fold_case: true, loopback_range_127: true };
    assert!(hosts_equivalent_with("LocalHost", "127.0.0.2", llm));
    assert!(hosts_equivalent_with("::1", "localhost", llm));
    assert!(!hosts_equivalent_with("example.com", "127.0.0.1", llm));
    assert!(hosts_equivalent_with("Example.COM", "example.com", llm));
    let router = HostEquivalence { fold_case: false, loopback_range_127: false };
    assert!(hosts_equivalent_with("localhost", "127.0.0.1", router));
    assert!(hosts_equivalent_with("::1", "127.0.0.1", router));
    assert!(!hosts_equivalent_with("LocalHost", "localhost", router));
    assert!(!hosts_equivalent_with("127.0.0.2", "127.0.0.1", router));
    assert!(!hosts_equivalent_with("[::1]", "::1", router));
}

#[test]
fn is_local_host_matches_wrapper_over_table() {
    // `is_local_host` is a thin wrapper: local(h) == equivalent(h, 127.0.0.1).
    let llm = HostEquivalence { fold_case: true, loopback_range_127: true };
    for h in [
        "localhost", "LocalHost", "  localhost  ", "127.0.0.1", "127.0.0.2",
        "::1", "example.com", "Example.COM", "", "0.0.0.0", "10.0.0.1",
    ] {
        assert_eq!(
            is_local_host(h),
            hosts_equivalent_with(h, "127.0.0.1", llm),
            "wrapper parity for {h:?}"
        );
    }
}

#[test]
fn is_private_ip_ranges() {
    assert!(is_private_ip("10.0.0.1"));
    assert!(is_private_ip("192.168.1.1"));
    assert!(is_private_ip("172.16.0.1"));
    assert!(!is_private_ip("8.8.8.8"));
}

#[test]
fn validate_local_http_accepts() {
    assert!(validate_https_or_local_http("http://localhost:11434").is_ok());
}

#[test]
fn validate_rejects_bare_hostname() {
    assert!(validate_https_or_local_http("localhost").is_err());
}

#[test]
fn validate_rejects_empty() {
    assert!(validate_https_or_local_http("").is_err());
}

#[test]
fn allows_localhost_http() {
    assert!(validate_https_or_local_http("http://localhost:11434/api/embed").is_ok());
}

#[test]
fn chat_completions_url_already_suffixed() {
    assert_eq!(
        chat_completions_url("http://localhost:11434/v1/chat/completions"),
        "http://localhost:11434/v1/chat/completions"
    );
}

#[test]
fn chat_completions_url_appends() {
    assert_eq!(
        chat_completions_url("http://localhost:11434/v1"),
        "http://localhost:11434/v1/chat/completions"
    );
}

#[test]
fn chat_completions_url_trims_trailing_slash() {
    assert_eq!(
        chat_completions_url("http://localhost:11434/v1/"),
        "http://localhost:11434/v1/chat/completions"
    );
}

#[test]
fn derive_embeddings_url_v1_chat() {
    assert_eq!(
        derive_embeddings_url("http://host:port/v1/chat/completions"),
        "http://host:port/v1"
    );
}

#[test]
fn derive_embeddings_url_plain_chat() {
    assert_eq!(
        derive_embeddings_url("http://host:port/chat/completions"),
        "http://host:port/v1"
    );
}

#[test]
fn derive_embeddings_url_passthrough() {
    assert_eq!(
        derive_embeddings_url("http://host:port/v1"),
        "http://host:port/v1"
    );
}
