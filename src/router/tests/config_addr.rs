use super::*;

#[test]
fn hosts_equivalent_same_host() {
    assert!(hosts_equivalent("localhost", "localhost"));
}

#[test]
fn hosts_equivalent_localhost_and_ip() {
    assert!(hosts_equivalent("localhost", "127.0.0.1"));
}

#[test]
fn hosts_equivalent_ipv6_local() {
    assert!(hosts_equivalent("::1", "127.0.0.1"));
}

#[test]
fn hosts_equivalent_different_hosts() {
    assert!(!hosts_equivalent("upstream.test", "127.0.0.1"));
    assert!(!hosts_equivalent("0.0.0.0", "127.0.0.1"));
}

#[test]
fn hosts_equivalent_works_with_whitespace() {
    assert!(hosts_equivalent("  localhost  ", "127.0.0.1"));
}

#[test]
fn hosts_equivalent_divergences_locked() {
    // Characterization (M5): the router equivalence class — trim-only, three
    // exact loopback forms. Each divergence from fluent_llm::url is
    // security-relevant (self-routing-loop vs SSRF threat models) and locked.
    assert!(!hosts_equivalent("LocalHost", "localhost"), "no case folding");
    assert!(!hosts_equivalent("LOCALHOST", "127.0.0.1"));
    assert!(!hosts_equivalent("127.0.0.2", "127.0.0.1"), "no 127/8 range");
    assert!(hosts_equivalent("127.0.0.1", "127.0.0.1"));
    assert!(!hosts_equivalent("[::1]", "::1"), "no bracket unwrap");
    assert!(!hosts_equivalent("127.0.0.1 ", "upstream.test"));
}

#[test]
fn parse_bind_addr_simple() {
    let (host, port) = parse_bind_addr("0.0.0.0:8079").unwrap();
    assert_eq!(host, "0.0.0.0");
    assert_eq!(port, 8079);
}

#[test]
fn parse_bind_addr_empty_fails() {
    assert!(parse_bind_addr("").is_err());
}

#[test]
fn parse_bind_addr_missing_port_fails() {
    assert!(parse_bind_addr("localhost").is_err());
}

#[test]
fn parse_bind_addr_ipv6() {
    let (host, port) = parse_bind_addr("[::1]:8079").unwrap();
    assert_eq!(host, "::1");
    assert_eq!(port, 8079);
}

#[test]
fn validate_ok_when_no_models() {
    let models = HashMap::new();
    assert!(validate_no_self_routing("0.0.0.0:8079", &models).is_ok());
}

#[test]
fn validate_ok_when_models_point_upstream() {
    let mut models = HashMap::new();
    models.insert(
        "fast".into(),
        ModelEntry {
            endpoint: "http://upstream.test:8080/v1/chat/completions".into(),
            name: Some("fast".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            instances: None,
            weights: None,
            hf_repo: None,
            hf_file: None,
            api_key: None,
        },
    );
    assert!(validate_no_self_routing("0.0.0.0:8079", &models).is_ok());
}

#[test]
fn validate_rejects_self_loop_localhost() {
    let mut models = HashMap::new();
    models.insert(
        "fast".into(),
        ModelEntry {
            endpoint: "http://localhost:8079/v1/chat/completions".into(),
            name: Some("fast".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            instances: None,
            weights: None,
            hf_repo: None,
            hf_file: None,
            api_key: None,
        },
    );
    let err = validate_no_self_routing("127.0.0.1:8079", &models)
        .expect_err("should reject self-routing model");
    assert!(
        err.to_string().contains("routing loop"),
        "error should mention routing loop: {err}"
    );
}

#[test]
fn validate_rejects_self_loop_exact_match() {
    let mut models = HashMap::new();
    models.insert(
        "fast".into(),
        ModelEntry {
            endpoint: "http://127.0.0.1:8079/v1/chat/completions".into(),
            name: Some("fast".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            instances: None,
            weights: None,
            hf_repo: None,
            hf_file: None,
            api_key: None,
        },
    );
    let err = validate_no_self_routing("127.0.0.1:8079", &models)
        .expect_err("should reject self-routing model");
    assert!(err.to_string().contains("routing loop"));
}

#[test]
fn validate_rejects_when_port_differs_but_host_is_same() {
    let mut models = HashMap::new();
    models.insert(
        "fast".into(),
        ModelEntry {
            endpoint: "http://127.0.0.1:8080/v1/chat/completions".into(),
            name: Some("fast".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            instances: None,
            weights: None,
            hf_repo: None,
            hf_file: None,
            api_key: None,
        },
    );
    // Different port (8080 vs 8079) should be OK
    assert!(validate_no_self_routing("127.0.0.1:8079", &models).is_ok());
}

#[test]
fn validate_empty_bind_addr_errors() {
    let models = HashMap::new();
    let err = validate_no_self_routing("", &models).expect_err("empty bind_addr should error");
    assert!(err.to_string().contains("must be set"));
}

#[test]
fn validate_skips_managed_models_with_no_endpoint() {
    // A managed model (weights/hf_repo/instances) may declare no endpoint;
    // Coral Router assigns and rewrites it at boot. It must not be treated
    // as a self-routing risk.
    let mut models = HashMap::new();
    models.insert(
        "managed".into(),
        ModelEntry {
            endpoint: String::new(),
            name: Some("managed".into()),
            intelligence: 1,
            cost_input: 0.0,
            cost_output: 0.0,
            cost_cached_read: 0.0,
            speed: 10,
            total_timeout_ms: 5000,
            idle_timeout_ms: 2000,
            stream: false,
            filter_thinking: false,
            retry_count: 0,
            retry_base_interval_s: 1,
            params: None,
            instances: Some(
                [("ledger".to_string(), crate::config::InstanceProfile {
                    name: None,
                    group: None,
                    count: 1,
                    num_ctx: 65536,
                    parallel: None,
                    pinned: true,
                    no_sleep: false,
                    sleep_idle_seconds: None,
                    default: false,
                    resume: false,
                    params: None,
                    max_ctx: None,
                })]
                .into_iter()
                .collect(),
            ),
            weights: Some("/models/m.gguf".into()),
            hf_repo: None,
            hf_file: None,
            api_key: None,
        },
    );
    assert!(
        validate_no_self_routing("127.0.0.1:8079", &models).is_ok(),
        "managed model with no configured endpoint is not self-routing"
    );
}
