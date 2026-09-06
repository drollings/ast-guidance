//! Orphan-adoption tests: argv parsing, identity matching, HTTP probing, the
//! adopt-model flow (with self-healing adoption records), the grammar-less
//! manager fallback, and the persisted fleet map round-trip.
//!
//! All hermetic: the fork is replaced by the `StubServer` harness (no
//! inference, no real `llama-server`), and `/proc` is only read, never
//! written.

use super::*;
use std::sync::Arc;
use std::time::Duration;

use crate::config::{DefaultModelParams, InstanceProfile, SidecarConfig};
use crate::instances::stub::StubServer;
use crate::instances::{InstanceClient, InstanceManager};
use crate::supervisor::{LlamaServerSpec, LlamaServerSupervisor, ManagedServer};

fn code_spec(port: u16) -> LlamaServerSpec {
    LlamaServerSpec {
        model_key: "code".to_string(),
        name: "code".to_string(),
        weights: Some("/app/ai/models/gguf/code/latest.gguf".to_string()),
        hf_repo: None,
        hf_file: None,
        port,
        instances: vec![],
        boot: true,
        slot_save_path: None,
        api_key: None,
        instance_wait_s: None,
        defaults: DefaultModelParams::default(),
        extra_args: vec![],
    }
}

fn code_identity() -> ServerIdentity {
    ServerIdentity {
        alias: "code".to_string(),
        model_path: "/app/ai/models/gguf/code/latest.gguf".to_string(),
        instances_supported: true,
    }
}

#[test]
fn argv_parses_both_flag_spellings() {
    let argv = [
        "--host", "127.0.0.1", "--port", "53577", "--alias", "code", "-m",
        "/w.gguf",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect::<Vec<_>>();
    let found = parse_llama_argv(&argv).expect("parses");
    assert_eq!(found.port, 53577);
    assert_eq!(found.alias, "code");
    assert_eq!(found.weights.as_deref(), Some("/w.gguf"));

    let argv_eq = ["--port=42869", "--alias=swarm"]
        .iter()
        .map(|s| (*s).to_string())
        .collect::<Vec<_>>();
    let found_eq = parse_llama_argv(&argv_eq).expect("=-spelling parses");
    assert_eq!(found_eq.port, 42869);
    assert_eq!(found_eq.alias, "swarm");
}

#[test]
fn argv_rejects_missing_alias_and_non_loopback_binds() {
    let no_alias = ["--port", "1"].iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
    assert!(parse_llama_argv(&no_alias).is_none());

    // A stock router-mode server on 0.0.0.0 is never ours, even with an alias.
    let public = ["--host", "0.0.0.0", "--port", "8080", "--alias", "code"]
        .iter()
        .map(|s| (*s).to_string())
        .collect::<Vec<_>>();
    assert!(parse_llama_argv(&public).is_none());

    // Absent --host defaults to loopback in llama-server, so it is adoptable.
    let implicit = ["--port", "2", "--alias", "code"]
        .iter()
        .map(|s| (*s).to_string())
        .collect::<Vec<_>>();
    assert!(parse_llama_argv(&implicit).is_some());
}

#[test]
fn pid_guard_rejects_foreign_processes() {
    // The test runner itself is not a llama-server.
    let me = std::process::id();
    assert!(!pid_still_ours(me, "code", 0));
    // PID 1 is init, never a llama-server with our alias.
    assert!(!pid_still_ours(1, "code", 0));
}

#[test]
fn server_matches_weights_on_either_channel() {
    let spec = code_spec(1111);
    let identity = code_identity();

    // Cmdline weights agree.
    let via_cmd = DiscoveredServer {
        pid: 9,
        port: 53577,
        alias: "code".to_string(),
        weights: Some("/app/ai/models/gguf/code/latest.gguf".to_string()),
        hf_repo: None,
    };
    assert!(server_matches(&spec, &via_cmd, &identity));

    // Cmdline silent, /props path agrees.
    let via_props = DiscoveredServer {
        pid: 9,
        port: 53577,
        alias: "code".to_string(),
        weights: None,
        hf_repo: None,
    };
    assert!(server_matches(&spec, &via_props, &identity));

    // Same alias, different weights: a different deployment, never adopted.
    let other_weights = DiscoveredServer {
        pid: 9,
        port: 53577,
        alias: "code".to_string(),
        weights: Some("/other/code.gguf".to_string()),
        hf_repo: None,
    };
    let other_identity = ServerIdentity {
        alias: "code".to_string(),
        model_path: "/other/code.gguf".to_string(),
        instances_supported: true,
    };
    assert!(!server_matches(&spec, &other_weights, &other_identity));

    // Alias mismatch is always a refusal.
    let other_alias = DiscoveredServer {
        pid: 9,
        port: 53577,
        alias: "swarm".to_string(),
        weights: Some("/app/ai/models/gguf/code/latest.gguf".to_string()),
        hf_repo: None,
    };
    assert!(!server_matches(&spec, &other_alias, &identity));
}

#[test]
fn server_matches_hf_and_refuses_sourceless_specs() {
    let mut spec = code_spec(1111);
    spec.weights = None;
    spec.hf_repo = Some("org/model".to_string());
    let identity = ServerIdentity {
        alias: "code".to_string(),
        model_path: String::new(),
        instances_supported: true,
    };
    let found = DiscoveredServer {
        pid: 9,
        port: 53577,
        alias: "code".to_string(),
        weights: None,
        hf_repo: Some("org/model".to_string()),
    };
    assert!(server_matches(&spec, &found, &identity));

    // No weights source configured: alias alone never identifies a deployment.
    spec.hf_repo = None;
    assert!(!server_matches(&spec, &found, &identity));
}

fn probe_stub(instances_status: u16) -> StubServer {
    type Handler = Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync>;
    let handler: Handler =
        Arc::new(move |method, path, _body| {
            if method == "GET" && path == "/health" {
                (200, r#"{"status":"ok"}"#.to_string())
            } else if method == "GET" && path == "/props" {
                (
                    200,
                    serde_json::json!({
                        "model_alias": "code",
                        "model_path": "/app/ai/models/gguf/code/latest.gguf",
                    })
                    .to_string(),
                )
            } else if method == "GET" && path == "/instances" {
                (instances_status, r#"{"instances":[]}"#.to_string())
            } else {
                (404, r#"{"error":{"message":"not found"}}"#.to_string())
            }
        });
    StubServer::start(handler)
}

#[tokio::test]
async fn probe_identity_reports_the_grammar_verdict() {
    let client = reqwest::Client::new();

    let with_instances = probe_stub(200);
    let identity = probe_identity(&client, &with_instances.base_url(), None)
        .await
        .expect("healthy stub probes");
    assert_eq!(identity.alias, "code");
    assert_eq!(identity.model_path, "/app/ai/models/gguf/code/latest.gguf");
    assert!(identity.instances_supported);

    // 404 on /instances = a grammar-less (single-context) server.
    let grammar_less = probe_stub(404);
    let identity = probe_identity(&client, &grammar_less.base_url(), None)
        .await
        .expect("grammar-less stub probes");
    assert!(!identity.instances_supported);

    // Nothing listening: never adopt on a partial read.
    assert!(
        probe_identity(&client, "http://127.0.0.1:1", None)
            .await
            .is_none()
    );
}

fn pinned_profile(name: &str) -> InstanceProfile {
    InstanceProfile {
        name: Some(name.to_string()),
        group: Some(name.to_string()),
        count: 1,
        num_ctx: 8192,
        parallel: None,
        pinned: true,
        no_sleep: false,
        sleep_idle_seconds: None,
        default: true,
        resume: false,
        params: None,
        max_ctx: None,
    }
}

#[tokio::test]
async fn grammar_less_manager_skips_management_without_http() {
    // A dead base URL: any management call would fail, so reaching this
    // assertion proves the grammar-less path performs zero HTTP.
    let client = InstanceClient::new(
        reqwest::Client::new(),
        "http://127.0.0.1:1",
        None,
    );
    let manager = InstanceManager::new(
        "code",
        client,
        vec![pinned_profile("default")],
        SidecarConfig::default(),
    );
    manager.set_instances_supported(false);
    assert!(!manager.has_instances_api());

    manager.reconcile().await.expect("reconcile suspends");
    manager.ensure_instance("default").await.expect("on-demand skips");
    manager.ensure_group("default").await.expect("group alloc skips");
    manager
        .ensure_group_ready("default")
        .await
        .expect("group ready skips");
}

#[test]
fn adopt_model_rewires_the_entry_and_self_heals() {
    let sup = LlamaServerSupervisor::with_client_for_test(
        std::path::PathBuf::from("/bin/false"),
        reqwest::Client::new(),
    );
    let spec = code_spec(1111);
    sup.servers.insert(
        "code".to_string(),
        ManagedServer::with_liveness(spec.clone(), Duration::from_secs(30), 3, 5),
    );

    let identity = ServerIdentity {
        alias: "code".to_string(),
        model_path: "/app/ai/models/gguf/code/latest.gguf".to_string(),
        instances_supported: false,
    };
    sup.adopt_model("code", &spec, 4242, 53577, &identity);

    assert_eq!(
        sup.adoption_info("code"),
        Some(AdoptionInfo {
            pid: 4242,
            instances_supported: false,
        })
    );
    assert!(sup.is_adopted("code"));
    let server = sup.server_for("code").expect("adopted entry");
    assert_eq!(server.base_url(), "http://127.0.0.1:53577");
    assert_eq!(server.adopted_pid(), Some(4242));
    assert!(server.is_running());

    // The orphan dies mid-life and a fresh child takes over: the stale record
    // self-heals to None instead of suppressing the respawned server's
    // grammar.
    server.clear_adoption();
    assert!(!sup.is_adopted("code"));
    assert!(sup.adoption_info("code").is_none());
}

#[tokio::test]
async fn persist_state_round_trips_the_fleet_map() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state_path = dir.path().join("servers.json");
    let mut config = crate::config::RouterConfig::default();
    config.sidecar.server_state_path =
        Some(state_path.to_string_lossy().to_string());

    let sup = LlamaServerSupervisor::with_client_for_test(
        std::path::PathBuf::from("/bin/false"),
        reqwest::Client::new(),
    );
    // Only running servers persist: the adopted entry qualifies, the idle one
    // does not.
    let adopted = ManagedServer::with_liveness(
        code_spec(53577),
        Duration::from_secs(30),
        3,
        5,
    );
    adopted.mark_adopted(4242);
    sup.servers.insert("code".to_string(), adopted);
    sup.servers.insert(
        "idle".to_string(),
        ManagedServer::with_liveness(code_spec(1111), Duration::from_secs(30), 3, 5),
    );

    fluent_concurrency::scope::CURRENT_CAPS
        .scope(
            fluent_concurrency::capability::default_capability_set(),
            sup.persist_state(&config),
        )
        .await;

    let text = std::fs::read_to_string(&state_path).expect("state file written");
    let value: serde_json::Value = serde_json::from_str(&text).expect("valid json");
    assert_eq!(value["servers"]["code"]["port"], 53577);
    assert_eq!(value["servers"]["code"]["pid"], 4242);
    assert!(value["servers"].get("idle").is_none());
}
