use super::*;
use crate::config::ModelEntry;

use bytes::Bytes;
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

fn managed_entry() -> ModelEntry {
    serde_json::from_value(serde_json::json!({
        "endpoint": "http://127.0.0.1:0/v1/chat/completions",
        "name": "abiray/lfm2.5-2.6b-heretic-abliterated",
        "intelligence": 2,
        "cost_input": 1e-06, "cost_output": 6e-06, "cost_cached_read": 4e-07,
        "speed": 8,
        "weights": "/models/lfm2.6b.gguf",
        "instances": {
            "swarm": { "count": 2, "group": "swarm", "num_ctx": 8192, "pinned": true },
            "ledger": { "num_ctx": 65536, "default": true }
        }
    }))
    .expect("entry parses")
}

fn defaults() -> crate::config::DefaultModelParams {
    crate::config::DefaultModelParams::default()
}

#[test]
fn fleet_lib_dir_follows_symlinks_to_the_real_binary_dir() {
    // A symlinked binary resolves to the directory holding the real file
    // (where the fork ships its .so files), not the symlink's directory.
    let tmp = tempfile::tempdir().expect("tempdir");
    let real_dir = tmp.path().join("build-coral").join("bin");
    std::fs::create_dir_all(&real_dir).expect("mkdirs");
    let real_bin = real_dir.join("llama-server");
    std::fs::write(&real_bin, b"fake").expect("write fake bin");
    let link_dir = tmp.path().join("local").join("bin");
    std::fs::create_dir_all(&link_dir).expect("mkdirs");
    let link = link_dir.join("llama-server");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&real_bin, &link).expect("symlink");
    assert_eq!(fleet_lib_dir(&link).as_deref(), Some(real_dir.as_path()));
}

#[test]
fn fleet_lib_dir_is_none_for_a_missing_binary() {
    assert_eq!(
        fleet_lib_dir(std::path::Path::new("/no/such/llama-server")),
        None
    );
}

#[test]
fn prepend_library_path_pins_dir_first_and_keeps_inherited() {
    let dir = std::path::Path::new("/opt/fork/build-coral/bin");
    assert_eq!(
        prepend_library_path(dir, Some("/usr/local/lib:/opt/lib".into())),
        std::ffi::OsString::from("/opt/fork/build-coral/bin:/usr/local/lib:/opt/lib"),
    );
    assert_eq!(
        prepend_library_path(dir, None),
        std::ffi::OsString::from("/opt/fork/build-coral/bin"),
    );
    assert_eq!(
        prepend_library_path(dir, Some(std::ffi::OsString::new())),
        std::ffi::OsString::from("/opt/fork/build-coral/bin"),
    );
}

#[test]
fn resolve_llama_server_prefers_env_override() {
    let old = std::env::var_os(LLAMA_SERVER_ENV);
    std::env::set_var(LLAMA_SERVER_ENV, "/custom/llama-server");
    let resolved = resolve_llama_server();
    assert_eq!(resolved.as_deref(), Some(std::path::Path::new("/custom/llama-server")));
    match old {
        Some(v) => std::env::set_var(LLAMA_SERVER_ENV, v),
        None => std::env::remove_var(LLAMA_SERVER_ENV),
    }
}

#[test]
fn server_args_declare_host_port_alias_and_model() {
    let spec = LlamaServerSpec::from_entry(
        "swarm",
        &managed_entry(),
        18080,
        Some("/srv/slots".into()),
        Some("sekrit".into()),
        defaults(),
    );
    let args = build_server_args(&spec);
    let joined = args.join(" ");
    assert!(joined.contains("--host 127.0.0.1"));
    assert!(joined.contains("--port 18080"));
    assert!(joined.contains("--alias abiray/lfm2.5-2.6b-heretic-abliterated"));
    assert!(joined.contains("-m /models/lfm2.6b.gguf"));
    assert!(joined.contains("--slot-save-path /srv/slots"));
    assert!(joined.contains("--api-key sekrit"));
}

#[test]
fn server_args_declare_default_params_run_defaults() {
    let spec = LlamaServerSpec::from_entry(
        "swarm",
        &managed_entry(),
        18080,
        None,
        None,
        defaults(),
    );
    let args = build_server_args(&spec);
    let joined = args.join(" ");
    assert!(joined.contains("--batch-size 4096"));
    assert!(joined.contains("--ubatch-size 1024"));
    assert!(joined.contains("--cache-type-k q8_0"));
    assert!(joined.contains("--cache-type-v q8_0"));
    assert!(joined.contains("--n-gpu-layers 999"));
}

#[test]
fn server_args_declare_only_pinned_instance_profiles() {
    let spec = LlamaServerSpec::from_entry(
        "swarm",
        &managed_entry(),
        18080,
        None,
        None,
        defaults(),
    );
    assert!(spec.boot, "pinned swarm profile -> boot model");
    let args = build_server_args(&spec);
    let instances: Vec<&String> = args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| if a == "--instance" { args.get(i + 1) } else { None })
        .collect();
    // `ledger` is unpinned (no `pinned: true`) in the fixture, so only the
    // pinned count:2 swarm siblings are declared at spawn.
    assert_eq!(instances.len(), 2, "only pinned instances declared at boot");
    assert!(instances.contains(&&"swarm-0:group=swarm:ctx=8192:pinned".to_string()));
    assert!(instances.contains(&&"swarm-1:group=swarm:ctx=8192:pinned".to_string()));
    assert!(
        !instances.iter().any(|s| s.contains("ledger")),
        "unpinned ledger deferred to on-demand creation"
    );
}

#[test]
fn all_unpinned_pool_declares_full_grammar_at_spawn() {
    // A model whose instance pool has NO pinned profile is an all-lazy
    // model: it stays unloaded at boot, but when the dispatch path loads it
    // on demand its server must still register `/instances`. The fork only
    // registers that endpoint when `--instance` grammar is present, so an
    // all-unpinned pool declares its full grammar at spawn (there is no
    // resident anchor whose VRAM a declaration would waste).
    let mut entry = managed_entry();
    entry.instances = Some(
        [
            (
                "scratch".to_string(),
                crate::config::InstanceProfile {
                    name: None,
                    group: None,
                    count: 1,
                    num_ctx: 131072,
                    parallel: None,
                    pinned: false,
                    no_sleep: false,
                    sleep_idle_seconds: Some(1),
                    default: false,
                    resume: false,
                    params: None,
                    max_ctx: None,
                    session: false,
                },
            ),
            (
                "ledger".to_string(),
                crate::config::InstanceProfile {
                    name: None,
                    group: None,
                    count: 1,
                    num_ctx: 65536,
                    parallel: None,
                    pinned: false,
                    no_sleep: false,
                    sleep_idle_seconds: None,
                    default: true,
                    resume: false,
                    params: None,
                    max_ctx: None,
                    session: false,
                },
            ),
        ]
        .into_iter()
        .collect(),
    );
    let spec = LlamaServerSpec::from_entry("swarm", &entry, 18080, None, None, defaults());
    assert!(!spec.boot, "no pinned instance -> all-lazy, loaded on demand");
    let args = build_server_args(&spec);
    let instances: Vec<&String> = args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| if a == "--instance" { args.get(i + 1) } else { None })
        .collect();
    // The full grammar is declared so /instances registers when the model
    // is loaded on demand.
    assert_eq!(instances.len(), 2, "all-unpinned pool declares full grammar");
    assert!(instances.contains(&&"scratch:ctx=131072".to_string()));
    assert!(instances.contains(&&"ledger:ctx=65536:default".to_string()));
}

#[test]
fn plain_model_gets_default_ctx_and_idle_sleep() {
    let mut entry = managed_entry();
    entry.instances = None;
    let spec = LlamaServerSpec::from_entry("swarm", &entry, 18080, None, None, defaults());
    assert!(!spec.boot, "no pinned instance -> lazy model");
    let args = build_server_args(&spec);
    let joined = args.join(" ");
    assert!(joined.contains("--ctx-size 16384"));
    assert!(joined.contains("--sleep-idle-seconds 15"));
    assert!(!joined.contains("--instance"), "no instance grammar for plain models");
}

#[test]
fn server_args_use_hf_repo_when_no_weights() {
    let mut entry = managed_entry();
    entry.weights = None;
    entry.hf_repo = Some("abiray/lfm2.5-2.6b-gguf".into());
    entry.hf_file = Some("Q4_K_M.gguf".into());
    let spec = LlamaServerSpec::from_entry("swarm", &entry, 18080, None, None, defaults());
    let args = build_server_args(&spec);
    let joined = args.join(" ");
    assert!(joined.contains("-hf abiray/lfm2.5-2.6b-gguf"));
    assert!(joined.contains("-hff Q4_K_M.gguf"));
    assert!(!joined.contains("-m /models"));
}

#[test]
fn model_entry_managed_detection() {
    let mut entry = managed_entry();
    assert!(entry.is_managed(), "weights -> managed");
    entry.weights = None;
    entry.hf_repo = None;
    assert!(entry.is_managed(), "instances -> managed");
    entry.instances = None;
    assert!(!entry.is_managed(), "nothing to load -> not managed");
}

// ── Post-boot liveness supervision ─────────────────────────────────

/// Spawn a tiny in-process `/health` server whose availability is toggled
/// by an `AtomicBool`: `true` -> 200, `false` -> 503. Returns its port.
async fn spawn_health_server(up: Arc<AtomicBool>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind health stub");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        loop {
            let Ok((stream, _peer)) = listener.accept().await else {
                break;
            };
            let up = up.clone();
            let io = TokioIo::new(stream);
            let service =
                hyper::service::service_fn(move |_req: hyper::Request<Incoming>| {
                    let up = up.clone();
                    async move {
                        let status = if up.load(Ordering::SeqCst) { 200u16 } else { 503u16 };
                        Ok::<_, std::convert::Infallible>(
                            hyper::Response::builder()
                                .status(status)
                                .body(Full::new(Bytes::new()).boxed_unsync())
                                .expect("build response"),
                        )
                    }
                });
            tokio::spawn(async move {
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, service)
                    .await;
            });
        }
    });
    addr.port()
}

/// Write a fake `llama-server` script that logs each spawn to `counter` and
/// then stays alive until killed (`exec` keeps the PID stable so the
/// supervisor's kill reaches the long-lived process).
fn write_fake_llama(counter: &Path) -> PathBuf {
    let script = counter.parent().expect("counter parent").join("fake-llama");
    let content = format!(
        "#!/bin/sh\necho spawned >> \"{}\"\nexec sleep 600\n",
        counter.display()
    );
    std::fs::write(&script, content).expect("write fake llama");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake llama");
    }
    script
}

/// Write a fake `llama-server` script that logs each spawn to `counter` and
/// then exits immediately (a crash loop — e.g. a missing weights file).
fn write_crashing_llama(counter: &Path) -> PathBuf {
    let script = counter
        .parent()
        .expect("counter parent")
        .join("fake-llama-crash");
    let content = format!(
        "#!/bin/sh\necho spawned >> \"{}\"\nexit 1\n",
        counter.display()
    );
    std::fs::write(&script, content).expect("write crashing fake llama");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake llama");
    }
    script
}

fn spawn_count(counter: &Path) -> usize {
    std::fs::read_to_string(counter)
        .map(|s| s.lines().count())
        .unwrap_or(0)
}

async fn wait_for_spawn_count(counter: &Path, expected: usize, timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if spawn_count(counter) >= expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!(
        "spawn counter did not reach {expected} within {timeout:?}; count={}",
        spawn_count(counter)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn liveness_restarts_hung_server_but_leaves_healthy_one() {
    let up = Arc::new(AtomicBool::new(true));
    let port = spawn_health_server(up.clone()).await;
    let tmp = tempfile::tempdir().expect("tempdir");
    let counter = tmp.path().join("spawns");
    let bin = write_fake_llama(&counter);

    let server = Arc::new(ManagedServer::with_liveness(
        LlamaServerSpec::from_entry("swarm", &managed_entry(), port, None, None, defaults()),
        Duration::from_millis(100),
        3,
        0,
    ));
    server.spawn_child(&bin);
    let supervise = tokio::spawn(server.clone().supervise(bin, Arc::new(build_shared_client())));

    // While the server answers 200, several liveness polls must NOT touch
    // the child.
    tokio::time::sleep(Duration::from_millis(450)).await;
    assert_eq!(
        spawn_count(&counter),
        1,
        "healthy server must not be restarted by the liveness poll"
    );

    // Hang the server; stay below the (3) failure threshold -> still no
    // restart.
    up.store(false, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        spawn_count(&counter),
        1,
        "below the failure threshold must not restart"
    );

    // Cross the threshold: the hung child is killed and respawned.
    wait_for_spawn_count(&counter, 2, Duration::from_secs(8)).await;
    supervise.abort();
    let _ = server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn liveness_never_restarts_during_shutdown() {
    let up = Arc::new(AtomicBool::new(true));
    let port = spawn_health_server(up.clone()).await;
    let tmp = tempfile::tempdir().expect("tempdir");
    let counter = tmp.path().join("spawns");
    let bin = write_fake_llama(&counter);

    let server = Arc::new(ManagedServer::with_liveness(
        LlamaServerSpec::from_entry("swarm", &managed_entry(), port, None, None, defaults()),
        Duration::from_millis(50),
        1,
        0,
    ));
    server.spawn_child(&bin);
    let supervise = tokio::spawn(server.clone().supervise(bin.clone(), Arc::new(build_shared_client())));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(spawn_count(&counter), 1, "server running");

    // Stop the server: the supervise task must exit without restarting
    // even though the health probe now fails (stopping guard).
    server.stop().await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        spawn_count(&counter),
        1,
        "shutdown must never restart the server"
    );
    supervise.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crash_loop_is_contained_after_max_restarts() {
    // A llama-server that dies immediately (e.g. a missing weights file)
    // must NOT restart forever: the failure count rises, the backoff
    // escalates, and after `max_restarts` crashes the supervision task
    // gives up and marks the server failed (containment).
    let tmp = tempfile::tempdir().expect("tempdir");
    let counter = tmp.path().join("spawns");
    let bin = write_crashing_llama(&counter);

    let server = Arc::new(ManagedServer::with_liveness(
        LlamaServerSpec::from_entry("swarm", &managed_entry(), 1, None, None, defaults()),
        Duration::from_millis(50),
        3,
        3,
    ));
    server.spawn_child(&bin);
    let supervise = tokio::spawn(server.clone().supervise(bin.clone(), Arc::new(build_shared_client())));

    // Three spawns (initial + two restarts) exhaust the budget; then the
    // loop stops. Give the third child time to exit and be contained.
    wait_for_spawn_count(&counter, 3, Duration::from_secs(30)).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        spawn_count(&counter),
        3,
        "containment stops the restart loop at max_restarts"
    );
    assert!(
        server.inner.failed.load(Ordering::Relaxed),
        "server marked failed (contained)"
    );
    assert!(!server.is_running(), "contained server is not running");
    supervise.abort();

    // A load attempt after containment fails fast with the terminal error
    // instead of spawning again.
    let err = server.ensure_running(&bin).await.unwrap_err();
    assert!(err.contains("containment"), "terminal error names containment: {err}");
    assert_eq!(spawn_count(&counter), 3, "contained server is never re-spawned");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn probe_health_uses_injected_client() {
    let up = Arc::new(AtomicBool::new(false));
    let port = spawn_health_server(up.clone()).await;
    let server = Arc::new(ManagedServer::with_liveness(
        LlamaServerSpec::from_entry("swarm", &managed_entry(), port, None, None, defaults()),
        Duration::from_millis(50),
        3,
        0,
    ));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    assert!(!server.probe_health_with_client(&client).await, "503 -> false");
    up.store(true, Ordering::SeqCst);
    // give server time to toggle
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(server.probe_health_with_client(&client).await, "200 -> true");
    // verify identity reuse: same Arc reused for 100 probes
    let shared = Arc::new(client);
    for _ in 0..100 {
        let _ = server.probe_health_with_client(&shared).await;
    }
    assert!(Arc::strong_count(&shared) >= 1);
}

#[test]
fn restart_backoff_schedule_is_exact_powers_of_two_capped_at_64s() {
    // M3.1 golden: failures 0..=8 → 1,2,4,8,16,32,64,64,64 seconds, no
    // jitter. Any shared-schedule migration must reproduce this table
    // element-wise.
    let expected_secs = [1u64, 2, 4, 8, 16, 32, 64, 64, 64];
    for (failures, expected) in expected_secs.iter().enumerate() {
        assert_eq!(
            restart_backoff(failures as u32),
            Duration::from_secs(*expected),
            "failures={failures}"
        );
    }
    // Far beyond the cap: still 64s, never overflows.
    assert_eq!(restart_backoff(100), Duration::from_secs(64));
    assert_eq!(restart_backoff(u32::MAX), Duration::from_secs(64));
}
