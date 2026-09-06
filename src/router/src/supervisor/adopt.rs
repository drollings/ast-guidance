//! Orphan adoption: rediscover live `llama-server` processes across router restarts.
//!
//! Coral Router assigns each managed model a fresh free localhost port at every
//! boot (`LlamaServerSupervisor::build`) and rewrites the model's endpoint to
//! it. When the router process dies without a graceful shutdown (SIGKILL,
//! OOM-kill, crash) its spawned servers survive — still holding their VRAM and
//! their instance pools — but the next boot picks new ports and can never find
//! them. The result is duplicated VRAM pressure plus orphaned KV state.
//!
//! This module closes that gap with an adopt-before-spawn pass
//! (`LlamaServerSupervisor::adopt_orphans`, run after `build`, before
//! `start_all`):
//!
//! 1. **Discover** candidates: the persisted state file
//!    (`sidecar.server_state_path`, fast path) plus a `/proc` scan for
//!    `llama-server` cmdlines (Linux; fallback when the state file is absent).
//! 2. **Verify** each candidate over HTTP: `/health` must answer, and `/props`
//!    must report the expected `--alias` and weights path. Loopback binds only
//!    — a stock router-mode server on `0.0.0.0` (or any foreign process that
//!    happened to take the port) is never adopted.
//! 3. **Converge, never wipe**: the adopted server's entry is rebuilt with the
//!    discovered port and the normal boot reconcile runs against it. Pinned
//!    instances missing from the live pool are created; `n_ctx` drift is
//!    resized; everything else — including on-demand instances the previous
//!    router lifetime created — is left untouched. No live instance that
//!    matches the configuration is ever destroyed by adoption.
//!
//! A grammar-less orphan (no `--instance` flags, so no `/instances` route)
//! adopted for a model that declares an instance pool is kept for generation
//! traffic with a loud drift warning; its manager skips instance reconcile
//! until an explicit unload lets the next load respawn it with the correct
//! grammar (see `InstanceManager`'s grammar marking).

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::RouterConfig;

use super::{LlamaServerSpec, LlamaServerSupervisor, ManagedServer};

/// How long an identity probe waits per HTTP call.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// One live `llama-server` process found by discovery (state file or `/proc`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredServer {
    /// OS pid of the process.
    pub pid: u32,
    /// `--port` from its cmdline (0 when absent/unparseable).
    pub port: u16,
    /// `--alias` from its cmdline (the llama.cpp model name).
    pub alias: String,
    /// `-m` weights path from its cmdline, when present.
    pub weights: Option<String>,
    /// `-hf` repo from its cmdline, when present.
    pub hf_repo: Option<String>,
}

/// The verified identity of a reachable server: what `/props` reports plus
/// whether the `/instances` management route exists on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerIdentity {
    /// `model_alias` from `/props` (empty when the field is absent).
    pub alias: String,
    /// `model_path` from `/props` (empty when the field is absent).
    pub model_path: String,
    /// Whether `GET /instances` answers 2xx (instance grammar) as opposed to
    /// 404 (the server was spawned without `--instance` flags).
    pub instances_supported: bool,
}

/// What adoption recorded for one model: the non-child pid owned by adoption
/// and whether its management API speaks `/instances`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdoptionInfo {
    /// OS pid of the adopted process (never a child of this router).
    pub pid: u32,
    /// `GET /instances` verdict at adopt time.
    pub instances_supported: bool,
}

/// The per-boot adoption outcome, logged by the caller.
#[derive(Debug, Default)]
pub struct AdoptReport {
    /// `(model_key, base_url, instances_supported)` for every adopted model.
    pub adopted: Vec<(String, String, bool)>,
}

/// Parse a `llama-server` argv (without argv[0]) into its serving identity.
/// Returns `None` when no `--alias` is present (nothing to match a managed
/// model against) or the bind is not loopback. Both `--flag value` and
/// `--flag=value` spellings are accepted.
pub fn parse_llama_argv(argv: &[String]) -> Option<DiscoveredServer> {
    fn next_value(argv: &[String], i: usize, flag: &str) -> Option<(String, usize)> {
        let arg = argv.get(i)?;
        if let Some(v) = arg.strip_prefix(&format!("{flag}=")) {
            return Some((v.to_string(), i));
        }
        if *arg == flag {
            return Some((argv.get(i + 1)?.clone(), i + 1));
        }
        None
    }

    let mut port: u16 = 0;
    let mut host: Option<String> = None;
    let mut alias: Option<String> = None;
    let mut weights: Option<String> = None;
    let mut hf_repo: Option<String> = None;
    let mut i = 0;
    while i < argv.len() {
        if let Some((v, j)) = next_value(argv, i, "--port") {
            port = v.parse().unwrap_or(0);
            i = j + 1;
            continue;
        }
        if let Some((v, j)) = next_value(argv, i, "--host") {
            host = Some(v);
            i = j + 1;
            continue;
        }
        if let Some((v, j)) = next_value(argv, i, "--alias") {
            alias = Some(v);
            i = j + 1;
            continue;
        }
        if let Some((v, j)) = next_value(argv, i, "-m") {
            weights = Some(v);
            i = j + 1;
            continue;
        }
        if let Some((v, j)) = next_value(argv, i, "-hf") {
            hf_repo = Some(v);
            i = j + 1;
            continue;
        }
        i += 1;
    }
    let alias = alias.filter(|a| !a.is_empty())?;
    // Only loopback binds are adoptable: a stock router-mode server on
    // 0.0.0.0 (or any non-loopback bind) is never ours. llama-server
    // defaults to 127.0.0.1 when --host is absent.
    if !matches!(
        host.as_deref(),
        None | Some("127.0.0.1" | "localhost" | "::1")
    ) {
        return None;
    }
    Some(DiscoveredServer {
        pid: 0,
        port,
        alias,
        weights,
        hf_repo,
    })
}

/// Read one `/proc/<pid>/cmdline` (NUL-separated argv). `None` when the
/// process is gone or unreadable.
fn read_proc_cmdline(pid: u32) -> Option<Vec<String>> {
    let path = format!("/proc/{pid}/cmdline");
    let text = fluent_wvr::capability::capability_aware_fs::read_to_string(path).ok()?;
    let args: Vec<String> = text
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if args.is_empty() {
        return None;
    }
    Some(args)
}

/// Whether `pid` is still our `llama-server` (guards against pid reuse before
/// signalling an adopted process). When `expect_port` is nonzero the cmdline
/// `--port` must also agree.
pub fn pid_still_ours(pid: u32, expect_alias: &str, expect_port: u16) -> bool {
    let Some(args) = read_proc_cmdline(pid) else {
        return false;
    };
    let Some(exe) = args.first().map(|a| {
        Path::new(a)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string()
    }) else {
        return false;
    };
    if exe != "llama-server" {
        return false;
    }
    let Some(found) = parse_llama_argv(&args[1..]) else {
        return false;
    };
    if found.alias != expect_alias {
        return false;
    }
    if expect_port != 0 && found.port != 0 && found.port != expect_port {
        return false;
    }
    true
}

/// Scan `/proc` for loopback-bound `llama-server` processes with an `--alias`.
/// Capability-gated (`FsCapability`): the caller must run inside a capability
/// scope (boot does). Non-Linux hosts return an empty set.
#[cfg(not(target_os = "linux"))]
pub fn scan_proc() -> Vec<DiscoveredServer> {
    Vec::new()
}

/// Scan `/proc` for loopback-bound `llama-server` processes with an `--alias`.
/// Capability-gated (`FsCapability`): the caller must run inside a capability
/// scope (boot does).
#[cfg(target_os = "linux")]
pub fn scan_proc() -> Vec<DiscoveredServer> {
    let mut out = Vec::new();
    let Ok(entries) = fluent_wvr::capability::capability_aware_fs::read_dir("/proc") else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid_str) = name.to_str() else {
            continue;
        };
        let Ok(pid) = pid_str.parse::<u32>() else {
            continue;
        };
        let Some(args) = read_proc_cmdline(pid) else {
            continue;
        };
        let exe_ok = args.first().is_some_and(|a| {
            Path::new(a)
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == "llama-server")
        });
        if !exe_ok {
            continue;
        }
        if let Some(mut found) = parse_llama_argv(&args[1..]) {
            found.pid = pid;
            out.push(found);
        }
    }
    out.sort_by_key(|d| d.pid);
    out
}

/// One persisted server record: where the router last left a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SavedServer {
    port: u16,
    pid: u32,
}

/// The persisted fleet map (`sidecar.server_state_path`): model key to the
/// port/pid the router last spawned or adopted for it.
#[derive(Debug, Default, Serialize, Deserialize)]
struct ServerStateFile {
    #[serde(default)]
    servers: HashMap<String, SavedServer>,
}

impl ServerStateFile {
    fn load(path: &str) -> Self {
        let Ok(text) = fluent_wvr::capability::capability_aware_fs::read_to_string(path) else {
            return Self::default();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }
}

/// Probe one candidate base URL: `/health` must answer 2xx, then `/props`
/// supplies the alias/path identity and `GET /instances` decides the grammar
/// verdict (2xx = instance pool, 404 = plain/grammar-less server). Any other
/// outcome is `None` (never adopt on a partial read).
pub async fn probe_identity(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
) -> Option<ServerIdentity> {
    let mut health = client.get(format!("{base_url}/health"));
    let mut props = client.get(format!("{base_url}/props"));
    let mut instances = client.get(format!("{base_url}/instances"));
    if let Some(key) = api_key {
        let key = key.to_string();
        health = health.bearer_auth(&key);
        props = props.bearer_auth(key.clone());
        instances = instances.bearer_auth(key);
    }
    let health_ok = health
        .send()
        .await
        .is_ok_and(|r| r.status().is_success());
    if !health_ok {
        return None;
    }
    let props_value: serde_json::Value = props.send().await.ok()?.json().await.ok()?;
    let alias = props_value
        .get("model_alias")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let model_path = props_value
        .get("model_path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let instances_supported = match instances.send().await {
        Ok(r) if r.status().is_success() => true,
        Ok(r) if r.status().as_u16() == 404 => false,
        _ => return None,
    };
    Some(ServerIdentity {
        alias,
        model_path,
        instances_supported,
    })
}

/// Whether a verified candidate is the server for `spec`: the alias must
/// match, and the weights source must agree on at least one channel (the
/// cmdline `-m`/`-hf` or the `/props` model path). Strict on purpose — a
/// same-alias server with different weights is a different deployment.
pub fn server_matches(
    spec: &LlamaServerSpec,
    discovered: &DiscoveredServer,
    identity: &ServerIdentity,
) -> bool {
    if discovered.alias != spec.name || identity.alias != spec.name {
        return false;
    }
    if let Some(want) = spec.weights.as_deref() {
        let cmd_ok = discovered.weights.as_deref() == Some(want);
        let props_ok = !identity.model_path.is_empty() && identity.model_path == want;
        cmd_ok || props_ok
    } else if let Some(want) = spec.hf_repo.as_deref() {
        discovered.hf_repo.as_deref() == Some(want)
    } else {
        // No weights source configured: alias alone cannot identify the
        // deployment; refuse rather than adopt a stranger.
        false
    }
}

impl LlamaServerSupervisor {
    /// Adopt-before-spawn: for every managed model, look for an already-live
    /// `llama-server` that is verifiably ours (state file first, then the
    /// `/proc` scan) and take it over instead of spawning a duplicate.
    ///
    /// An adopted entry keeps its discovered port (so the later endpoint
    /// rewrite and instance clients target the live server), is marked with
    /// its non-child pid, and skips spawning in `start_all`. The normal boot
    /// reconcile then converges it: missing pinned instances are created,
    /// drift is resized/warned, and matching live instances are never
    /// destroyed. Must run before `start_all`.
    pub async fn adopt_orphans(&self, config: &RouterConfig) -> AdoptReport {
        let mut report = AdoptReport::default();
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .timeout(PROBE_TIMEOUT)
            .build()
            .expect("adopt http client build");
        let api_key = config
            .sidecar
            .api_key_env
            .as_deref()
            .map(std::env::var)
            .and_then(Result::ok)
            .filter(|k| !k.is_empty());
        let saved = config
            .sidecar
            .server_state_path
            .as_deref()
            .map(ServerStateFile::load)
            .unwrap_or_default();
        let scanned = scan_proc();

        let mut keys: Vec<String> = self.model_keys();
        keys.sort();
        for key in keys {
            let Some(server) = self.server_for(&key) else {
                continue;
            };
            let spec = server.spec().clone();
            // Candidate ports in priority order: the state file record, then
            // every scanned process whose alias matches (full verification
            // happens per candidate via HTTP + weights comparison).
            let mut candidates: Vec<DiscoveredServer> = Vec::new();
            if let Some(entry) = saved.servers.get(&key) {
                if entry.port != 0
                    && pid_still_ours(entry.pid, &spec.name, entry.port)
                    && !candidates.iter().any(|c| c.port == entry.port)
                {
                    candidates.push(DiscoveredServer {
                        pid: entry.pid,
                        port: entry.port,
                        alias: spec.name.clone(),
                        weights: spec.weights.clone(),
                        hf_repo: spec.hf_repo.clone(),
                    });
                }
            }
            for found in &scanned {
                if found.alias == spec.name
                    && found.port != 0
                    && !candidates.iter().any(|c| c.port == found.port)
                {
                    candidates.push(found.clone());
                }
            }
            for candidate in candidates {
                let base_url = format!("http://127.0.0.1:{}", candidate.port);
                let Some(identity) =
                    probe_identity(&client, &base_url, api_key.as_deref()).await
                else {
                    continue;
                };
                if !server_matches(&spec, &candidate, &identity) {
                    tracing::warn!(
                        target: "router.supervisor",
                        model = %key,
                        base_url = %base_url,
                        alias = %identity.alias,
                        model_path = %identity.model_path,
                        "orphan identity mismatch - leaving process alone",
                    );
                    continue;
                }
                self.adopt_model(&key, &spec, candidate.pid, candidate.port, &identity);
                report
                    .adopted
                    .push((key.clone(), base_url.clone(), identity.instances_supported));
                if !identity.instances_supported && !spec.instances.is_empty() {
                    tracing::warn!(
                        target: "router.supervisor",
                        model = %key,
                        base_url = %base_url,
                        "adopted grammar-less server for an instance-pool model - generation works, instance reconcile is suspended until an explicit unload respawns it with --instance flags",
                    );
                } else {
                    tracing::info!(
                        target: "router.supervisor",
                        model = %key,
                        base_url = %base_url,
                        instances_supported = identity.instances_supported,
                        "adopted live llama-server - spawn skipped, reconcile will converge the pool",
                    );
                }
                break;
            }
        }
        report
    }

    /// Swap one managed entry for an adopted one: same spec on the discovered
    /// port, marked with the non-child pid. Called only pre-`start_all`, when
    /// no supervision task or child exists for the entry.
    fn adopt_model(
        &self,
        key: &str,
        spec: &LlamaServerSpec,
        pid: u32,
        port: u16,
        identity: &ServerIdentity,
    ) {
        let mut adopted_spec = spec.clone();
        adopted_spec.port = port;
        let old = self.server_for(key).expect("managed key from registry");
        let replacement = ManagedServer::with_liveness(
            adopted_spec,
            old.liveness_poll(),
            old.liveness_failures_before_restart(),
            old.max_restarts(),
        );
        replacement.mark_adopted(pid);
        self.servers.insert(key.to_string(), replacement);
        self.adoptions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                key.to_string(),
                AdoptionInfo {
                    pid,
                    instances_supported: identity.instances_supported,
                },
            );
    }

    /// The adoption record for a model, if it was adopted this boot and the
    /// same process still owns the entry. Self-healing: when the orphan died
    /// mid-life and a fresh child took over (`adopted_pid` cleared), the stale
    /// record is dropped and `None` is returned, so callers (instance-manager
    /// construction, residency) see the respawned server for what it is.
    pub fn adoption_info(&self, model_key: &str) -> Option<AdoptionInfo> {
        let info = self
            .adoptions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(model_key)
            .cloned()?;
        let live = self
            .server_for(model_key)
            .and_then(|s| s.adopted_pid())
            == Some(info.pid);
        if live {
            Some(info)
        } else {
            self.clear_adoption(model_key);
            None
        }
    }

    /// Clear one model's adoption record (after its orphan died or was
    /// unloaded and a fresh child takes over).
    pub(super) fn clear_adoption(&self, model_key: &str) {
        self.adoptions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(model_key);
    }

    /// Persist the fleet map (`{model: {port, pid}}` for every running
    /// server, spawned or adopted) to `sidecar.server_state_path`. Best
    /// effort: a missing path config disables persistence, failures are
    /// logged. Runs inside a capability scope (gated fs + gated HTTP-free).
    pub async fn persist_state(&self, config: &RouterConfig) {
        let Some(path) = config.sidecar.server_state_path.as_deref() else {
            return;
        };
        let mut servers = HashMap::new();
        for key in self.model_keys() {
            let Some(server) = self.server_for(&key) else {
                continue;
            };
            if !server.is_running() {
                continue;
            }
            let pid = server
                .adopted_pid()
                .or_else(|| server.child_pid())
                .unwrap_or(0);
            if pid == 0 {
                continue;
            }
            servers.insert(key, SavedServer {
                port: server.spec().port,
                pid,
            });
        }
        let state = ServerStateFile { servers };
        let Ok(text) = serde_json::to_string_pretty(&state) else {
            return;
        };
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fluent_wvr::capability::capability_aware_fs::create_dir_all(parent);
            }
        }
        let cap = fluent_wvr::capability::FsCapability::new();
        if let Err(e) = cap.write(path, text).await {
            tracing::warn!(
                target: "router.supervisor",
                state_path = %path,
                error = %e,
                "server state persist failed - adoption falls back to /proc scan",
            );
        }
    }
}

#[cfg(test)]
#[path = "../../tests/supervisor_adopt.rs"]
mod tests;
