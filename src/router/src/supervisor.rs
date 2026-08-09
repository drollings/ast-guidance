//! Managed `llama-server` process supervision.
//!
//! Coral Router is the process owner of one `llama-server` per model weights
//! file (the llama.cpp router mode is never used). The [`LlamaServerSupervisor`]
//! finds `llama-server` on `$PATH`, spawns one process per managed model on a
//! free localhost port, waits for `/health`, and supervises each child
//! (logging its output, restarting it with backoff if it dies unexpectedly).
//!
//! The spawned servers bind to `127.0.0.1` only and are never exposed directly;
//! every generation and management call goes through Coral Router.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

use crate::config::{InstanceProfile, RouterConfig};
use crate::instances::instance_grammar_string;

/// The `llama-server` binary name resolved from `$PATH`.
pub const LLAMA_SERVER_BIN: &str = "llama-server";

/// Env var that overrides the resolved `llama-server` binary path.
pub const LLAMA_SERVER_ENV: &str = "LLAMA_SERVER";

/// How long a freshly-spawned server has to answer `/health`.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(120);

/// Poll interval while waiting for `/health`.
const HEALTH_POLL: Duration = Duration::from_secs(1);

/// Resolve the `llama-server` binary path: the `LLAMA_SERVER` env override
/// first, then a `$PATH` search. `None` when not found.
pub fn resolve_llama_server() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os(LLAMA_SERVER_ENV) {
        return Some(PathBuf::from(explicit));
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(LLAMA_SERVER_BIN);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Reserve a free localhost port (bind `:0`, read the port, drop the listener).
/// The returned port is a hint - a race is possible but practically negligible
/// at boot time.
pub fn free_port() -> Option<u16> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
}

/// Everything needed to spawn and supervise one model's `llama-server`.
#[derive(Debug, Clone)]
pub struct LlamaServerSpec {
    /// Coral Router model id (public).
    pub model_key: String,
    /// llama.cpp model name (`--alias`; also the server's primary model id).
    pub name: String,
    /// Local GGUF path (`-m`). `None` uses `hf_repo`.
    pub weights: Option<String>,
    /// HuggingFace repo (`-hf`).
    pub hf_repo: Option<String>,
    /// HuggingFace file within `hf_repo` (`-hff`), optional.
    pub hf_file: Option<String>,
    /// Localhost port the server binds.
    pub port: u16,
    /// The declared instance pool (`--instance` grammar flags).
    pub instances: Vec<InstanceProfile>,
    /// `--slot-save-path` for KV snapshots.
    pub slot_save_path: Option<String>,
    /// `--api-key` for the server's management + generation endpoints.
    pub api_key: Option<String>,
    /// `--instance-wait` (group wait seconds); `None` keeps the server default.
    pub instance_wait_s: Option<i64>,
    /// Additional raw args passed through verbatim.
    pub extra_args: Vec<String>,
}

impl LlamaServerSpec {
    /// Build the spec for a managed model entry on `port`.
    pub fn from_entry(
        model_key: &str,
        entry: &crate::config::ModelEntry,
        port: u16,
        slot_save_path: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            model_key: model_key.to_string(),
            name: entry.llama_model_name(model_key),
            weights: entry.weights.clone(),
            hf_repo: entry.hf_repo.clone(),
            hf_file: entry.hf_file.clone(),
            port,
            instances: entry.instance_profiles(),
            slot_save_path,
            api_key,
            instance_wait_s: None,
            extra_args: Vec::new(),
        }
    }
}

/// Render the exact argv for a spawned server (unit-testable, no side effects).
///
/// A model with a `weights` path loads it via `-m`; an `hf_repo` loads
/// on-demand via `-hf`/`-hff`. The instance pool is declared with one
/// `--instance` flag per expanded profile (the comma-joined grammar would also
/// be accepted). `--slot-save-path` and `--api-key` enable snapshots and auth.
pub fn build_server_args(spec: &LlamaServerSpec) -> Vec<String> {
    let mut args = vec![
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        spec.port.to_string(),
    ];
    if !spec.name.is_empty() {
        args.push("--alias".into());
        args.push(spec.name.clone());
    }
    if let Some(weights) = &spec.weights {
        args.push("-m".into());
        args.push(weights.clone());
    }
    if let Some(repo) = &spec.hf_repo {
        args.push("-hf".into());
        args.push(repo.clone());
    }
    if let Some(file) = &spec.hf_file {
        args.push("-hff".into());
        args.push(file.clone());
    }
    for profile in &spec.instances {
        args.push("--instance".into());
        args.push(instance_grammar_string(std::slice::from_ref(profile)));
    }
    if let Some(path) = &spec.slot_save_path {
        args.push("--slot-save-path".into());
        args.push(path.clone());
    }
    if let Some(key) = &spec.api_key {
        args.push("--api-key".into());
        args.push(key.clone());
    }
    if let Some(wait) = spec.instance_wait_s {
        args.push("--instance-wait".into());
        args.push(wait.to_string());
    }
    args.extend(spec.extra_args.iter().cloned());
    args
}

/// Backoff between restart attempts: 1s, 2s, 4s, ... capped at 64s.
fn restart_backoff(consecutive_failures: u32) -> Duration {
    Duration::from_secs(1u64 << consecutive_failures.min(6))
}

/// One supervised server: the spawn spec, its localhost address, and the shared
/// process state guarded for the supervision task.
pub struct ManagedServer {
    spec: LlamaServerSpec,
    base_url: String,
    inner: Arc<ServerInner>,
}

struct ServerInner {
    /// The live child while one is running (moved into the supervision task
    /// while it awaits `wait`).
    child: Mutex<Option<Child>>,
    /// Set by `stop()` so the supervision task never restarts.
    stopping: AtomicBool,
    /// Abort handle for the supervision task.
    supervisor: Mutex<Option<tokio::task::AbortHandle>>,
    /// Consecutive spawn failures (drives restart backoff).
    spawn_failures: AtomicU32,
}

impl ManagedServer {
    fn new(spec: LlamaServerSpec) -> Self {
        let base_url = format!("http://127.0.0.1:{}", spec.port);
        Self {
            spec,
            base_url,
            inner: Arc::new(ServerInner {
                child: Mutex::new(None),
                stopping: AtomicBool::new(false),
                supervisor: Mutex::new(None),
                spawn_failures: AtomicU32::new(0),
            }),
        }
    }

    pub fn spec(&self) -> &LlamaServerSpec {
        &self.spec
    }

    /// The server's management base URL (`http://127.0.0.1:<port>`).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Whether the process was intentionally stopped.
    pub fn stopping(&self) -> bool {
        self.inner.stopping.load(Ordering::Relaxed)
    }

    /// Spawn the child and wait for it to answer `/health`. Called once at
    /// boot; the supervision task handles later exits and restarts.
    pub async fn start(self: &Arc<Self>, bin: &Path) -> Result<(), String> {
        self.spawn_child(bin);
        self.wait_healthy().await
    }

    /// Wait for `/health` to answer 2xx, up to [`HEALTH_TIMEOUT`].
    pub async fn wait_healthy(&self) -> Result<(), String> {
        let client = reqwest::Client::new();
        let deadline = Instant::now() + HEALTH_TIMEOUT;
        loop {
            match client
                .get(format!("{}/health", self.base_url))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!(
                        target: "router.supervisor",
                        model = %self.spec.model_key,
                        base_url = %self.base_url,
                        "llama-server healthy",
                    );
                    return Ok(());
                }
                _ => {}
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "llama-server for model '{}' did not become healthy on {} within {}s",
                    self.spec.model_key,
                    self.base_url,
                    HEALTH_TIMEOUT.as_secs(),
                ));
            }
            tokio::time::sleep(HEALTH_POLL).await;
        }
    }

    /// Spawn the child process (no health wait). Failure is logged and counted
    /// so the supervision loop backs off and retries.
    fn spawn_child(self: &Arc<Self>, bin: &Path) {
        if self.inner.stopping.load(Ordering::Relaxed) {
            return;
        }
        let args = build_server_args(&self.spec);
        match Command::new(bin)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
        {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                {
                    let mut guard = match self.inner.child.lock() {
                        Ok(g) => g,
                        Err(e) => e.into_inner(),
                    };
                    *guard = Some(child);
                }
                self.inner.spawn_failures.store(0, Ordering::Relaxed);
                // Log the server's output (best-effort, detached).
                if let Some(out) = stdout {
                    let tag = self.spec.model_key.clone();
                    tokio::spawn(async move {
                        let mut reader = BufReader::new(out);
                        let mut line = String::new();
                        while reader.read_line(&mut line).await.is_ok() && !line.is_empty() {
                            tracing::debug!(
                                target: "router.supervisor.child",
                                model = %tag,
                                line = line.trim_end(),
                                "server stdout",
                            );
                            line.clear();
                        }
                    });
                }
                if let Some(err) = stderr {
                    let tag = self.spec.model_key.clone();
                    tokio::spawn(async move {
                        let mut reader = BufReader::new(err);
                        let mut line = String::new();
                        while reader.read_line(&mut line).await.is_ok() && !line.is_empty() {
                            tracing::info!(
                                target: "router.supervisor.child",
                                model = %tag,
                                line = line.trim_end(),
                                "server stderr",
                            );
                            line.clear();
                        }
                    });
                }
                tracing::info!(
                    target: "router.supervisor",
                    model = %self.spec.model_key,
                    base_url = %self.base_url,
                    args = ?args,
                    "llama-server spawned",
                );
            }
            Err(e) => {
                let failures = self.inner.spawn_failures.fetch_add(1, Ordering::Relaxed) + 1;
                tracing::error!(
                    target: "router.supervisor",
                    model = %self.spec.model_key,
                    error = %e,
                    failures = failures,
                    "llama-server spawn failed",
                );
            }
        }
    }

    /// The supervision loop: wait for the child to exit, and unless `stop()`
    /// was called, restart it with backoff. Runs as a spawned task for the
    /// life of the server.
    async fn supervise(self: Arc<Self>, bin: PathBuf) {
        loop {
            let child = {
                let mut guard = match self.inner.child.lock() {
                    Ok(g) => g,
                    Err(e) => e.into_inner(),
                };
                guard.take()
            };
            let Some(mut child) = child else {
                // No child to watch (spawn failed); retry with backoff.
                if self.inner.stopping.load(Ordering::Relaxed) {
                    return;
                }
                let failures = self.inner.spawn_failures.load(Ordering::Relaxed);
                tokio::time::sleep(restart_backoff(failures.max(1))).await;
                self.spawn_child(&bin);
                continue;
            };
            let status = child.wait().await;
            if self.inner.stopping.load(Ordering::Relaxed) {
                tracing::info!(
                    target: "router.supervisor",
                    model = %self.spec.model_key,
                    "llama-server stopped",
                );
                return;
            }
            let failures = self.inner.spawn_failures.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::error!(
                target: "router.supervisor",
                model = %self.spec.model_key,
                status = ?status,
                failures = failures,
                "llama-server exited unexpectedly - restarting with backoff",
            );
            tokio::time::sleep(restart_backoff(failures)).await;
            self.spawn_child(&bin);
        }
    }

    /// Stop the server: mark stopped, kill the child (the supervision task
    /// sees `stopping` and exits without restarting).
    pub async fn stop(self: &Arc<Self>) {
        self.inner.stopping.store(true, Ordering::Relaxed);
        if let Some(handle) = self.inner.supervisor.lock().ok().and_then(|mut g| g.take()) {
            handle.abort();
        }
        let child = {
            let mut guard = match self.inner.child.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            guard.take()
        };
        if let Some(mut child) = child {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        tracing::info!(
            target: "router.supervisor",
            model = %self.spec.model_key,
            "llama-server stopped by router",
        );
    }
}

/// The supervisor: one [`ManagedServer`] per managed model.
pub struct LlamaServerSupervisor {
    bin: PathBuf,
    servers: HashMap<String, Arc<ManagedServer>>,
}

impl LlamaServerSupervisor {
    /// Resolve the binary and build one server per managed model (`weights` /
    /// `hf_repo` / `instances` declared), each on its own free localhost port.
    /// Fails fast when `llama-server` is not found on `$PATH`.
    pub fn build(config: &RouterConfig) -> Result<Self, String> {
        let bin = resolve_llama_server().ok_or_else(|| {
            format!(
                "llama-server binary not found in $PATH (set the {LLAMA_SERVER_ENV} env var or install it)"
            )
        })?;
        let api_key = config
            .sidecar
            .api_key_env
            .as_deref()
            .map(std::env::var)
            .and_then(Result::ok)
            .filter(|k| !k.is_empty());
        let mut servers = HashMap::new();
        let mut keys: Vec<&String> = config.models.keys().collect();
        keys.sort();
        for key in keys {
            let entry = &config.models[key];
            if !entry.is_managed() {
                continue;
            }
            let port = free_port().ok_or_else(|| "no free localhost port".to_string())?;
            let spec = LlamaServerSpec::from_entry(
                key,
                entry,
                port,
                config.sidecar.slot_save_path.clone(),
                api_key.clone(),
            );
            let server = Arc::new(ManagedServer::new(spec));
            servers.insert(key.clone(), server);
        }
        tracing::info!(
            target: "router.supervisor",
            binary = %bin.display(),
            server_count = servers.len(),
            "llama-server supervisor built",
        );
        Ok(Self { bin, servers })
    }

    /// The resolved binary path.
    pub fn bin(&self) -> &Path {
        &self.bin
    }

    /// The server for a model id.
    pub fn server_for(&self, model_key: &str) -> Option<&Arc<ManagedServer>> {
        self.servers.get(model_key)
    }

    /// The management base URL for a model id.
    pub fn base_url_for(&self, model_key: &str) -> Option<&str> {
        self.server_for(model_key).map(|s| s.base_url())
    }

    /// Every managed model key.
    pub fn model_keys(&self) -> impl Iterator<Item = &String> {
        self.servers.keys()
    }

    /// Spawn every managed server and wait for each to become healthy.
    /// Spawns happen first (processes load weights concurrently), then health
    /// is awaited in parallel, so boot time is bounded by the slowest model
    /// load rather than the sum. Returns the first health failure so boot
    /// aborts loudly (a server that cannot load its weights must not be
    /// silently skipped).
    pub async fn start_all(self: &Arc<Self>) -> Result<(), String> {
        let bin = self.bin.clone();
        let mut keys: Vec<String> = self.servers.keys().cloned().collect();
        keys.sort();
        for key in &keys {
            let server = &self.servers[key];
            // `--slot-save-path` must name an existing directory: ensure it
            // exists so snapshot support works out of the box.
            if let Some(dir) = &server.spec.slot_save_path {
                if let Err(e) = tokio::fs::create_dir_all(dir).await {
                    tracing::warn!(
                        target: "router.supervisor",
                        model = %key,
                        slot_save_path = %dir,
                        error = %e,
                        "slot-save-path create failed (snapshots disabled)",
                    );
                }
            }
            server.spawn_child(&bin);
            let handle = {
                let me = Arc::clone(server);
                let bin = bin.clone();
                tokio::spawn(async move { me.supervise(bin).await; })
            };
            if let Ok(mut guard) = server.inner.supervisor.lock() {
                *guard = Some(handle.abort_handle());
            }
        }
        // Await every server's /health concurrently; the first failure aborts.
        let mut health = Vec::new();
        for key in &keys {
            let server = Arc::clone(&self.servers[key]);
            health.push(async move { (key.clone(), server.wait_healthy().await) });
        }
        for fut in health {
            let (key, result) = fut.await;
            if let Err(e) = result {
                return Err(format!("model '{key}': {e}"));
            }
        }
        Ok(())
    }

    /// Stop every managed server (used on shutdown).
    pub async fn shutdown(&self) {
        let mut keys: Vec<String> = self.servers.keys().cloned().collect();
        keys.sort();
        for key in &keys {
            if let Some(server) = self.servers.get(key) {
                server.stop().await;
            }
        }
    }
}

impl Drop for LlamaServerSupervisor {
    fn drop(&mut self) {
        // Best-effort process cleanup when the supervisor is dropped outside an
        // async context (each child has kill_on_drop(true), so dropping the
        // supervision task's child handle kills the process).
        for server in self.servers.values() {
            server.inner.stopping.store(true, Ordering::Relaxed);
            if let Some(handle) = server
                .inner
                .supervisor
                .lock()
                .ok()
                .and_then(|mut g| g.take())
            {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelEntry;

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
    fn server_args_emit_instance_grammar_per_profile() {
        let spec = LlamaServerSpec::from_entry(
            "swarm",
            &managed_entry(),
            18080,
            None,
            None,
        );
        let args = build_server_args(&spec);
        let instances: Vec<&String> = args
            .iter()
            .enumerate()
            .filter_map(|(i, a)| if a == "--instance" { args.get(i + 1) } else { None })
            .collect();
        assert_eq!(instances.len(), 3, "count:2 swarm + ledger = 3 instances");
        assert!(instances.contains(&&"swarm-0:group=swarm:ctx=8192:pinned".to_string()));
        assert!(instances.contains(&&"ledger:ctx=65536:default".to_string()));
    }

    #[test]
    fn server_args_use_hf_repo_when_no_weights() {
        let mut entry = managed_entry();
        entry.weights = None;
        entry.hf_repo = Some("abiray/lfm2.5-2.6b-gguf".into());
        entry.hf_file = Some("Q4_K_M.gguf".into());
        let spec = LlamaServerSpec::from_entry("swarm", &entry, 18080, None, None);
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
}
