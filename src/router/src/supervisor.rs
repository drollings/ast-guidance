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

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common_core::registry::ConcurrentRegistry;
use common_core::retry::{PollResult, PollWithBackoff};
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
    /// The declared instance pool (`--instance` grammar flags). Only pinned
    /// profiles are declared at spawn; unpinned instances are created on
    /// demand by the sidecar.
    pub instances: Vec<InstanceProfile>,
    /// Whether the model is spawned at boot: true when it declares at least
    /// one pinned instance. Models without a pinned instance are loaded on
    /// demand by the router (see `LlamaServerSupervisor::ensure_running`).
    pub boot: bool,
    /// `--slot-save-path` for KV snapshots.
    pub slot_save_path: Option<String>,
    /// `--api-key` for the server's management + generation endpoints.
    pub api_key: Option<String>,
    /// `--instance-wait` (group wait seconds); `None` keeps the server default.
    pub instance_wait_s: Option<i64>,
    /// `default_params` run defaults: batch sizes, KV cache types, flash
    /// attention, GPU offload, and the plain-model context size.
    pub defaults: crate::config::DefaultModelParams,
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
        defaults: crate::config::DefaultModelParams,
    ) -> Self {
        let instances = entry.instance_profiles();
        let boot = instances.iter().any(|p| p.pinned);
        Self {
            model_key: model_key.to_string(),
            name: entry.llama_model_name(model_key),
            weights: entry.weights.clone(),
            hf_repo: entry.hf_repo.clone(),
            hf_file: entry.hf_file.clone(),
            port,
            instances,
            boot,
            slot_save_path,
            api_key,
            instance_wait_s: None,
            defaults,
            extra_args: Vec::new(),
        }
    }
}

/// Render the exact argv for a spawned server (unit-testable, no side effects).
///
/// A model with a `weights` path loads it via `-m`; an `hf_repo` loads
/// on-demand via `-hf`/`-hff`. Run defaults from `default_params` (batch
/// sizes, KV cache types, flash attention, GPU offload) are always emitted so
/// every managed server runs identically; a plain model (no instance pool)
/// also gets the default context size and idle-sleep timeout. Only **pinned**
/// instance profiles are declared as `--instance` flags at spawn — unpinned
/// instances are created on demand by the sidecar. `--slot-save-path` and
/// `--api-key` enable snapshots and auth.
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
    // default_params run defaults (the "how a model is run" contract).
    args.push("--batch-size".into());
    args.push(spec.defaults.batch_size.to_string());
    args.push("--ubatch-size".into());
    args.push(spec.defaults.ubatch_size.to_string());
    args.push("--cache-type-k".into());
    args.push(spec.defaults.cache_type_k.clone());
    args.push("--cache-type-v".into());
    args.push(spec.defaults.cache_type_v.clone());
    if let Some(mode) = &spec.defaults.flash_attn {
        args.push("--flash-attn".into());
        args.push(mode.clone());
    }
    args.push("--n-gpu-layers".into());
    args.push(spec.defaults.n_gpu_layers.to_string());
    if spec.defaults.n_cpu_moe > 0 {
        args.push("--n-cpu-moe".into());
        args.push(spec.defaults.n_cpu_moe.to_string());
    }
    // Only pinned instances are declared at spawn; unpinned instances are
    // created on demand by the sidecar (see InstanceManager::ensure_instance).
    for profile in &spec.instances {
        if !profile.pinned {
            continue;
        }
        args.push("--instance".into());
        args.push(instance_grammar_string(std::slice::from_ref(profile)));
    }
    // A plain model (no instance pool) takes the default context size and
    // idle-sleep timeout from `default_params`.
    if spec.instances.is_empty() {
        args.push("--ctx-size".into());
        args.push(spec.defaults.num_ctx.to_string());
        args.push("--sleep-idle-seconds".into());
        args.push(spec.defaults.sleep_idle_seconds.to_string());
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
    /// Whether a child is currently expected to be alive (spawned, or being
    /// supervised). Cleared on stop and on unload; re-set by ensure_running.
    running: AtomicBool,
    /// Serializes spawn on the on-demand path so concurrent dispatches cannot
    /// double-spawn a lazy model.
    spawn_lock: tokio::sync::Mutex<()>,
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
                running: AtomicBool::new(false),
                spawn_lock: tokio::sync::Mutex::new(()),
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

    /// Whether a spawned child is currently expected to be alive. `false` for
    /// a lazy model that has never been loaded (or was unloaded for VRAM).
    pub fn is_running(&self) -> bool {
        self.inner.running.load(Ordering::Relaxed)
    }

    /// Spawn the child and wait for it to answer `/health`. Called at boot for
    /// boot models; the supervision task handles later exits and restarts.
    pub async fn start(self: &Arc<Self>, bin: &Path) -> Result<(), String> {
        self.spawn_child(bin);
        self.wait_healthy().await
    }

    /// Bring the server up on demand: spawn (if not already running) and wait
    /// for `/health`. Idempotent and safe under concurrent dispatch — the
    /// spawn lock prevents a lazy model from being double-spawned. Called by
    /// the router when a dispatch targets a managed model that is not loaded.
    pub async fn ensure_running(self: &Arc<Self>, bin: &Path) -> Result<(), String> {
        if self.inner.running.load(Ordering::Relaxed) {
            return Ok(());
        }
        let _guard = self.inner.spawn_lock.lock().await;
        // Re-check under the lock: a concurrent caller may have spawned.
        if self.inner.running.load(Ordering::Relaxed) {
            return Ok(());
        }
        if self.inner.stopping.load(Ordering::Relaxed) {
            return Err(format!("model '{}' is stopped", self.spec.model_key));
        }
        self.spawn_child(bin);
        self.inner.running.store(true, Ordering::Relaxed);
        tracing::info!(
            target: "router.supervisor",
            model = %self.spec.model_key,
            base_url = %self.base_url,
            "llama-server loaded on demand",
        );
        // Start the supervision task if none is running (e.g. after an unload).
        {
            let mut guard = match self.inner.supervisor.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            if guard.is_none() {
                let me = Arc::clone(self);
                let bin = bin.to_path_buf();
                let handle = tokio::spawn(async move { me.supervise(bin).await; });
                *guard = Some(handle.abort_handle());
            }
        }
        self.wait_healthy().await
    }

    /// Unload the server (on-demand teardown): kill the child and stop the
    /// supervision task so it does not restart. The spec stays registered, so
    /// a later [`Self::ensure_running`] re-spawns the model. Used by the
    /// sidecar when a model's weights must be freed for VRAM.
    pub async fn unload(self: &Arc<Self>) {
        self.inner.running.store(false, Ordering::Relaxed);
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
            "llama-server unloaded (on-demand eviction)",
        );
    }

    /// Wait for `/health` to answer 2xx, up to `HEALTH_TIMEOUT`.
    pub async fn wait_healthy(&self) -> Result<(), String> {
        let client = reqwest::Client::new();
        // `cap=1` keeps the poll interval constant at HEALTH_POLL (the poll
        // deadline is a fixed wall-clock, not a growing backoff), and
        // `max_failures = HEALTH_TIMEOUT/HEALTH_POLL` bounds the loop to the
        // deadline, preserving the original hard-stop behavior.
        let poll = PollWithBackoff::new(HEALTH_POLL, 1)
            .with_max_failures(HEALTH_TIMEOUT.as_secs() as u32);
        match poll
            .run(|| async {
                let healthy = client
                    .get(format!("{}/health", self.base_url))
                    .send()
                    .await
                    .is_ok_and(|resp| resp.status().is_success());
                if healthy {
                    tracing::info!(
                        target: "router.supervisor",
                        model = %self.spec.model_key,
                        base_url = %self.base_url,
                        "llama-server healthy",
                    );
                }
                healthy
            })
            .await
        {
            PollResult::Ready => Ok(()),
            PollResult::Exhausted { .. } => Err(format!(
                "llama-server for model '{}' did not become healthy on {} within {}s",
                self.spec.model_key,
                self.base_url,
                HEALTH_TIMEOUT.as_secs(),
            )),
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
        self.inner.running.store(false, Ordering::Relaxed);
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
    servers: ConcurrentRegistry<String, ManagedServer>,
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
        // `--slot-save-path` must name an existing directory: llama-server
        // rejects a nonexistent path at argv-parse time and exits, which would
        // crash-loop every managed server (and with it the whole boot). Resolve
        // the directory once here; if it cannot be created (e.g. an unwritable
        // mount), snapshots are disabled for every managed server — the flag is
        // omitted entirely rather than handed to a process that will die on it.
        let slot_save_path = match &config.sidecar.slot_save_path {
            Some(dir) => match std::fs::create_dir_all(dir) {
                Ok(()) => Some(dir.clone()),
                Err(e) => {
                    tracing::warn!(
                        target: "router.supervisor",
                        slot_save_path = %dir,
                        error = %e,
                        "slot-save-path create failed (snapshots disabled)",
                    );
                    None
                }
            },
            None => None,
        };
        let servers = ConcurrentRegistry::new();
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
                slot_save_path.clone(),
                api_key.clone(),
                config.default_params.clone(),
            );
            servers.insert(key.clone(), ManagedServer::new(spec));
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
    pub fn server_for(&self, model_key: &str) -> Option<Arc<ManagedServer>> {
        self.servers.get(&model_key.to_string())
    }

    /// The management base URL for a model id.
    pub fn base_url_for(&self, model_key: &str) -> Option<String> {
        self.server_for(model_key).map(|s| s.base_url().to_string())
    }

    /// Every managed model key.
    pub fn model_keys(&self) -> Vec<String> {
        self.servers.keys()
    }

    /// Spawn every **boot** managed server and wait for each to become healthy.
    /// Only models declaring at least one pinned instance are booted; the rest
    /// (lazy models) are loaded on demand via [`Self::ensure_running`]. Spawns
    /// happen first (processes load weights concurrently), then health is
    /// awaited in parallel, so boot time is bounded by the slowest boot model
    /// rather than the sum. Returns the first health failure so boot aborts
    /// loudly (a boot model that cannot load its weights must not be silently
    /// skipped).
    pub async fn start_all(self: &Arc<Self>) -> Result<(), String> {
        let bin = self.bin.clone();
        let mut keys: Vec<String> = self.servers.keys();
        keys.sort();
        let mut boot_keys = Vec::new();
        for key in &keys {
            let server = self.servers.get(key).expect("managed key from registry");
            if !server.spec.boot {
                tracing::info!(
                    target: "router.supervisor",
                    model = %key,
                    "model deferred - no pinned instance, loaded on demand",
                );
                continue;
            }
            boot_keys.push(key.clone());
            server.spawn_child(&bin);
            server.inner.running.store(true, Ordering::Relaxed);
            let handle = {
                let me = Arc::clone(&server);
                let bin = bin.clone();
                tokio::spawn(async move { me.supervise(bin).await; })
            };
            let supervisor_guard = server.inner.supervisor.lock();
            if let Ok(mut guard) = supervisor_guard {
                *guard = Some(handle.abort_handle());
            }
        }
        // Await every boot server's /health concurrently; the first failure
        // aborts (lazy models are not health-checked at boot).
        let mut health = Vec::new();
        for key in &boot_keys {
            let server = self.servers.get(key).expect("managed key from registry");
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

    /// Load a model on demand: spawn its `llama-server` (if not already
    /// running) and wait for `/health`. Returns `None` when the model is not
    /// managed. Used by the dispatch path before targeting a lazy model.
    pub async fn ensure_running(&self, model_key: &str) -> Result<(), String> {
        let server = self.servers.get(&model_key.to_string()).ok_or_else(|| {
            format!("model '{model_key}' is not managed by the supervisor")
        })?;
        server.ensure_running(&self.bin).await
    }

    /// Whether a managed model's server is currently running. `None` when the
    /// model is not managed.
    pub fn is_running(&self, model_key: &str) -> Option<bool> {
        self.servers.get(&model_key.to_string()).map(|s| s.is_running())
    }

    /// Unload a model's server on demand (frees its VRAM). The spec stays
    /// registered so [`Self::ensure_running`] can re-load it later.
    pub async fn unload(&self, model_key: &str) {
        if let Some(server) = self.servers.get(&model_key.to_string()) {
            server.unload().await;
        }
    }

    /// Stop every managed server (used on shutdown).
    pub async fn shutdown(&self) {
        let mut keys: Vec<String> = self.servers.keys();
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
        for key in self.servers.keys() {
            if let Some(server) = self.servers.get(&key) {
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

    fn defaults() -> crate::config::DefaultModelParams {
        crate::config::DefaultModelParams::default()
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
}
