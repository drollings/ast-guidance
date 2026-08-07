//! Instance-pool grammar generation and validation.
//!
//! Mirrors the fork's `common_instances_parse` / `common_instances_to_string`
//! (`common/common.cpp`) so the router can emit the exact `--instance` flags the
//! operator hands to `llama-server`. The grammar is colon-separated:
//!
//! `name[:group=G][:ctx=N][:parallel=M][:pinned][:sleep=0|N][:default]`
//!
//! The router never reads the raw KV bytes — it only declares instances; the
//! fork owns the weights and the instance pool.
//!
//! This module also hosts the sidecar: [`InstanceClient`] wraps the fork's
//! management API (`/instances`, `/memory`, ...) with `HttpClass`-classified
//! errors, and [`InstanceManager`] owns boot reconciliation, the `/memory`
//! residency loop (LRU eviction of unpinned instances when free VRAM is low),
//! and allocate-on-503 (a fresh instance when a group-miss request retries).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use common_core::hash::uuid_v4;
use crate::config::InstanceProfile;

use fluent_llm::HttpClass;

/// Render a flat list of (expanded) `InstanceProfile`s as the fork's
/// comma-joined `--instance` grammar, matching `common_instances_to_string`
/// byte-for-byte for equivalent inputs.
pub fn instance_grammar_string(profiles: &[InstanceProfile]) -> String {
    let parts: Vec<String> = profiles.iter().map(render_one).collect();
    parts.join(",")
}

fn render_one(profile: &InstanceProfile) -> String {
    let name = profile.name.clone().unwrap_or_default();
    let mut s = name.clone();

    if let Some(group) = &profile.group {
        if *group != name {
            s.push_str(":group=");
            s.push_str(group);
        }
    }

    if profile.num_ctx > 0 {
        s.push_str(":ctx=");
        s.push_str(&profile.num_ctx.to_string());
    }

    if let Some(parallel) = profile.parallel {
        if parallel > 0 {
            s.push_str(":parallel=");
            s.push_str(&parallel.to_string());
        }
    }

    if profile.pinned {
        // pinned implies sleep=0 (never auto-sleep); a declared sleep is ignored.
        s.push_str(":pinned");
    } else if profile.no_sleep {
        s.push_str(":sleep=0");
    } else if let Some(sleep) = profile.sleep_idle_seconds {
        if sleep > 0 {
            s.push_str(":sleep=");
            s.push_str(&sleep.to_string());
        }
    }

    if profile.default {
        s.push_str(":default");
    }

    s
}

/// Validate a flat instance list the way the fork's parser does: no duplicate
/// names, and no instance's group colliding with another instance's name.
/// The group==own-name default is permitted.
pub fn validate_instances(profiles: &[InstanceProfile]) -> Result<(), String> {
    for (i, pi) in profiles.iter().enumerate() {
        let ni = pi.name.as_deref().unwrap_or("");
        let gi = pi.group.as_deref().unwrap_or("");
        for other in profiles.iter().skip(i + 1) {
            let nj = other.name.as_deref().unwrap_or("");
            let gj = other.group.as_deref().unwrap_or("");
            if !ni.is_empty() && ni == nj {
                return Err(format!("duplicate instance name '{ni}'"));
            }
            if (!gi.is_empty() && gi == nj) || (!gj.is_empty() && gj == ni) {
                return Err(format!(
                    "instance group '{gi}' collides with instance name '{nj}'"
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Management client — the fork's management API over raw reqwest
// ---------------------------------------------------------------------------

/// Derive the management base URL from a model's chat-completions endpoint:
/// `http://host:port/v1/chat/completions` -> `http://host:port`. The management
/// endpoints (`/instances`, `/memory`, ...) live at the same host as the
/// generation endpoint.
pub fn management_base_url(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    for suffix in ["/v1/chat/completions", "/chat/completions"] {
        if let Some(base) = trimmed.strip_suffix(suffix) {
            return base.trim_end_matches('/').to_string();
        }
    }
    trimmed.to_string()
}

/// Classification of a management-API failure, mirroring `HttpClass` with the
/// extra `Duplicate` (409) and evict-trigger (507/503) distinctions the sidecar
/// cares about.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InstanceError {
    /// 429/503/504/507/other 5xx — transient; a 507/503 also signals an
    /// allocation/eviction trigger.
    #[error("transient management error: {status} {body}")]
    Transient { status: u16, body: String },
    /// 409 duplicate name — tolerated during reconciliation.
    #[error("duplicate instance (409)")]
    Duplicate,
    /// Permanent 4xx (except 409) — no retry.
    #[error("management request rejected: {status} {body}")]
    Rejected { status: u16, body: String },
    /// Transport / network failure before an HTTP status was received.
    #[error("management network error: {0}")]
    Network(String),
    /// A 2xx whose payload did not match the expected shape.
    #[error("management response parse error: {0}")]
    Other(String),
}

impl InstanceError {
    /// Whether the failure merits a retry with backoff.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            InstanceError::Transient { .. } | InstanceError::Network(_)
        )
    }

    /// Whether this is a 409 duplicate-name collision (tolerable in reconcile).
    pub fn is_duplicate(&self) -> bool {
        matches!(self, InstanceError::Duplicate)
    }

    /// Whether a low-memory allocation (507) or 503 signals the residency loop
    /// should consider evicting.
    pub fn is_evict_trigger(&self) -> bool {
        matches!(
            self,
            InstanceError::Transient { status, .. } if *status == 503 || *status == 507
        )
    }
}

/// One instance as reported by `GET /instances`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub id: String,
    pub group: String,
    pub n_ctx: u64,
    pub parallel: u32,
    pub pinned: bool,
    pub no_sleep: bool,
    #[serde(default)]
    pub state: String,
    pub vram_bytes: u64,
    /// Most recent slot use (`-1` when never used).
    pub last_used: i64,
}

/// Per-instance memory footprint from `/memory`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemorySlot {
    pub id: String,
    pub n_ctx: u64,
    pub state: String,
    pub model: u64,
    pub context: u64,
    pub compute: u64,
    pub total: u64,
}

/// Aggregated memory report from `/memory` (shared weights counted once).
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryReport {
    #[serde(default)]
    pub slots: Vec<MemorySlot>,
    pub total: MemoryTotal,
}

/// The `total` object of `/memory`.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryTotal {
    pub model: u64,
    pub context: u64,
    pub compute: u64,
    pub total: u64,
}

/// One snapshot entry from `GET /instances/:name/snapshots`.
#[derive(Debug, Clone, Deserialize)]
pub struct SnapshotInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub mtime: String,
    #[serde(default)]
    pub n_ctx_seq: u64,
}

/// Typed management client against the fork's `/instances` + `/memory` API.
///
/// Mirrors the raw-reqwest pattern of `OpenAiChatBackend`: a plain
/// `reqwest::Client` and explicit status classification via `HttpClass`.
#[derive(Clone)]
pub struct InstanceClient {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl InstanceClient {
    pub fn new(
        client: reqwest::Client,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        Self {
            client,
            base_url: base_url.into(),
            api_key,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Issue a management request, classify the status, and return the parsed
    /// JSON on 2xx. 409 -> `Duplicate`; 429/503/504/507/5xx -> `Transient`;
    /// other 4xx -> `Rejected`.
    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, InstanceError> {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);
        let mut builder = self.client.request(method, &url);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        if let Some(body) = body {
            builder = builder.json(body);
        }

        let response = builder
            .send()
            .await
            .map_err(|e| InstanceError::Network(e.to_string()))?;
        let status = response.status();
        if status.is_success() {
            return response
                .json()
                .await
                .map_err(|e| InstanceError::Other(e.to_string()));
        }
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| String::new());
        let status_u16 = status.as_u16();
        if status_u16 == 409 {
            return Err(InstanceError::Duplicate);
        }
        // Classify via the shared `HttpClass` taxonomy; a 507 (low device
        // memory) is an evict trigger like 503.
        let class = HttpClass::from_status(status_u16);
        if class.is_retryable() || status_u16 == 507 {
            Err(InstanceError::Transient {
                status: status_u16,
                body,
            })
        } else {
            Err(InstanceError::Rejected {
                status: status_u16,
                body,
            })
        }
    }

    /// `GET /instances` — the current instance set.
    pub async fn list(&self) -> Result<Vec<InstanceInfo>, InstanceError> {
        let value = self
            .request(reqwest::Method::GET, "/instances", None)
            .await?;
        serde_json::from_value(value)
            .map_err(|e| InstanceError::Other(format!("list: {e}")))
    }

    /// `POST /instances` — allocate a fresh context from the shared weights.
    /// Only KV + compute are allocated; the model weights stay loaded.
    pub async fn create(
        &self,
        name: &str,
        group: &str,
        ctx_size: u64,
        parallel: Option<u32>,
        pinned: bool,
    ) -> Result<(), InstanceError> {
        let mut body = serde_json::json!({
            "name": name,
            "group": group,
            "ctx_size": ctx_size,
            "pinned": pinned,
        });
        if let Some(parallel) = parallel {
            body["parallel"] = Value::Number(parallel.into());
        }
        self.request(reqwest::Method::POST, "/instances", Some(&body))
            .await
            .map(|_| ())
    }

    /// `DELETE /instances/:name` — free KV + compute (the primary eviction
    /// path). `force` overrides `pinned`.
    pub async fn destroy(&self, name: &str, force: bool) -> Result<(), InstanceError> {
        let path = if force {
            format!("/instances/{name}?force=true")
        } else {
            format!("/instances/{name}")
        };
        self.request(reqwest::Method::DELETE, &path, None)
            .await
            .map(|_| ())
    }

    /// `POST /instances/:name/pin` — protect residency.
    pub async fn pin(&self, name: &str) -> Result<(), InstanceError> {
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{name}/pin"),
            None,
        )
        .await
        .map(|_| ())
    }

    /// `POST /instances/:name/unpin` — release residency protection.
    pub async fn unpin(&self, name: &str) -> Result<(), InstanceError> {
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{name}/unpin"),
            None,
        )
        .await
        .map(|_| ())
    }

    /// `POST /instances/:name/resize` — re-create the context at a new size.
    pub async fn resize(&self, name: &str, ctx_size: u64) -> Result<(), InstanceError> {
        let body = serde_json::json!({ "ctx_size": ctx_size });
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{name}/resize"),
            Some(&body),
        )
        .await
        .map(|_| ())
    }

    /// `GET /memory` — per-instance + total VRAM (shared weights counted once).
    pub async fn memory(&self) -> Result<MemoryReport, InstanceError> {
        let value = self
            .request(reqwest::Method::GET, "/memory", None)
            .await?;
        serde_json::from_value(value)
            .map_err(|e| InstanceError::Other(format!("memory: {e}")))
    }

    /// `POST /instances/:name/snapshot` — save the slot-0 KV to a named snapshot.
    pub async fn save_snapshot(&self, instance: &str, name: &str) -> Result<(), InstanceError> {
        let body = serde_json::json!({ "name": name });
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{instance}/snapshot"),
            Some(&body),
        )
        .await
        .map(|_| ())
    }

    /// `GET /instances/:name/snapshots` — list the instance's snapshots.
    pub async fn list_snapshots(&self, instance: &str) -> Result<Vec<SnapshotInfo>, InstanceError> {
        let value = self
            .request(
                reqwest::Method::GET,
                &format!("/instances/{instance}/snapshots"),
                None,
            )
            .await?;
        serde_json::from_value(value)
            .map_err(|e| InstanceError::Other(format!("list_snapshots: {e}")))
    }

    /// `DELETE /instances/:name/snapshot/:snapshot` — remove a snapshot file.
    pub async fn delete_snapshot(
        &self,
        instance: &str,
        name: &str,
    ) -> Result<(), InstanceError> {
        self.request(
            reqwest::Method::DELETE,
            &format!("/instances/{instance}/snapshot/{name}"),
            None,
        )
        .await
        .map(|_| ())
    }
}

// ---------------------------------------------------------------------------
// InstanceManager — sidecar: reconcile, residency, allocate-on-503
// ---------------------------------------------------------------------------

/// The sidecar owner of instance lifecycle. Holds the management client, the
/// expanded configured profiles for one `(endpoint)` pool, and the residency
/// policy. Runs as a task on the router's tokio runtime (owned by the server).
pub struct InstanceManager {
    client: InstanceClient,
    profiles: Vec<InstanceProfile>,
    policy: crate::config::SidecarConfig,
}

impl InstanceManager {
    pub fn new(
        client: InstanceClient,
        profiles: Vec<InstanceProfile>,
        policy: crate::config::SidecarConfig,
    ) -> Self {
        Self {
            client,
            profiles,
            policy,
        }
    }

    pub fn client(&self) -> &InstanceClient {
        &self.client
    }

    /// Boot reconciliation: create configured instances missing from
    /// `GET /instances`, resize `n_ctx` mismatches, and warn on
    /// `parallel`/`pinned` drift. A duplicate-create (409) is tolerated.
    /// Emits an audit record of the result.
    pub async fn reconcile(&self) -> Result<(), InstanceError> {
        let existing = match self.client.list().await {
            Ok(list) => list,
            Err(e) => {
                tracing::warn!(
                    target: "router.instances",
                    base_url = %self.client.base_url(),
                    error = %e,
                    "instance reconcile aborted — management API unreachable",
                );
                return Err(e);
            }
        };
        let by_name: HashMap<&str, &InstanceInfo> =
            existing.iter().map(|i| (i.id.as_str(), i)).collect();

        let mut created = 0usize;
        let mut resized = 0usize;
        for profile in &self.profiles {
            let name = profile.name.as_deref().unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let group = profile.group.as_deref().unwrap_or(name);
            match by_name.get(name) {
                Some(info) => {
                    // Resize on n_ctx drift; warn on parallel/pinned drift.
                    if profile.num_ctx > 0 && info.n_ctx != profile.num_ctx {
                        match self.client.resize(name, profile.num_ctx).await {
                            Ok(()) => {
                                resized += 1;
                                tracing::info!(
                                    target: "router.instances",
                                    instance = %name,
                                    from_ctx = info.n_ctx,
                                    to_ctx = profile.num_ctx,
                                    "instance resized",
                                );
                            }
                            Err(e) => tracing::warn!(
                                target: "router.instances",
                                instance = %name,
                                error = %e,
                                "instance resize failed",
                            ),
                        }
                    }
                    if info.pinned != profile.pinned {
                        tracing::warn!(
                            target: "router.instances",
                            instance = %name,
                            expected_pinned = profile.pinned,
                            actual_pinned = info.pinned,
                            "instance pinned drift",
                        );
                    }
                    if let Some(parallel) = profile.parallel {
                        if info.parallel != parallel {
                            tracing::warn!(
                                target: "router.instances",
                                instance = %name,
                                expected_parallel = parallel,
                                actual_parallel = info.parallel,
                                "instance parallel drift",
                            );
                        }
                    }
                }
                None => match self.client.create(name, group, profile.num_ctx, profile.parallel, profile.pinned).await {
                    Ok(()) => {
                        created += 1;
                        tracing::info!(
                            target: "router.instances",
                            instance = %name,
                            group = %group,
                            n_ctx = profile.num_ctx,
                            pinned = profile.pinned,
                            "instance created at boot",
                        );
                    }
                    Err(InstanceError::Duplicate) => {
                        // Another reconciler won the race; tolerate.
                    }
                    Err(e) => tracing::warn!(
                        target: "router.instances",
                        instance = %name,
                        error = %e,
                        "instance create failed",
                    ),
                },
            }
        }
        crate::audit::emit(
            "instances",
            serde_json::json!({
                "action": "reconcile",
                "created": created,
                "resized": resized,
                "base_url": self.client.base_url(),
            }),
        );
        Ok(())
    }

    /// One residency pass: poll `/memory`, and when free VRAM (device ceiling
    /// minus used) drops below the watermark, evict up to `evict_batch`
    /// least-recently-used unpinned instances. Pinned instances are never
    /// evicted. A missing `vram_total_bytes` ceiling disables eviction (the
    /// pass still polls and logs).
    pub async fn residency_cycle(&self) -> Result<(), InstanceError> {
        let Some(vram_total) = self.policy.vram_total_bytes else {
            return Ok(());
        };
        let mem = self.client.memory().await?;
        let used = mem.total.total;
        let free = vram_total.saturating_sub(used);
        if free >= self.policy.vram_low_watermark_bytes {
            return Ok(());
        }
        tracing::warn!(
            target: "router.instances",
            free_bytes = free,
            used_bytes = used,
            watermark_bytes = self.policy.vram_low_watermark_bytes,
            "free VRAM below watermark — evicting LRU unpinned instances",
        );
        let infos = self.client.list().await?;
        let mut candidates: Vec<&InstanceInfo> = infos.iter().filter(|i| !i.pinned).collect();
        candidates.sort_by_key(|i| i.last_used);
        let mut evicted = 0usize;
        for info in candidates {
            if evicted >= self.policy.evict_batch {
                break;
            }
            match self.client.destroy(&info.id, false).await {
                Ok(()) => {
                    evicted += 1;
                    tracing::info!(
                        target: "router.instances",
                        instance = %info.id,
                        last_used = info.last_used,
                        vram_bytes = info.vram_bytes,
                        "unpinned instance evicted for low VRAM",
                    );
                    crate::audit::emit(
                        "instances",
                        serde_json::json!({
                            "action": "evict",
                            "instance": info.id,
                            "reason": "low_vram",
                        }),
                    );
                }
                Err(e) => tracing::warn!(
                    target: "router.instances",
                    instance = %info.id,
                    error = %e,
                    "instance eviction failed",
                ),
            }
        }
        Ok(())
    }

    /// The residency loop: poll `/memory` every `poll_interval_s`, evicting on
    /// low free VRAM, forever. Runs as a spawned task owned by the server.
    pub async fn run_residency(&self) {
        let interval = Duration::from_secs(self.policy.poll_interval_s.max(1));
        loop {
            if let Err(e) = self.residency_cycle().await {
                tracing::warn!(
                    target: "router.instances",
                    error = %e,
                    "residency poll failed (retrying next interval)",
                );
            }
            tokio::time::sleep(interval).await;
        }
    }

    /// Allocate a fresh instance for `group` on a 503 group-miss. Uses the
    /// group's configured profile (name/group/ctx/parallel/pinned) with a
    /// unique `<group>-<uuid>` name. No-op when no profile configures the
    /// group (there is nothing to allocate).
    pub async fn ensure_group(&self, group: &str) -> Result<(), InstanceError> {
        let Some(profile) = self
            .profiles
            .iter()
            .find(|p| p.group.as_deref() == Some(group))
        else {
            tracing::debug!(
                target: "router.instances",
                group = %group,
                "no configured profile for group — nothing to allocate",
            );
            return Ok(());
        };
        let name = format!("{group}-{}", &uuid_v4()[..8]);
        let profile_group = profile.group.as_deref().unwrap_or(group);
        let result = self
            .client
            .create(&name, profile_group, profile.num_ctx, profile.parallel, profile.pinned)
            .await;
        if result.is_ok() {
            tracing::info!(
                target: "router.instances",
                instance = %name,
                group = %profile_group,
                "instance allocated on group miss",
            );
        }
        result
    }
}

/// Build one `InstanceManager` per distinct model endpoint that declares an
/// instance pool, grouping that endpoint's expanded profiles under a single
/// manager. The management base URL is derived from each endpoint. Empty when
/// no model declares instances (the sidecar is inactive).
pub fn build_instance_managers(
    config: &crate::config::RouterConfig,
) -> HashMap<String, Arc<InstanceManager>> {
    // endpoint -> (profiles, model keys with instances at that endpoint)
    let mut per_endpoint: HashMap<String, Vec<InstanceProfile>> = HashMap::new();
    for entry in config.models.values() {
        let profiles = entry.instance_profiles();
        if profiles.is_empty() {
            continue;
        }
        per_endpoint.entry(entry.endpoint.clone()).or_default().extend(profiles);
    }

    let mut managers = HashMap::new();
    for (endpoint, profiles) in per_endpoint {
        let base_url = management_base_url(&endpoint);
        let api_key = config
            .sidecar
            .api_key_env
            .as_deref()
            .map(std::env::var)
            .and_then(Result::ok)
            .filter(|k| !k.is_empty());
        let client = InstanceClient::new(reqwest::Client::new(), base_url, api_key);
        let manager = Arc::new(InstanceManager::new(
            client,
            profiles,
            config.sidecar.clone(),
        ));
        managers.insert(endpoint, manager);
    }
    managers
}

/// A tiny HTTP/1.1 stub that records every request and answers from a shared
/// handler closure `(method, path, body) -> (status, body)`. Used to exercise
/// `InstanceClient`/`InstanceManager` (and the dispatch allocate-on-503 path)
/// against fixture `/instances`, `/memory`, and `/chat/completions` JSON
/// without a real llama-server.
#[cfg(test)]
pub(crate) mod stub {
    use std::sync::Arc;

    pub struct StubServer {
        #[allow(dead_code)]
        _handle: tokio::task::JoinHandle<()>,
        pub addr: std::net::SocketAddr,
        requests: Arc<std::sync::Mutex<Vec<(String, String, String)>>>,
    }

    impl StubServer {
        pub fn start(
            handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync>,
        ) -> Self {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
            let reqs = requests.clone();
            let handle = tokio::spawn(async move {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                loop {
                    let (mut stream, _) = match listener.accept().await {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let handler = handler.clone();
                    let reqs = reqs.clone();
                    tokio::spawn(async move {
                        use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
                        let mut reader = BufReader::new(&mut stream);
                        let mut request_line = String::new();
                        if reader.read_line(&mut request_line).await.is_err() {
                            return;
                        }
                        let mut content_length: usize = 0;
                        loop {
                            let mut line = String::new();
                            if reader.read_line(&mut line).await.is_err() {
                                return;
                            }
                            if line == "\r\n" {
                                break;
                            }
                            if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                                content_length = v.trim().parse().unwrap_or(0);
                            }
                        }
                        let mut body = vec![0u8; content_length];
                        if content_length > 0 && reader.read_exact(&mut body).await.is_err() {
                            return;
                        }
                        let body_str = String::from_utf8_lossy(&body).into_owned();
                        let parts: Vec<&str> = request_line.split_whitespace().collect();
                        let (method, path) = if parts.len() >= 2 {
                            (parts[0].to_string(), parts[1].to_string())
                        } else {
                            ("GET".into(), "/".into())
                        };
                        if let Ok(mut r) = reqs.lock() {
                            r.push((method.clone(), path.clone(), body_str.clone()));
                        }
                        let (status, resp_body) = handler(&method, &path, &body_str);
                        let reason = match status {
                            200 => "OK",
                            201 => "Created",
                            204 => "No Content",
                            400 => "Bad Request",
                            404 => "Not Found",
                            409 => "Conflict",
                            503 => "Service Unavailable",
                            507 => "Insufficient Storage",
                            _ => "Error",
                        };
                        let resp = format!(
                            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            resp_body.len(),
                            resp_body
                        );
                        let _ = stream.write_all(resp.as_bytes()).await;
                        let _ = stream.flush().await;
                    });
                }
            });
            Self {
                _handle: handle,
                addr,
                requests,
            }
        }

        pub fn base_url(&self) -> String {
            format!("http://{}", self.addr)
        }

        pub fn recorded(&self) -> Vec<(String, String, String)> {
            self.requests
                .lock()
                .map(|r| r.clone())
                .unwrap_or_default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::stub::StubServer;
    use super::*;

    fn profile(name: &str, group: &str) -> InstanceProfile {
        InstanceProfile {
            name: Some(name.into()),
            group: Some(group.into()),
            count: 1,
            num_ctx: 16384,
            parallel: None,
            pinned: false,
            no_sleep: false,
            sleep_idle_seconds: None,
            default: false,
            params: None,
        }
    }

    fn assert_round_trip(profiles: &[InstanceProfile]) {
        let s = instance_grammar_string(profiles);
        let reparsed: Vec<InstanceProfile> = s
            .split(',')
            .filter(|p| !p.is_empty())
            .map(|p| parse_one(p).into_instance_profile())
            .collect();
        assert_eq!(reparsed.len(), profiles.len());
        for (a, b) in profiles.iter().zip(reparsed.iter()) {
            assert_eq!(a.name.as_deref(), b.name.as_deref());
            assert_eq!(a.group.as_deref(), b.group.as_deref());
            assert_eq!(a.num_ctx, b.num_ctx);
            assert_eq!(a.parallel, b.parallel);
            assert_eq!(a.pinned, b.pinned);
            assert_eq!(a.no_sleep, b.no_sleep);
            assert_eq!(a.default, b.default);
        }
    }

    // A minimal fork-parser equivalent used only to validate round-trip shape.
    #[derive(Default)]
    struct Raw {
        name: String,
        group: Option<String>,
        ctx: u64,
        parallel: u32,
        pinned: bool,
        no_sleep: bool,
        sleep: Option<i32>,
        default: bool,
    }

    impl Raw {
        fn into_instance_profile(self) -> InstanceProfile {
            let group = self
                .group
                .clone()
                .unwrap_or_else(|| self.name.clone());
            InstanceProfile {
                name: Some(self.name),
                group: Some(group),
                count: 1,
                num_ctx: self.ctx,
                parallel: if self.parallel > 0 {
                    Some(self.parallel)
                } else {
                    None
                },
                pinned: self.pinned,
                no_sleep: self.no_sleep,
                sleep_idle_seconds: self.sleep,
                default: self.default,
                params: None,
            }
        }
    }

    fn parse_one(spec: &str) -> Raw {
        let mut comps = spec.split(':');
        let mut raw = Raw {
            name: comps.next().unwrap_or_default().to_string(),
            ..Default::default()
        };
        for comp in comps {
            if comp == "pinned" {
                raw.pinned = true;
            } else if comp == "default" {
                raw.default = true;
            } else if let Some(v) = comp.strip_prefix("group=") {
                raw.group = Some(v.to_string());
            } else if let Some(v) = comp.strip_prefix("ctx=") {
                raw.ctx = v.parse().unwrap();
            } else if let Some(v) = comp.strip_prefix("parallel=") {
                raw.parallel = v.parse().unwrap();
            } else if let Some(v) = comp.strip_prefix("sleep=") {
                raw.sleep = Some(v.parse().unwrap());
            }
        }
        raw
    }

    #[test]
    fn grammar_matches_reference_deployment() {
        let swarm = InstanceProfile {
            name: Some("swarm".into()),
            group: Some("swarm".into()),
            count: 3,
            num_ctx: 16384,
            parallel: None,
            pinned: false,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            params: None,
        };
        // Expand the count as instance_profiles() would.
        let profiles: Vec<InstanceProfile> = (0..3)
            .map(|i| InstanceProfile {
                name: Some(format!("swarm{i}")),
                group: Some("swarm".into()),
                ..swarm.clone()
            })
            .collect();
        assert_eq!(
            instance_grammar_string(&profiles),
            "swarm0:group=swarm:ctx=16384:sleep=0,swarm1:group=swarm:ctx=16384:sleep=0,swarm2:group=swarm:ctx=16384:sleep=0"
        );
    }

    #[test]
    fn grammar_matches_reference_ledger_scratch() {
        let ledger = InstanceProfile {
            name: Some("ledger".into()),
            group: None,
            num_ctx: 131072,
            pinned: true,
            default: true,
            ..profile("x", "x")
        };
        let scratch = InstanceProfile {
            name: Some("scratch".into()),
            group: None,
            num_ctx: 131072,
            sleep_idle_seconds: Some(30),
            ..profile("x", "x")
        };
        assert_eq!(
            instance_grammar_string(&[ledger, scratch]),
            "ledger:ctx=131072:pinned:default,scratch:ctx=131072:sleep=30"
        );
    }

    #[test]
    fn round_trip_preserves_shape() {
        let profiles = vec![
            profile("swarm0", "swarm"),
            profile("ledger", "ledger"),
            profile("scratch", "scratch"),
        ];
        assert_round_trip(&profiles);
    }

    #[test]
    fn pinned_implies_sleep_zero() {
        // A pinned profile with a positive declared sleep emits only `:pinned`.
        let pinned = InstanceProfile {
            name: Some("p".into()),
            group: Some("p".into()),
            num_ctx: 0,
            pinned: true,
            sleep_idle_seconds: Some(30),
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[pinned]), "p:pinned");
    }

    #[test]
    fn sleep_minus_one_is_inherited_and_not_emitted() {
        let inherit = InstanceProfile {
            name: Some("a".into()),
            group: Some("a".into()),
            num_ctx: 0,
            sleep_idle_seconds: Some(-1),
            ..profile("x", "x")
        };
        // -1 (inherit) is omitted; a positive N is emitted; absent is omitted.
        assert_eq!(instance_grammar_string(&[inherit]), "a");
        let positive = InstanceProfile {
            name: Some("b".into()),
            group: Some("b".into()),
            num_ctx: 0,
            sleep_idle_seconds: Some(5),
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[positive]), "b:sleep=5");
        let absent = InstanceProfile {
            name: Some("c".into()),
            group: Some("c".into()),
            num_ctx: 0,
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[absent]), "c");
    }

    #[test]
    fn validates_duplicate_names() {
        let a = profile("dup", "g1");
        let b = profile("dup", "g2");
        assert!(validate_instances(&[a, b]).is_err());
    }

    #[test]
    fn validates_group_name_collision() {
        let a = profile("foo", "bar");
        let b = profile("bar", "baz");
        assert!(validate_instances(&[a, b]).is_err());
    }

    #[test]
    fn allows_group_equals_own_name() {
        let a = profile("foo", "foo");
        let b = profile("bar", "bar");
        assert!(validate_instances(&[a, b]).is_ok());
    }

    // ── M4 sidecar ─────────────────────────────────────────────────────────

    fn sidecar_policy() -> crate::config::SidecarConfig {
        crate::config::SidecarConfig {
            poll_interval_s: 5,
            vram_low_watermark_bytes: 1024,
            evict_batch: 2,
            vram_total_bytes: Some(10000),
            slot_save_path: Some("/srv/slots".into()),
            api_key_env: None,
        }
    }

    fn instance_info(id: &str, group: &str, pinned: bool, last_used: i64) -> InstanceInfo {
        InstanceInfo {
            id: id.into(),
            group: group.into(),
            n_ctx: 16384,
            parallel: 1,
            pinned,
            no_sleep: false,
            state: "loaded".into(),
            vram_bytes: 1000,
            last_used,
        }
    }

    fn management_base(endpoint: &str) -> String {
        super::management_base_url(endpoint)
    }

    #[test]
    fn management_base_url_strips_chat_completions_suffix() {
        assert_eq!(
            management_base("http://localhost:8080/v1/chat/completions"),
            "http://localhost:8080"
        );
        assert_eq!(
            management_base("http://localhost:8080/chat/completions"),
            "http://localhost:8080"
        );
        assert_eq!(management_base("http://localhost:8080/v1"), "http://localhost:8080/v1");
    }

    #[tokio::test]
    async fn client_list_and_memory_parse_fixture() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!([
                            instance_info("swarm0", "swarm", false, 5),
                            instance_info("ledger", "ledger", true, 1),
                        ])
                        .to_string(),
                    )
                } else if method == "GET" && path == "/memory" {
                    (
                        200,
                        serde_json::json!({
                            "slots": [
                                { "id": "swarm0", "n_ctx": 16384, "state": "loaded", "model": 0, "context": 262144, "compute": 1048576, "total": 1310720 }
                            ],
                            "total": { "model": 115343360, "context": 262144, "compute": 1048576, "total": 118226080 }
                        })
                        .to_string(),
                    )
                } else {
                    (404, r#"{"error":{"message":"not found"}}"#.into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(
            reqwest::Client::new(),
            stub.base_url(),
            None,
        );

        let list = client.list().await.expect("list");
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "swarm0");
        assert_eq!(list[0].group, "swarm");
        assert_eq!(list[1].pinned, true);

        let mem = client.memory().await.expect("memory");
        assert_eq!(mem.total.total, 118226080);
        assert_eq!(mem.slots.len(), 1);
    }

    #[tokio::test]
    async fn client_mutating_calls_hit_expected_paths() {
        let requests_c = Arc::new(std::sync::Mutex::new(Vec::<(String, String)>::new()));
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                requests_c.lock().unwrap().push((method.into(), path.into()));
                if path.ends_with("/snapshots") {
                    (200, "[]".into())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        client
            .create("work", "swarm", 32768, Some(2), true)
            .await
            .expect("create");
        client.destroy("work", false).await.expect("destroy");
        client.destroy("work", true).await.expect("destroy force");
        client.pin("work").await.expect("pin");
        client.unpin("work").await.expect("unpin");
        client.resize("work", 49152).await.expect("resize");
        client.save_snapshot("work", "readfiles").await.expect("save");
        client.delete_snapshot("work", "readfiles").await.expect("delete");
        client.list_snapshots("work").await.expect("list snapshots");

        let recorded = stub.recorded();
        assert_eq!(recorded.len(), 9);
        let paths: Vec<&str> = recorded.iter().map(|(_, p, _)| p.as_str()).collect();
        assert!(paths.contains(&"/instances"));
        assert!(paths.contains(&"/instances/work"));
        assert!(paths.contains(&"/instances/work?force=true"));
        assert!(paths.contains(&"/instances/work/pin"));
        assert!(paths.contains(&"/instances/work/unpin"));
        assert!(paths.contains(&"/instances/work/resize"));
        assert!(paths.contains(&"/instances/work/snapshot"));
        assert!(paths.contains(&"/instances/work/snapshot/readfiles"));
        assert!(paths.contains(&"/instances/work/snapshots"));
        // create body carries the declared fields.
        let create_req = recorded
            .iter()
            .find(|(m, p, _)| m == "POST" && p == "/instances")
            .unwrap();
        let body: serde_json::Value = serde_json::from_str(&create_req.2).unwrap();
        assert_eq!(body["name"], "work");
        assert_eq!(body["group"], "swarm");
        assert_eq!(body["ctx_size"], 32768);
        assert_eq!(body["parallel"], 2);
        assert_eq!(body["pinned"], true);
    }

    #[tokio::test]
    async fn client_classifies_409_duplicate_and_5xx_transient() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if path == "/instances" && method == "POST" {
                    (409, r#"{"error":{"message":"duplicate"}}"#.into())
                } else if path == "/instances/boom" {
                    (507, r#"{"error":{"message":"oom"}}"#.into())
                } else {
                    (404, r#"{"error":{"message":"not found"}}"#.into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        let dup = client.create("dup", "g", 16384, None, false).await;
        assert!(dup.unwrap_err().is_duplicate());

        let transient = client.destroy("boom", false).await.unwrap_err();
        assert!(transient.is_retryable());
        assert!(transient.is_evict_trigger());

        let rejected = client.destroy("missing", false).await.unwrap_err();
        assert!(matches!(rejected, InstanceError::Rejected { .. }));
    }

    #[tokio::test]
    async fn reconcile_creates_missing_and_resizes_n_ctx_drift() {
        // Server already has ledger (correct) and swarm0 (wrong n_ctx); swarm1
        // is missing entirely.
        let existing = [
            instance_info("ledger", "ledger", true, 1),
            instance_info("swarm0", "swarm", false, 5),
        ];
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, serde_json::to_string(&existing).unwrap())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        // Profiles: ledger (pinned, n_ctx 131072), swarm0 + swarm1 (group swarm, n_ctx 16384).
        let profiles = vec![
            InstanceProfile {
                name: Some("ledger".into()),
                group: Some("ledger".into()),
                count: 1,
                num_ctx: 131072,
                parallel: None,
                pinned: true,
                no_sleep: false,
                sleep_idle_seconds: None,
                default: true,
                params: None,
            },
            InstanceProfile {
                name: Some("swarm0".into()),
                group: Some("swarm".into()),
                count: 1,
                num_ctx: 16384,
                parallel: None,
                pinned: false,
                no_sleep: true,
                sleep_idle_seconds: None,
                default: false,
                params: None,
            },
            InstanceProfile {
                name: Some("swarm1".into()),
                group: Some("swarm".into()),
                count: 1,
                num_ctx: 16384,
                parallel: None,
                pinned: false,
                no_sleep: true,
                sleep_idle_seconds: None,
                default: false,
                params: None,
            },
        ];
        let manager = InstanceManager::new(client, profiles, sidecar_policy());
        manager.reconcile().await.expect("reconcile");

        let recorded = stub.recorded();
        // One POST /instances (swarm1 missing) and one resize (swarm0 ctx drift).
        let creates = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .count();
        let resizes = recorded
            .iter()
            .filter(|(_, p, _)| p.ends_with("/resize"))
            .count();
        assert_eq!(creates, 1, "exactly one missing profile is created");
        assert_eq!(resizes, 1, "n_ctx drift triggers exactly one resize");
        let create_body: serde_json::Value = serde_json::from_str(
            &recorded
                .iter()
                .find(|(m, p, _)| m == "POST" && p == "/instances")
                .unwrap()
                .2,
        )
        .unwrap();
        assert_eq!(create_body["name"], "swarm1");
        assert_eq!(create_body["group"], "swarm");
    }

    #[tokio::test]
    async fn reconcile_tolerates_duplicate_create() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, "[]".into())
                } else if method == "POST" && path == "/instances" {
                    // A concurrent reconciler created it first.
                    (409, "{}".into())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("swarm0".into()),
            group: Some("swarm".into()),
            count: 1,
            num_ctx: 16384,
            parallel: None,
            pinned: false,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            params: None,
        }];
        let manager = InstanceManager::new(client, profiles, sidecar_policy());
        // A 409 during reconcile is tolerated — reconcile completes Ok.
        manager.reconcile().await.expect("reconcile tolerates 409");
    }

    #[tokio::test]
    async fn residency_evicts_lru_unpinned_and_never_pinned() {
        // Free VRAM is far below the watermark: ceiling 10000 - used 9000 = 1000 < 1024.
        let memory = serde_json::json!({
            "slots": [],
            "total": { "model": 5000, "context": 2000, "compute": 2000, "total": 9000 }
        });
        let existing = [
            instance_info("pinned", "g", true, 0),    // exempt
            instance_info("lru1", "g", false, 100),   // oldest unpinned
            instance_info("lru2", "g", false, 200),
            instance_info("recent", "g", false, 9000),
        ];
        let memory = memory.to_string();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/memory" {
                    (200, memory.to_string())
                } else if method == "GET" && path == "/instances" {
                    (200, serde_json::to_string(&existing).unwrap())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let manager = InstanceManager::new(client, Vec::new(), sidecar_policy());
        manager.residency_cycle().await.expect("residency");

        // evict_batch = 2: the two oldest unpinned (lru1, lru2) are deleted;
        // pinned is never touched.
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert_eq!(deletes.len(), 2);
        assert!(deletes.contains(&"/instances/lru1"));
        assert!(deletes.contains(&"/instances/lru2"));
        assert!(!deletes.iter().any(|p| p.contains("pinned")));
    }

    #[tokio::test]
    async fn residency_no_eviction_when_free_vram_above_watermark() {
        let memory = serde_json::json!({
            "slots": [],
            "total": { "model": 1000, "context": 200, "compute": 200, "total": 1400 }
        });
        // free = 10000 - 1400 = 8600 >= 1024 watermark -> no eviction.
        let memory = memory.to_string();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/memory" {
                    (200, memory.clone())
                } else if method == "GET" && path == "/instances" {
                    (200, "[]".into())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let manager = InstanceManager::new(client, Vec::new(), sidecar_policy());
        manager.residency_cycle().await.expect("residency");
        assert!(
            stub.recorded()
                .iter()
                .all(|(m, _, _)| m != "DELETE"),
            "no eviction above the watermark"
        );
    }

    #[tokio::test]
    async fn ensure_group_allocates_fresh_instance_from_profile() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "POST" && path == "/instances" {
                    (201, "{}".into())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("swarm0".into()),
            group: Some("swarm".into()),
            count: 1,
            num_ctx: 16384,
            parallel: None,
            pinned: false,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            params: None,
        }];
        let manager = InstanceManager::new(client, profiles, sidecar_policy());
        manager.ensure_group("swarm").await.expect("ensure_group");

        let recorded = stub.recorded();
        let creates = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .collect::<Vec<_>>();
        assert_eq!(creates.len(), 1);
        let body: serde_json::Value = serde_json::from_str(&creates[0].2).unwrap();
        assert_eq!(body["group"], "swarm");
        assert_eq!(body["ctx_size"], 16384);
        let name = body["name"].as_str().unwrap();
        assert!(name.starts_with("swarm-"), "unique name generated: {name}");
    }
}
