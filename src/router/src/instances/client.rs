//! The typed management client against one spawned `llama-server`'s `/instances`
//! API, plus the wire types it returns. Talks DIRECTLY to the pool's server —
//! no router routing, so no `model` field is carried (each server owns exactly
//! one model's weights).
//!
//! Mirrors the raw-reqwest pattern of `OpenAiChatBackend`: a plain
//! `reqwest::Client` and explicit status classification via `HttpClass`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use fluent_llm::HttpClass;

/// Classification of a management-API failure, mirroring `HttpClass` with the
/// extra `Duplicate` (409) and evict-trigger (507/503) distinctions the sidecar
/// cares about.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InstanceError {
    /// 429/503/504/507/other 5xx - transient; a 507/503 also signals an
    /// allocation/eviction trigger.
    #[error("transient management error: {status} {body}")]
    Transient { status: u16, body: String },
    /// 409 duplicate name - tolerated during reconciliation.
    #[error("duplicate instance (409)")]
    Duplicate,
    /// Permanent 4xx (except 409) - no retry.
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

/// One instance as reported by `GET /instances` (the branch-spec envelope).
/// `state` is always `"loaded"` - an instance is either allocated (and
/// immediately ready) or deleted; there is no sleeping state. `model_bytes` is
/// the shared weights, reported on the first loaded instance only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub group: String,
    #[serde(default)]
    pub n_ctx: u64,
    #[serde(default)]
    pub parallel: u32,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub is_default: bool,
    /// Router-side preserve-on-evict flag: when set, the router snapshots this
    /// context's KV (and keeps its ledger transcript durable) before it drops
    /// the context to free VRAM, so a later request can resume it with
    /// `snapshot=<name>-resume`. Not understood by the fork - Coral Router
    /// tracks it and overlays it on the aggregate envelope.
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub model_bytes: u64,
    #[serde(default)]
    pub context_bytes: u64,
    #[serde(default)]
    pub compute_bytes: u64,
    #[serde(default)]
    pub total_bytes: u64,
    #[serde(default)]
    pub vram_bytes: u64,
    /// Most recent slot use (`-1` when never used).
    #[serde(default = "default_never_used")]
    pub last_used: i64,
}

const fn default_never_used() -> i64 {
    -1
}

/// The summed memory `total` object of `GET /instances`. `model` counts a
/// model's weights once regardless of how many of its instances are loaded.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstanceTotals {
    #[serde(default)]
    pub model: u64,
    #[serde(default)]
    pub context: u64,
    #[serde(default)]
    pub compute: u64,
    #[serde(default)]
    pub total: u64,
}

/// The full `GET /instances` envelope: the running instances, the pool's
/// on-disk KV snapshots (even when no instance is loaded), and the summed
/// memory. A cold pool returns `instances: []` and a zeroed `total`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceList {
    #[serde(default)]
    pub instances: Vec<InstanceInfo>,
    #[serde(default)]
    pub snapshots: Vec<SnapshotInfo>,
    #[serde(default)]
    pub total: InstanceTotals,
}

/// One snapshot entry from `GET /instances` or `GET /instances/:name/snapshots`.
/// Snapshots are per-instance on the fork (`<slot_save_path>/<model>/<instance>/`):
/// `instance` names the owning namespace (`None` for pre-scoping legacy flat
/// files, which read back for migration).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SnapshotInfo {
    pub name: String,
    #[serde(default)]
    pub instance: Option<String>,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub mtime: serde_json::Value,
    #[serde(default)]
    pub n_ctx_seq: u64,
}

/// Typed management client against one spawned `llama-server`'s `/instances`
/// API. Talks DIRECTLY to the pool's server - no router routing, so no `model`
/// field is carried (each server owns exactly one model's weights).
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

    /// `GET /instances` - the current instance set + snapshots + summed memory.
    /// Tolerates a bare array (pre-envelope forks) and a bare object.
    pub async fn list(&self) -> Result<InstanceList, InstanceError> {
        let value = self
            .request(reqwest::Method::GET, "/instances", None)
            .await?;
        if let Some(arr) = value.as_array() {
            let instances: Vec<InstanceInfo> =
                serde_json::from_value(Value::Array(arr.clone()))
                    .map_err(|e| InstanceError::Other(format!("list: {e}")))?;
            return Ok(InstanceList {
                instances,
                snapshots: vec![],
                total: InstanceTotals::default(),
            });
        }
        serde_json::from_value(value)
            .map_err(|e| InstanceError::Other(format!("list: {e}")))
    }

    /// `POST /instances` - allocate a NEW context from the shared weights.
    /// Only KV + compute are allocated; the model weights stay loaded.
    /// `default` marks the target of a bare `<base>` request. Returns the
    /// created instance on 201. A 2xx whose body is not a full instance object
    /// (an older fork returning `{}`) degrades to a synthesized record so the
    /// caller still has the instance identity.
    pub async fn create(
        &self,
        name: &str,
        group: &str,
        ctx_size: u64,
        parallel: Option<u32>,
        pinned: bool,
        is_default: bool,
    ) -> Result<InstanceInfo, InstanceError> {
        let mut body = serde_json::json!({
            "name": name,
            "group": group,
            "ctx_size": ctx_size,
            "pinned": pinned,
            "default": is_default,
        });
        if let Some(parallel) = parallel {
            body["parallel"] = Value::Number(parallel.into());
        }
        let value = self
            .request(reqwest::Method::POST, "/instances", Some(&body))
            .await?;
        Ok(serde_json::from_value::<InstanceInfo>(value).unwrap_or_else(|_| InstanceInfo {
            id: name.to_string(),
            aliases: vec![],
            group: group.to_string(),
            n_ctx: ctx_size,
            parallel: parallel.unwrap_or(1),
            pinned,
            is_default,
            resume: false,
            state: "loaded".into(),
            model_bytes: 0,
            context_bytes: 0,
            compute_bytes: 0,
            total_bytes: 0,
            vram_bytes: 0,
            last_used: -1,
        }))
    }

    /// `DELETE /instances/:name` - free KV + compute (the primary eviction
    /// path). `force` is accepted for compatibility and ignored (nothing
    /// enforces `pinned` in the branch).
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

    /// `POST /instances/:name/pin` - set the advisory `pinned` flag.
    pub async fn pin(&self, name: &str) -> Result<(), InstanceError> {
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{name}/pin"),
            None,
        )
        .await
        .map(|_| ())
    }

    /// `POST /instances/:name/unpin` - clear the advisory `pinned` flag.
    pub async fn unpin(&self, name: &str) -> Result<(), InstanceError> {
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{name}/unpin"),
            None,
        )
        .await
        .map(|_| ())
    }

    /// `POST /instances/:name/resize` - destroy and re-create the context at a
    /// new size (the current KV and any snapshot binding are discarded).
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

    /// `POST /instances/:name/snapshot` - save one slot's KV to a named
    /// snapshot in the instance's own namespace. Defaults to slot 0 (see
    /// `save_snapshot_slot` for an explicit slot).
    pub async fn save_snapshot(&self, instance: &str, name: &str) -> Result<(), InstanceError> {
        self.save_snapshot_slot(instance, name, 0).await
    }

    /// `POST /instances/:name/snapshot` with an explicit `id_slot`: save that
    /// slot's KV under `name` in the instance's namespace and bind the slot to
    /// it. The fork rejects an out-of-range slot loudly (400).
    pub async fn save_snapshot_slot(
        &self,
        instance: &str,
        name: &str,
        id_slot: i32,
    ) -> Result<(), InstanceError> {
        let body = serde_json::json!({ "name": name, "id_slot": id_slot });
        self.request(
            reqwest::Method::POST,
            &format!("/instances/{instance}/snapshot"),
            Some(&body),
        )
        .await
        .map(|_| ())
    }

    /// `GET /instances/:name/snapshots` - list the instance's snapshots. The
    /// server wraps them as `{"snapshots": [...]}`; a bare array is tolerated.
    /// Entries carry their owning `instance` (`None` for legacy flat files).
    pub async fn list_snapshots(&self, instance: &str) -> Result<Vec<SnapshotInfo>, InstanceError> {
        let value = self
            .request(
                reqwest::Method::GET,
                &format!("/instances/{instance}/snapshots"),
                None,
            )
            .await?;
        let arr = value.get("snapshots").cloned().unwrap_or(value);
        serde_json::from_value(arr)
            .map_err(|e| InstanceError::Other(format!("list_snapshots: {e}")))
    }

    /// `DELETE /instances/:name/snapshot/:snapshot` - remove a snapshot file.
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

    /// `GET /props` - the server's property envelope. Best-effort: `None` when
    /// the endpoint is unreachable or returns a non-2xx. A plain (no-instance
    /// grammar) server exposes no `/instances`, but `/props` still reports the
    /// context size and idle-sleep state Coral Router uses to synthesize its
    /// resident footprint.
    pub async fn props(&self) -> Option<Value> {
        self.request(reqwest::Method::GET, "/props", None).await.ok()
    }

    /// `POST /abort` - ask the server to stop the generation running in a
    /// slot. `id_slot` defaults to 0 (the default slot). The standard
    /// llama.cpp abort contract; the fork answers non-2xx (or an error JSON
    /// body) when no matching task is running, which surfaces as an
    /// [`InstanceError`] for the caller to log and ignore.
    pub async fn abort(&self, id_slot: Option<i32>) -> Result<(), InstanceError> {
        let body = serde_json::json!({ "id_slot": id_slot.unwrap_or(0) });
        self.request(reqwest::Method::POST, "/abort", Some(&body))
            .await
            .map(|_| ())
    }
}