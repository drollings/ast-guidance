//! Instance-pool grammar generation, management client, sidecar, and the
//! router's aggregate `/instances` facade.
//!
//! Coral Router is the process owner of one `llama-server` per model weights
//! file (see `supervisor`): it spawns each server on a free localhost port and
//! talks to it DIRECTLY - the llama.cpp router mode is never used. This module
//! declares the instance pool it hands each server as `--instance` grammar
//! (matching `common_instances_parse`/`common_instances_to_string`), wraps the
//! server's management API (`/instances`), and hosts the sidecar:
//!
//! - [`InstanceClient`] - one server's `/instances` management API over raw
//!   reqwest, with `HttpClass`-classified errors.
//! - [`InstanceManager`] - boot reconciliation, the residency loop (LRU
//!   eviction of unpinned instances when free VRAM is low), and
//!   allocate-on-503.
//! - [`InstancePool`] - the router's public `/instances` facade: aggregates
//!   every managed server's envelope under `<model_id>:<instance_name>` ids,
//!   sums `total` with 64-bit arithmetic, and proxies per-model operations.

pub mod api;
pub mod client;
pub mod manager;
pub mod pool;
pub mod traits;

pub use client::{
    InstanceClient, InstanceError, InstanceInfo, InstanceList, InstanceTotals, SnapshotInfo,
};
pub use manager::{build_instance_managers, resume_snapshot_name, weights_identity, InstanceManager, WeightsIdentity};
pub use pool::InstancePool;
pub use traits::{LlamaBackend, LlamaContext, LlamaKVCache, LlamaWeights, LlmFleet};

use crate::config::InstanceProfile;

/// Render a flat list of (expanded) `InstanceProfile`s as the fork's
/// comma-joined `--instance` grammar, matching `common_instances_to_string`
/// byte-for-byte for equivalent inputs.
///
/// Grammar (the minimal branch spec): `name[:group=G][:ctx=N][:parallel=M]
/// [:pinned][:default]`. There is no sleep component: the branch has no
/// auto-sleep, so `sleep_idle_seconds` never reaches the server - it is an
/// eviction-priority hint the sidecar reads from config.
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
        s.push_str(":pinned");
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

/// Whether a string is a valid instance/snapshot name (`[A-Za-z0-9._-]+`).
pub fn is_valid_instance_name(name: &str) -> bool {
    fluent_types::instance_id::is_valid_instance_name(name)
}

/// The alias set for one public instance id: the bare model id and `latest`
/// form on the default instance, the group form, and the exact id.
fn instance_aliases(model_key: &str, instance_id: &str, group: &str, is_default: bool) -> Vec<String> {
    let mut aliases = Vec::new();
    if is_default {
        aliases.push(model_key.to_string());
        aliases.push(format!("{model_key}:latest"));
    }
    aliases.push(format!("{model_key}:{group}"));
    aliases.push(instance_id.to_string());
    aliases.sort();
    aliases.dedup();
    aliases
}

/// A tiny HTTP/1.1 stub that records every request and answers from a shared
/// handler closure `(method, path, body) -> (status, body)`. Used to exercise
/// `InstanceClient`/`InstanceManager` (and the dispatch allocate-on-503 path)
/// against fixture `/instances`, `/memory`, and `/chat/completions` JSON

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
#[path = "../../tests/instances_mod.rs"]
mod tests;
#[cfg(test)]
#[path = "../../tests/residency_engine_golden.rs"]
mod residency_engine_golden_tests;
