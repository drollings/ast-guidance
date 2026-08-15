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

pub use client::{
    InstanceClient, InstanceError, InstanceInfo, InstanceList, InstanceTotals, SnapshotInfo,
};
pub use manager::{build_instance_managers, resume_snapshot_name, InstanceManager};
pub use pool::InstancePool;

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
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
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
mod tests {
    use super::stub::StubServer;
    use super::*;

    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use crate::config::InstanceProfile;

    use serde_json::Value;

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
            resume: false,
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
                resume: false,
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
            pinned: true,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        };
        // Expand the count as instance_profiles() would.
        let profiles: Vec<InstanceProfile> = (0..3)
            .map(|i| InstanceProfile {
                name: Some(format!("swarm-{i}")),
                group: Some("swarm".into()),
                ..swarm.clone()
            })
            .collect();
        assert_eq!(
            instance_grammar_string(&profiles),
            "swarm-0:group=swarm:ctx=16384:pinned,swarm-1:group=swarm:ctx=16384:pinned,swarm-2:group=swarm:ctx=16384:pinned"
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
            resume: false,
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
            // `sleep` is a sidecar eviction hint only - never emitted (the
            // branch has no auto-sleep).
            "ledger:ctx=131072:pinned:default,scratch:ctx=131072"
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
    fn pinned_emits_only_pinned_flag() {
        // A pinned profile with a positive declared sleep emits only `:pinned`
        // (sleep is never forwarded - it is a sidecar eviction hint).
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
    fn sleep_is_never_emitted() {
        // `sleep`/`no_sleep` are config eviction hints only; the grammar has no
        // sleep component (the minimal branch has no auto-sleep).
        let with_sleep = InstanceProfile {
            name: Some("a".into()),
            group: Some("a".into()),
            num_ctx: 0,
            sleep_idle_seconds: Some(5),
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[with_sleep]), "a");
        let warm = InstanceProfile {
            name: Some("b".into()),
            group: Some("b".into()),
            num_ctx: 0,
            no_sleep: true,
            ..profile("x", "x")
        };
        assert_eq!(instance_grammar_string(&[warm]), "b");
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

    fn sidecar_policy() -> crate::config::SidecarConfig {
        crate::config::SidecarConfig {
            poll_interval_s: 5,
            vram_low_watermark_bytes: 1024,
            evict_batch: 2,
            vram_total_bytes: Some(10000),
            minimum_remaining_vram: Some(2000),
            slot_save_path: Some("/srv/slots".into()),
            resume_ttl_s: None,
            api_key_env: None,
            liveness_poll_interval_s: 30,
            liveness_failures_before_restart: 3,
            max_restarts: 5,
        }
    }

    fn instance_info(id: &str, group: &str, pinned: bool, last_used: i64) -> InstanceInfo {
        InstanceInfo {
            id: id.into(),
            aliases: vec![],
            group: group.into(),
            n_ctx: 16384,
            parallel: 1,
            pinned,
            is_default: false,
            resume: false,
            state: "loaded".into(),
            model_bytes: 0,
            context_bytes: 262144,
            compute_bytes: 1048576,
            total_bytes: 1310720,
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
    async fn client_list_parses_envelope_and_bare_array() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [
                                instance_info("swarm0", "swarm", false, 5),
                                instance_info("ledger", "ledger", true, 1),
                            ],
                            "snapshots": [],
                            "total": { "model": 115343360, "context": 524288, "compute": 2097152, "total": 118226080 },
                        })
                        .to_string(),
                    )
                } else {
                    (404, r#"{"error":{"message":"not found"}}"#.into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        let list = client.list().await.expect("list");
        assert_eq!(list.instances.len(), 2);
        assert_eq!(list.instances[0].id, "swarm0");
        assert_eq!(list.instances[0].group, "swarm");
        assert_eq!(list.instances[1].pinned, true);
        assert_eq!(list.total.total, 118226080);
        assert_eq!(list.total.model, 115343360);
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
            .create("work", "swarm", 32768, Some(2), true, true)
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

        let dup = client.create("dup", "g", 16384, None, false, false).await;
        assert!(dup.unwrap_err().is_duplicate());

        let transient = client.destroy("boom", false).await.unwrap_err();
        assert!(transient.is_retryable());
        assert!(transient.is_evict_trigger());

        let rejected = client.destroy("missing", false).await.unwrap_err();
        assert!(matches!(rejected, InstanceError::Rejected { .. }));
    }

    #[tokio::test]
    async fn client_talks_directly_to_its_server() {
        // The client targets one spawned server directly: no `model` in the
        // create body and no `?model=` on per-instance ops (the llama.cpp
        // router mode is never used - Coral Router owns the processes).
        let seen: Arc<std::sync::Mutex<Vec<(String, String, String)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, body| {
                seen2.lock().unwrap().push((method.into(), path.into(), body.into()));
                (200, "{}".into())
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);

        client.create("swarm-0", "swarm", 16384, Some(1), false, false).await.unwrap();
        client.destroy("swarm-0", false).await.unwrap();
        client.resize("swarm-0", 32768).await.unwrap();

        let rec = seen.lock().unwrap();
        let create = rec
            .iter()
            .find(|(m, p, _)| m == "POST" && p == "/instances")
            .expect("create recorded");
        let body: Value = serde_json::from_str(&create.2).unwrap();
        assert!(
            body.get("model").is_none(),
            "no router-routing model in the create body"
        );
        assert!(
            rec.iter().all(|(_, p, _)| !p.contains("model=")),
            "no ?model= routing query on per-instance ops"
        );
    }

    #[tokio::test]
    async fn reconcile_creates_missing_pinned_and_resizes_n_ctx_drift() {
        // Server already has ledger (correct) and swarm0 (wrong n_ctx); swarm1
        // is missing entirely.
        let existing = [
            instance_info("ledger", "ledger", true, 1),
            instance_info("swarm0", "swarm", true, 5),
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

        // Profiles: ledger (pinned, n_ctx 131072), swarm0 + swarm1 (group swarm,
        // n_ctx 16384). swarm1 is unpinned -> deferred to on-demand creation.
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
            resume: false,
                params: None,
            },
            InstanceProfile {
                name: Some("swarm0".into()),
                group: Some("swarm".into()),
                count: 1,
                num_ctx: 16384,
                parallel: None,
                pinned: true,
                no_sleep: true,
                sleep_idle_seconds: None,
                default: false,
            resume: false,
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
            resume: false,
                params: None,
            },
        ];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        manager.reconcile().await.expect("reconcile");

        let recorded = stub.recorded();
        // One resize (swarm0 ctx drift). No POST: swarm1 is unpinned and
        // deferred, and every pinned profile already exists.
        let creates = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .count();
        let resizes = recorded
            .iter()
            .filter(|(_, p, _)| p.ends_with("/resize"))
            .count();
        assert_eq!(creates, 0, "unpinned swarm1 deferred to on-demand creation");
        assert_eq!(resizes, 1, "n_ctx drift triggers exactly one resize");
        assert_eq!(
            recorded
                .iter()
                .filter(|(_, p, _)| p.ends_with("/resize"))
                .map(|(_, p, _)| p.as_str())
                .next(),
            Some("/instances/ledger/resize"),
            "ledger's n_ctx drift (131072 profile vs 16384 present) is resized"
        );
    }

    #[tokio::test]
    async fn ensure_instance_creates_missing_unpinned_on_demand() {
        // Server is empty; `scratch` is configured but unpinned -> absent at
        // boot, created on demand by ensure_instance.
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, r#"{"instances":[],"snapshots":[],"total":{"total":0}}"#.into())
                } else {
                    (201, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("scratch".into()),
            group: Some("scratch".into()),
            count: 1,
            num_ctx: 131072,
            parallel: None,
            pinned: false,
            no_sleep: false,
            sleep_idle_seconds: Some(1),
            default: false,
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        manager.ensure_instance("scratch").await.expect("ensure_instance");

        let recorded = stub.recorded();
        let creates: Vec<&(String, String, String)> = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .collect();
        assert_eq!(creates.len(), 1, "scratch created on demand");
        let body: serde_json::Value = serde_json::from_str(&creates[0].2).unwrap();
        assert_eq!(body["name"], "scratch");
        assert_eq!(body["group"], "scratch");
    }

    #[tokio::test]
    async fn ensure_instance_skips_when_already_present_or_unknown() {
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, serde_json::to_string(&[instance_info("scratch", "scratch", false, 7)]).unwrap())
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("scratch".into()),
            group: Some("scratch".into()),
            count: 1,
            num_ctx: 131072,
            parallel: None,
            pinned: false,
            no_sleep: false,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        // Already present -> no create.
        manager.ensure_instance("scratch").await.expect("present");
        // Unknown name -> nothing to create, no error.
        manager.ensure_instance("nope").await.expect("unknown is a no-op");
        let recorded = stub.recorded();
        assert!(
            recorded
                .iter()
                .all(|(m, _, _)| m != "POST"),
            "no create when already present or unknown: {recorded:?}"
        );
    }

    #[tokio::test]
    async fn list_accepts_wrapped_instances_object() {
        // The fork returns GET /instances as {"instances":[...]}; list() must
        // unwrap it (it also tolerates a bare array).
        let existing = [
            instance_info("ledger", "ledger", true, 1),
            instance_info("swarm0", "swarm", false, 5),
        ];
        let payload = serde_json::json!({ "instances": existing });
        let payload_str = payload.to_string();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> =
            Arc::new(move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, payload_str.clone())
                } else {
                    (404, "{}".into())
                }
            });
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let list = client.list().await.expect("list parses wrapped shape");
        assert_eq!(list.instances.len(), 2);
        assert_eq!(list.instances[0].id, "ledger");
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
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        // A 409 during reconcile is tolerated - reconcile completes Ok.
        manager.reconcile().await.expect("reconcile tolerates 409");
    }

    #[tokio::test]
    async fn residency_evicts_lru_unpinned_and_never_pinned() {
        // Device budget = 10000 - 2000 = 8000; used 10000 -> over budget.
        let envelope = serde_json::json!({
            "instances": [
                instance_info("pinned", "g", true, 0),    // exempt
                instance_info("lru1", "g", false, 100),   // oldest unpinned
                instance_info("lru2", "g", false, 200),
                instance_info("recent", "g", false, 9000),
            ],
            "snapshots": [],
            "total": { "model": 5000, "context": 2500, "compute": 2500, "total": 10000 }
        });
        let stub = residency_stub(envelope);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");

        // evict_batch = 2 and two 1000-byte evictions are needed to reach the
        // budget (10000 -> 8000): the two oldest unpinned (lru1, lru2) are
        // deleted; pinned is never touched.
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
    async fn residency_eviction_frees_largest_lru_context_first() {
        // Two candidates with the same last_used: the one with more VRAM is
        // evicted first (freeing the most VRAM from the coldest context). A
        // pinned instance keeps the model's weights resident (no whole-model
        // candidate), isolating the context-level ordering.
        let mut big = instance_info("big", "g", false, 100);
        big.vram_bytes = 5000;
        let mut small = instance_info("small", "g", false, 100);
        small.vram_bytes = 1000;
        let envelope = serde_json::json!({
            "instances": [ small, big, instance_info("keep", "g", true, 0) ],
            "snapshots": [],
            "total": { "model": 5000, "context": 2000, "compute": 2000, "total": 9000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.evict_batch = 1;
        let manager = Arc::new(InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");

        let deletes: Vec<String> = stub
            .recorded()
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.clone())
            .collect();
        assert_eq!(deletes, vec!["/instances/big"], "largest LRU context evicted first");
    }

    #[tokio::test]
    async fn residency_no_eviction_when_free_vram_within_budget() {
        let envelope = serde_json::json!({
            "instances": [],
            "snapshots": [],
            "total": { "model": 1000, "context": 200, "compute": 200, "total": 1400 }
        });
        // used 1400 <= budget 8000 -> no eviction.
        let stub = residency_stub(envelope);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");
        assert!(
            stub.recorded()
                .iter()
                .all(|(m, _, _)| m != "DELETE"),
            "no eviction within budget"
        );
    }

    #[tokio::test]
    async fn residency_polls_without_budget_and_never_evicts() {
        // No vram_total_bytes and no minimum_remaining_vram -> no budget; the
        // pass must still GET /instances and report, but must never DELETE.
        let envelope = serde_json::json!({
            "instances": [],
            "snapshots": [],
            "total": { "model": 5000, "context": 2000, "compute": 2000, "total": 9000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = None;
        policy.minimum_remaining_vram = None;
        let manager = Arc::new(InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        // `residency_cycle` reads the ROCm sysfs VRAM total through the
        // capability-gated fs helper; grant `FsCapability` as the serving
        // path does.
        fluent_concurrency::scope::CURRENT_CAPS
            .scope(
                fluent_concurrency::capability::default_capability_set(),
                async { pool.residency_cycle().await.expect("residency") },
            )
            .await;
        let recorded = stub.recorded();
        assert!(
            recorded.iter().any(|(m, p, _)| m == "GET" && p == "/instances"),
            "instances are always polled, budget or not"
        );
        assert!(
            recorded.iter().all(|(m, _, _)| m != "DELETE"),
            "no eviction without a budget"
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
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
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

    #[tokio::test]
    async fn ensure_group_ready_is_noop_when_member_present() {
        // A resident member -> no management write, idempotent.
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [ instance_info("swarm-0", "swarm", true, 1) ],
                            "snapshots": [],
                            "total": { "model": 1000, "context": 200, "compute": 200, "total": 1400 }
                        })
                        .to_string(),
                    )
                } else {
                    (200, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("swarm".into()),
            group: Some("swarm".into()),
            count: 2,
            num_ctx: 16384,
            parallel: None,
            pinned: true,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        manager
            .ensure_group_ready("swarm")
            .await
            .expect("group already resident");
        let writes = stub
            .recorded()
            .into_iter()
            .filter(|(m, _, _)| m == "POST")
            .collect::<Vec<_>>();
        assert!(writes.is_empty(), "no allocation when a member exists");
    }

    #[tokio::test]
    async fn ensure_group_ready_reconciles_pinned_group_when_absent() {
        // The pinned `swarm` group has no resident member while the default
        // `ledger` instance is resident -> reconcile creates only the missing
        // pinned member and never re-creates a resident one.
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [ instance_info("ledger", "ledger", true, 0) ],
                            "snapshots": [],
                            "total": { "model": 1000, "context": 200, "compute": 200, "total": 1400 }
                        })
                        .to_string(),
                    )
                } else if method == "POST" && path == "/instances" {
                    (201, "{}".into())
                } else {
                    (404, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![
            InstanceProfile {
                name: Some("ledger".into()),
                group: Some("ledger".into()),
                count: 1,
                num_ctx: 131072,
                parallel: None,
                pinned: true,
                no_sleep: true,
                sleep_idle_seconds: None,
                default: true,
                resume: false,
                params: None,
            },
            InstanceProfile {
                name: Some("swarm".into()),
                group: Some("swarm".into()),
                count: 2,
                num_ctx: 16384,
                parallel: None,
                pinned: true,
                no_sleep: true,
                sleep_idle_seconds: None,
                default: false,
                resume: false,
                params: None,
            },
        ];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        manager
            .ensure_group_ready("swarm")
            .await
            .expect("reconcile creates the missing pinned member");
        let creates = stub
            .recorded()
            .into_iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .map(|(_, _, b)| serde_json::from_str::<serde_json::Value>(&b).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(creates.len(), 1, "reconcile creates only the missing pinned member");
        assert!(
            creates.iter().any(|b| b["name"] == "swarm" && b["group"] == "swarm"),
            "pinned swarm member created"
        );
        assert!(
            creates.iter().all(|b| b["name"] != "ledger"),
            "resident ledger member is not needlessly re-created"
        );
    }

    #[tokio::test]
    async fn ensure_group_ready_allocates_unpinned_group_when_absent() {
        // Unpinned group with no resident member -> a fresh member is allocated
        // on demand (the "even unpinned, load on demand" guarantee).
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [],
                            "snapshots": [],
                            "total": { "model": 1000, "context": 200, "compute": 200, "total": 1400 }
                        })
                        .to_string(),
                    )
                } else if method == "POST" && path == "/instances" {
                    (201, "{}".into())
                } else {
                    (404, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let client = InstanceClient::new(reqwest::Client::new(), stub.base_url(), None);
        let profiles = vec![InstanceProfile {
            name: Some("scratch".into()),
            group: Some("scratch".into()),
            count: 1,
            num_ctx: 131072,
            parallel: None,
            pinned: false,
            no_sleep: true,
            sleep_idle_seconds: None,
            default: false,
            resume: false,
            params: None,
        }];
        let manager = InstanceManager::new("base", client, profiles, sidecar_policy());
        manager
            .ensure_group_ready("scratch")
            .await
            .expect("unpinned group allocated on demand");
        let creates = stub
            .recorded()
            .into_iter()
            .filter(|(m, p, _)| m == "POST" && p == "/instances")
            .collect::<Vec<_>>();
        assert_eq!(creates.len(), 1, "one fresh member allocated");
    }

    #[tokio::test]
    async fn pool_aggregates_models_with_rewritten_ids() {
        use crate::instances::stub::StubServer;

        // Two managed models, each served by its own stub: `swarm` (ledger
        // default + scratch) and `qwen` (work). Envelopes use the server's own
        // (bare) instance ids and byte totals.
        let handler_a: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    let ledger = InstanceInfo {
                        is_default: true,
                        ..instance_info("ledger", "ledger", true, 1)
                    };
                    (
                        200,
                        serde_json::json!({
                            "instances": [
                                ledger,
                                instance_info("scratch", "scratch", false, 9),
                            ],
                            "snapshots": [ { "name": "readfiles", "size": 4194304 } ],
                            "total": { "model": 2428416000u64, "context": 2148925440u64, "compute": 2220361792u64, "total": 6797703232u64 },
                        })
                        .to_string(),
                    )
                } else {
                    (404, "{}".into())
                }
            },
        );
        let handler_b: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [ instance_info("work", "work", false, 3) ],
                            "snapshots": [],
                            "total": { "model": 5000000000u64, "context": 1000, "compute": 1000, "total": 5000002000u64 },
                        })
                        .to_string(),
                    )
                } else {
                    (404, "{}".into())
                }
            },
        );
        let stub_a = StubServer::start(handler_a);
        let stub_b = StubServer::start(handler_b);
        let mut managers = HashMap::new();
        managers.insert(
            "swarm".into(),
            Arc::new(InstanceManager::new(
                "swarm",
                InstanceClient::new(reqwest::Client::new(), stub_a.base_url(), None),
                Vec::new(),
                sidecar_policy(),
            )),
        );
        managers.insert(
            "qwen".into(),
            Arc::new(InstanceManager::new(
                "qwen",
                InstanceClient::new(reqwest::Client::new(), stub_b.base_url(), None),
                Vec::new(),
                sidecar_policy(),
            )),
        );
        let pool = InstancePool::from_managers(managers, None);

        let agg = pool.aggregate(None).await.expect("aggregate");
        let instances = agg["instances"].as_array().unwrap();
        assert_eq!(instances.len(), 3);
        let ids: Vec<&str> = instances
            .iter()
            .filter_map(|i| i["id"].as_str())
            .collect();
        assert!(ids.contains(&"swarm:ledger"), "ids: {ids:?}");
        assert!(ids.contains(&"swarm:scratch"));
        assert!(ids.contains(&"qwen:work"));
        // The default instance's aliases carry the bare model id + latest.
        let ledger = instances.iter().find(|i| i["id"] == "swarm:ledger").unwrap();
        let aliases: Vec<&str> = ledger["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|a| a.as_str())
            .collect();
        assert!(aliases.contains(&"swarm"));
        assert!(aliases.contains(&"swarm:latest"));
        assert!(aliases.contains(&"swarm:ledger"));
        // Snapshots tagged with the owning model; totals summed with 64-bit
        // arithmetic (each model's weights counted once).
        let snaps = agg["snapshots"].as_array().unwrap();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0]["model"], "swarm");
        let total = &agg["total"];
        assert_eq!(total["model"], 2428416000u64 + 5000000000u64);
        assert_eq!(total["total"], 6797703232u64 + 5000002000u64);
    }

    #[tokio::test]
    async fn pool_scopes_aggregate_to_one_model() {
        use crate::instances::stub::StubServer;

        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (
                        200,
                        serde_json::json!({
                            "instances": [ instance_info("ledger", "ledger", true, 1) ],
                            "snapshots": [],
                            "total": { "model": 1, "context": 1, "compute": 1, "total": 3 },
                        })
                        .to_string(),
                    )
                } else {
                    (404, "{}".into())
                }
            },
        );
        let stub = StubServer::start(handler);
        let mut managers = HashMap::new();
        managers.insert(
            "swarm".into(),
            Arc::new(InstanceManager::new(
                "swarm",
                InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
                Vec::new(),
                sidecar_policy(),
            )),
        );
        let pool = InstancePool::from_managers(managers, None);
        let agg = pool.aggregate(Some("swarm")).await.expect("scoped aggregate");
        assert_eq!(agg["instances"].as_array().unwrap().len(), 1);
        let unknown = pool.aggregate(Some("nope")).await.expect("unknown scope");
        assert_eq!(unknown["instances"].as_array().unwrap().len(), 0);
        assert!(pool.resolve_instance_id("swarm:ledger").is_some());
        assert!(pool.resolve_instance_id("swarm:ledger:x").is_none());
        assert!(pool.resolve_instance_id("nope:x").is_none());
    }

    /// Build a stub server that serves a fixed `/instances` envelope and
    /// records destroys; DELETE answers 200.
    fn residency_stub(envelope: serde_json::Value) -> StubServer {
        let envelope = envelope.to_string();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/instances" {
                    (200, envelope.clone())
                } else {
                    (200, "{}".into())
                }
            },
        );
        StubServer::start(handler)
    }

    fn manager_for_stub(stub: &StubServer) -> Arc<InstanceManager> {
        Arc::new(InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            sidecar_policy(),
        ))
    }

    #[tokio::test]
    async fn pool_residency_evicts_lru_largest_unpinned_across_managers() {
        // Device budget = 10000 (vram_total) - 2000 (minimum_remaining) = 8000.
        // Both managers together report used 10000 -> over budget. The coldest
        // largest unpinned context (old) is evicted first; pinned never is.
        let env_a = serde_json::json!({
            "instances": [
                instance_info("pinned", "g", true, 0),    // exempt
                instance_info("old", "g", false, 100),    // LRU, vram 1000
                instance_info("big", "g", false, 200),    // larger vram 1000
            ],
            "snapshots": [],
            "total": { "model": 5000, "context": 2500, "compute": 2500, "total": 10000 }
        });
        let env_b = serde_json::json!({
            "instances": [],
            "snapshots": [],
            "total": { "model": 0, "context": 0, "compute": 0, "total": 0 }
        });
        let stub_a = residency_stub(env_a);
        let stub_b = residency_stub(env_b);
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager_for_stub(&stub_a));
        managers.insert("other".into(), manager_for_stub(&stub_b));
        let pool = InstancePool::from_managers(managers, None);

        pool.residency_cycle().await.expect("residency");

        let deletes_a = stub_a
            .recorded()
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.clone())
            .collect::<Vec<_>>();
        // evict_batch = 2 (sidecar_policy) -> both unpinned candidates go,
        // ordered LRU (old) before big; pinned never is.
        assert_eq!(deletes_a.len(), 2, "evict_batch = 2 per pass");
        assert!(deletes_a[0].ends_with("/old"), "LRU evicted first: {deletes_a:?}");
        assert!(
            !deletes_a.iter().any(|p| p.contains("pinned")),
            "pinned instance never evicted"
        );
        assert!(
            stub_b.recorded().iter().all(|(m, _, _)| m != "DELETE"),
            "other manager has no unpinned candidates"
        );
    }

    #[tokio::test]
    async fn pool_residency_no_eviction_within_budget() {
        // used 5000 <= budget 8000 -> no eviction.
        let env = serde_json::json!({
            "instances": [ instance_info("warm", "g", false, 1) ],
            "snapshots": [],
            "total": { "model": 4000, "context": 500, "compute": 500, "total": 5000 }
        });
        let stub = residency_stub(env);
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");
        assert!(
            stub.recorded().iter().all(|(m, _, _)| m != "DELETE"),
            "no eviction within budget"
        );
    }

    #[tokio::test]
    async fn pool_residency_without_budget_never_evicts() {
        let mut policy = sidecar_policy();
        policy.minimum_remaining_vram = None;
        policy.vram_total_bytes = None;
        let env = serde_json::json!({
            "instances": [ instance_info("warm", "g", false, 1) ],
            "snapshots": [],
            "total": { "model": 4000, "context": 500, "compute": 500, "total": 5000 }
        });
        let stub = residency_stub(env);
        let manager = Arc::new(InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        // `residency_cycle` reads the ROCm sysfs VRAM total via the gated
        // fs helper; grant `FsCapability` as the serving path does.
        fluent_concurrency::scope::CURRENT_CAPS
            .scope(
                fluent_concurrency::capability::default_capability_set(),
                async { pool.residency_cycle().await.expect("residency") },
            )
            .await;
        assert!(
            stub.recorded().iter().all(|(m, _, _)| m != "DELETE"),
            "no budget -> no eviction"
        );
    }

    #[test]
    fn build_instance_managers_rejects_duplicate_name_within_model() {
        // Two profiles in ONE model resolve to the same instance name: the
        // profile key `swarm0` and another profile whose explicit `name` is
        // also `swarm0`. The pool grammar is invalid and boot fails fast.
        let config: crate::config::RouterConfig =
            serde_json::from_value(serde_json::json!({
                "models": {
                    "a": {
                        "endpoint": "http://x/v1/chat/completions",
                        "intelligence": 1,
                        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 1,
                        "instances": {
                            "swarm0": { "num_ctx": 16384 },
                            "x": { "name": "swarm0", "num_ctx": 32768 }
                        }
                    }
                }
            }))
            .unwrap();
        let err = match build_instance_managers(&config, None) {
            Err(e) => e,
            Ok(_) => panic!("duplicate-name config must fail validation"),
        };
        assert!(
            err.contains("duplicate instance name 'swarm0'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn residency_backoff_progresses_and_caps() {
        // The residency/bootstrap poll backoff is `common_core::retry::
        // capped_backoff_ms` (base * min(failures+1, cap)). Healthy -> base
        // interval; first failure -> 2x; second -> 3x; ... capped at 12x.
        use common_core::retry::capped_backoff_ms;
        let base_ms = Duration::from_secs(5).as_millis() as u64;
        assert_eq!(capped_backoff_ms(base_ms, 0, 12), 5_000);
        assert_eq!(capped_backoff_ms(base_ms, 1, 12), 10_000);
        assert_eq!(capped_backoff_ms(base_ms, 2, 12), 15_000);
        assert_eq!(capped_backoff_ms(base_ms, 11, 12), 60_000, "cap at 12x base");
        assert_eq!(capped_backoff_ms(base_ms, 100, 12), 60_000, "capped regardless of further failures");
    }

    #[test]
    fn build_instance_managers_ok_on_valid_config() {
        let config: crate::config::RouterConfig =
            serde_json::from_value(serde_json::json!({
                "models": {
                    "a": {
                        "endpoint": "http://x/v1/chat/completions",
                        "intelligence": 1,
                        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 1,
                        "instances": { "swarm": { "num_ctx": 16384, "count": 2 } }
                    }
                }
            }))
            .unwrap();
        let pool = build_instance_managers(&config, None).expect("valid config builds managers");
        assert_eq!(pool.managers_iter().len(), 1);
        // Keyed by the Coral Router model id.
        assert!(pool.manager("a").is_some());
    }

    #[test]
    fn build_instance_managers_is_empty_without_instances() {
        let config: crate::config::RouterConfig =
            serde_json::from_value(serde_json::json!({
                "models": {
                    "a": {
                        "endpoint": "http://x/v1/chat/completions",
                        "intelligence": 1,
                        "cost_input": 0.0, "cost_output": 0.0, "cost_cached_read": 0.0, "speed": 1,
                    }
                }
            }))
            .unwrap();
        let pool = build_instance_managers(&config, None).expect("no managers");
        assert!(pool.is_empty());
    }

    // -- plain (no-instance-grammar) model footprint --------------------------

    /// A stub that 404s `/instances` (the fork's behavior for a server started
    /// without `--instance` grammar) and answers `/props`.
    fn plain_stub(props: serde_json::Value) -> StubServer {
        let props = props.to_string();
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> = Arc::new(
            move |method, path, _body| {
                if method == "GET" && path == "/props" {
                    (200, props.clone())
                } else {
                    (404, r#"{"error":{"message":"File Not Found"}}"#.into())
                }
            },
        );
        StubServer::start(handler)
    }

    fn plain_manager(stub: &StubServer, weights: u64) -> Arc<InstanceManager> {
        Arc::new(
            InstanceManager::new(
                "qwen",
                InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
                Vec::new(),
                sidecar_policy(),
            )
            .with_weights_bytes(weights),
        )
    }

    #[tokio::test]
    async fn plain_model_footprint_reports_weights_when_awake() {
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 16384 }
        });
        let stub = plain_stub(props);
        let manager = plain_manager(&stub, 10_000_000_000);
        let (envelope, plain) = manager.list_with_fallback().await.expect("fallback");
        assert!(plain, "a 404 on /instances is synthesized");
        assert_eq!(envelope.instances.len(), 1);
        let inst = &envelope.instances[0];
        assert_eq!(inst.id, "qwen:default");
        assert_eq!(inst.state, "loaded");
        assert_eq!(inst.model_bytes, 10_000_000_000);
        assert_eq!(inst.n_ctx, 16384);
        assert_eq!(envelope.total.model, 10_000_000_000);
        assert_eq!(envelope.total.total, 10_000_000_000);
    }

    #[tokio::test]
    async fn plain_model_footprint_zeroes_weights_when_sleeping() {
        let props = serde_json::json!({
            "is_sleeping": true,
            "default_generation_settings": { "n_ctx": 16384 }
        });
        let stub = plain_stub(props);
        let manager = plain_manager(&stub, 10_000_000_000);
        let (envelope, plain) = manager.list_with_fallback().await.expect("fallback");
        assert!(plain);
        let inst = &envelope.instances[0];
        assert_eq!(inst.state, "sleeping");
        assert_eq!(inst.model_bytes, 0, "sleeping plain model freed its weights");
        assert_eq!(envelope.total.total, 0);
    }

    #[tokio::test]
    async fn plain_model_footprint_none_when_server_down() {
        // A down (never-loaded) plain server: /props is unreachable -> None.
        let handler: Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync> =
            Arc::new(|_m, _p, _b| (404, "{}".into()));
        let stub = StubServer::start(handler);
        let manager = plain_manager(&stub, 1_000);
        assert!(manager.list_with_fallback().await.is_none());
    }

    #[tokio::test]
    async fn aggregate_includes_plain_model_footprint() {
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let mut managers = HashMap::new();
        managers.insert("qwen".into(), plain_manager(&stub, 5_000_000_000));
        let pool = InstancePool::from_managers(managers, None);

        let agg = pool.aggregate(None).await.expect("aggregate");
        let instances = agg["instances"].as_array().unwrap();
        assert_eq!(instances.len(), 1);
        let entry = &instances[0];
        assert_eq!(entry["id"], "qwen:default");
        assert_eq!(entry["model_bytes"], 5_000_000_000u64);
        let aliases: Vec<&str> = entry["aliases"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|a| a.as_str())
            .collect();
        assert!(aliases.contains(&"qwen"), "aliases: {aliases:?}");
        assert_eq!(agg["total"]["model"], 5_000_000_000u64);
        assert_eq!(agg["total"]["total"], 5_000_000_000u64);

        let scoped = pool.aggregate(Some("qwen")).await.expect("scoped");
        assert_eq!(scoped["instances"].as_array().unwrap().len(), 1);
        let unknown = pool.aggregate(Some("nope")).await.expect("unknown scope");
        assert_eq!(unknown["instances"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn list_models_includes_plain_model_entry() {
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let mut managers = HashMap::new();
        managers.insert("qwen".into(), plain_manager(&stub, 5_000_000_000));
        let pool = InstancePool::from_managers(managers, None);

        let models = pool.list_models().await;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["id"], "qwen:default");
        assert_eq!(models[0]["state"], "loaded");
    }

    #[tokio::test]
    async fn touch_advances_plain_model_last_used() {
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let manager = plain_manager(&stub, 1_000);
        let before = manager.plain_footprint().await.expect("footprint").last_used;
        manager.touch();
        let after = manager.plain_footprint().await.expect("footprint").last_used;
        assert!(after >= before, "touch must advance last_used");
    }

    #[tokio::test]
    async fn residency_polls_plain_models_and_survives_without_supervisor() {
        // A plain model awake at 10_000 bytes, budget 2000: over budget. With
        // no supervisor the plain-model unload is a no-op break; the pass must
        // still complete Ok and poll the plain server's /props.
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = Some(4000);
        policy.minimum_remaining_vram = Some(2000);
        let manager = Arc::new(
            InstanceManager::new(
                "qwen",
                InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
                Vec::new(),
                policy,
            )
            .with_weights_bytes(10_000),
        );
        let mut managers = HashMap::new();
        managers.insert("qwen".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency completes without supervisor");
        assert!(
            stub.recorded().iter().any(|(m, p, _)| m == "GET" && p == "/props"),
            "the plain-model branch must poll /props"
        );
    }

    // -- load-time admission control (make_room_for) -------------------------

    #[tokio::test]
    async fn make_room_for_no_eviction_within_budget() {
        // swarm holds one unpinned instance (used 3000); loading gemma (1000)
        // projects 4000 <= budget 8000 -> nothing evicted.
        let swarm_envelope = serde_json::json!({
            "instances": [ instance_info("scratch", "scratch", false, 5) ],
            "snapshots": [],
            "total": { "model": 2000, "context": 500, "compute": 500, "total": 3000 }
        });
        let swarm_stub = residency_stub(swarm_envelope);
        let gemma_stub = plain_stub(serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        }));
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager_for_stub(&swarm_stub));
        managers.insert("gemma".into(), plain_manager(&gemma_stub, 1000));
        let pool = InstancePool::from_managers(managers, None);
        pool.make_room_for("gemma", 1000).await;
        assert!(
            swarm_stub
                .recorded()
                .iter()
                .all(|(m, _, _)| m != "DELETE"),
            "no eviction within budget"
        );
    }

    #[tokio::test]
    async fn make_room_for_evicts_unpinned_instance_over_budget() {
        // Budget 4000 - 2000 = 2000; used 3000 + gemma 1000 = 4000 -> over.
        // The only freeable chunk is the unpinned `scratch` instance.
        let envelope = serde_json::json!({
            "instances": [ instance_info("scratch", "scratch", false, 5) ],
            "snapshots": [],
            "total": { "model": 2000, "context": 500, "compute": 500, "total": 3000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = Some(4000);
        let manager = Arc::new(InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.make_room_for("gemma", 1000).await;
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(
            deletes.contains(&"/instances/scratch"),
            "unpinned instance evicted to make room: {deletes:?}"
        );
    }

    #[tokio::test]
    async fn make_room_for_excludes_target_and_survives_without_supervisor() {
        // qwen (plain, awake, 10_000) resident; gemma (plain, 7_000) is the
        // cold target. Budget 2000 -> over budget. Without a supervisor the
        // plain unload is a no-op break; the pass must complete Ok and never
        // poll the excluded target.
        let props = serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let qwen_stub = plain_stub(props.clone());
        let gemma_stub = plain_stub(props);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = Some(4000);
        let qwen = Arc::new(
            InstanceManager::new(
                "qwen",
                InstanceClient::new(reqwest::Client::new(), qwen_stub.base_url(), None),
                Vec::new(),
                policy.clone(),
            )
            .with_weights_bytes(10_000),
        );
        let gemma = Arc::new(
            InstanceManager::new(
                "gemma",
                InstanceClient::new(reqwest::Client::new(), gemma_stub.base_url(), None),
                Vec::new(),
                policy,
            )
            .with_weights_bytes(7_000),
        );
        let mut managers = HashMap::new();
        managers.insert("qwen".into(), qwen);
        managers.insert("gemma".into(), gemma);
        let pool = InstancePool::from_managers(managers, None);
        pool.make_room_for("gemma", 7_000).await;
        assert!(
            gemma_stub.recorded().is_empty(),
            "the cold target must be excluded from the gather: {:?}",
            gemma_stub.recorded()
        );
    }

    #[tokio::test]
    async fn is_sleeping_reflects_fork_state_and_skips_instance_models() {
        let props = serde_json::json!({
            "is_sleeping": true,
            "default_generation_settings": { "n_ctx": 8192 }
        });
        let stub = plain_stub(props);
        let manager = plain_manager(&stub, 1000);
        assert_eq!(manager.is_sleeping().await, Some(true), "fork reports sleeping");

        let stub2 = plain_stub(serde_json::json!({
            "is_sleeping": false,
            "default_generation_settings": { "n_ctx": 8192 }
        }));
        let manager2 = plain_manager(&stub2, 1000);
        assert_eq!(manager2.is_sleeping().await, Some(false), "awake");

        // Instance models never poll /props: their pinned contexts keep the
        // weights resident, so the answer is always Some(false).
        let stub3 = plain_stub(serde_json::json!({ "is_sleeping": true }));
        let instance_manager = InstanceManager::new(
            "swarm",
            InstanceClient::new(reqwest::Client::new(), stub3.base_url(), None),
            vec![profile("ledger", "ledger")],
            sidecar_policy(),
        );
        assert_eq!(instance_manager.is_sleeping().await, Some(false));
        assert!(
            stub3.recorded().is_empty(),
            "instance models must not poll /props: {:?}",
            stub3.recorded()
        );
    }

    // -- resume (preserve-on-evict) ------------------------------------------

    #[test]
    fn eviction_score_weights_size_and_coldness() {
        use common_core::cache::eviction_score;
        let now = 1_000_000i64;
        // Equal coldness: the larger footprint scores higher (evicted first) -
        // a 10 GB weight pool outranks a context buffer of any recency within
        // reach, which is the OOM-avoidance priority.
        assert!(eviction_score(10_000_000_000, 100, now) > eviction_score(1_000, 100, now));
        assert!(eviction_score(10_000_000_000, now - 1, now) > eviction_score(1_000, now - 3600, now));
        // Equal size: the colder (older last_used) scores higher.
        assert!(eviction_score(1_000, 50, now) > eviction_score(1_000, 900, now));
        // Coldness scales within a size class: the same 10 GB pool idle a
        // minute is far more evictable than when used a second ago, so active
        // work is relatively protected by recency.
        assert!(eviction_score(10_000_000_000, now - 60, now) > eviction_score(10_000_000_000, now - 1, now));
        // Never used = maximally cold.
        assert!(eviction_score(1_000, -1, now) > eviction_score(1_000, 1, now));
    }

    #[test]
    fn resume_flag_round_trips_through_profiles() {
        // A profile with `resume: true` seeds the manager's map so the
        // aggregate and eviction see it.
        let manager = InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), "http://x", None),
            vec![
                InstanceProfile {
                    resume: true,
                    ..profile("agent", "g")
                },
                profile("scratch", "g"),
            ],
            sidecar_policy(),
        );
        assert!(manager.resume_for("agent"));
        assert!(!manager.resume_for("scratch"));
        manager.set_resume("scratch", true);
        assert!(manager.resume_for("scratch"));
        manager.set_resume("agent", false);
        assert!(!manager.resume_for("agent"));
    }

    #[tokio::test]
    async fn aggregate_reports_resume_overlay() {
        let envelope = serde_json::json!({
            "instances": [ instance_info("agent", "g", false, 1) ],
            "snapshots": [],
            "total": { "model": 1000, "context": 500, "compute": 500, "total": 2000 }
        });
        let stub = residency_stub(envelope);
        let manager = manager_for_stub(&stub);
        manager.set_resume("agent", true);
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        let agg = pool.aggregate(None).await.expect("aggregate");
        let entry = agg["instances"]
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i["id"] == "swarm:agent")
            .expect("aggregated entry");
        assert_eq!(entry["resume"], true, "aggregate overlays the router-side flag");
    }

    #[tokio::test]
    async fn eviction_snapshots_resume_context_before_destroy() {
        // Budget 2000; over budget by a resume-marked unpinned context with a
        // pinned sibling keeping the weights resident. The resume context is
        // snapshotted (`POST .../agent/snapshot`) before it is destroyed, and
        // the pinned sibling is never touched.
        let mut agent = instance_info("agent", "g", false, 100);
        agent.vram_bytes = 2000;
        let envelope = serde_json::json!({
            "instances": [ instance_info("keep", "g", true, 0), agent ],
            "snapshots": [],
            "total": { "model": 5000, "context": 3000, "compute": 3000, "total": 11000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.vram_total_bytes = Some(4000); // budget = 4000 - 2000 = 2000
        let manager = Arc::new(InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        manager.set_resume("agent", true);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");

        let recorded = stub.recorded();
        let snapshot_posts: Vec<&str> = recorded
            .iter()
            .filter(|(m, p, _)| m == "POST" && *p == "/instances/agent/snapshot")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert_eq!(snapshot_posts.len(), 1, "resume context snapshotted: {recorded:?}");
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(deletes.contains(&"/instances/agent"), "resume context evicted");
        assert!(
            !deletes.iter().any(|p| p.contains("keep")),
            "pinned context never evicted"
        );
    }

    #[tokio::test]
    async fn expire_resume_clears_idle_context_and_deletes_snapshot() {
        // `resume_ttl_s = 60`: an ancient (idle) resume context has its flag
        // cleared and its `-resume` snapshot deleted - the router concluding
        // the work is done. Within budget, so no eviction happens.
        let envelope = serde_json::json!({
            "instances": [
                instance_info("agent", "g", false, 100), // idle ~50 years
                instance_info("keep", "g", true, 0),
            ],
            "snapshots": [],
            "total": { "model": 1000, "context": 500, "compute": 500, "total": 2000 }
        });
        let stub = residency_stub(envelope);
        let mut policy = sidecar_policy();
        policy.resume_ttl_s = Some(60);
        let manager = Arc::new(InstanceManager::new(
            "base",
            InstanceClient::new(reqwest::Client::new(), stub.base_url(), None),
            Vec::new(),
            policy,
        ));
        manager.set_resume("agent", true);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager);
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");

        assert!(
            !pool.manager("base").unwrap().resume_for("agent"),
            "idle resume cleared"
        );
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(
            deletes.contains(&"/instances/agent/snapshot/agent-resume"),
            "resume snapshot deleted on expiry: {deletes:?}"
        );
    }

    #[tokio::test]
    async fn set_resume_false_deletes_snapshot() {
        let envelope = serde_json::json!({
            "instances": [ instance_info("agent", "g", false, 1) ],
            "snapshots": [],
            "total": { "model": 1000, "context": 500, "compute": 500, "total": 2000 }
        });
        let stub = residency_stub(envelope);
        let mut managers = HashMap::new();
        managers.insert("swarm".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.set_resume("swarm", "agent", true).await.expect("enable");
        assert!(pool.manager("swarm").unwrap().resume_for("agent"));
        pool.set_resume("swarm", "agent", false).await.expect("disable");
        assert!(!pool.manager("swarm").unwrap().resume_for("agent"));
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(
            deletes.contains(&"/instances/agent/snapshot/agent-resume"),
            "disable deletes the resume snapshot: {deletes:?}"
        );
    }

    #[tokio::test]
    async fn whole_model_is_largest_footprint_candidate() {
        // A model with NO pinned instances is a whole-model candidate: its
        // weights + all contexts. Without a supervisor the model eviction
        // still drops every context before breaking; the point is that the
        // whole-model unit outranks the individual contexts.
        let envelope = serde_json::json!({
            "instances": [
                instance_info("ctx-a", "g", false, 100),
                instance_info("ctx-b", "g", false, 200),
            ],
            "snapshots": [],
            "total": { "model": 8000, "context": 2000, "compute": 2000, "total": 12000 }
        });
        let stub = residency_stub(envelope);
        let mut managers = HashMap::new();
        managers.insert("base".into(), manager_for_stub(&stub));
        let pool = InstancePool::from_managers(managers, None);
        pool.residency_cycle().await.expect("residency");
        let recorded = stub.recorded();
        let deletes: Vec<&str> = recorded
            .iter()
            .filter(|(m, _, _)| m == "DELETE")
            .map(|(_, p, _)| p.as_str())
            .collect();
        assert!(
            deletes.contains(&"/instances/ctx-a") && deletes.contains(&"/instances/ctx-b"),
            "whole-model eviction drops every context: {deletes:?}"
        );
    }
}
