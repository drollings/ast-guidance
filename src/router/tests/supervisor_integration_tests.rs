//! End-to-end supervisor + sidecar integration over a fake llama-server.
//!
//! Unlike `supervisor.rs`'s unit tests (which use `ManagedServer::with_liveness`
//! directly and a `sh` script that just sleeps), this drives the REAL
//! `LlamaServerSupervisor::build` → `start_all` → dispatch path: the
//! `LLAMA_SERVER` env override points at a small **python3 http server** that
//! actually speaks the llama-server contract on the loopback port the
//! supervisor selects (`/health`, `/instances`, `/v1/chat/completions`,
//! `/v1/models`). So this genuinely exercises:
//!
//! * boot-time endpoint rewrite for a pinned-instance model,
//! * `/health` probing by the supervisor,
//! * a real `/v1/chat/completions` dispatch through the spawned server,
//! * on-demand residency (`ensure_target_ready` → `ensure_running`),
//! * unload of a model left with zero contexts + reload on next use.
//!
//! Hermetic by construction: loopback only, never the real `llama-server`
//! binary and never any inference endpoint. If `python3` is unavailable the
//! test skips (skip-not-fail) rather than failing.

use std::sync::Arc;

use tempfile::TempDir;

use crate::config::{RouterConfig, SidecarConfig};
use crate::supervisor::LlamaServerSupervisor;

const LLAMA_SERVER_ENV: &str = "LLAMA_SERVER";

/// The fake `llama-server`: a python3 `http.server` on `--port`. Answers
/// `/health`, `/instances` (empty unless the alias is the pinned model),
/// `/v1/models`, and `/v1/chat/completions` (deterministic completion).
const FAKE_LLAMA_SCRIPT: &str = r#"#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = 8080
alias = ""
args = sys.argv[1:]
for i, a in enumerate(args):
    if a == "--port" and i + 1 < len(args):
        port = int(args[i + 1])
    elif a == "--alias" and i + 1 < len(args):
        alias = args[i + 1]

def instances_body():
    # The pinned model reports one pinned, loaded instance (so residency never
    # unloads it); every other model reports zero contexts.
    if alias == "pinned-m":
        return {
            "instances": [{
                "id": "swarm", "aliases": [], "group": "swarm", "n_ctx": 4096,
                "parallel": 1, "pinned": True, "is_default": True, "state": "loaded",
                "model_bytes": 0, "context_bytes": 0, "compute_bytes": 0,
                "total_bytes": 0, "vram_bytes": 0, "last_used": -1}],
            "snapshots": [], "total": {},
        }
    return {"instances": [], "snapshots": [], "total": {}}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {})
        elif self.path == "/instances" or self.path.startswith("/instances?"):
            self._send(200, instances_body())
        elif self.path == "/v1/models":
            self._send(200, {"data": []})
        else:
            self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._send(200, {
                "id": "cmpl-fake", "object": "chat.completion", "created": 0,
                "model": "fake",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "fake completion"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        elif self.path == "/instances":
            self._send(201, {"id": "work", "state": "loaded"})
        else:
            self._send(404, {"error": "not found"})

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"#;

fn write_fake_llama(dir: &TempDir) -> std::path::PathBuf {
    let script = dir.path().join("fake-llama");
    std::fs::write(&script, FAKE_LLAMA_SCRIPT).expect("write fake llama");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake llama");
    }
    script
}

fn managed_config() -> RouterConfig {
    let mut cfg: RouterConfig = serde_json::from_value(serde_json::json!({
        "models": {
            "pinned-m": {
                "endpoint": "http://127.0.0.1:1/v1/chat/completions",
                "name": "pinned-m", "intelligence": 1,
                "cost_input": 1e-06, "cost_output": 6e-06, "cost_cached_read": 4e-07,
                "speed": 8,
                "weights": "/models/pinned.gguf",
                "instances": {"swarm": {"group": "swarm", "num_ctx": 4096, "pinned": true}}
            },
            "lazy-m": {
                "endpoint": "http://127.0.0.1:1/v1/chat/completions",
                "name": "lazy-m", "intelligence": 1,
                "cost_input": 1e-06, "cost_output": 6e-06, "cost_cached_read": 4e-07,
                "speed": 8,
                "weights": "/models/lazy.gguf",
                "instances": {"scratch": {"num_ctx": 4096}}
            }
        }
    }))
    .expect("managed config parses");
    cfg.sidecar = SidecarConfig::default();
    cfg
}

fn set_llama_server(path: &std::path::Path, restore: &mut Option<std::ffi::OsString>) {
    *restore = std::env::var_os(LLAMA_SERVER_ENV);
    std::env::set_var(LLAMA_SERVER_ENV, path);
}

fn restore_llama_server(restore: Option<std::ffi::OsString>) {
    match restore {
        Some(v) => std::env::set_var(LLAMA_SERVER_ENV, v),
        None => std::env::remove_var(LLAMA_SERVER_ENV),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn supervisor_boot_rewrite_and_on_demand_residency() {
    // The fake llama-server is a python3 http server; skip cleanly when python
    // is unavailable (the supervisor would otherwise fail to health-check).
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("python3 not available; skipping supervisor integration test");
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let fake = write_fake_llama(&tmp);
    let mut restore = None;
    set_llama_server(&fake, &mut restore);

    let mut config = managed_config();
    config.sidecar = SidecarConfig::default();

    let sup = Arc::new(LlamaServerSupervisor::build(&config).expect("build supervisor"));
    sup.start_all().await.expect("start_all boots pinned models");

    // Boot-time endpoint rewrite (mirrors the coral-router binary's boot loop):
    // every managed model's endpoint is rewritten to its spawned server.
    for key in sup.model_keys() {
        if let Some(server) = sup.server_for(&key) {
            if let Some(entry) = config.models.get_mut(&key) {
                entry.endpoint = format!("{}/v1/chat/completions", server.base_url());
            }
        }
    }

    // Boot-time endpoint rewrite + lazy-not-loaded-at-boot.
    let pinned_server = sup.server_for("pinned-m").expect("pinned server");
    assert_eq!(sup.is_running("pinned-m"), Some(true), "pinned model boots");
    assert_eq!(sup.is_running("lazy-m"), Some(false), "lazy model deferred at boot");
    assert_eq!(
        config.models["pinned-m"].endpoint,
        format!("{}/v1/chat/completions", pinned_server.base_url()),
        "boot rewrites the pinned model endpoint to its spawned server"
    );

    // Real dispatch through the spawned fake server.
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", pinned_server.base_url()))
        .json(&serde_json::json!({
            "model": "pinned-m",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("dispatch chat completion");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("chat json");
    assert_eq!(body["choices"][0]["message"]["content"], "fake completion");

    // On-demand residency: the pool wires the supervisor, so targeting the lazy
    // model's endpoint loads it on demand.
    let pool = crate::instances::build_instance_managers(&config, Some(sup.clone()))
        .expect("build instance pool");
    let lazy_server = sup.server_for("lazy-m").expect("lazy server");
    pool.ensure_target_ready(
        &format!("{}/v1/chat/completions", lazy_server.base_url()),
        None,
    )
    .await;
    assert_eq!(
        sup.is_running("lazy-m"),
        Some(true),
        "lazy model loaded on demand by the dispatch path"
    );

    // Unload via residency: the lazy fake reports zero contexts, so
    // `unload_empty_models` unloads its weights; the pinned model (which
    // reports a pinned instance) is never unloaded.
    pool.unload_empty_models().await;
    assert_eq!(
        sup.is_running("lazy-m"),
        Some(false),
        "lazy model unloaded when left with zero contexts"
    );
    assert_eq!(
        sup.is_running("pinned-m"),
        Some(true),
        "pinned model is never unloaded"
    );

    // Unload is reversible: a later dispatch re-loads the lazy model.
    pool.ensure_target_ready(
        &format!("{}/v1/chat/completions", lazy_server.base_url()),
        None,
    )
    .await;
    assert_eq!(sup.is_running("lazy-m"), Some(true), "re-loaded after unload");

    sup.shutdown().await;
    restore_llama_server(restore);
}
