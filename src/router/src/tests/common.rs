//! Crate-typed HTTP test fixtures shared by fluent-router Tier-1 suites.
//!
//! **Why Tier 1, not `tests/common/mod.rs`:** these fixtures drive the real
//! `serve_http` accept loop, which is `pub(crate)` (`server.rs`), and build
//! `ServerDeps` from crate-internal pipeline/config types — a Tier-2 `tests/`
//! directory is a separate crate linked against the public API and cannot
//! reach them. Per `ROADMAP_20260816_TESTS.md` §2.1, crate-internal e2e
//! fixtures live in a Tier-1 module. (A Tier-2 `tests/common` would only be
//! appropriate for fixtures using exclusively public API.)
//!
//! Migrated homes of the former `server_http_tests.rs` /
//! `config_route_tests.rs` duplicates: `TestServer`, `post_chat`/`get`,
//! `make_config`, the `ServerDeps` builder, and the spawn helpers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokio::net::TcpListener;

use crate::config::RouterConfig;
use crate::pipeline::PipelineOrchestrator;
use crate::routes::plan::PlanRoute;
use crate::routes::rigor::RigorRoute;
use crate::server::handler::ServerDeps;
use crate::server::responses::{ResponseBody, ServerStats};
use crate::server::serve_http;
use crate::testing::mock::{MockDispatchContext, TranscriptProvider};
use fluent_llm::client::ChatBackend;

/// Upstream responder: given the parsed request body, produce an HTTP response.
pub type UpstreamRespond = Arc<dyn Fn(&Value) -> hyper::Response<ResponseBody> + Send + Sync>;

/// A running router server bound to an ephemeral port.
pub struct TestServer {
    pub addr: std::net::SocketAddr,
    pub handle: tokio::task::JoinHandle<()>,
}

impl TestServer {
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

/// POST an OpenAI-style chat completion body, bounded by an overall timeout.
pub async fn post_chat(
    base_url: &str,
    body: Value,
    timeout_ms: u64,
) -> Result<reqwest::Response, String> {
    let client = reqwest::Client::new();
    tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        client
            .post(format!("{base_url}/v1/chat/completions"))
            .json(&body)
            .send(),
    )
    .await
    .map_err(|_| "request timed out".to_string())?
    .map_err(|e| format!("request failed: {e}"))
}

/// GET a path, bounded by an overall timeout.
pub async fn get(base_url: &str, path: &str, timeout_ms: u64) -> Result<reqwest::Response, String> {
    let client = reqwest::Client::new();
    tokio::time::timeout(
        Duration::from_millis(timeout_ms),
        client.get(format!("{base_url}{path}")).send(),
    )
    .await
    .map_err(|_| "request timed out".to_string())?
    .map_err(|e| format!("request failed: {e}"))
}

/// Build a `RouterConfig` with a single `default` pipeline, a single `fast`
/// model/route/group, and the given upstream + dispatch settings.
pub fn make_config(
    endpoint: &str,
    stream: bool,
    filter_thinking: bool,
    total_timeout_ms: u64,
    idle_timeout_ms: u64,
) -> RouterConfig {
    let value = json!({
        "pipelines": {"default": {"deterministic_prefilter": true, "classifier": true}},
        "models": {"fast": {
            "endpoint": endpoint,
            "name": "fast",
            "intelligence": 1,
            "cost_input": 0.000001,
            "cost_output": 0.000006,
            "cost_cached_read": 0.0000004,
            "speed": 10,
            "total_timeout_ms": total_timeout_ms,
            "idle_timeout_ms": idle_timeout_ms,
            "stream": stream,
            "filter_thinking": filter_thinking,
            "retry_count": 0,
            "retry_base_interval_s": 1
        }},
        "model_groups": {"fast": ["fast"]},
        "routes": {"fast": {"group": "fast", "pipelines": ["default"]}},
        "default_route": "fast"
    });
    serde_json::from_value(value).expect("valid test config")
}

/// Defaults for every `ServerDeps` field except `pipelines`/`routes`/`models`
/// (which come from the config) and whatever the caller overrides.
#[allow(clippy::too_many_arguments)]
pub fn test_deps(
    pipelines: Arc<HashMap<String, Arc<PipelineOrchestrator>>>,
    config: &RouterConfig,
    mock: Option<Arc<MockDispatchContext>>,
    sessions: Option<Arc<crate::dag_session::SessionRegistry>>,
    plan_route: Option<Arc<PlanRoute>>,
    ladders: HashMap<String, Arc<crate::dispatch::escalation::Ladder>>,
    context_cache: Option<Arc<dyn fluent_types::ContextCache>>,
) -> ServerDeps {
    ServerDeps {
        pipelines,
        routes: Arc::new(config.routes.clone()),
        models: Arc::new(config.models.clone()),
        stats: Arc::new(ServerStats::new()),
        max_payload: config.server.max_payload,
        classifier: None,
        mock_dispatch: mock,
        ledger: None,
        cache: None,
        plan_route,
        rigor_route: None,
        sessions,
        http_client: Arc::new(reqwest::Client::new()),
        ladders,
        context_cache,
        instance_pool: None,
        api_key_env_name: None,
        supervisor: None,
        coordinator: None,
    }
}

/// `test_deps` with a session registry and/or ledger wired.
pub fn test_deps_with_ledger(
    pipelines: Arc<HashMap<String, Arc<PipelineOrchestrator>>>,
    config: &RouterConfig,
    mock: Option<Arc<MockDispatchContext>>,
    sessions: Option<Arc<crate::dag_session::SessionRegistry>>,
    ledger: Option<Arc<crate::ledger::ContentNodeLedger>>,
) -> ServerDeps {
    let mut deps = test_deps(pipelines, config, mock, sessions, None, HashMap::new(), None);
    deps.ledger = ledger;
    deps
}

/// `test_deps` with a rigor route + session registry + ledger wired
/// (checkpoint/rewind + red-team view are load-bearing in the server path).
pub fn rigor_test_deps(
    pipelines: Arc<HashMap<String, Arc<PipelineOrchestrator>>>,
    config: &RouterConfig,
    rigor_route: Option<Arc<RigorRoute>>,
    sessions: Option<Arc<crate::dag_session::SessionRegistry>>,
    ledger: Option<Arc<crate::ledger::ContentNodeLedger>>,
) -> ServerDeps {
    let mut deps = test_deps(pipelines, config, None, sessions, None, HashMap::new(), None);
    deps.rigor_route = rigor_route;
    deps.ledger = ledger;
    deps
}

/// Spawn the real server (ephemeral port) with a transcript classifier and an
/// optional dispatch mock. The default transcript classifier routes to the
/// `fast` target.
pub async fn spawn_test_server(
    config: RouterConfig,
    mock: Option<MockDispatchContext>,
) -> TestServer {
    spawn_test_server_with_sessions(config, mock, None).await
}

/// `spawn_test_server` with an optional `SessionRegistry` (session-step
/// tracking on the dispatch path).
pub async fn spawn_test_server_with_sessions(
    config: RouterConfig,
    mock: Option<MockDispatchContext>,
    sessions: Option<Arc<crate::dag_session::SessionRegistry>>,
) -> TestServer {
    spawn_test_server_with_ledger(config, mock, sessions, None).await
}

/// `spawn_test_server` with a session registry and/or ledger wired
/// (Answer-recording tests wire both).
#[allow(clippy::too_many_arguments)]
pub async fn spawn_test_server_with_ledger(
    config: RouterConfig,
    mock: Option<MockDispatchContext>,
    sessions: Option<Arc<crate::dag_session::SessionRegistry>>,
    ledger: Option<Arc<crate::ledger::ContentNodeLedger>>,
) -> TestServer {
    let provider = TranscriptProvider::new(HashMap::new());
    let backend: Arc<dyn ChatBackend> = Arc::new(provider);
    let pipelines = Arc::new(config.build_all_pipelines_with_backend(Some(&backend)));
    let mock = mock.map(Arc::new);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");

    let mut deps = test_deps(pipelines, &config, mock, sessions, None, HashMap::new(), None);
    deps.ledger = ledger;
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "test server failed");
        }
    });

    TestServer { addr, handle }
}

/// Spawn a server from prebuilt `ServerDeps` (escalation tests need ladders
/// and/or a context cache that `spawn_test_server` does not wire).
pub async fn spawn_test_server_with_deps(deps: ServerDeps) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        if let Err(e) = serve_http(listener, deps, None).await {
            tracing::error!(target: "router.test", error = %e, "test server failed");
        }
    });
    TestServer { addr, handle }
}

/// A `MockDispatchContext` preloaded with a single (user_message →
/// dispatch_response) transcript entry.
pub fn mock_for(user_message: &str, dispatch_response: &str) -> MockDispatchContext {
    MockDispatchContext::new(
        vec![crate::testing::mock::MockTranscriptEntry {
            user_message: user_message.to_string(),
            classifier_response: String::new(),
            expected_route: None,
            expect_model_group: None,
            dispatch_response: Some(dispatch_response.to_string()),
            rejected: false,
            reject_reason_contains: None,
        }],
        vec![],
    )
}

/// Spawn an in-process mock OpenAI upstream that answers every request via
/// `respond`. Returns its base URL.
pub async fn spawn_mock_upstream(respond: UpstreamRespond) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");

    tokio::spawn(async move {
        loop {
            let Ok((stream, _peer)) = listener.accept().await else {
                break;
            };
            let io = TokioIo::new(stream);
            let respond = respond.clone();
            let service = hyper::service::service_fn(move |req: hyper::Request<Incoming>| {
                let respond = respond.clone();
                async move {
                    let body_bytes = req
                        .collect()
                        .await
                        .map(http_body_util::Collected::to_bytes)
                        .unwrap_or_default();
                    let value = serde_json::from_slice::<Value>(&body_bytes).unwrap_or(Value::Null);
                    Ok::<_, std::convert::Infallible>(respond(&value))
                }
            });
            tokio::spawn(async move {
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(io, service)
                    .await;
            });
        }
    });

    format!("http://{addr}")
}

/// Install the process-wide global subscriber and return the shared buffer
/// (audit/log assertions read it).
pub fn install_audit_capture() -> Arc<Mutex<Vec<String>>> {
    crate::test_support::install_global_subscriber()
}