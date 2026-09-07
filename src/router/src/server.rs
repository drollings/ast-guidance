//! HTTP server exposing the router pipeline as an OpenAI-compatible endpoint.
//! Uses hyper for HTTP with SSE streaming support via http-body-util::channel.

pub mod admin;
pub mod cors;
pub mod dispatch;
pub mod entity_link;
pub mod handler;
pub mod instances_api;
pub mod responses;
pub mod review;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fluent_llm::cache::ResponseCache;
use fluent_wvr::prelude::*;
use tokio::net::TcpListener;
use tokio::sync::watch;

use crate::config::{ModelEntry, RoleEntry, RouteRef, ServerConfig};
use crate::dag_session::SessionRegistry;
use crate::dispatch::escalation::Ladder;
use crate::ledger::ContentNodeLedger;
use crate::pipeline::PipelineOrchestrator;
use crate::routes::plan::PlanRoute;
use crate::routes::rigor::RigorRoute;
use crate::server::review::{ReviewWorker, ReviewFetch};
use crate::testing::mock::MockDispatchContext;

pub struct RouterServer {
    name: ArcIntern<str>,
    pipelines: HashMap<String, Arc<PipelineOrchestrator>>,
    routes: HashMap<String, RouteRef>,
    models: HashMap<String, ModelEntry>,
    /// Fleet-default instance map (mirrors
    /// `RouterConfig.default_params.instances`), carried so per-request paths
    /// inherit it through the same code path.
    default_instances: Option<HashMap<String, crate::config::InstanceProfile>>,
    /// Routing-vocabulary table (mirrors `RouterConfig.roles`), carried so
    /// per-request paths can resolve roles without the full config.
    roles: HashMap<String, RoleEntry>,
    bind_addr: String,
    max_payload: usize,
    classifier: Option<(String, ModelEntry)>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
    /// Chart store + selector host (boot-loaded; dispatch to it).
    plan_route: Option<Arc<PlanRoute>>,
    /// Rigor route: blue/red/judge protocol. `None` → `/v1/rigor`
    /// returns an explicit "not configured" response.
    rigor_route: Option<Arc<RigorRoute>>,
    /// Per-`session_id` `DependencySession` registry (canonical session).
    sessions: Option<Arc<SessionRegistry>>,
    /// Per-model-group escalation ladders.
    ladders: HashMap<String, Arc<Ladder>>,
    /// Deterministic-fact cache consulted before escalating.
    context_cache: Option<Arc<dyn fluent_types::ContextCache>>,
    /// Sidecar instance pool: one manager per managed model, aggregating
    /// the public `/instances` API and consulting the manager on a 503
    /// group-miss to allocate fresh KV before retrying.
    instance_pool: Option<Arc<crate::instances::InstancePool>>,
    /// Managed llama-server supervisor (the process owner). Backs
    /// `POST /models/unload` and the `/metrics` aggregation.
    supervisor: Option<Arc<crate::supervisor::LlamaServerSupervisor>>,
    /// Env var naming the management API key (enforced on `/instances`).
    api_key_env_name: Option<String>,
    /// Background `LedgerTierWorker` join handle. Held so the worker task
    /// lives for the process lifetime.
    tier_worker: Option<tokio::task::JoinHandle<()>>,
    /// The `LedgerAgentCoordinator`, when the operator opts in. `None`
    /// keeps dispatch unchanged.
    coordinator: Option<Arc<crate::ledger::orchestrator::LedgerAgentCoordinator>>,
    /// The async review worker (ROADMAP §12.6, C4). Owned by the server,
    /// spawned into the tracked task set, drained on graceful shutdown.
    review_worker: Option<Arc<ReviewWorker>>,
    /// The review-model fetch seam for the review worker.
    review_fetch: Option<ReviewFetch>,
    /// The async entity-link overlay worker (ROADMAP_20260827_ORT §6.2).
    /// Owned by the server, drained on graceful shutdown.
    entity_link_worker: Option<Arc<crate::server::entity_link::EntityLinkWorker>>,
    /// The arc_ready annotation-overlay worker (OVERLAYS §7). Owned by the
    /// server, drained on graceful shutdown so in-flight derivations complete.
    overlay_worker: Option<Arc<crate::ledger::overlay_worker::OverlayWorker>>,
    /// The ort ONNX session registry (ROADMAP_20260827_ORT §0.5): one session
    /// per onnx-declared model. The llama.cpp supervisor, sidecar, and
    /// `/instances` never touch onnx models; the registry backs
    /// `/models/unload` refusals for `Always`-resident models.
    onnx: Option<Arc<fluent_llm::onnx_session::OrtSessionRegistry>>,
    /// The unified weights facade (ROADMAP M4): llama adapters + onnx
    /// implementors behind the shared `LlmWeights` surface. Backs the
    /// `/instances` + `/v1/models` aggregation (onnx rows), `ps`, and the
    /// onnx branch of `POST /models/unload`.
    fleet: Option<Arc<crate::instances::traits::LlmFleet>>,
    /// The in-process onnx generative `ChatBackend` (the `onnx/llm` routing
    /// model). Routes whose `model_groups` resolve to an onnx role dispatch
    /// through it. Wired by the composition root alongside the onnx registry.
    onnx_llm_backend: Option<Arc<dyn fluent_llm::client::ChatBackend>>,
    /// The shared residency engine (ROADMAP M5): ONE loop over the fleet's
    /// weights (llama adapters + onnx implementors), replacing both the llama
    /// sidecar task and the onnx residency sibling. Built in the composition
    /// root (where the sidecar knobs live), spawned in `serve`, aborted on
    /// graceful shutdown.
    residency_engine: Option<Arc<fluent_llm::runtime::LlmResidencyEngine>>,
    depends: Vec<ArcIntern<str>>,
    provides: Vec<ArcIntern<str>>,
}

impl RouterServer {
    pub fn new(
        pipelines: HashMap<String, Arc<PipelineOrchestrator>>,
        routes: HashMap<String, RouteRef>,
        models: HashMap<String, ModelEntry>,
        config: &ServerConfig,
        classifier: Option<(String, ModelEntry)>,
    ) -> Self {
        Self {
            name: ArcIntern::from("router.server"),
            pipelines,
            routes,
            models,
            default_instances: None,
            roles: HashMap::new(),
            bind_addr: config.bind_addr.clone(),
            max_payload: config.max_payload,
            classifier,
            mock_dispatch: None,
            ledger: None,
            cache: None,
            plan_route: None,
            rigor_route: None,
            sessions: None,
            ladders: HashMap::new(),
            context_cache: None,
            instance_pool: None,
            supervisor: None,
            api_key_env_name: None,
            tier_worker: None,
            coordinator: None,
            review_worker: None,
            review_fetch: None,
            entity_link_worker: None,
            overlay_worker: None,
            onnx: None,
            fleet: None,
            onnx_llm_backend: None,
            residency_engine: None,
            depends: vec![],
            provides: vec![ArcIntern::from("http.endpoint")],
        }
    }

    /// Attach the fleet-default instance map (`default_params.instances`) so
    /// per-request paths inherit it through the same code path.
    #[must_use]
    pub fn with_default_instances(
        mut self,
        instances: Option<HashMap<String, crate::config::InstanceProfile>>,
    ) -> Self {
        self.default_instances = instances;
        self
    }

    /// Attach the routing-vocabulary table (`RouterConfig.roles`) so
    /// per-request paths resolve roles through the single precedence.
    /// Absent (the default) leaves key-based routing untouched.
    #[must_use]
    pub fn with_roles(mut self, roles: HashMap<String, RoleEntry>) -> Self {
        self.roles = roles;
        self
    }

    #[must_use]
    pub fn with_ledger(mut self, ledger: Arc<ContentNodeLedger>) -> Self {
        self.ledger = Some(ledger);
        self
    }

    #[must_use]
    pub fn with_cache(mut self, cache: Arc<ResponseCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    #[must_use]
    pub fn with_plan_route(mut self, plan_route: Arc<PlanRoute>) -> Self {
        self.plan_route = Some(plan_route);
        self
    }

    /// Attach the rigor route. `None` (default) leaves `/v1/rigor`
    /// present but unconfigured — requests return an explicit error.
    #[must_use]
    pub fn with_rigor_route(mut self, rigor_route: Arc<RigorRoute>) -> Self {
        self.rigor_route = Some(rigor_route);
        self
    }

    /// Attach the per-session `DependencySession` registry (canonical
    /// session). Each chat-completion request then tracks a step in the
    /// session keyed by its `session_id`.
    #[must_use]
    pub fn with_sessions(mut self, sessions: Arc<SessionRegistry>) -> Self {
        self.sessions = Some(sessions);
        self
    }

    /// Attach the per-model-group escalation ladders.
    #[must_use]
    pub fn with_ladders(mut self, ladders: HashMap<String, Arc<Ladder>>) -> Self {
        tracing::info!(
            target: "router.server",
            ladder_count = ladders.len(),
            "escalation ladders attached",
        );
        self.ladders = ladders;
        self
    }

    /// Attach the deterministic context cache consulted before escalating.
    #[must_use]
    pub fn with_context_cache(
        mut self,
        context_cache: Arc<dyn fluent_types::ContextCache>,
    ) -> Self {
        tracing::info!(
            target: "router.server",
            "context cache attached — escalation short-circuits on hits",
        );
        self.context_cache = Some(context_cache);
        self
    }

    /// Attach the sidecar instance pool: one manager per managed model.
    /// `serve` runs each manager's boot reconciliation and residency loop as a
    /// task; dispatch consults the owning manager on a 503 group-miss to
    /// allocate KV, and the public `/instances` API aggregates the pool.
    #[must_use]
    pub fn with_instance_pool(mut self, pool: crate::instances::InstancePool) -> Self {
        if !pool.is_empty() {
            tracing::info!(
                target: "router.server",
                manager_count = pool.managers_iter().len(),
                "sidecar instance pool attached",
            );
        }
        self.instance_pool = Some(Arc::new(pool));
        self
    }

    /// Attach the management API key env var name (enforced on `/instances`).
    #[must_use]
    pub fn with_management_api_key(mut self, env_name: Option<String>) -> Self {
        self.api_key_env_name = env_name;
        self
    }

    /// Attach the managed llama-server supervisor. Enables
    /// `POST /models/unload` and the `/metrics` aggregation.
    #[must_use]
    pub fn with_supervisor(mut self, supervisor: Option<Arc<crate::supervisor::LlamaServerSupervisor>>) -> Self {
        self.supervisor = supervisor;
        self
    }

    /// Hold the background `LedgerTierWorker` join handle so the worker
    /// task lives for the process lifetime.
    #[must_use]
    pub fn with_tier_worker(mut self, handle: tokio::task::JoinHandle<()>) -> Self {
        self.tier_worker = Some(handle);
        self
    }

    /// Attach the `LedgerAgentCoordinator`. `None` (the default) leaves
    /// dispatch unchanged.
    #[must_use]
    pub fn with_coordinator(
        mut self,
        coordinator: Arc<crate::ledger::orchestrator::LedgerAgentCoordinator>,
    ) -> Self {
        self.coordinator = Some(coordinator);
        self
    }

    /// Attach the async review worker (ROADMAP §12.6, C4). The worker is
    /// spawned into the server's tracked task set and drained on graceful
    /// shutdown so in-flight corrections + status writes complete.
    #[must_use]
    pub fn with_review_worker(mut self, worker: Arc<ReviewWorker>) -> Self {
        self.review_worker = Some(worker);
        self
    }

    /// Attach the review-model fetch seam for the review worker.
    #[must_use]
    pub fn with_review_fetch(mut self, fetch: ReviewFetch) -> Self {
        self.review_fetch = Some(fetch);
        self
    }

    /// Attach the async entity-link overlay worker (ROADMAP_20260827_ORT §6.2).
    /// Drained on graceful shutdown like the review worker.
    #[must_use]
    pub fn with_entity_link_worker(
        mut self,
        worker: Arc<crate::server::entity_link::EntityLinkWorker>,
    ) -> Self {
        self.entity_link_worker = Some(worker);
        self
    }

    /// Attach the arc_ready annotation-overlay worker (OVERLAYS §7). Drained
    /// on graceful shutdown so in-flight derivations complete before exit.
    #[must_use]
    pub fn with_overlay_worker(
        mut self,
        worker: Arc<crate::ledger::overlay_worker::OverlayWorker>,
    ) -> Self {
        self.overlay_worker = Some(worker);
        self
    }

    /// Attach the ort ONNX session registry. Backs `/models/unload` refusals
    /// for `Always`-resident onnx models and the onnx `EmbeddingProvider`
    /// branch of the chart embedder. `None` (the default) leaves dispatch
    /// unchanged — onnx models degrade to plain HTTP entries (fail-open).
    #[must_use]
    pub fn with_onnx_registry(
        mut self,
        onnx: Option<Arc<fluent_llm::onnx_session::OrtSessionRegistry>>,
    ) -> Self {
        self.onnx = onnx;
        self
    }

    /// Attach the in-process onnx generative `ChatBackend` (the `onnx/llm`
    /// routing model) so routes whose `model_groups` resolve to an onnx role
    /// dispatch through it. Wired by the composition root alongside the onnx
    /// registry; `None` leaves onnx targets unserved (mock intercepts first in
    /// tests).
    #[must_use]
    pub fn with_onnx_llm_backend(
        mut self,
        backend: Option<Arc<dyn fluent_llm::client::ChatBackend>>,
    ) -> Self {
        self.onnx_llm_backend = backend;
        self
    }

    /// Attach the shared residency engine (ROADMAP M5). `None` (the default)
    /// leaves both fleets resident until the process exits. When attached,
    /// `serve` spawns it as the single residency loop over the fleet's weights
    /// (llama adapters + onnx implementors) and aborts it on graceful shutdown.
    #[must_use]
    pub fn with_residency_engine(
        mut self,
        engine: Option<Arc<fluent_llm::runtime::LlmResidencyEngine>>,
    ) -> Self {
        self.residency_engine = engine;
        self
    }

    /// Attach the unified weights facade (ROADMAP M4): llama adapters + onnx
    /// implementors behind the shared `LlmWeights` surface. Backs the
    /// `/instances` + `/v1/models` aggregation (onnx rows), `ps`, and the
    /// onnx branch of `POST /models/unload`.
    #[must_use]
    pub fn with_fleet(mut self, fleet: Arc<crate::instances::traits::LlmFleet>) -> Self {
        self.fleet = Some(fleet);
        self
    }

    #[must_use]
    pub fn with_mock(mut self, mock_dispatch: MockDispatchContext) -> Self {
        tracing::info!(
            target: "router.server",
            except_count = mock_dispatch.except_models.len(),
            "mock dispatch enabled"
        );
        self.mock_dispatch = Some(Arc::new(mock_dispatch));
        self
    }

    /// Serve until the `shutdown` watch fires. Coral Router owns the local
    /// inference fleet and the server's serving tasks, so this method runs the
    /// HTTP accept loop to completion and then **drains** (aborts + awaits,
    /// within a timeout) the process-lifetime background tasks it spawned —
    /// the per-manager pinned-instance reconcile and the device-wide residency
    /// loop — so nothing is left detached on shutdown.
    pub async fn serve(
        &self,
        shutdown: watch::Receiver<bool>,
    ) -> Result<(), crate::error::ServerError> {
        let chart_count = self
            .plan_route
            .as_ref()
            .map_or(0, |p| p.chart_store().len());
        tracing::info!(
            target: "router.server",
            bind_addr = %self.bind_addr,
            has_mock = self.mock_dispatch.is_some(),
            has_ledger = self.ledger.is_some(),
            has_cache = self.cache.is_some(),
            has_plan_route = self.plan_route.is_some(),
            has_rigor_route = self.rigor_route.is_some(),
            chart_count = chart_count,
            ladder_count = self.ladders.len(),
            "serving HTTP"
        );
        let deps = handler::ServerDeps {
            pipelines: Arc::new(self.pipelines.clone()),
            routes: Arc::new(self.routes.clone()),
            models: Arc::new(self.models.clone()),
            stats: Arc::new(responses::ServerStats::new()),
            max_payload: self.max_payload,
            classifier: self.classifier.clone(),
            mock_dispatch: self.mock_dispatch.clone(),
            ledger: self.ledger.clone(),
            cache: self.cache.clone(),
            plan_route: self.plan_route.clone(),
            rigor_route: self.rigor_route.clone(),
            sessions: self.sessions.clone(),
            http_client: Arc::new(
                reqwest::Client::builder()
                    .connect_timeout(Duration::from_secs(10))
                    .build()
                    .map_err(|e| {
                        crate::error::ServerError::Http(format!("HTTP client build failed: {e}"))
                    })?,
            ),
            ladders: self.ladders.clone(),
            context_cache: self.context_cache.clone(),
            instance_pool: self.instance_pool.clone(),
            api_key_env_name: self.api_key_env_name.clone(),
            supervisor: self.supervisor.clone(),
            coordinator: self.coordinator.clone(),
            review_worker: self.review_worker.clone(),
            review_fetch: self.review_fetch.clone(),
            entity_link_worker: self.entity_link_worker.clone(),
            onnx: self.onnx.clone(),
            onnx_llm_backend: self.onnx_llm_backend.clone(),
            fleet: self.fleet.clone(),
            roles: Arc::new(self.roles.clone()),
            default_instances: self.default_instances.clone(),
        };

        // Reconcile configured pinned instances at boot (retrying until the
        // managed server's management API is reachable) per manager. These are
        // process-lifetime background tasks owned by the server: they run in a
        // `JoinSet` that is drained on shutdown, never detached.
        let mut background = tokio::task::JoinSet::new();
        if let Some(pool) = &self.instance_pool {
            for manager in pool.managers_iter() {
                let manager = manager.clone();
                background.spawn(async move {
                    manager.bootstrap().await;
                });
            }
        }
        // The shared residency engine (M5): ONE loop over the fleet's weights
        // (llama adapters + onnx implementors), replacing both the llama
        // sidecar task and the onnx residency sibling. It runs on an injected
        // `Runtime` (no ambient `tokio::spawn`); its join handle is held and
        // aborted at graceful shutdown so no detached task survives the server.
        let mut residency_engine_handle: Option<tokio::task::JoinHandle<()>> = None;
        if let (Some(engine), Some(fleet)) = (&self.residency_engine, &self.fleet) {
            let weights = fleet.weights();
            if !weights.is_empty() {
                let weights = Arc::new(weights.to_vec());
                let runtime = fluent_concurrency::tokio_runtime();
                residency_engine_handle = Some(engine.start(&runtime, weights));
                tracing::info!(
                    target: "router.server",
                    weights_count = fleet.weights().len(),
                    "shared residency engine started (llama + onnx in one loop)",
                );
            }
        }

        let result = run_http(&self.bind_addr, deps, shutdown).await;

        // Graceful shutdown: drain the review worker (C4) so in-flight
        // corrections + status writes complete before the router exits.
        if let Some(worker) = &self.review_worker {
            Arc::clone(worker).drain().await;
        }

        // Graceful shutdown: drain the entity-link overlay worker (M6.2) so
        // in-flight candidates land before the router exits.
        if let Some(worker) = &self.entity_link_worker {
            Arc::clone(worker).drain().await;
        }

        // Graceful shutdown: drain the arc_ready annotation-overlay worker
        // (OVERLAYS §9) so in-flight derivations complete before exit.
        if let Some(worker) = &self.overlay_worker {
            Arc::clone(worker).drain().await;
        }

        // Graceful shutdown: abort the background tasks and await their
        // completion so no process-lifetime task is left detached.
        background.abort_all();
        let _ = tokio::time::timeout(Duration::from_secs(5), async {
            while background.join_next().await.is_some() {}
        })
        .await;

        // Graceful shutdown: abort the shared residency engine (its task runs on
        // the injected `Runtime`, outside the JoinSet, so it is aborted
        // explicitly — never left detached).
        if let Some(handle) = residency_engine_handle {
            handle.abort();
            let _ = handle.await;
        }

        result
    }
}

impl WorkUnit for RouterServer {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }

    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        let bind_addr = self.bind_addr.clone();
        let max_payload = self.max_payload;
        let deps = handler::ServerDeps {
            pipelines: Arc::new(self.pipelines.clone()),
            routes: Arc::new(self.routes.clone()),
            models: Arc::new(self.models.clone()),
            stats: Arc::new(responses::ServerStats::new()),
            max_payload,
            classifier: self.classifier.clone(),
            mock_dispatch: self.mock_dispatch.clone(),
            ledger: self.ledger.clone(),
            cache: self.cache.clone(),
            plan_route: self.plan_route.clone(),
            rigor_route: self.rigor_route.clone(),
            sessions: self.sessions.clone(),
            http_client: Arc::new(reqwest::Client::new()),
            ladders: self.ladders.clone(),
            context_cache: self.context_cache.clone(),
            instance_pool: self.instance_pool.clone(),
            api_key_env_name: self.api_key_env_name.clone(),
            supervisor: self.supervisor.clone(),
            coordinator: self.coordinator.clone(),
            review_worker: self.review_worker.clone(),
            review_fetch: self.review_fetch.clone(),
            entity_link_worker: self.entity_link_worker.clone(),
            onnx: self.onnx.clone(),
            onnx_llm_backend: self.onnx_llm_backend.clone(),
            fleet: self.fleet.clone(),
            roles: Arc::new(self.roles.clone()),
            default_instances: self.default_instances.clone(),
        };
        let rt = ctx.rt.clone();

        // The `WorkUnit`/`SupervisedBatch` entry runs the server as a
        // supervised task; it holds its own (never-fired) shutdown watch so
        // `run_http`'s accept loop runs for the task's lifetime.
        let (_, shutdown_rx) = watch::channel(false);
        let _handle = rt.spawn(Box::pin(async move {
            if let Err(e) = run_http(&bind_addr, deps, shutdown_rx).await {
                tracing::error!(target: "router.server", error = %e, "HTTP server error");
            }
        }));

        Ok(WorkOutput::ok(format!(
            "HTTP server bound to {}",
            self.bind_addr
        )))
    }
}

impl_fieldless!(RouterServer);

impl Describable for RouterServer {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {}, "required": [] })
    }
}

impl_component!(RouterServer);

async fn run_http(
    bind_addr: &str,
    deps: handler::ServerDeps,
    shutdown: watch::Receiver<bool>,
) -> Result<(), crate::error::ServerError> {
    let listener =
        TcpListener::bind(bind_addr)
            .await
            .map_err(|source| crate::error::ServerError::Bind {
                addr: bind_addr.to_string(),
                source,
            })?;

    tracing::info!(target: "router.server", addr = %bind_addr, "HTTP server listening (hyper)");

    serve_http(listener, deps, Some(shutdown)).await
}

/// Accept loop over an already-bound listener. Public(crate) so integration
/// tests can bind an ephemeral listener themselves (`127.0.0.1:0`) and drive
/// a real server with no rebind race; production entry is `run_http`.
///
/// `shutdown` is an optional graceful-stop signal. `None` runs the accept loop
/// forever (the test-helper default). `Some(receiver)`: when the watch fires
/// the loop stops accepting, then drains the in-flight per-connection
/// tasks (abort + await, within a timeout) so no per-request task is left
/// detached.
pub(crate) async fn serve_http(
    listener: TcpListener,
    deps: handler::ServerDeps,
    shutdown: Option<watch::Receiver<bool>>,
) -> Result<(), crate::error::ServerError> {
    use hyper_util::rt::TokioIo;

    let mut connections: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();

    loop {
        let accepted = match &shutdown {
            Some(sh) => {
                let mut sh = sh.clone();
                tokio::select! {
                    _ = sh.changed() => break,
                    accepted = listener.accept() => accepted,
                }
            }
            None => listener.accept().await,
        };

        let (stream, _peer) = match accepted {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(target: "router.server", error = %e, "accept error");
                continue;
            }
        };

        let deps = deps.clone();
        connections.spawn(async move {
            let io = TokioIo::new(stream);
            let service = hyper::service::service_fn(move |req| {
                let deps = deps.clone();
                handler::handle_request(req, deps)
            });

            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                if !e.to_string().contains("connection closed")
                    && !e.to_string().contains("shutdown")
                {
                    tracing::error!(target: "router.server", error = %e, "hyper connection error");
                }
            }
        });
    }

    // Graceful shutdown: abort in-flight connections and await their
    // tasks within a timeout, so tracked per-request tasks are never detached.
    connections.abort_all();
    let _ = tokio::time::timeout(Duration::from_secs(5), async {
        while connections.join_next().await.is_some() {}
    })
    .await;

    Ok(())
}
