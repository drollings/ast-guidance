//! HTTP server exposing the router pipeline as an OpenAI-compatible endpoint.
//! Uses hyper for HTTP with SSE streaming support via http-body-util::channel.

pub mod dispatch;
pub mod handler;
pub mod responses;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use common_core::ResponseCache;
use fluent_wvr::prelude::*;
use tokio::net::TcpListener;

use crate::config::{ModelEntry, RouteRef, ServerConfig};
use crate::dag_session::SessionRegistry;
use crate::dispatch::escalation::EscalationLadder;
use crate::ledger::ContentNodeLedger;
use crate::pipeline::PipelineOrchestrator;
use crate::routes::plan::PlanRoute;
use crate::testing::mock::MockDispatchContext;

pub struct RouterServer {
    name: ArcIntern<str>,
    pipelines: HashMap<String, Arc<PipelineOrchestrator>>,
    routes: HashMap<String, RouteRef>,
    models: HashMap<String, ModelEntry>,
    bind_addr: String,
    max_payload: usize,
    classifier: Option<(String, ModelEntry)>,
    mock_dispatch: Option<Arc<MockDispatchContext>>,
    ledger: Option<Arc<ContentNodeLedger>>,
    cache: Option<Arc<ResponseCache>>,
    /// Chart store + selector host (boot-loaded; M7/M8 dispatch to it).
    plan_route: Option<Arc<PlanRoute>>,
    /// Per-`session_id` `DependencySession` registry (D6 canonical session).
    sessions: Option<Arc<SessionRegistry>>,
    /// Per-model-group escalation ladders (M3).
    ladders: HashMap<String, Arc<EscalationLadder>>,
    /// Deterministic-fact cache consulted before escalating (M3).
    context_cache: Option<Arc<dyn fluent_types::ContextCache>>,
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
            bind_addr: config.bind_addr.clone(),
            max_payload: config.max_payload,
            classifier,
            mock_dispatch: None,
            ledger: None,
            cache: None,
            plan_route: None,
            sessions: None,
            ladders: HashMap::new(),
            context_cache: None,
            depends: vec![],
            provides: vec![ArcIntern::from("http.endpoint")],
        }
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

    /// Attach the per-session `DependencySession` registry (D6 canonical
    /// session). Each chat-completion request then tracks a step in the
    /// session keyed by its `session_id`.
    #[must_use]
    pub fn with_sessions(mut self, sessions: Arc<SessionRegistry>) -> Self {
        self.sessions = Some(sessions);
        self
    }

    /// Attach the per-model-group escalation ladders (M3).
    #[must_use]
    pub fn with_ladders(mut self, ladders: HashMap<String, Arc<EscalationLadder>>) -> Self {
        tracing::info!(
            target: "router.server",
            ladder_count = ladders.len(),
            "escalation ladders attached",
        );
        self.ladders = ladders;
        self
    }

    /// Attach the deterministic context cache consulted before escalating (M3).
    #[must_use]
    pub fn with_context_cache(mut self, context_cache: Arc<dyn fluent_types::ContextCache>) -> Self {
        tracing::info!(
            target: "router.server",
            "context cache attached — escalation short-circuits on hits",
        );
        self.context_cache = Some(context_cache);
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

    pub async fn serve(&self) -> Result<(), crate::error::ServerError> {
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
            sessions: self.sessions.clone(),
            http_client: Arc::new(
                reqwest::Client::builder()
                    .connect_timeout(Duration::from_secs(10))
                    .build()
                    .map_err(|e| {
                        crate::error::ServerError::Http(format!(
                            "HTTP client build failed: {e}"
                        ))
                    })?,
            ),
            ladders: self.ladders.clone(),
            context_cache: self.context_cache.clone(),
        };
        run_http(&self.bind_addr, deps).await
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
            sessions: self.sessions.clone(),
            http_client: Arc::new(reqwest::Client::new()),
            ladders: self.ladders.clone(),
            context_cache: self.context_cache.clone(),
        };
        let rt = ctx.rt.clone();

        let _handle = rt.spawn(Box::pin(async move {
            if let Err(e) = run_http(&bind_addr, deps).await {
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
) -> Result<(), crate::error::ServerError> {
    let listener =
        TcpListener::bind(bind_addr)
            .await
            .map_err(|source| crate::error::ServerError::Bind {
                addr: bind_addr.to_string(),
                source,
            })?;

    tracing::info!(target: "router.server", addr = %bind_addr, "HTTP server listening (hyper)");

    serve_http(listener, deps).await
}

/// Accept loop over an already-bound listener. Public(crate) so integration
/// tests can bind an ephemeral listener themselves (`127.0.0.1:0`) and drive
/// a real server with no rebind race; production entry is `run_http`.
pub(crate) async fn serve_http(
    listener: TcpListener,
    deps: handler::ServerDeps,
) -> Result<(), crate::error::ServerError> {
    use hyper_util::rt::TokioIo;

    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(target: "router.server", error = %e, "accept error");
                continue;
            }
        };

        let deps = deps.clone();

        tokio::spawn(async move {
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
}
