#[cfg(test)]
mod server_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use fluent_wvr::prelude::*;

    use crate::config::ServerConfig;
    use crate::pipeline::PipelineOrchestrator;
    use crate::server::RouterServer;

    fn make_echo_pipeline() -> Arc<PipelineOrchestrator> {
        let stages: Vec<Arc<dyn Component>> = vec![];
        Arc::new(PipelineOrchestrator::new(stages))
    }

    #[test]
    fn router_server_creates_with_config() {
        let mut pipelines = HashMap::new();
        pipelines.insert("default".into(), make_echo_pipeline());
        let routes = HashMap::new();
        let models = HashMap::new();
        let config = ServerConfig::default();
        let server = RouterServer::new(pipelines, routes, models, &config, None);
        assert_eq!(server.name(), "router.server");
        assert!(server
            .provides()
            .contains(&ArcIntern::from("http.endpoint")));
    }

    #[test]
    fn server_config_defaults() {
        let config = ServerConfig::default();
        assert!(
            config.bind_addr.is_empty(),
            "bind_addr must be provided by config or CLI, not hardcoded"
        );
        assert_eq!(config.max_payload, 1_048_576);
    }
}
