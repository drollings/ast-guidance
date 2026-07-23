#[cfg(test)]
mod server_tests {
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
        let pipeline = make_echo_pipeline();
        let config = ServerConfig::default();
        let server = RouterServer::new(pipeline, &config);
        assert_eq!(server.name(), "router.server");
        assert!(server.provides().contains(&ArcIntern::from("http.endpoint")));
    }

    #[test]
    fn server_config_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.bind_addr, "127.0.0.1:8080");
        assert_eq!(config.max_payload, 1_048_576);
    }
}