use std::sync::Arc;

use crate::agent::{AgentConfig, AgentError, AgentIdentity, AgentRegistry, AgentTask};
use crate::kv_cache::{HotKvCache, KvSnapshot};
use crate::pipeline_types::RoutingDestination;
use crate::types::{RouterMessageContent, RouterRequest};
use common_core::watchdog::WatchdogSet;

#[derive(Debug, Clone)]
pub struct AgentResponse {
    pub content: String,
    pub tokens_generated: u32,
    pub watchdog_fired: bool,
    pub kv_cache_restored: bool,
}

pub struct AgentDispatcher {
    registry: Arc<AgentRegistry>,
    kv_cache: Arc<HotKvCache>,
    watchdogs: WatchdogSet,
}

impl AgentDispatcher {
    pub fn new(
        registry: Arc<AgentRegistry>,
        kv_cache: Arc<HotKvCache>,
        watchdogs: WatchdogSet,
    ) -> Self {
        Self {
            registry,
            kv_cache,
            watchdogs,
        }
    }

    pub fn registry(&self) -> &Arc<AgentRegistry> {
        &self.registry
    }

    pub fn kv_cache(&self) -> &Arc<HotKvCache> {
        &self.kv_cache
    }

    fn notify_kv_cache_restore(&self, identity: &AgentIdentity) -> bool {
        if let Some(snapshot) = self.kv_cache.get(&identity.model, identity.adapter.as_deref(), &identity.session_id) {
            tracing::info!(
                "restored KV cache for model={} adapter={:?} session={} ({} tokens)",
                snapshot.model,
                snapshot.adapter,
                snapshot.session_id,
                snapshot.token_count,
            );
            true
        } else {
            false
        }
    }

    fn notify_kv_cache_save(&self, identity: &AgentIdentity, content: &str) {
        let snapshot = KvSnapshot {
            model: identity.model.clone(),
            adapter: identity.adapter.clone(),
            session_id: identity.session_id.clone(),
            file_path: std::path::PathBuf::new(),
            token_count: content.split_whitespace().count(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            last_used_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            llama_cpp_version: String::new(),
            model_quant: None,
            base_model_hash: String::new(),
        };
        self.kv_cache.put(snapshot);
    }

    fn build_agent_request(
        destination: &RoutingDestination,
        request: &RouterRequest,
    ) -> AgentTask {
        let (model, adapter, session_id) = match destination {
            RoutingDestination::LocalAgent {
                model,
                adapter,
                session_id,
            } => (model.clone(), adapter.clone(), session_id.clone()),
            RoutingDestination::Frontier { .. } => {
                return AgentTask {
                    messages: vec![],
                    config: AgentConfig::default(),
                };
            }
        };

        let messages: Vec<guidance_llm::ChatMessage> = request
            .messages
            .iter()
            .map(|m| {
                let content = match &m.content {
                    RouterMessageContent::Text(s) => s.clone(),
                    RouterMessageContent::Parts(_) => String::new(),
                };
                guidance_llm::ChatMessage::builder()
                    .role(m.role.clone())
                    .content(content)
                    .build()
            })
            .collect();

        AgentTask {
            messages,
            config: AgentConfig {
                model,
                adapter,
                session_id,
                timeout_ms: 30_000,
                max_tokens: request.max_tokens,
                temperature: request.temperature,
            },
        }
    }

    pub async fn dispatch(
        &self,
        destination: &RoutingDestination,
        request: &RouterRequest,
    ) -> Result<AgentResponse, AgentError> {
        match destination {
            RoutingDestination::LocalAgent {
                model,
                adapter,
                session_id,
            } => {
                let mut identity = AgentIdentity::new(model.clone(), session_id.clone());
                if let Some(ref adapter_name) = adapter {
                    identity = identity.with_adapter(adapter_name.clone());
                }

                let agent_task = Self::build_agent_request(destination, request);
                let kv_cache_restored = self.notify_kv_cache_restore(&identity);

                if let Some(ref adapter_name) = adapter {
                    if !self.registry.is_adapter_loaded(model, adapter_name) {
                        return Err(AgentError::AdapterNotLoaded(format!(
                            "adapter '{adapter_name}' not loaded for model '{model}'"
                        )));
                    }
                }

                let mut accumulated = String::new();
                let mut tokens_generated: u32 = 0;

                let result = self.registry.submit(&identity, agent_task).await?;

                for token_text in result.split(' ') {
                    if let Some(event) = self.watchdogs.check(Some(token_text)) {
                        WatchdogSet::log_event(&event);
                        return Ok(AgentResponse {
                            content: accumulated,
                            tokens_generated,
                            watchdog_fired: true,
                            kv_cache_restored,
                        });
                    }
                    accumulated.push_str(token_text);
                    accumulated.push(' ');
                    tokens_generated += 1;
                }

                self.notify_kv_cache_save(&identity, &accumulated);

                Ok(AgentResponse {
                    content: accumulated.trim().to_string(),
                    tokens_generated,
                    watchdog_fired: false,
                    kv_cache_restored,
                })
            }
            RoutingDestination::Frontier { .. } => Err(AgentError::Execution(
                "AgentDispatcher cannot handle frontier destinations".into(),
            )),
        }
    }
}
