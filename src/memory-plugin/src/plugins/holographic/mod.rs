//! Holographic memory plugin — local SQLite + HRR compositional algebra.
//!
//! This is the primary local memory backend. It provides:
//! - FTS5 keyword search
//! - HRR compositional algebraic probing
//! - Trust scoring with asymmetric feedback
//! - Entity extraction and resolution
//! - Hybrid retrieval (FTS + Jaccard + HRR)

pub mod hrr;
pub mod store;

use crate::traits::MemoryOps;
use crate::types::*;
use fluent_wvr::{
    impl_component, Describable, FieldAccess, FieldError, WorkContext, WorkError, WorkOutput,
    WorkUnit,
};
use internment::ArcIntern;
use serde_json::json;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use store::{HolographicStore, StoreConfig};

/// Holographic memory plugin configuration.
#[derive(Debug, Clone)]
pub struct HolographicConfig {
    /// SQLite database path.
    pub db_path: PathBuf,
    /// HRR vector dimensions.
    pub hrr_dim: usize,
    /// Default trust score for new facts.
    pub default_trust: f64,
    /// Auto-extract facts at session end.
    pub auto_extract: bool,
    /// Minimum trust threshold for retrieval.
    pub min_trust_threshold: f64,
}

impl Default for HolographicConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("memory_store.db"),
            hrr_dim: 1024,
            default_trust: 0.5,
            auto_extract: false,
            min_trust_threshold: 0.3,
        }
    }
}

/// Holographic memory plugin.
///
/// Local embedded state with FTS5 search, HRR compositional algebra,
/// trust scoring, and entity resolution. Implements `Component` via
/// `FieldAccess + Describable + WorkUnit + Send + Sync`.
pub struct HolographicMemory {
    config: HolographicConfig,
    store: Option<Arc<HolographicStore>>,
}

impl HolographicMemory {
    /// Create a new holographic memory plugin.
    pub fn new(config: HolographicConfig) -> Self {
        Self {
            config,
            store: None,
        }
    }
}

// ── MemoryOps implementation ──────────────────────────────────

impl MemoryOps for HolographicMemory {
    fn name(&self) -> &'static str {
        "holographic"
    }

    fn is_available(&self) -> bool {
        true // SQLite is always available
    }

    fn initialize(&mut self, ctx: &MemoryInitContext) -> Result<(), MemoryError> {
        let db_path = if self.config.db_path.is_relative() {
            ctx.memory_root.join(&self.config.db_path)
        } else {
            self.config.db_path.clone()
        };

        let store_config = StoreConfig {
            db_path,
            default_trust: self.config.default_trust,
            hrr_dim: self.config.hrr_dim,
        };

        let store = HolographicStore::open(store_config)?;
        self.store = Some(Arc::new(store));
        Ok(())
    }

    fn shutdown(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {
            // Store is reference-counted; drop happens when Arc reaches 0.
            // No explicit cleanup needed — SQLite closes on drop.
        })
    }

    fn prefetch(
        &self,
        query: &str,
        _ctx: &MemoryQueryContext,
    ) -> Pin<Box<dyn Future<Output = String> + Send + '_>> {
        let store = self.store.as_ref().map(Arc::clone);
        let query = query.to_string();
        let min_trust = self.config.min_trust_threshold;

        Box::pin(async move {
            let store = match store {
                Some(s) => s,
                None => return String::new(),
            };

            if query.is_empty() {
                return String::new();
            }

            let results = store
                .search_facts(&query, None, min_trust, 5)
                .await
                .unwrap_or_default();

            if results.is_empty() {
                return String::new();
            }

            let lines: Vec<String> = results
                .iter()
                .map(|f| format!("- [{:.1}] {}", f.trust_score, f.content))
                .collect();

            format!("## Holographic Memory\n{}", lines.join("\n"))
        })
    }

    fn queue_prefetch(&self, _query: &str, _ctx: &MemoryQueryContext) {
        // No-op: prefetch is synchronous in this plugin.
    }

    fn search(
        &self,
        req: &MemorySearchRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryResult>, MemoryError>> + Send + '_>> {
        let store = self.store.as_ref().map(Arc::clone);
        let req = req.clone();

        Box::pin(async move {
            let store =
                store.ok_or_else(|| MemoryError::NotAvailable("store not initialized".into()))?;

            let facts = store
                .search_facts(
                    &req.query,
                    req.category.as_deref(),
                    req.min_trust,
                    req.limit,
                )
                .await?;

            Ok(facts
                .into_iter()
                .map(|f| MemoryResult {
                    content: f.content,
                    score: f.trust_score,
                    trust: f.trust_score,
                    source: "holographic".into(),
                    metadata: json!({
                        "fact_id": f.fact_id,
                        "category": f.category,
                        "tags": f.tags,
                        "retrieval_count": f.retrieval_count,
                    }),
                })
                .collect())
        })
    }

    fn sync_turn(
        &self,
        _user_content: &str,
        _assistant_content: &str,
        _ctx: &MemoryQueryContext,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {
            // Holographic memory stores explicit facts via tools, not auto-sync.
        })
    }

    fn on_session_end(
        &self,
        messages: &[TurnMessage],
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let store = self.store.as_ref().map(Arc::clone);
        let auto_extract = self.config.auto_extract;
        let messages: Vec<TurnMessage> = messages.to_vec();

        Box::pin(async move {
            if !auto_extract {
                return;
            }
            let store = match store {
                Some(s) => s,
                None => return,
            };

            let pref_patterns = [
                regex::Regex::new(r"(?i)\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)").ok(),
                regex::Regex::new(r"(?i)\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)")
                    .ok(),
                regex::Regex::new(r"(?i)\bI\s+(?:always|never|usually)\s+(.+)").ok(),
            ];

            let decision_patterns = [
                regex::Regex::new(r"(?i)\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)").ok(),
                regex::Regex::new(r"(?i)\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)").ok(),
            ];

            for msg in &messages {
                if msg.role != "user" {
                    continue;
                }
                if msg.content.len() < 10 {
                    continue;
                }

                for pattern in pref_patterns.iter().flatten() {
                    if pattern.is_match(&msg.content) {
                        let _ = store
                            .add_fact(&msg.content[..msg.content.len().min(400)], "user_pref", "")
                            .await;
                        break;
                    }
                }

                for pattern in decision_patterns.iter().flatten() {
                    if pattern.is_match(&msg.content) {
                        let _ = store
                            .add_fact(&msg.content[..msg.content.len().min(400)], "project", "")
                            .await;
                        break;
                    }
                }
            }
        })
    }

    fn handle_tool_call(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<String, MemoryError> {
        match tool_name {
            "fact_store" => self.handle_fact_store(args),
            "fact_feedback" => self.handle_fact_feedback(args),
            _ => Err(MemoryError::ToolError(format!("unknown tool: {tool_name}"))),
        }
    }

    fn tool_schemas(&self) -> Vec<ToolSchema> {
        vec![
            ToolSchema {
                name: "fact_store".into(),
                description: "Deep structured memory with algebraic reasoning. \
                    Use alongside the memory tool — memory for always-on context, \
                    fact_store for deep recall and compositional queries.\n\n\
                    ACTIONS: add, search, probe, related, reason, contradict, update, remove, list"
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "search", "probe", "related", "reason",
                                     "contradict", "update", "remove", "list"]
                        },
                        "content": { "type": "string" },
                        "query": { "type": "string" },
                        "entity": { "type": "string" },
                        "entities": { "type": "array", "items": { "type": "string" } },
                        "fact_id": { "type": "integer" },
                        "category": { "type": "string" },
                        "tags": { "type": "string" },
                        "min_trust": { "type": "number" },
                        "limit": { "type": "integer" }
                    },
                    "required": ["action"]
                }),
            },
            ToolSchema {
                name: "fact_feedback".into(),
                description: "Rate a fact after using it. Mark 'helpful' if accurate, \
                    'unhelpful' if outdated."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": { "type": "string", "enum": ["helpful", "unhelpful"] },
                        "fact_id": { "type": "integer" }
                    },
                    "required": ["action", "fact_id"]
                }),
            },
        ]
    }
}

impl HolographicMemory {
    /// Internal: handle fact_store tool call.
    fn handle_fact_store(&self, args: &serde_json::Value) -> Result<String, MemoryError> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MemoryError::ToolError("missing 'action'".into()))?;

        let store = self
            .store
            .as_ref()
            .ok_or_else(|| MemoryError::NotAvailable("store not initialized".into()))?;

        let rt = tokio::runtime::Handle::current();

        match action {
            "add" => {
                let content = args
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| MemoryError::ToolError("missing 'content'".into()))?;
                let category = args
                    .get("category")
                    .and_then(|v| v.as_str())
                    .unwrap_or("general");
                let tags = args.get("tags").and_then(|v| v.as_str()).unwrap_or("");

                let fact_id = rt.block_on(store.add_fact(content, category, tags))?;
                Ok(serde_json::json!({"fact_id": fact_id, "status": "added"}).to_string())
            }
            "search" => {
                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| MemoryError::ToolError("missing 'query'".into()))?;
                let min_trust = args
                    .get("min_trust")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(self.config.min_trust_threshold);
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
                let category = args.get("category").and_then(|v| v.as_str());

                let results = rt.block_on(store.search_facts(query, category, min_trust, limit))?;
                Ok(serde_json::json!({"results": results, "count": results.len()}).to_string())
            }
            "feedback" => {
                let fact_id = args
                    .get("fact_id")
                    .and_then(|v| v.as_i64())
                    .ok_or_else(|| MemoryError::ToolError("missing 'fact_id'".into()))?;
                let helpful = args
                    .get("action")
                    .and_then(|v| v.as_str())
                    .map(|a| a == "helpful")
                    .unwrap_or(false);

                let result = rt.block_on(store.record_feedback(fact_id, helpful))?;
                Ok(result.to_string())
            }
            _ => Err(MemoryError::ToolError(format!(
                "unknown fact_store action: {action}"
            ))),
        }
    }

    /// Internal: handle fact_feedback tool call.
    fn handle_fact_feedback(&self, args: &serde_json::Value) -> Result<String, MemoryError> {
        self.handle_fact_store(args)
    }
}

// ── Component trait implementations ───────────────────────────

impl FieldAccess for HolographicMemory {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        match name {
            "db_path" => {
                self.config.db_path = PathBuf::from(value);
                Ok(())
            }
            "hrr_dim" => {
                self.config.hrr_dim = value
                    .parse()
                    .map_err(|e| FieldError::Parse(format!("invalid hrr_dim: {e}")))?;
                Ok(())
            }
            "default_trust" => {
                self.config.default_trust = value
                    .parse()
                    .map_err(|e| FieldError::Parse(format!("invalid default_trust: {e}")))?;
                Ok(())
            }
            "auto_extract" => {
                self.config.auto_extract = value
                    .parse()
                    .map_err(|e| FieldError::Parse(format!("invalid auto_extract: {e}")))?;
                Ok(())
            }
            "min_trust_threshold" => {
                self.config.min_trust_threshold = value
                    .parse()
                    .map_err(|e| FieldError::Parse(format!("invalid min_trust_threshold: {e}")))?;
                Ok(())
            }
            _ => Err(FieldError::NotFound(name.into())),
        }
    }

    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "db_path" => Ok(self.config.db_path.display().to_string()),
            "hrr_dim" => Ok(self.config.hrr_dim.to_string()),
            "default_trust" => Ok(self.config.default_trust.to_string()),
            "auto_extract" => Ok(self.config.auto_extract.to_string()),
            "min_trust_threshold" => Ok(self.config.min_trust_threshold.to_string()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }

    fn field_names(&self) -> &'static [&'static str] {
        &[
            "db_path",
            "hrr_dim",
            "default_trust",
            "auto_extract",
            "min_trust_threshold",
        ]
    }
}

impl Describable for HolographicMemory {
    fn describe(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "db_path": {
                    "type": "string",
                    "description": "SQLite database path"
                },
                "hrr_dim": {
                    "type": "integer",
                    "description": "HRR vector dimensions"
                },
                "default_trust": {
                    "type": "number",
                    "description": "Default trust score for new facts"
                },
                "auto_extract": {
                    "type": "boolean",
                    "description": "Auto-extract facts at session end"
                },
                "min_trust_threshold": {
                    "type": "number",
                    "description": "Minimum trust threshold for retrieval"
                }
            },
            "required": ["db_path"]
        })
    }
}

impl_component!(HolographicMemory);

impl WorkUnit for HolographicMemory {
    fn name(&self) -> &str {
        "holographic"
    }

    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }

    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("holographic memory"))
    }
}

// Compile-time assertion: HolographicMemory satisfies all bounds
const _: () = {
    fn _assert_send_sync<T: Send + Sync>() {}
    fn _assert() {
        _assert_send_sync::<HolographicMemory>();
    }
};
