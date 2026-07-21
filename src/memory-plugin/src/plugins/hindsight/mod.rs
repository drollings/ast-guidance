//! Hindsight memory plugin — structured knowledge graph.
//!
//! Multi-strategy retrieval with strict structural entity resolution.
//! Code symbols (functions, structs, modules) map to graph entities.
//! Tracks symbol location/hash shifts during codebase iteration.

use crate::traits::MemoryOps;
use crate::types::*;
use fluent_wvr::prelude::*;
use internment::ArcIntern;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

// ── Graph data model ──────────────────────────────────────────

/// Typed entity in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEntity {
    /// A function or method definition.
    Function {
        /// Function name.
        name: String,
        /// Module path (e.g. `crate::module::func`).
        module_path: String,
        /// Source file path.
        file_path: String,
        /// First line number (1-indexed).
        line_start: u32,
        /// Last line number (inclusive).
        line_end: u32,
        /// SHA-256 hash of the function body.
        content_hash: String,
    },
    /// A struct or enum definition.
    Struct {
        /// Type name.
        name: String,
        /// Module path.
        module_path: String,
        /// Source file path.
        file_path: String,
        /// Field names.
        fields: Vec<String>,
        /// SHA-256 hash of the type definition.
        content_hash: String,
    },
    /// A module declaration.
    Module {
        /// Module name.
        name: String,
        /// Module path.
        path: String,
        /// Source file path.
        file_path: String,
        /// Names of symbols defined in this module.
        child_symbols: Vec<String>,
    },
    /// An observed fact from cross-session reasoning.
    Observation {
        /// Observation content.
        content: String,
        /// Category label.
        category: String,
        /// Trust score in [0.0, 1.0].
        trust: f64,
        /// Entity names that support this observation.
        evidence: Vec<String>,
        /// Creation timestamp (Unix seconds).
        created_at: i64,
    },
}

/// A directed edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source entity name.
    pub from: String,
    /// Target entity name.
    pub to: String,
    /// Edge relation type.
    pub relation: Relation,
}

/// Relation type for knowledge graph edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Relation {
    /// Parent contains child (module → function).
    Contains,
    /// Function calls function.
    Calls,
    /// Function uses type or constant.
    Uses,
    /// Module depends on module.
    DependsOn,
    /// Entity supports an observation.
    Supports,
    /// Entity contradicts an observation.
    Contradicts,
}

/// The full knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeGraph {
    /// Named entities keyed by unique name.
    pub entities: BTreeMap<String, GraphEntity>,
    /// Directed edges between entities.
    pub edges: Vec<GraphEdge>,
    /// Monotonically increasing version counter.
    pub version: u64,
}

/// Configuration for the hindsight plugin.
#[derive(Debug, Clone)]
pub struct HindsightConfig {
    /// Maximum graph traversal depth.
    pub max_depth: usize,
    /// Path for persisting the graph.
    pub graph_path: Option<std::path::PathBuf>,
}

impl Default for HindsightConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            graph_path: None,
        }
    }
}

/// Hindsight memory plugin — structured knowledge graph.
pub struct HindsightMemory {
    config: HindsightConfig,
    graph: Arc<RwLock<KnowledgeGraph>>,
}

impl HindsightMemory {
    /// Create a new hindsight memory plugin with the given configuration.
    pub fn new(config: HindsightConfig) -> Self {
        Self {
            config,
            graph: Arc::new(RwLock::new(KnowledgeGraph::default())),
        }
    }
}

impl MemoryOps for HindsightMemory {
    fn name(&self) -> &'static str {
        "hindsight"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn initialize(&mut self, _ctx: &MemoryInitContext) -> Result<(), MemoryError> {
        if let Some(path) = &self.config.graph_path {
            if path.exists() {
                let data = common_core::io::read_to_string_err(path)
                    .map_err(|e| MemoryError::InitFailed(e.to_string()))?;
                let graph: KnowledgeGraph = serde_json::from_str(&data)
                    .map_err(|e| MemoryError::InitFailed(e.to_string()))?;
                tracing::info!(
                    "loaded knowledge graph v{} from {}",
                    graph.version,
                    path.display()
                );
            }
        }
        Ok(())
    }

    fn shutdown(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let graph = Arc::clone(&self.graph);
        let path = self.config.graph_path.clone();
        Box::pin(async move {
            if let Some(path) = path {
                if let Some(parent) = path.parent() {
                    let _ = common_core::io::ensure_dir(parent);
                }
                let g = graph.read().await;
                if let Ok(data) = serde_json::to_string_pretty(&*g) {
                    let _ = std::fs::write(path, data);
                }
            }
        })
    }

    fn prefetch(
        &self,
        query: &str,
        _ctx: &MemoryQueryContext,
    ) -> Pin<Box<dyn Future<Output = String> + Send + '_>> {
        let graph = Arc::clone(&self.graph);
        let query = query.to_string();
        Box::pin(async move {
            if query.is_empty() {
                return String::new();
            }

            let g = graph.read().await;

            let matching: Vec<&String> = g
                .entities
                .keys()
                .filter(|name| common_core::string::contains_ignore_case(name, &query))
                .take(5)
                .collect();

            if matching.is_empty() {
                return String::new();
            }

            let lines: Vec<String> = matching.iter().map(|name| format!("- {name}")).collect();
            format!("## Hindsight Knowledge Graph\n{}", lines.join("\n"))
        })
    }

    fn queue_prefetch(&self, _query: &str, _ctx: &MemoryQueryContext) {}

    fn search(
        &self,
        req: &MemorySearchRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<MemoryResult>, MemoryError>> + Send + '_>> {
        let graph = Arc::clone(&self.graph);
        let req = req.clone();
        Box::pin(async move {
            let g = graph.read().await;

            let results: Vec<MemoryResult> = g
                .entities
                .iter()
                .filter(|(name, _)| common_core::string::contains_ignore_case(name, &req.query))
                .take(req.limit)
                .map(|(name, entity)| {
                    let content = match entity {
                        GraphEntity::Function { module_path, .. } => {
                            format!("function {name} in {module_path}")
                        }
                        GraphEntity::Struct { module_path, .. } => {
                            format!("struct {name} in {module_path}")
                        }
                        GraphEntity::Module { path, .. } => format!("module {path}"),
                        GraphEntity::Observation { content, .. } => content.clone(),
                    };
                    MemoryResult {
                        content,
                        score: 1.0,
                        trust: 1.0,
                        source: "hindsight".into(),
                        metadata: json!({ "entity_name": name }),
                    }
                })
                .collect();

            Ok(results)
        })
    }

    fn sync_turn(
        &self,
        user_content: &str,
        assistant_content: &str,
        _ctx: &MemoryQueryContext,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let graph = self.graph.clone();
        let user = user_content.to_string();
        let assistant = assistant_content.to_string();
        Box::pin(async move {
            // Extract observations from assistant responses
            // Look for patterns like "X is Y", "we use X", "the function X does Y"
            let patterns = [
                (r"(?i)\bwe use\b (.+)", "Uses"),
                (r"(?i)\bthe function\b (.+?)\b does\b", "Calls"),
                (r"(?i)\bX depends on\b (.+)", "DependsOn"),
            ];

            let mut g = graph.write().await;
            let mut new_observations = Vec::new();

            for (pattern_str, _relation) in &patterns {
                if let Ok(re) = regex::Regex::new(pattern_str) {
                    for cap in re.captures_iter(&assistant) {
                        if let Some(m) = cap.get(1) {
                            let content = m.as_str().trim().to_string();
                            if content.len() > 5 && content.len() < 200 {
                                new_observations.push(content);
                            }
                        }
                    }
                }
            }

            // Also extract simple factual statements from user messages
            let user_lower = user.to_lowercase();
            if user_lower.contains("is a") || user_lower.contains("means") {
                let observation = GraphEntity::Observation {
                    content: user.clone(),
                    category: "user_stated".into(),
                    trust: 0.7,
                    evidence: Vec::new(),
                    created_at: simple_now(),
                };
                let key = format!("obs_{}", simple_now());
                g.entities.insert(key, observation);
            }

            // Store assistant observations
            for content in new_observations {
                let entity = GraphEntity::Observation {
                    content: content.clone(),
                    category: "inferred".into(),
                    trust: 0.6,
                    evidence: Vec::new(),
                    created_at: simple_now(),
                };
                let key = format!("obs_{}_{}", simple_now(), g.entities.len());
                g.entities.insert(key, entity);
            }

            if !g.entities.is_empty() {
                g.version += 1;
            }
        })
    }

    fn on_session_end(
        &self,
        messages: &[TurnMessage],
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let graph = self.graph.clone();
        let messages: Vec<TurnMessage> = messages.to_vec();
        Box::pin(async move {
            if messages.is_empty() {
                return;
            }

            let mut g = graph.write().await;
            let mut topic_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();

            // Extract topic entities from user messages
            for msg in &messages {
                if msg.role == "user" {
                    let words: Vec<&str> = msg.content.split_whitespace().collect();
                    for window in words.windows(2) {
                        let bigram = window.join(" ");
                        if bigram.len() > 5 {
                            *topic_counts.entry(bigram).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Store frequent topics as observations
            for (topic, count) in topic_counts {
                if count >= 2 {
                    let entity = GraphEntity::Observation {
                        content: format!("Recurring topic: {topic}"),
                        category: "session_topic".into(),
                        trust: 0.8,
                        evidence: Vec::new(),
                        created_at: simple_now(),
                    };
                    let key = format!("topic_{}_{}", simple_now(), g.entities.len());
                    g.entities.insert(key, entity);
                }
            }

            g.version += 1;
        })
    }

    fn handle_tool_call(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<String, MemoryError> {
        match tool_name {
            "knowledge_add" => {
                let name = args
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| MemoryError::ToolError("missing 'name'".into()))?
                    .to_string();
                let content = args
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let category = args
                    .get("category")
                    .and_then(|v| v.as_str())
                    .unwrap_or("general")
                    .to_string();
                let entity_type = args
                    .get("entity_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("observation")
                    .to_string();
                let from = args.get("from").and_then(|v| v.as_str()).map(String::from);
                let relation = args
                    .get("relation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Supports");

                let graph = self.graph.clone();
                let rt = tokio::runtime::Handle::current();
                let result = rt.block_on(async {
                    let mut g = graph.write().await;

                    // Create the entity
                    let entity = match entity_type.as_str() {
                        "function" => GraphEntity::Function {
                            name: name.clone(),
                            module_path: content.clone(),
                            file_path: String::new(),
                            line_start: 0,
                            line_end: 0,
                            content_hash: String::new(),
                        },
                        "struct" => GraphEntity::Struct {
                            name: name.clone(),
                            module_path: content.clone(),
                            file_path: String::new(),
                            fields: Vec::new(),
                            content_hash: String::new(),
                        },
                        "module" => GraphEntity::Module {
                            name: name.clone(),
                            path: content.clone(),
                            file_path: String::new(),
                            child_symbols: Vec::new(),
                        },
                        _ => GraphEntity::Observation {
                            content: content.clone(),
                            category: category.clone(),
                            trust: 0.5,
                            evidence: Vec::new(),
                            created_at: simple_now(),
                        },
                    };

                    g.entities.insert(name.clone(), entity);
                    g.version += 1;

                    // Add edge if `from` is specified
                    if let Some(from_name) = from {
                        let rel = match relation {
                            "Calls" => Relation::Calls,
                            "Uses" => Relation::Uses,
                            "DependsOn" => Relation::DependsOn,
                            "Contradicts" => Relation::Contradicts,
                            _ => Relation::Supports,
                        };
                        g.edges.push(GraphEdge {
                            from: from_name,
                            to: name.clone(),
                            relation: rel,
                        });
                    }

                    g.entities.len()
                });

                Ok(serde_json::json!({
                    "status": "added",
                    "entity": name,
                    "category": category,
                    "total_entities": result,
                })
                .to_string())
            }
            "knowledge_search" => {
                let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

                let graph = self.graph.clone();
                let max_depth = self.config.max_depth.min(depth);
                let rt = tokio::runtime::Handle::current();
                let result = rt.block_on(async {
                    let g = graph.read().await;

                    // Phase 1: find matching entities by name
                    let matching: Vec<String> = g
                        .entities
                        .keys()
                        .filter(|name| common_core::string::contains_ignore_case(name, query))
                        .take(10)
                        .cloned()
                        .collect();

                    if matching.is_empty() {
                        return serde_json::json!({
                            "results": [],
                            "query": query,
                            "graph_version": g.version,
                        });
                    }

                    // Phase 2: BFS traversal from matching entities
                    let mut visited = std::collections::HashSet::new();
                    let mut results: Vec<serde_json::Value> = Vec::new();
                    let mut frontier: Vec<(String, usize)> =
                        matching.into_iter().map(|n| (n, 0)).collect();

                    while let Some((name, depth)) = frontier.pop() {
                        if depth > max_depth || !visited.insert(name.clone()) {
                            continue;
                        }

                        if let Some(entity) = g.entities.get(&name) {
                            let (entity_type, summary) = match entity {
                                GraphEntity::Function { module_path, .. } => {
                                    ("function", format!("fn {name} in {module_path}"))
                                }
                                GraphEntity::Struct { module_path, .. } => {
                                    ("struct", format!("struct {name} in {module_path}"))
                                }
                                GraphEntity::Module { path, .. } => {
                                    ("module", format!("mod {path}"))
                                }
                                GraphEntity::Observation {
                                    content, category, ..
                                } => ("observation", format!("[{category}] {content}")),
                            };

                            // Find connected entities
                            let neighbors: Vec<String> = g
                                .edges
                                .iter()
                                .filter(|e| e.from == name || e.to == name)
                                .map(|e| {
                                    if e.from == name {
                                        e.to.clone()
                                    } else {
                                        e.from.clone()
                                    }
                                })
                                .collect();

                            results.push(serde_json::json!({
                                "name": name,
                                "type": entity_type,
                                "summary": summary,
                                "depth": depth,
                                "neighbors": neighbors,
                            }));

                            // Enqueue neighbors for traversal
                            for neighbor in neighbors {
                                if !visited.contains(&neighbor) {
                                    frontier.push((neighbor, depth + 1));
                                }
                            }
                        }
                    }

                    serde_json::json!({
                        "results": results,
                        "query": query,
                        "graph_version": g.version,
                        "traversal_depth": max_depth,
                    })
                });

                Ok(result.to_string())
            }
            _ => Err(MemoryError::ToolError(format!("unknown tool: {tool_name}"))),
        }
    }

    fn tool_schemas(&self) -> Vec<ToolSchema> {
        vec![
            ToolSchema {
                name: "knowledge_add".into(),
                description: "Add an entity or observation to the knowledge graph. Optionally link it to an existing entity."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Entity name (unique identifier)" },
                        "content": { "type": "string", "description": "Entity content or description" },
                        "category": { "type": "string", "description": "Category label (e.g. 'general', 'user_pref')" },
                        "entity_type": { "type": "string", "enum": ["function", "struct", "module", "observation"], "description": "Type of entity" },
                        "from": { "type": "string", "description": "Name of existing entity to link FROM" },
                        "relation": { "type": "string", "enum": ["Calls", "Uses", "DependsOn", "Supports", "Contradicts"], "description": "Edge relation type" }
                    },
                    "required": ["name"]
                }),
            },
            ToolSchema {
                name: "knowledge_search".into(),
                description: "Search the structured knowledge graph for entities and relationships. Performs BFS traversal up to specified depth."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query (matches entity names)" },
                        "depth": { "type": "integer", "description": "Maximum BFS traversal depth (default: 3)" }
                    },
                    "required": ["query"]
                }),
            },
        ]
    }
}

impl FieldAccess for HindsightMemory {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        match name {
            "max_depth" => {
                self.config.max_depth = value
                    .parse()
                    .map_err(|e| FieldError::Parse(format!("invalid max_depth: {e}")))?;
                Ok(())
            }
            _ => Err(FieldError::NotFound(name.into())),
        }
    }

    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "max_depth" => Ok(self.config.max_depth.to_string()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }

    fn field_names(&self) -> &'static [&'static str] {
        &["max_depth"]
    }
}

impl Describable for HindsightMemory {
    fn describe(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum graph traversal depth"
                }
            }
        })
    }
}

impl_component!(HindsightMemory);

impl WorkUnit for HindsightMemory {
    fn name(&self) -> &str {
        "hindsight"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("hindsight memory"))
    }
}

const _: () = {
    fn _assert<T: Send + Sync>() {}
    fn _a() {
        _assert::<HindsightMemory>();
    }
};

/// Simple timestamp without chrono dependency.
fn simple_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
