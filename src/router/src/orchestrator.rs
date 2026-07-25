//! Long-lived orchestrator session — manages session context as a list of
//! `SessionNode`s, supports compaction, checkpointing, and rewind.

use fluent_types::NodeId;
use guidance_llm::{ChatMessage, LlmClient, LlmConfig};
use serde::{Deserialize, Serialize};

use crate::compaction::{CompactionStrategy, RecencyCompaction};
use crate::session::SessionNode;

/// Errors produced by orchestrator session operations.
#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("session error: {0}")]
    Session(String),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    #[error("node error: {0}")]
    Node(String),
}

/// Long-lived orchestrator session. Default context 131072, never evicted.
/// Manages the session context as a list of `SessionNode`s.
pub struct OrchestratorSession {
    pub session_id: String,
    pub model: String,
    nodes: Vec<SessionNode>,
    llm_client: LlmClient,
    compaction_strategy: Box<dyn CompactionStrategy>,
    max_nodes: usize,
    turn_index: u64,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    name: String,
    node_id: Option<NodeId>,
    turn_index: u64,
}

impl OrchestratorSession {
    /// Creates a new orchestrator session.
    pub fn new(
        session_id: impl Into<String>,
        model: impl Into<String>,
        llm_config: LlmConfig,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            model: model.into(),
            nodes: Vec::new(),
            llm_client: LlmClient::with_config(llm_config),
            compaction_strategy: Box::new(RecencyCompaction),
            max_nodes: 100,
            turn_index: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Set a custom compaction strategy (defaults to `RecencyCompaction`).
    #[must_use]
    pub fn with_compaction(
        mut self,
        strategy: Box<dyn CompactionStrategy>,
    ) -> Self {
        self.compaction_strategy = strategy;
        self
    }

    /// Set the maximum number of nodes before compaction triggers.
    #[must_use]
    pub fn with_max_nodes(mut self, max: usize) -> Self {
        self.max_nodes = max;
        self
    }

    /// Add a user message node. Returns the created `SessionNode`.
    pub fn add_user_message(
        &mut self,
        _content: &str,
    ) -> Result<SessionNode, OrchestratorError> {
        let node = SessionNode {
            node_id: NodeId::from_int(self.turn_index as i64),
            role: "user".into(),
            turn_index: self.turn_index,
            accepted: true,
            acceptance_score: None,
            active_lod: 0,
            parent_id: None,
            step_id: None,
            step_status: None,
        };
        self.turn_index += 1;
        self.nodes.push(node.clone());
        Ok(node)
    }

    /// Add an assistant response node. Only stored if accepted.
    /// Rejected nodes are kept in `nodes` but excluded from working context.
    pub fn add_assistant_response(
        &mut self,
        _content: &str,
        accepted: bool,
        score: Option<f64>,
    ) -> Result<SessionNode, OrchestratorError> {
        let node = SessionNode {
            node_id: NodeId::from_int(self.turn_index as i64),
            role: "assistant".into(),
            turn_index: self.turn_index,
            accepted,
            acceptance_score: score,
            active_lod: 0,
            parent_id: None,
            step_id: None,
            step_status: None,
        };
        self.turn_index += 1;
        self.nodes.push(node.clone());
        Ok(node)
    }

    /// Compact the session: demote older nodes to lower LOD levels using
    /// the configured `CompactionStrategy`.
    pub fn compact(&mut self) {
        let lods = self
            .compaction_strategy
            .select_lod(&self.nodes, self.max_nodes);
        for (node, lod) in self.nodes.iter_mut().zip(lods) {
            node.active_lod = lod;
        }
    }

    /// Build the current context window from accepted nodes for the orchestrator.
    /// Rejected nodes are excluded from the working context.
    pub fn build_context(&self) -> Vec<ChatMessage> {
        self.nodes
            .iter()
            .filter(|n| n.accepted)
            .map(|n| ChatMessage {
                role: n.role.clone(),
                content: format!("[{}/LOD{}]", n.role, n.active_lod),
            })
            .collect()
    }

    /// Create a named checkpoint at the current turn index.
    pub fn checkpoint(&mut self, name: &str) -> Result<(), OrchestratorError> {
        let last_node_id = self.nodes.last().map(|n| n.node_id);
        self.checkpoints.push(Checkpoint {
            name: name.into(),
            node_id: last_node_id,
            turn_index: self.turn_index,
        });
        Ok(())
    }

    /// Rewind to a named checkpoint, discarding any nodes added after it.
    pub fn rewind(&mut self, checkpoint_name: &str) -> Result<(), OrchestratorError> {
        let cp = self
            .checkpoints
            .iter()
            .find(|c| c.name == checkpoint_name)
            .ok_or_else(|| {
                OrchestratorError::Checkpoint(format!(
                    "checkpoint not found: {checkpoint_name}"
                ))
            })?;

        let target_turn = cp.turn_index;
        self.nodes.retain(|n| n.turn_index < target_turn);
        self.turn_index = target_turn;
        self.checkpoints.retain(|c| c.turn_index <= target_turn);
        Ok(())
    }

    /// Get all checkpoint names in order of creation.
    pub fn checkpoints(&self) -> Vec<String> {
        self.checkpoints.iter().map(|c| c.name.clone()).collect()
    }

    /// Current number of nodes (including rejected ones).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Current turn index.
    pub fn turn_index(&self) -> u64 {
        self.turn_index
    }

    /// Access the LLM client for orchestrator calls.
    pub fn llm_client(&self) -> &LlmClient {
        &self.llm_client
    }

    /// The session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The model name.
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Reference to the session nodes.
    pub fn nodes(&self) -> &[SessionNode] {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_llm_config() -> LlmConfig {
        LlmConfig {
            api_url: "http://router.test:0000/v1/chat/completions".into(),
            model: "test-model".into(),
            think: None,
            timeout_ms: 5000,
            debug: false,
            show_prompts: false,
        }
    }

    #[test]
    fn test_add_user_message_increments_turn() {
        let mut session = OrchestratorSession::new("sess-1", "test-model", test_llm_config());
        assert_eq!(session.turn_index(), 0);

        let node = session.add_user_message("hello").unwrap();
        assert_eq!(node.turn_index, 0);
        assert_eq!(node.role, "user");
        assert_eq!(session.turn_index(), 1);
        assert_eq!(session.node_count(), 1);
    }

    #[test]
    fn test_accepted_nodes_in_context() {
        let mut session = OrchestratorSession::new("sess-1", "test-model", test_llm_config());

        session.add_user_message("query").unwrap();
        session
            .add_assistant_response("good answer", true, Some(0.9))
            .unwrap();
        session
            .add_assistant_response("bad answer", false, Some(0.2))
            .unwrap();

        let ctx = session.build_context();
        assert_eq!(ctx.len(), 2, "rejected node excluded from context");
    }

    #[test]
    fn test_checkpoint_and_rewind() {
        let mut session = OrchestratorSession::new("sess-1", "test-model", test_llm_config());

        session.add_user_message("turn 1").unwrap();
        session.checkpoint("after-turn-1").unwrap();

        session.add_user_message("turn 2").unwrap();
        session.add_user_message("turn 3").unwrap();
        assert_eq!(session.node_count(), 3);

        session.rewind("after-turn-1").unwrap();
        assert_eq!(session.node_count(), 1);
        assert_eq!(session.turn_index(), 1);
    }

    #[test]
    fn test_rewind_missing_checkpoint() {
        let mut session = OrchestratorSession::new("sess-1", "test-model", test_llm_config());
        let result = session.rewind("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_names() {
        let mut session = OrchestratorSession::new("sess-1", "test-model", test_llm_config());

        session.checkpoint("cp1").unwrap();
        session.checkpoint("cp2").unwrap();

        let names = session.checkpoints();
        assert_eq!(names, vec!["cp1", "cp2"]);
    }

    #[test]
    fn test_compact_applies_lods() {
        let mut session = OrchestratorSession::new("sess-1", "test-model", test_llm_config())
            .with_max_nodes(4);

        for i in 0..5 {
            session.add_user_message(&format!("msg {i}")).unwrap();
        }

        session.compact();

        let lods: Vec<u8> = session.nodes().iter().map(|n| n.active_lod).collect();
        assert_eq!(lods.len(), 5);
        // oldest nodes should have higher LOD level (less detail)
        assert!(lods[0] > 0, "oldest node should be compacted");
        // newest nodes should have LOD 0 (full detail)
        assert_eq!(lods[4], 0, "newest node should have full detail");
    }
}