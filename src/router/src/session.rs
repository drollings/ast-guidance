//! Session context node schema — extends the existing `ContextNode` in
//! `fluent-types` with session-specific metadata.

use fluent_types::NodeId;
use serde::{Deserialize, Serialize};

/// A session context entry. Backed by a ContentNode but carries
/// session-specific metadata: role, turn index, acceptance status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionNode {
    /// The underlying content node in the graph DB.
    pub node_id: NodeId,

    /// Role: "user", "assistant", "system", "tool"
    pub role: String,

    /// Monotonically increasing turn index within the session.
    pub turn_index: u64,

    /// Whether this node's content was accepted into the orchestrator's
    /// working context. Rejected nodes remain in storage for audit.
    pub accepted: bool,

    /// Acceptance score (0.0-1.0) from the summarization stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_score: Option<f64>,

    /// LOD level currently active for this node (driven by compaction policy).
    #[serde(default)]
    pub active_lod: u8,

    /// Parent node in the hierarchy. None = root-level entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<NodeId>,

    /// The step this node represents progress toward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_id: Option<String>,

    /// Status of this step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_status: Option<StepStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}