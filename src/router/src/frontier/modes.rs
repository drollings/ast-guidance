/// Four frontier involvement modes per MOA_ROUTER_SPEC §8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontierMode {
    /// Anonymized hypothetical fallback: minimal decontextualized question,
    /// rubric-validated before acceptance, cached for HNSW lookup.
    AnonymizedHypothetical,
    /// Authorized directory/code review: whitelist-verified subtree only.
    AuthorizedCodeReview,
    /// Workflow composition: frontier produces a reusable WorkflowConfig JSON.
    WorkflowComposition,
    /// Copilot/judge over local reasoning: frontier reviews at checkpoints.
    CopilotJudge,
}

/// Result of a frontier interaction.
#[derive(Debug, Clone)]
pub struct FrontierResult {
    pub mode: FrontierMode,
    pub response: String,
    pub audit_entry: AuditEntry,
}

#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub payload: String,
    pub raw_response: String,
    pub trigger: String,
    pub timestamp: u64,
}

/// Execute a frontier interaction in the given mode.
/// Full implementation deferred until agent/orchestrator integration is mature.
#[allow(clippy::unnecessary_wraps)]
pub fn execute_frontier_mode(
    _mode: FrontierMode,
    _payload: &str,
    _model_endpoint: &str,
) -> Result<FrontierResult, String> {
    Err("Frontier mode execution not yet implemented — requires agent wiring".into())
}
