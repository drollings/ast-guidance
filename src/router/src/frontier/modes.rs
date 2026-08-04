//! Frontier escalation ladder — VISION §"The escalation ladder".
//!
//! `EscalationMode` is the canonical taxonomy of frontier-engagement stages
//! (decision D8 — replaces the old `FrontierMode` "four involvement modes"
//! enum, which never matched the VISION ladder). Stages are tried in order
//! (filter → question → team → turnover) after every local model in a
//! `model_group` fails. The ladder *runtime* is forward track
//! (ROADMAP_20260804_DRY §0.5); these types are the reconciled spec for the
//! future dispatch loop, and the audit types it will write to.

use crate::error::ServerError;

/// A stage of the frontier escalation ladder.
///
/// Each stage is a discrete policy governing how much context, data, and
/// agency the frontier model receives (doc/router/VISION.md §"The four
/// modes"). Stages escalate filter → question → team → turnover: the
/// less-permissive stages are tried first so frontier calls are
/// progressively more expensive, never all-or-nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationMode {
    /// Deterministic PII/anonymization rules strip sensitive content; the
    /// filtered query is sent as a one-shot prompt to frontier. Frontier
    /// sees filtered/de-identified text only — low risk, no raw data
    /// crosses the boundary.
    Filter,
    /// A `decomposer_model` (fast local LLM) breaks the problem into generic
    /// hypothetical questions; the frontier answers each independently; an
    /// `assembler_model` synthesizes the responses into the final answer.
    /// Frontier sees abstract hypotheticals with no personal data and no
    /// session context — low risk.
    Question,
    /// `classifier_parallel` instances of a `classifier_model` run in
    /// parallel slots and vote on approach; a `draft_model` attempts the
    /// easier sub-steps locally; a `judge_model` reviews the draft and
    /// crafts a precise frontier prompt containing only the unsolved
    /// sub-problem and verified partial work. Frontier sees partial
    /// solution structure — medium risk.
    Team,
    /// Full context handoff: the frontier receives the entire session
    /// ledger and all tool access and continues autonomously; subsequent
    /// messages in the session go through frontier. Frontier has full
    /// agency and raw data — high risk, the most permissive stage.
    Turnover,
}

/// Result of a frontier interaction.
#[derive(Debug, Clone)]
pub struct FrontierResult {
    pub mode: EscalationMode,
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

/// Execute a frontier interaction in the given escalation stage.
/// Full implementation deferred until the escalation-ladder dispatch loop
/// lands (forward track — ROADMAP_20260804_DRY §0.5).
#[allow(clippy::unnecessary_wraps)]
pub fn execute_frontier_mode(
    _mode: EscalationMode,
    _payload: &str,
    _model_endpoint: &str,
) -> Result<FrontierResult, ServerError> {
    Err(ServerError::FrontierNotImplemented(
        "frontier mode execution not yet implemented — requires agent wiring".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_stages_all_defer_to_the_not_implemented_stub() {
        for stage in [
            EscalationMode::Filter,
            EscalationMode::Question,
            EscalationMode::Team,
            EscalationMode::Turnover,
        ] {
            let err = execute_frontier_mode(stage, "payload", "endpoint").unwrap_err();
            assert!(
                matches!(err, ServerError::FrontierNotImplemented(_)),
                "stage {stage:?} must remain a forward-track stub"
            );
        }
    }

    #[test]
    fn frontier_result_carries_the_stage_that_produced_it() {
        let result = FrontierResult {
            mode: EscalationMode::Team,
            response: "answer".into(),
            audit_entry: AuditEntry {
                payload: "prompt".into(),
                raw_response: "raw".into(),
                trigger: "low judge confidence".into(),
                timestamp: 1,
            },
        };
        assert_eq!(result.mode, EscalationMode::Team);
        assert_eq!(result.response, "answer");
        assert_eq!(result.audit_entry.trigger, "low judge confidence");
    }
}
