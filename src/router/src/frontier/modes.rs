//! Frontier escalation ladder — VISION §"The escalation ladder".
//!
//!  `EscalationMode` is the canonical taxonomy of frontier-engagement
//! stages (replaces the old `FrontierMode` "four involvement
//! modes" enum, which never matched the VISION ladder).  Stages are tried
//! in order (filter → question → team → turnover) after every local model
//! in a `model_group` fails.  The ladder *runtime* lives in
//! `crate::dispatch::escalation`; this module owns the taxonomy plus the
//! audit types it writes to.

use serde::{Deserialize, Serialize};

/// A stage of the frontier escalation ladder.
///
/// Each stage is a discrete policy governing how much context, data, and
/// agency the frontier model receives (doc/router/VISION.md §"The four
/// modes"). Stages escalate filter → question → team → turnover: the
/// less-permissive stages are tried first so frontier calls are
/// progressively more expensive, never all-or-nothing.
///
/// `serde` is derived so a configured `model_groups[g].escalation.modes`
/// list deserializes from config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// The audit payload shape for one frontier interaction — the same fields the
/// durable audit record carries (`payload`/`raw_response`/`trigger`/
/// `timestamp`). `crate::dispatch::escalation` builds one per mode run and
/// emits it via `crate::audit::emit` with `kind = "escalation"` plus the
/// `mode`/`accepted` fields.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub payload: String,
    pub raw_response: String,
    pub trigger: String,
    pub timestamp: u64,
}
#[cfg(test)]
#[path = "../../tests/frontier_modes.rs"]
mod tests;
