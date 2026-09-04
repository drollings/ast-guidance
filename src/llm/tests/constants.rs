//! ROADMAP_20260903_LLM M6.2 — LLM constants goldens.
//!
//! Canonical home for the four LLM-domain constant goldens: the locked
//! values plus calibration notes (see `src/llm/CALIBRATION.md`).
//!
//! NOTE (M11): the `dual_path_equality` test died with the
//! `common_core::constants` shims it pinned. The locked values below are
//! the lasting contract.
//!
//! Calibration (roadmap §1, M10): these are task-value budgets (context
//! width, wall-clock budgets, retry cadence), not producer confidence —
//! a generous timeout is never "the model is sure", and the values move
//! unchanged here; retuning them is M10.

use fluent_llm::constants::*;

// ── Locked values ─────────────────────────────────────────────────────────

#[test]
fn values_match_spec() {
    assert_eq!(MAX_EMBEDDING_DIMENSIONS, 4_096);
    assert_eq!(DEFAULT_TOTAL_TIMEOUT_MS, 300_000);
    assert_eq!(DEFAULT_IDLE_TIMEOUT_MS, 30_000);
    assert_eq!(DEFAULT_RETRY_INTERVAL_S, 1);
}
