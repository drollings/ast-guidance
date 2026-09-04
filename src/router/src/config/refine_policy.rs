//! Router DTO for `spacy_rs::RefinePolicy` (F5).
//!
//! The DTO lives in `fluent-router` only — `spacy-rs` never imports it (crate-
//! boundary invariant). The router deserializes this shape from `RouterConfig`
//! JSON; the builder converts it `From`/`Into` `spacy_rs::RefinePolicy` at the
//! wiring site.
//!
//! Intentionally **not** promoted to `common-core` per
//! `doc/skills/common-core/SKILL.md:3-4` zero-domain rule: `RefinePolicy`
//! knows about `PROPN`/ArcEager thresholds (domain logic). The canonical type
//! remains `spacy_rs::RefinePolicy`; this DTO is the router-local serde mirror.

use serde::{Deserialize, Serialize};

fn default_min_overall() -> f64 { 0.7 }
fn default_min_role_coverage() -> f64 { 0.5 }
fn default_min_token_score() -> f64 { 0.5 }
fn default_true() -> bool { true }
fn default_unresolved_token_threshold() -> f64 { 0.3 }

/// Mirrors `spacy_rs::RefineMode` field-for-field, owned here so the router
/// config surface never deserializes a `spacy-rs` type directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RouterRefineMode {
    #[default]
    Off,
    OnUncertain,
    Always,
}

impl From<RouterRefineMode> for spacy_rs::RefineMode {
    fn from(v: RouterRefineMode) -> Self {
        match v {
            RouterRefineMode::Off => spacy_rs::RefineMode::Off,
            RouterRefineMode::OnUncertain => spacy_rs::RefineMode::OnUncertain,
            RouterRefineMode::Always => spacy_rs::RefineMode::Always,
        }
    }
}

impl From<spacy_rs::RefineMode> for RouterRefineMode {
    fn from(v: spacy_rs::RefineMode) -> Self {
        match v {
            spacy_rs::RefineMode::Off => RouterRefineMode::Off,
            spacy_rs::RefineMode::OnUncertain => RouterRefineMode::OnUncertain,
            spacy_rs::RefineMode::Always => RouterRefineMode::Always,
        }
    }
}

/// Router serde surface for `spacy_rs::RefinePolicy` — field-for-field mirror
/// (F5). `spacy_rs::RefinePolicy` is never deserialized directly from router
/// config; the builder converts `dto.into()`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
#[serde(default)]
pub struct RouterRefinePolicy {
    #[serde(default)]
    pub mode: RouterRefineMode,
    #[serde(default = "default_min_overall")]
    pub min_overall: f64,
    #[serde(default = "default_min_role_coverage")]
    pub min_role_coverage: f64,
    #[serde(default = "default_true")]
    pub refine_on_ties: bool,
    #[serde(default = "default_min_token_score")]
    pub min_token_score: f64,
    #[serde(default = "default_true")]
    pub refine_on_unresolved_critical_role: bool,
    #[serde(default = "default_true")]
    pub refine_on_unresolved_propn: bool,
    #[serde(default = "default_true")]
    pub refine_on_collision_note: bool,
    #[serde(default = "default_unresolved_token_threshold")]
    pub unresolved_token_threshold: f64,
}

impl Default for RouterRefinePolicy {
    fn default() -> Self {
        Self {
            mode: RouterRefineMode::Off,
            min_overall: default_min_overall(),
            min_role_coverage: default_min_role_coverage(),
            refine_on_ties: true,
            min_token_score: default_min_token_score(),
            refine_on_unresolved_critical_role: true,
            refine_on_unresolved_propn: true,
            refine_on_collision_note: true,
            unresolved_token_threshold: default_unresolved_token_threshold(),
        }
    }
}

impl From<RouterRefinePolicy> for spacy_rs::RefinePolicy {
    fn from(dto: RouterRefinePolicy) -> Self {
        spacy_rs::RefinePolicy {
            mode: dto.mode.into(),
            min_overall: dto.min_overall,
            min_role_coverage: dto.min_role_coverage,
            refine_on_ties: dto.refine_on_ties,
            min_token_score: dto.min_token_score,
            refine_on_unresolved_critical_role: dto.refine_on_unresolved_critical_role,
            refine_on_unresolved_propn: dto.refine_on_unresolved_propn,
            refine_on_collision_note: dto.refine_on_collision_note,
            unresolved_token_threshold: dto.unresolved_token_threshold,
        }
    }
}

impl From<spacy_rs::RefinePolicy> for RouterRefinePolicy {
    fn from(v: spacy_rs::RefinePolicy) -> Self {
        Self {
            mode: v.mode.into(),
            min_overall: v.min_overall,
            min_role_coverage: v.min_role_coverage,
            refine_on_ties: v.refine_on_ties,
            min_token_score: v.min_token_score,
            refine_on_unresolved_critical_role: v.refine_on_unresolved_critical_role,
            refine_on_unresolved_propn: v.refine_on_unresolved_propn,
            refine_on_collision_note: v.refine_on_collision_note,
            unresolved_token_threshold: v.unresolved_token_threshold,
        }
    }
}
#[cfg(test)]
#[path = "../../tests/config_refine_policy.rs"]
mod tests;
