//! `Target → WorkUnit` bridge.
//!
//! A [`Target`] (from the target registry / resolver world, with bitset
//! capabilities) adapted to the fluent-wvr [`Component`] world so the
//! resolver's `ExecutionPlan` can run under `SupervisedBatch` / `WorkUnit` semantics.
//!
//! This module replaces the pruned `DagExecutor` Its `execute` mirrors
//! `CommandUnit`'s semantics exactly via the shared `run_shell_command`
//! helper — sequential determinism comes from walking
//! `ExecutionPlan::order`, and per-target result linkage is preserved via
//! `Target::id`.

use crate::target::{CapabilityRegistry, Target};
use crate::work_unit::run_shell_command;
use fluent_wvr::prelude::*;
use internment::ArcIntern;

/// A [`Target`] adapted to the `WorkUnit`/`Component` interface.
///
/// `depends` / `provides` are the `Target`'s bitsets resolved to capability
/// names via a [`CapabilityRegistry`] (bit indices are per-registry; indices
/// that map to no name are skipped with a `tracing::warn!`).
#[derive(Debug, Clone)]
pub struct TargetWorkUnit {
    pub name: ArcIntern<str>,
    pub depends: Vec<ArcIntern<str>>,
    pub provides: Vec<ArcIntern<str>>,
    pub command: String,
    pub essential: bool,
}

impl TargetWorkUnit {
    /// Maps the `Target`'s `BitVec` `depends`/`provides` to capability names.
    ///
    /// Bit indices are per-registry; indices with no registered name are
    /// skipped with a `tracing::warn!` rather than failing.
    pub fn from_target(target: &Target, caps: &CapabilityRegistry) -> Self {
        let depends = bitvec_to_names(target.depends.iter_ones(), caps, "depends", &target.name);
        let provides = bitvec_to_names(target.provides.iter_ones(), caps, "provides", &target.name);
        Self {
            name: target.name.clone(),
            depends,
            provides,
            command: target.command.clone(),
            essential: target.essential,
        }
    }
}

fn bitvec_to_names(
    bits: impl Iterator<Item = usize>,
    caps: &CapabilityRegistry,
    kind: &str,
    target: &str,
) -> Vec<ArcIntern<str>> {
    bits.filter_map(|idx| {
        if let Some(name) = caps.get_name(idx) {
            Some(name)
        } else {
            tracing::warn!(
                target,
                kind,
                bit_index = idx,
                "bit index not mapped in CapabilityRegistry; skipping"
            );
            None
        }
    })
    .collect()
}

impl WorkUnit for TargetWorkUnit {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &self.depends
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &self.provides
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        run_shell_command(&self.name, &self.command, ctx)
    }
}

impl FieldAccess for TargetWorkUnit {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        match name {
            "name" => {
                self.name = ArcIntern::from(value);
                Ok(())
            }
            "command" => {
                self.command = value.to_string();
                Ok(())
            }
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "name" => Ok(self.name.to_string()),
            "command" => Ok(self.command.clone()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["name", "command"]
    }
}

impl fluent_wvr::Describable for TargetWorkUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Target name" },
                "command": { "type": "string", "description": "Shell command to execute" }
            },
            "required": ["name", "command"]
        })
    }
}

impl_component!(TargetWorkUnit);

#[cfg(test)]
#[path = "../tests/target_work_unit.rs"]
mod tests;
