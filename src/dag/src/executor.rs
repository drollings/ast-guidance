use std::collections::HashMap;

use crate::target::{Target, TargetRegistry};
use fluent_types::ExecutorKind;
use thiserror::Error;
use tracing::{info, span, Level};

use crate::resolver::ExecutionPlan;

#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("target not found: {0}")]
    TargetNotFound(String),
    #[error("execution failed for '{target}': {message}")]
    ExecutionFailed { target: String, message: String },
    #[error("WASM execution not yet implemented: {0}")]
    WasmNotImplemented(String),
    #[error("Docker execution not yet implemented: {0}")]
    DockerNotImplemented(String),
}

pub struct DagExecutor<'a> {
    registry: &'a TargetRegistry,
    results: HashMap<usize, ExecutionResult>,
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub bit_index: usize,
    pub name: String,
    pub success: bool,
    pub output: String,
}

impl<'a> DagExecutor<'a> {
    pub fn new(registry: &'a TargetRegistry) -> Self {
        Self {
            registry,
            results: HashMap::new(),
        }
    }

    pub fn execute(
        &mut self,
        plan: &ExecutionPlan,
    ) -> Result<Vec<ExecutionResult>, ExecutionError> {
        let span = span!(Level::INFO, "dag_execute", size = plan.len());
        let _enter = span.enter();
        let mut results = Vec::with_capacity(plan.len());
        for &bit_idx in &plan.order {
            let target = self
                .registry
                .get_by_bit_index(bit_idx)
                .ok_or_else(|| ExecutionError::TargetNotFound(format!("bit_index {bit_idx}")))?;
            info!(target = %target.name, "executing");
            let result = Self::execute_target(target, plan)?;
            results.push(result.clone());
            self.results.insert(bit_idx, result);
        }
        Ok(results)
    }

    fn execute_target(
        target: &Target,
        _plan: &ExecutionPlan,
    ) -> Result<ExecutionResult, ExecutionError> {
        match target.executor {
            ExecutorKind::Native => {
                if target.command.is_empty() {
                    return Ok(ExecutionResult {
                        bit_index: target.id as usize,
                        name: target.name.to_string(),
                        success: true,
                        output: String::new(),
                    });
                }
                let output =
                    common_core::shell::run_shell_capture(&target.command).map_err(|e| {
                        ExecutionError::ExecutionFailed {
                            target: target.name.to_string(),
                            message: e.to_string(),
                        }
                    })?;
                let success = output.success;
                let output_str = if success {
                    output.stdout
                } else {
                    output.stderr
                };
                Ok(ExecutionResult {
                    bit_index: target.id as usize,
                    name: target.name.to_string(),
                    success,
                    output: output_str,
                })
            }
            ExecutorKind::Wasm => Err(ExecutionError::WasmNotImplemented(target.name.to_string())),
            ExecutorKind::Docker => Err(ExecutionError::DockerNotImplemented(
                target.name.to_string(),
            )),
        }
    }

    pub fn results(&self) -> &HashMap<usize, ExecutionResult> {
        &self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolver::DependencyResolver;
    use crate::target::{Target, TargetRegistry};
    use bitvec::prelude::*;
    use fluent_types::{ExecutorKind, TargetType};

    fn make_bitset(bits: &[usize]) -> BitVec {
        let max = bits.iter().max().copied().unwrap_or(0) + 1;
        let mut bv = BitVec::with_capacity(max);
        bv.resize(max, false);
        for &bit in bits {
            if bit < bv.len() {
                bv.set(bit, true);
            }
        }
        bv
    }

    #[test]
    fn test_execute_noop_targets() {
        let targets = vec![
            Target::new()
                .id(0)
                .name("init".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(BitVec::new())
                .provides(make_bitset(&[0]))
                .build(),
            Target::new()
                .id(1)
                .name("process".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[0]))
                .provides(make_bitset(&[1]))
                .build(),
            Target::new()
                .id(2)
                .name("finalize".into())
                .target_type(TargetType::File)
                .executor(ExecutorKind::Native)
                .depends(make_bitset(&[1]))
                .provides(make_bitset(&[2]))
                .build(),
        ];
        let mut reg = TargetRegistry::new();
        for t in targets {
            reg.register(t).unwrap();
        }
        let resolver = DependencyResolver::new(&reg);
        let plan = resolver.resolve(&["finalize"]).expect("resolve");
        assert_eq!(plan.order.len(), 3);
        let mut executor = DagExecutor::new(&reg);
        let results = executor.execute(&plan).expect("execute");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].name, "init");
        assert!(results[0].success);
        assert_eq!(results[1].name, "process");
        assert!(results[1].success);
        assert_eq!(results[2].name, "finalize");
        assert!(results[2].success);
    }
}
