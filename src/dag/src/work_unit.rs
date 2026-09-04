use bon::Builder;
use fluent_wvr::prelude::*;
use internment::ArcIntern;

#[derive(Builder)]
#[builder(start_fn = new)]
pub struct CommandUnit {
    name: String,
    command: String,
    #[builder(default)]
    depends: Vec<ArcIntern<str>>,
    #[builder(default)]
    provides: Vec<ArcIntern<str>>,
}

/// Shared shell-execution semantics for `CommandUnit` and `TargetWorkUnit`.
///
/// Single source of truth for the `dry_run` / no-op / `run_shell_capture`
/// contract so the two shell-backed work units never diverge.
pub(crate) fn run_shell_command(
    name: &str,
    command: &str,
    ctx: &WorkContext,
) -> Result<WorkOutput, WorkError> {
    if ctx.dry_run {
        return Ok(WorkOutput::ok(format!(
            "[DRY-RUN] would execute: {command}"
        )));
    }
    if command.is_empty() {
        return Ok(WorkOutput::ok(format!("no-op: {name}")));
    }
    let output = common_core::shell::run_shell_capture(command)
        .map_err(|e| WorkError::Execution(format!("command failed: {e}")))?;
    if output.success {
        Ok(WorkOutput::ok_with_data(
            format!("{name} completed"),
            serde_json::json!({"stdout": output.stdout}),
        ))
    } else {
        Err(WorkError::Execution(format!(
            "{name} failed: {}",
            output.stderr
        )))
    }
}

impl WorkUnit for CommandUnit {
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

impl FieldAccess for CommandUnit {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        match name {
            "name" => {
                self.name = value.to_string();
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
            "name" => Ok(self.name.clone()),
            "command" => Ok(self.command.clone()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["name", "command"]
    }
}

impl fluent_wvr::Describable for CommandUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Task name" },
                "command": { "type": "string", "description": "Shell command to execute" }
            },
            "required": ["name", "command"]
        })
    }
}

impl_component!(CommandUnit);

#[cfg(test)]
#[path = "../tests/work_unit.rs"]
mod tests;
