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
mod tests {
    use super::*;

    #[test]
    fn test_command_unit_noop() {
        let result = CommandUnit::new()
            .name("noop".into())
            .command(String::new())
            .build()
            .execute(&WorkContext::default())
            .unwrap();
        assert!(result.success);
    }
    #[test]
    fn test_command_unit_dry_run() {
        let result = CommandUnit::new()
            .name("dry".into())
            .command("echo hello".into())
            .build()
            .execute(&WorkContext {
                dry_run: true,
                ..WorkContext::default()
            })
            .unwrap();
        assert!(result.message.contains("DRY-RUN"));
    }
    #[test]
    fn test_command_unit_true() {
        let result = CommandUnit::new()
            .name("true_cmd".into())
            .command("true".into())
            .build()
            .execute(&WorkContext::default())
            .unwrap();
        assert!(result.success);
    }
    #[test]
    fn test_command_unit_false() {
        let result = CommandUnit::new()
            .name("false_cmd".into())
            .command("false".into())
            .build()
            .execute(&WorkContext::default());
        assert!(result.is_err());
    }
    #[test]
    fn test_command_unit_bon_builder() {
        let unit = CommandUnit::new()
            .name("build".into())
            .command("make".into())
            .depends(vec![ArcIntern::from("compile")])
            .provides(vec![ArcIntern::from("artifact")])
            .build();
        assert_eq!(unit.name(), "build");
        assert_eq!(&*unit.depends()[0], "compile");
        assert_eq!(&*unit.provides()[0], "artifact");
    }
    #[test]
    fn test_command_unit_field_access() {
        let mut unit = CommandUnit::new()
            .name("test".into())
            .command("echo hi".into())
            .build();
        assert_eq!(unit.get_field("name").unwrap(), "test");
        assert_eq!(unit.get_field("command").unwrap(), "echo hi");
        unit.set_field("name", "renamed").unwrap();
        assert_eq!(unit.get_field("name").unwrap(), "renamed");
        assert!(unit.set_field("nonexistent", "x").is_err());
    }
    #[test]
    fn test_command_unit_describable() {
        let unit = CommandUnit::new()
            .name("test".into())
            .command("echo hi".into())
            .build();
        let schema = unit.describe();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["name"]["type"], "string");
    }
    #[test]
    fn test_command_unit_is_component() {
        fn assert_component<T: Component>() {}
        assert_component::<CommandUnit>();
        let unit = CommandUnit::new()
            .name("test".into())
            .command("echo hi".into())
            .build();
        let _: &dyn Component = &unit;
    }
}
