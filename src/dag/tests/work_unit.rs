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
