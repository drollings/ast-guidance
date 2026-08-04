use crate::*;
use internment::ArcIntern;

struct TestComponent {
    name: ArcIntern<str>,
    value: i32,
}

impl FieldAccess for TestComponent {
    fn set_field(&mut self, name: &str, value: &str) -> Result<(), FieldError> {
        match name {
            "value" => {
                self.value = value.parse().map_err(|_| FieldError::Parse(value.into()))?;
                Ok(())
            }
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "value" => Ok(self.value.to_string()),
            _ => Err(FieldError::NotFound(name.into())),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["value"]
    }
}

impl Describable for TestComponent {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({"name": &*self.name, "value": self.value})
    }
}

impl WorkUnit for TestComponent {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok(format!("computed: {}", self.value * 2)))
    }
}

impl_component!(TestComponent);

#[test]
fn test_field_access() {
    let mut comp = TestComponent {
        name: ArcIntern::from("test"),
        value: 42,
    };
    assert_eq!(comp.get_field("value").unwrap(), "42");
    comp.set_field("value", "99").unwrap();
    assert_eq!(comp.get_field("value").unwrap(), "99");
    assert!(comp.set_field("nonexistent", "x").is_err());
}

#[test]
fn test_component_trait_object() {
    let comp = TestComponent {
        name: ArcIntern::from("test"),
        value: 10,
    };
    let boxed: Box<dyn Component> = Box::new(comp);
    assert_eq!(boxed.name(), "test");
}

// --- Derive macro tests ---

#[derive(FieldAccess, Describable)]
struct BasicConfig {
    name: String,
    count: u32,
    enabled: bool,
}

impl WorkUnit for BasicConfig {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("done"))
    }
}

impl_component!(BasicConfig);

#[test]
fn test_derive_field_access_basic() {
    let mut cfg = BasicConfig {
        name: "test".into(),
        count: 5,
        enabled: true,
    };
    assert_eq!(cfg.get_field("name").unwrap(), "test");
    assert_eq!(cfg.get_field("count").unwrap(), "5");
    assert_eq!(cfg.get_field("enabled").unwrap(), "true");
    cfg.set_field("count", "10").unwrap();
    assert_eq!(cfg.get_field("count").unwrap(), "10");
    assert!(cfg.set_field("nonexistent", "x").is_err());
}

#[test]
fn test_derive_field_names() {
    let cfg = BasicConfig {
        name: String::new(),
        count: 0,
        enabled: false,
    };
    let names = cfg.field_names();
    assert!(names.contains(&"name"));
    assert!(names.contains(&"count"));
    assert!(names.contains(&"enabled"));
}

#[test]
fn test_derive_describable_basic() {
    let cfg = BasicConfig {
        name: "test".into(),
        count: 5,
        enabled: true,
    };
    let schema = cfg.describe();
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["name"]["type"], "string");
    assert_eq!(schema["properties"]["count"]["type"], "integer");
    assert_eq!(schema["properties"]["enabled"]["type"], "boolean");
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("name")));
    assert!(required.contains(&serde_json::json!("count")));
}

#[derive(FieldAccess, Describable)]
struct ConstrainedConfig {
    #[field(desc = "TCP port", min = 1, max = 65535)]
    port: u16,
    #[field(desc = "Retry count", min = 0, max = 10)]
    retries: u32,
    #[field(desc = "Host name")]
    host: String,
}

impl WorkUnit for ConstrainedConfig {
    fn name(&self) -> &str {
        &self.host
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("done"))
    }
}

impl_component!(ConstrainedConfig);

#[test]
fn test_derive_field_access_constraint_valid() {
    let mut cfg = ConstrainedConfig {
        port: 8079,
        retries: 3,
        host: "localhost".into(),
    };
    cfg.set_field("port", "9000").unwrap();
    assert_eq!(cfg.port, 9000);
    cfg.set_field("retries", "5").unwrap();
    assert_eq!(cfg.retries, 5);
}

#[test]
fn test_derive_field_access_constraint_below_min() {
    let mut cfg = ConstrainedConfig {
        port: 8079,
        retries: 3,
        host: "localhost".into(),
    };
    let err = cfg.set_field("port", "0").unwrap_err();
    match err {
        FieldError::Constraint(msg) => {
            assert!(msg.contains("below minimum"), "unexpected: {msg}");
        }
        other => panic!("expected Constraint, got {other:?}"),
    }
}

#[test]
fn test_derive_field_access_constraint_above_max() {
    let mut cfg = ConstrainedConfig {
        port: 8079,
        retries: 3,
        host: "localhost".into(),
    };
    let err = cfg.set_field("port", "70000").unwrap_err();
    match err {
        FieldError::Constraint(msg) => {
            assert!(msg.contains("above maximum"), "unexpected: {msg}");
        }
        other => panic!("expected Constraint, got {other:?}"),
    }
}

#[test]
fn test_derive_field_access_constraint_zero_min() {
    let mut cfg = ConstrainedConfig {
        port: 8079,
        retries: 3,
        host: "localhost".into(),
    };
    cfg.set_field("retries", "0").unwrap();
    assert_eq!(cfg.retries, 0);
}

#[test]
fn test_derive_describable_with_constraints() {
    let cfg = ConstrainedConfig {
        port: 8079,
        retries: 3,
        host: "localhost".into(),
    };
    let schema = cfg.describe();
    let port_schema = &schema["properties"]["port"];
    assert_eq!(port_schema["type"], "integer");
    assert_eq!(port_schema["description"], "TCP port");
    assert_eq!(port_schema["minimum"], "1");
    assert_eq!(port_schema["maximum"], "65535");
}

#[test]
fn test_schema_provider() {
    let cfg = ConstrainedConfig {
        port: 8079,
        retries: 3,
        host: "localhost".into(),
    };
    let fields = cfg.schema();
    assert_eq!(fields.len(), 3);
    let port = &fields[0];
    assert_eq!(port.name, "port");
    assert_eq!(port.type_name, "u16");
    assert_eq!(port.description.as_deref(), Some("TCP port"));
    assert_eq!(port.min, Some(1.0));
    assert_eq!(port.max, Some(65535.0));
    assert!(port.required);
    let host = &fields[2];
    assert_eq!(host.name, "host");
    assert_eq!(host.type_name, "String");
    assert!(host.min.is_none());
}

#[test]
fn test_derive_component_blanket_impl() {
    let cfg = ConstrainedConfig {
        port: 8079,
        retries: 3,
        host: "localhost".into(),
    };
    let boxed: Box<dyn Component> = Box::new(cfg);
    assert_eq!(boxed.field_names().len(), 3);
}

#[derive(FieldAccess)]
struct FloatMinConfig {
    #[field(min = 1.5)]
    scale: f64,
}

#[test]
fn field_min_float_sets_min_not_max() {
    let mut c = FloatMinConfig { scale: 2.0 };
    assert!(c.set_field("scale", "0.5").is_err());
    assert!(c.set_field("scale", "2.0").is_ok());
    assert!(c.set_field("scale", "1.5").is_ok());
}

#[derive(FieldAccess, Describable)]
#[allow(dead_code)]
struct OptionalConfig {
    name: String,
    #[field(required = false)]
    nickname: Option<String>,
}

impl WorkUnit for OptionalConfig {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("done"))
    }
}

impl_component!(OptionalConfig);

#[test]
fn describable_required_false_excludes_from_required_array() {
    let c = OptionalConfig {
        name: "x".into(),
        nickname: None,
    };
    let desc = c.describe();
    let required = desc["required"].as_array().unwrap();
    assert!(
        required.iter().any(|v| v == "name"),
        "name should be required"
    );
    assert!(
        !required.iter().any(|v| v == "nickname"),
        "nickname should not be required"
    );
}

#[test]
fn schema_provider_required_false() {
    let c = OptionalConfig {
        name: "x".into(),
        nickname: None,
    };
    let fields = c.schema();
    let name_field = fields.iter().find(|f| f.name == "name").unwrap();
    assert!(name_field.required);
    let nick_field = fields.iter().find(|f| f.name == "nickname").unwrap();
    assert!(!nick_field.required);
}

#[derive(FieldAccess, Describable)]
#[allow(dead_code)]
struct FormatConfig {
    #[field(desc = "Endpoint URL", format = "url")]
    endpoint: String,
    #[field(desc = "Timeout", format = "duration")]
    timeout_ms: u64,
    name: String,
}

impl WorkUnit for FormatConfig {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("done"))
    }
}

impl_component!(FormatConfig);

#[test]
fn format_attribute_reaches_describable_json() {
    let c = FormatConfig {
        endpoint: "https://example.com".into(),
        timeout_ms: 5000,
        name: "test".into(),
    };
    let desc = c.describe();
    let endpoint_schema = &desc["properties"]["endpoint"];
    assert_eq!(endpoint_schema["description"], "Endpoint URL");
    assert_eq!(endpoint_schema["format"], "url");
    let timeout_schema = &desc["properties"]["timeout_ms"];
    assert_eq!(timeout_schema["format"], "duration");
}

#[test]
fn format_attribute_reaches_field_schema() {
    let c = FormatConfig {
        endpoint: "https://example.com".into(),
        timeout_ms: 5000,
        name: "test".into(),
    };
    let fields = c.schema();
    let endpoint_field = fields.iter().find(|f| f.name == "endpoint").unwrap();
    assert_eq!(endpoint_field.format.as_deref(), Some("url"));
    let timeout_field = fields.iter().find(|f| f.name == "timeout_ms").unwrap();
    assert_eq!(timeout_field.format.as_deref(), Some("duration"));
    let name_field = fields.iter().find(|f| f.name == "name").unwrap();
    assert!(name_field.format.is_none());
}

// --- FieldAccess derive: string sanitization tests ---

#[derive(FieldAccess, Describable)]
struct SanitizedConfig {
    #[field(max_len = 10)]
    short_name: String,
    #[field(sanitize = "trim")]
    trimmed: String,
    #[field(sanitize = "lowercase")]
    lowercased: String,
    #[field(pattern = "hello")]
    patterned: String,
    name: String,
}

impl WorkUnit for SanitizedConfig {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("done"))
    }
}

impl_component!(SanitizedConfig);

#[test]
fn max_len_rejects_long_string() {
    let mut cfg = SanitizedConfig {
        short_name: "x".into(),
        trimmed: "x".into(),
        lowercased: "x".into(),
        patterned: "hello".into(),
        name: "test".into(),
    };
    let err = cfg
        .set_field("short_name", "this is way too long")
        .unwrap_err();
    match err {
        FieldError::Constraint(msg) => {
            assert!(msg.contains("exceeds maximum"), "unexpected: {msg}");
        }
        other => panic!("expected Constraint, got {other:?}"),
    }
}

#[test]
fn max_len_allows_short_string() {
    let mut cfg = SanitizedConfig {
        short_name: "x".into(),
        trimmed: "x".into(),
        lowercased: "x".into(),
        patterned: "hello".into(),
        name: "test".into(),
    };
    cfg.set_field("short_name", "ok").unwrap();
    assert_eq!(cfg.short_name, "ok");
}

#[test]
fn sanitize_trim_strips_whitespace() {
    let mut cfg = SanitizedConfig {
        short_name: "x".into(),
        trimmed: "x".into(),
        lowercased: "x".into(),
        patterned: "hello".into(),
        name: "test".into(),
    };
    cfg.set_field("trimmed", "  hello  ").unwrap();
    assert_eq!(cfg.trimmed, "hello");
}

#[test]
fn sanitize_lowercase_lowercases_input() {
    let mut cfg = SanitizedConfig {
        short_name: "x".into(),
        trimmed: "x".into(),
        lowercased: "x".into(),
        patterned: "hello".into(),
        name: "test".into(),
    };
    cfg.set_field("lowercased", "HELLO World").unwrap();
    assert_eq!(cfg.lowercased, "hello world");
}

#[test]
fn pattern_rejects_non_matching() {
    let mut cfg = SanitizedConfig {
        short_name: "x".into(),
        trimmed: "x".into(),
        lowercased: "x".into(),
        patterned: "hello".into(),
        name: "test".into(),
    };
    let err = cfg.set_field("patterned", "goodbye").unwrap_err();
    match err {
        FieldError::Constraint(msg) => {
            assert!(msg.contains("does not match pattern"), "unexpected: {msg}");
        }
        other => panic!("expected Constraint, got {other:?}"),
    }
}

#[test]
fn pattern_allows_matching() {
    let mut cfg = SanitizedConfig {
        short_name: "x".into(),
        trimmed: "x".into(),
        lowercased: "x".into(),
        patterned: "hello".into(),
        name: "test".into(),
    };
    cfg.set_field("patterned", "say hello world").unwrap();
    assert_eq!(cfg.patterned, "say hello world");
}

#[derive(FieldAccess, Describable)]
struct OptionConfig {
    name: String,
    #[field(required = false)]
    nickname: Option<String>,
}

impl WorkUnit for OptionConfig {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("done"))
    }
}

impl_component!(OptionConfig);

#[derive(FieldAccess, Describable)]
struct EmptyIsNoneConfig {
    name: String,
    #[field(required = false)]
    nickname: Option<String>,
    #[field(required = false, empty_is_none = false)]
    email: Option<String>,
}

impl WorkUnit for EmptyIsNoneConfig {
    fn name(&self) -> &str {
        &self.name
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("done"))
    }
}

impl_component!(EmptyIsNoneConfig);

#[test]
fn option_string_empty_is_none_default() {
    let mut cfg = EmptyIsNoneConfig {
        name: "test".into(),
        nickname: Some("nick".into()),
        email: Some("e@x.com".into()),
    };
    cfg.set_field("nickname", "").unwrap();
    assert!(
        cfg.nickname.is_none(),
        "default empty_is_none should convert '' to None"
    );
}

#[test]
fn option_string_empty_is_none_false_keeps_some() {
    let mut cfg = EmptyIsNoneConfig {
        name: "test".into(),
        nickname: None,
        email: Some("e@x.com".into()),
    };
    cfg.set_field("email", "").unwrap();
    assert_eq!(
        cfg.email.as_deref(),
        Some(""),
        "empty_is_none=false should preserve Some('')"
    );
}

#[test]
fn option_string_empty_gives_none() {
    let mut cfg = OptionConfig {
        name: "test".into(),
        nickname: Some("nick".into()),
    };
    cfg.set_field("nickname", "").unwrap();
    assert!(cfg.nickname.is_none());
}

#[test]
fn option_string_non_empty_gives_some() {
    let mut cfg = OptionConfig {
        name: "test".into(),
        nickname: None,
    };
    cfg.set_field("nickname", "alice").unwrap();
    assert_eq!(cfg.nickname.as_deref(), Some("alice"));
}

#[test]
fn option_string_get_returns_empty_when_none() {
    let cfg = OptionConfig {
        name: "test".into(),
        nickname: None,
    };
    assert_eq!(cfg.get_field("nickname").unwrap(), "");
}

#[test]
fn option_string_get_returns_value_when_some() {
    let cfg = OptionConfig {
        name: "test".into(),
        nickname: Some("bob".into()),
    };
    assert_eq!(cfg.get_field("nickname").unwrap(), "bob");
}

#[test]
fn describable_includes_max_len_and_pattern() {
    let cfg = SanitizedConfig {
        short_name: "x".into(),
        trimmed: "x".into(),
        lowercased: "x".into(),
        patterned: "hello".into(),
        name: "test".into(),
    };
    let desc = cfg.describe();
    let short_schema = &desc["properties"]["short_name"];
    assert_eq!(short_schema["maxLength"], "10");
    let patterned_schema = &desc["properties"]["patterned"];
    assert_eq!(patterned_schema["pattern"], "hello");

    let fields = cfg.schema();
    let short_field = fields.iter().find(|f| f.name == "short_name").unwrap();
    assert_eq!(short_field.max_len, Some(10));
    let pattern_field = fields.iter().find(|f| f.name == "patterned").unwrap();
    assert_eq!(pattern_field.pattern.as_deref(), Some("hello"));
}

#[test]
fn describable_option_field_not_required() {
    let cfg = OptionConfig {
        name: "test".into(),
        nickname: None,
    };
    let desc = cfg.describe();
    let required = desc["required"].as_array().unwrap();
    assert!(required.iter().any(|v| v == "name"));
    assert!(!required.iter().any(|v| v == "nickname"));
}

// --- M3: Runtime type identification and typed data tests ---

#[test]
fn as_any_returns_correct_concrete_type() {
    let comp = TestComponent {
        name: ArcIntern::from("test"),
        value: 42,
    };
    let arc: Arc<dyn Component> = Arc::new(comp);
    let any_ref = arc.as_any();
    let downcasted = any_ref.downcast_ref::<TestComponent>();
    assert!(downcasted.is_some());
    assert_eq!(downcasted.unwrap().value, 42);
}

#[test]
fn downcast_ref_succeeds_for_correct_type() {
    let comp = TestComponent {
        name: ArcIntern::from("test"),
        value: 10,
    };
    let arc: Arc<dyn Component> = Arc::new(comp);
    assert!(component_downcast_ref::<TestComponent>(&*arc).is_some());
}

#[test]
fn downcast_ref_fails_for_wrong_type() {
    let comp = TestComponent {
        name: ArcIntern::from("test"),
        value: 10,
    };
    let arc: Arc<dyn Component> = Arc::new(comp);
    assert!(component_downcast_ref::<ConstrainedConfig>(&*arc).is_none());
}

#[test]
fn type_name_returns_concrete_type_through_arc_dyn() {
    let comp = TestComponent {
        name: ArcIntern::from("test"),
        value: 10,
    };
    let arc: Arc<dyn Component> = Arc::new(comp);
    let name = arc.type_name();
    assert!(
        name.contains("TestComponent"),
        "expected concrete type name, got: {name}"
    );
}

#[test]
fn try_as_any_mut_succeeds_with_single_owner() {
    let mut arc: Arc<dyn Component> = Arc::new(TestComponent {
        name: ArcIntern::from("t"),
        value: 1,
    });
    let result = ComponentArcExt::try_as_any_mut(&mut arc);
    assert!(result.is_some());
}

#[test]
fn try_as_any_mut_returns_none_with_multiple_owners() {
    let arc: Arc<dyn Component> = Arc::new(TestComponent {
        name: ArcIntern::from("t"),
        value: 1,
    });
    let _clone = Arc::clone(&arc);
    let mut arc_mut = arc;
    let result = ComponentArcExt::try_as_any_mut(&mut arc_mut);
    assert!(result.is_none());
}

#[test]
fn as_any_mut_panic_message_mentions_try_variant() {
    let arc: Arc<dyn Component> = Arc::new(TestComponent {
        name: ArcIntern::from("t"),
        value: 1,
    });
    let _clone = Arc::clone(&arc);
    let mut arc_mut = arc;
    let msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = arc_mut.as_any_mut();
    }))
    .unwrap_err();
    let payload = if let Some(s) = msg.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = msg.downcast_ref::<&str>() {
        s.to_string()
    } else {
        String::new()
    };
    assert!(
        payload.contains("ComponentArcExt::try_as_any_mut"),
        "panic message should mention ComponentArcExt::try_as_any_mut, got: {payload}"
    );
}

#[test]
fn test_work_unit_delegates_to_inner_for_arc_dyn_work_unit() {
    let inner = TestComponent {
        name: ArcIntern::from("test"),
        value: 10,
    };
    let arc: Arc<dyn WorkUnit> = Arc::new(inner);
    assert_eq!(arc.name(), "test");
    let ctx = WorkContext::default();
    let result = arc.execute(&ctx).unwrap();
    assert_eq!(result.message, "computed: 20");
}

#[test]
fn test_work_unit_delegates_to_inner_for_arc_dyn_component() {
    let inner = TestComponent {
        name: ArcIntern::from("test"),
        value: 10,
    };
    let arc: Arc<dyn Component> = Arc::new(inner);
    assert_eq!(arc.name(), "test");
    let ctx = WorkContext::default();
    let result = arc.execute(&ctx).unwrap();
    assert_eq!(result.message, "computed: 20");
}

#[test]
fn set_field_on_shared_arc_returns_readonly() {
    let arc: Arc<dyn Component> = Arc::new(TestComponent {
        name: ArcIntern::from("shared"),
        value: 1,
    });
    let _clone = Arc::clone(&arc);
    let mut arc_mut = arc;
    match arc_mut.set_field("value", "42") {
        Err(FieldError::ReadOnly(field, reason)) => {
            assert_eq!(field, "value");
            assert!(reason.contains("multiple owners"), "reason: {reason}");
        }
        other => panic!("expected ReadOnly, got: {other:?}"),
    }
}

// --- `impl_fieldless!` macro tests ---

struct FieldlessUnit;

impl WorkUnit for FieldlessUnit {
    fn name(&self) -> &str {
        "fieldless"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        Ok(WorkOutput::ok("no-op"))
    }
}

impl Describable for FieldlessUnit {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

impl_component!(FieldlessUnit);
impl_fieldless!(FieldlessUnit);

#[test]
fn impl_fieldless_produces_fieldless_no_ops() {
    let unit = FieldlessUnit;
    assert!(unit.field_names().is_empty());
    let err = unit.get_field("anything").unwrap_err();
    match err {
        FieldError::NotFound(msg) => {
            assert_eq!(msg, "FieldlessUnit has no configurable fields")
        }
        other => panic!("expected NotFound, got: {other:?}"),
    }
    let mut unit = FieldlessUnit;
    let err = unit.set_field("anything", "value").unwrap_err();
    match err {
        FieldError::NotFound(msg) => {
            assert_eq!(msg, "FieldlessUnit has no configurable fields")
        }
        other => panic!("expected NotFound, got: {other:?}"),
    }
}

struct GenericFieldlessWrapper<U> {
    inner: U,
}
impl<U: Component> WorkUnit for GenericFieldlessWrapper<U> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        self.inner.depends()
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        self.inner.provides()
    }
    fn execute(&self, ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        self.inner.execute(ctx)
    }
}
impl_fieldless!(generic (U: Component + 'static) for GenericFieldlessWrapper<U>);

#[test]
fn impl_fieldless_generic_arm_works() {
    let wrapped = GenericFieldlessWrapper {
        inner: FieldlessUnit,
    };
    assert!(wrapped.field_names().is_empty());
    assert!(wrapped.get_field("x").is_err());
}
