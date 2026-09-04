use super::*;

fn ctx(request: &str) -> RenderContext {
    RenderContext {
        request: request.to_string(),
        deps: HashMap::new(),
        upstream: HashMap::new(),
        chart: "test_chart".into(),
    }
}

#[test]
fn renders_request_and_for_loop() {
    let mut c = ctx("hello world");
    c.deps.insert(
        "files".into(),
        vec![
            BoundEntity {
                id: "f1".into(),
                kind: "file".into(),
                value: serde_json::json!({"name": "a.rs"}),
            },
            BoundEntity {
                id: "f2".into(),
                kind: "file".into(),
                value: serde_json::json!({"name": "b.rs"}),
            },
        ],
    );
    let out = render(
        "{{ request }}\n{% for e in deps.files %}- {{ e.value.name }}\n{% endfor %}",
        &c,
    )
    .unwrap();
    assert!(out.starts_with("hello world"));
    assert!(out.contains("- a.rs"));
    assert!(out.contains("- b.rs"));
}

#[test]
fn unknown_attribute_is_render_error() {
    let c = ctx("x");
    let err = render("{{ request }} {{ bogus }}", &c).unwrap_err();
    assert!(matches!(err, ChartError::Render { .. }));
}

#[test]
fn injection_string_neutralized() {
    let payload = "<script>alert(1)</script> {{ request }} {% if true %}x{% endif %}";
    let sanitized = sanitize_entity_value(serde_json::json!(payload));
    let s = sanitized.as_str().unwrap();
    // The template-open tokens must be broken by a zero-width space.
    assert!(!s.contains("{{"), "template markers must be escaped: {s:?}");
    assert!(!s.contains("{%"), "block markers must be escaped: {s:?}");
    // And it must not open a template block when rendered.
    let mut c = ctx("safe");
    c.deps.insert(
        "e".into(),
        vec![BoundEntity {
            id: "x".into(),
            kind: "e".into(),
            value: serde_json::json!(payload),
        }],
    );
    let out = render("{% for e in deps.e %}{{ e.value }}{% endfor %}", &c).unwrap();
    assert!(
        !out.contains("alert(1)"),
        "script must be neutralized: {out:?}"
    );
    // The zero-width escape is preserved as literal data (characters are
    // kept; only the template token is broken).
    assert!(out.contains('\u{200B}'));
}

#[test]
fn nul_and_control_chars_stripped() {
    let payload = "a\u{0}b\u{1}c\x1f\n   d";
    let sanitized = sanitize_entity_value(serde_json::json!(payload));
    assert_eq!(sanitized.as_str().unwrap(), "a b c d");
}

#[test]
fn over_cap_rendered_rejected() {
    let c = ctx("x");
    // A template that repeats the request more times than the cap.
    let template = format!(
        "{{% for i in range({CHART_RENDERED_MAX_CHARS} + 1) %}}{{ request }}{{% endfor %}}"
    );
    let err = render(&template, &c).unwrap_err();
    assert!(matches!(err, ChartError::Render { .. }));
}

#[test]
fn long_entity_truncated() {
    let long = "y".repeat(ENTITY_VALUE_MAX_CHARS * 2);
    let sanitized = sanitize_entity_value(serde_json::json!(long));
    let s = sanitized.as_str().unwrap();
    assert_eq!(s.chars().count(), ENTITY_VALUE_MAX_CHARS);
}

#[test]
fn golden_renders_appendix_a_target() {
    // Appendix A bug_triage "reproduce" target template with a fixed
    // request + one bound entity.
    let template = "Given the bug report {{ request }}, write a minimal reproduction plan.\n{% for e in deps.report %}Report entity: {{ e.value.title }}\n{% endfor %}";
    let mut c = RenderContext {
        request: "crash on startup when loading large projects".into(),
        deps: HashMap::new(),
        upstream: HashMap::new(),
        chart: "bug_triage".into(),
    };
    c.deps.insert(
        "report".into(),
        vec![BoundEntity {
            id: "issue-42".into(),
            kind: "report".into(),
            value: serde_json::json!({"title": "Segfault on project load"}),
        }],
    );
    let out = render(template, &c).unwrap();
    assert_eq!(
        out,
        "Given the bug report crash on startup when loading large projects, write a minimal reproduction plan.\nReport entity: Segfault on project load\n"
    );
}
