//! Chart template rendering — minijinja over a strict, sanitized context.
//!
//! A chart target's `template` is a minijinja source that may reference:
//!
//! - `request` — the user's message
//! - `deps.<name>` — the entities bound to a dependency name
//! - `upstream.<stage_id>.output` — a prior target's structured output
//! - `chart` — the chart name (provenance)
//!
//! Nothing else is exposed. The environment is strict (unknown attributes
//! are errors, not silent `undefined`), and every bound-entity string is
//! sanitized at the placeholder boundary before rendering.

use std::collections::HashMap;

use serde::Serialize;

use super::ChartError;

/// Max chars an entity string is truncated to before rendering.
pub const ENTITY_VALUE_MAX_CHARS: usize = 4_096;
/// Max chars a rendered template may produce; exceeding this is a
/// `ChartError::Render` (rejects pathological templates/entities at runtime,
/// not a panic).
pub const CHART_RENDERED_MAX_CHARS: usize = 32_768;

/// What a target's template may reference. Nothing else is exposed.
#[derive(Debug, Clone, Serialize)]
pub struct RenderContext {
    /// The user's message.
    pub request: String,
    /// dep name → bound entities.
    pub deps: HashMap<String, Vec<BoundEntity>>,
    /// Prior target outputs, keyed by stage id (M5).
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub upstream: HashMap<String, serde_json::Value>,
    /// Chart name (provenance).
    pub chart: String,
}

/// A single entity bound to a dep, sanitized before render.
#[derive(Debug, Clone, Serialize)]
pub struct BoundEntity {
    pub id: String,
    pub kind: String,
    /// Sanitized structured payload.
    pub value: serde_json::Value,
}

/// Build a fresh, strict minijinja environment. Renders are per-request and
/// short-lived; a fresh environment per render avoids any shared state.
/// The environment exposes no functions — only the context.
///
/// Note: minijinja 2.x folded `UnknownAttributes::Strict` into
/// `UndefinedBehavior::Strict` — under it, any unknown variable *or*
/// attribute access renders as an error instead of silently empty.
fn build_environment() -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env
}

/// Sanitize an entity's structured payload before it is exposed to a
/// template.
///
/// Policy (documented):
/// - Strings are truncated to `ENTITY_VALUE_MAX_CHARS` chars.
/// - Control characters and NUL are stripped.
/// - Excessive whitespace runs are collapsed to a single space.
/// - Strings (and string arrays) are escaped so literal `{{`, `{%`, or `{#`
///   in untrusted entity data cannot open a template construct.
/// - Numbers, booleans, nulls, and nested objects pass through with string
///   leaves sanitized recursively.
pub fn sanitize_entity_value(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => {
            serde_json::Value::String(escape_template_markers(&sanitize_string(&s)))
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(sanitize_entity_value).collect())
        }
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.into_iter()
                .map(|(k, val)| (k, sanitize_entity_value(val)))
                .collect(),
        ),
        other => other,
    }
}

/// Truncate, strip HTML/control chars/NUL, and collapse whitespace runs.
fn sanitize_string(s: &str) -> String {
    // Strip HTML tags (incl. <script>/<style>) + collapse whitespace — the
    // canonical untrusted-text sanitizer for LLM-bound content.
    let stripped = common_core::string::strip_html(s);
    let collapsed: String = stripped
        .chars()
        .map(|c| {
            if c == '\u{0}' || (c.is_control() && c != '\n' && c != '\t' && c != '\r') {
                ' '
            } else {
                c
            }
        })
        .collect();
    // Control chars replaced by spaces can create double-spaces; collapse
    // whitespace runs once more.
    let collapsed = collapse_whitespace(&collapsed);
    common_core::string::truncate_chars(&collapsed, ENTITY_VALUE_MAX_CHARS)
}

/// Collapse runs of whitespace (including newlines) into a single space.
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

/// Escape Jinja template-open markers so untrusted entity data cannot
/// introduce new template constructs.
///
/// Policy: a `{` that is the start of a `{{`, `{%`, or `{#` token gets a
/// zero-width space (`\u{200B}`) inserted **between** the opening braces.
/// The token is broken, so minijinja treats the data as literal text — but
/// the characters themselves are preserved (the ZWSP is invisible in most
/// renderings).
fn escape_template_markers(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' && matches!(chars.peek(), Some('{' | '%' | '#')) {
            out.push('{');
            out.push('\u{200B}'); // breaks the template token between the braces
        } else {
            out.push(c);
        }
    }
    out
}

/// Render a chart target's template against a sanitized context.
///
/// - Uses a fresh strict environment per call.
/// - The context is sanitized (entity strings escaped/truncated) before
///   rendering.
/// - Output is capped at `CHART_RENDERED_MAX_CHARS`; exceeding it returns
///   `ChartError::Render`.
pub fn render(template: &str, ctx: &RenderContext) -> Result<String, ChartError> {
    // Sanitize the context at the placeholder boundary: each bound entity's
    // value is run through `sanitize_entity_value`.
    let sanitized_deps: HashMap<String, Vec<BoundEntity>> = ctx
        .deps
        .iter()
        .map(|(dep, entities)| {
            (
                dep.clone(),
                entities
                    .iter()
                    .map(|e| BoundEntity {
                        id: e.id.clone(),
                        kind: e.kind.clone(),
                        value: sanitize_entity_value(e.value.clone()),
                    })
                    .collect(),
            )
        })
        .collect();
    let render_ctx = RenderContext {
        request: sanitize_string(&ctx.request),
        deps: sanitized_deps,
        upstream: ctx.upstream.clone(),
        chart: ctx.chart.clone(),
    };

    let env = build_environment();
    let tmpl = env
        .template_from_str(template)
        .map_err(|e| ChartError::Render {
            target: ctx.chart.clone(),
            detail: e.to_string(),
        })?;

    let rendered = tmpl.render(&render_ctx).map_err(|e| ChartError::Render {
        target: ctx.chart.clone(),
        detail: e.to_string(),
    })?;

    if rendered.chars().count() > CHART_RENDERED_MAX_CHARS {
        return Err(ChartError::Render {
            target: ctx.chart.clone(),
            detail: format!("rendered output exceeds {CHART_RENDERED_MAX_CHARS} chars"),
        });
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
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
        let out = render("{{ request }}\n{% for e in deps.files %}- {{ e.value.name }}\n{% endfor %}", &c).unwrap();
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
        assert!(!out.contains("alert(1)"), "script must be neutralized: {out:?}");
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
            "{{% for i in range({} + 1) %}}{{ request }}{{% endfor %}}",
            CHART_RENDERED_MAX_CHARS
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
}
