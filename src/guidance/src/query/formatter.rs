use std::fmt::Write;

use crate::query::synthesize::Stage;
use fluent_types::StageKind;

/// Output formatter — the fluent-wvr control plane for result presentation.
///
/// Each formatter converts a list of `Stage`s into a string representation.
/// The orchestrator dispatches to the registered formatter without branching
/// on format type.
pub trait Formatter: Send + Sync {
    /// Format stages into the target representation.
    fn format(&self, stages: &[Stage]) -> String;
}

/// Markdown table output.
pub struct MarkdownFormatter;

impl Formatter for MarkdownFormatter {
    fn format(&self, stages: &[Stage]) -> String {
        let mut out = String::new();
        for stage in stages {
            let kind = match stage.kind {
                StageKind::Prose => "Prose",
                StageKind::Code => "Code",
                StageKind::Metadata => "Metadata",
                StageKind::Insight => "Insight",
                StageKind::SkillDoc => "Skill",
                StageKind::NotFound => "Not Found",
                StageKind::CapabilityDoc => "Capability",
            };
            let _ = write!(out, "## {kind}\n\n");
            let _ = write!(
                out,
                "*Source: {}:{}*\n\n",
                stage.source,
                stage.line.unwrap_or(0)
            );
            out.push_str(&stage.content);
            out.push_str("\n\n---\n\n");
        }
        out
    }
}

/// JSON output.
pub struct JsonFormatter;

impl Formatter for JsonFormatter {
    fn format(&self, stages: &[Stage]) -> String {
        serde_json::json!({
            "stages": stages,
            "count": stages.len(),
        })
        .to_string()
    }
}

/// Compact JSON with previews.
pub struct CompactFormatter;

impl Formatter for CompactFormatter {
    fn format(&self, stages: &[Stage]) -> String {
        let summaries: Vec<serde_json::Value> = stages
            .iter()
            .map(|s| {
                serde_json::json!({
                    "kind": format!("{:?}", s.kind),
                    "source": s.source,
                    "line": s.line,
                    "preview": s.content.chars().take(80).collect::<String>(),
                })
            })
            .collect();
        serde_json::json!({
            "stages": summaries,
            "count": stages.len(),
        })
        .to_string()
    }
}

/// Debug output with full content.
pub struct DebugFormatter;

impl Formatter for DebugFormatter {
    fn format(&self, stages: &[Stage]) -> String {
        let mut out = String::new();
        out.push_str("=== Query Debug ===\n");
        let _ = writeln!(out, "Stages: {}\n", stages.len());
        for (i, stage) in stages.iter().enumerate() {
            let kind_str = format!("{:?}", stage.kind);
            let _ = writeln!(out, "[{}. {}]", i + 1, kind_str);
            let _ = writeln!(
                out,
                "  Source: {}:{}",
                stage.source,
                stage.line.unwrap_or(0)
            );
            let preview: String = stage.content.chars().take(120).collect();
            let _ = writeln!(
                out,
                "  Content ({} chars): {}",
                stage.content.len(),
                preview
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::synthesize::Synthesizer;
    use fluent_types::{GuidanceDoc, Member, MemberType, Meta};

    fn test_stages() -> Vec<Stage> {
        let doc = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "src/test.zig".into(),
                language: "zig".into(),
            },
            comment: Some("A test module.".into()),
            members: vec![Member {
                type_name: MemberType::FnDecl,
                name: "hello".into(),
                signature: Some("fn hello() void".into()),
                comment: Some("Says hello.".into()),
                line: Some(1),
                ..Member::default()
            }],
            ..GuidanceDoc::default()
        };
        Synthesizer::synthesize("hello", &doc, &["hello".to_string()])
    }

    #[test]
    fn test_markdown_formatter() {
        let stages = test_stages();
        let output = MarkdownFormatter.format(&stages);
        assert!(output.contains("## "));
        assert!(output.contains("hello"));
    }

    #[test]
    fn test_json_formatter() {
        let stages = test_stages();
        let output = JsonFormatter.format(&stages);
        assert!(output.contains("\"stages\""));
        assert!(output.contains("hello"));
    }

    #[test]
    fn test_compact_formatter() {
        let stages = test_stages();
        let output = CompactFormatter.format(&stages);
        assert!(output.contains("preview"));
    }

    #[test]
    fn test_debug_formatter() {
        let stages = test_stages();
        let output = DebugFormatter.format(&stages);
        assert!(output.contains("=== Query Debug ==="));
        assert!(output.contains("hello"));
    }
}
