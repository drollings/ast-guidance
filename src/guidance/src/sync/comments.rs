use std::path::Path;

use thiserror::Error;

use crate::ast_parser::AstParser;
use fluent_types::GuidanceDoc;

#[derive(Error, Debug)]
pub enum SyncError {
    #[error("IO error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("source file not found: {0}")]
    SourceNotFound(String),
}

impl From<std::io::Error> for SyncError {
    fn from(e: std::io::Error) -> Self {
        SyncError::Io(common_core::error::IoError(e))
    }
}

pub fn sync_comments(
    source_path: &Path,
    doc: &GuidanceDoc,
    parser: &mut AstParser,
) -> Result<(), SyncError> {
    let source = common_core::io::read_to_string_err(source_path)?;
    let fresh_doc = parser.parse_file(source_path, &source).ok();
    let insertions = collect_comment_insertions(&source, doc, fresh_doc.as_ref());
    if insertions.is_empty() {
        return Ok(());
    }
    let modified = apply_insertions(&source, insertions);
    if modified != source {
        common_core::io::write_atomic(source_path, modified.as_bytes())?;
    }
    Ok(())
}

fn collect_comment_insertions(
    source: &str,
    doc: &GuidanceDoc,
    fresh_doc: Option<&GuidanceDoc>,
) -> Vec<(usize, Vec<String>)> {
    let lines: Vec<&str> = source.lines().collect();
    let mut insertions = Vec::new();

    for member in &doc.members {
        let Some(ref comment_text) = member.comment else {
            continue;
        };
        if comment_text.is_empty() || member.comment_generated {
            continue;
        }
        let line = fresh_doc
            .and_then(|fd| fd.members.iter().find(|m| m.name == member.name))
            .and_then(|m| m.line)
            .or(member.line);
        let Some(line) = line else {
            continue;
        };
        let idx = (line as usize).saturating_sub(1);
        if idx > 0 && idx <= lines.len() {
            let prev_line = lines[idx - 1].trim();
            if prev_line.starts_with("///") || prev_line.starts_with("//!") {
                continue;
            }
        }
        let comment_lines: Vec<String> = comment_text
            .as_str()
            .split('\n')
            .map(|l| format!("/// {l}"))
            .collect();
        if !comment_lines.is_empty() {
            insertions.push((idx, comment_lines));
        }
    }
    insertions
}

fn apply_insertions(source: &str, mut insertions: Vec<(usize, Vec<String>)>) -> String {
    if insertions.is_empty() {
        return source.to_string();
    }
    let lines: Vec<&str> = source.lines().collect();
    insertions.sort_by_key(|(idx, _)| *idx);

    let mut result = String::new();
    let mut line_idx = 0;

    for (insert_idx, comment_lines) in &insertions {
        while line_idx < *insert_idx && line_idx < lines.len() {
            result.push_str(lines[line_idx]);
            result.push('\n');
            line_idx += 1;
        }
        for comment_line in comment_lines {
            result.push_str(comment_line);
            result.push('\n');
        }
    }

    while line_idx < lines.len() {
        result.push_str(lines[line_idx]);
        result.push('\n');
        line_idx += 1;
    }

    result.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_types::{GuidanceDoc, Member, MemberType, Meta};

    #[test]
    fn test_insert_comment_on_function() {
        let source = "pub fn hello() void {\n    return;\n}\n";
        let doc = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "test.zig".into(),
                language: "zig".into(),
            },
            members: vec![Member {
                type_name: MemberType::FnDecl,
                name: "hello".into(),
                comment: Some("Greets the user.".into()),
                line: Some(1),
                ..Member::default()
            }],
            ..GuidanceDoc::default()
        };

        let insertions = collect_comment_insertions(source, &doc, None);
        let result = apply_insertions(source, insertions);
        assert!(result.starts_with("/// Greets the user.\npub fn hello"));
    }

    #[test]
    fn test_skip_existing_comment() {
        let source = "/// Existing doc\npub fn foo() void {}\n";
        let doc = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "test.zig".into(),
                language: "zig".into(),
            },
            members: vec![Member {
                type_name: MemberType::FnDecl,
                name: "foo".into(),
                comment: Some("Existing doc".into()),
                line: Some(2),
                ..Member::default()
            }],
            ..GuidanceDoc::default()
        };

        let insertions = collect_comment_insertions(source, &doc, None);
        assert!(
            insertions.is_empty(),
            "should not insert when comment exists"
        );
    }

    #[test]
    fn test_no_comment_is_noop() {
        let source = "pub fn foo() void {}\n";
        let doc = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "test.zig".into(),
                language: "zig".into(),
            },
            members: vec![Member {
                type_name: MemberType::FnDecl,
                name: "foo".into(),
                comment: None,
                line: Some(1),
                ..Member::default()
            }],
            ..GuidanceDoc::default()
        };

        let insertions = collect_comment_insertions(source, &doc, None);
        assert!(insertions.is_empty());
    }
}
