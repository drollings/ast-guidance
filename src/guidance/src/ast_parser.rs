use std::path::Path;

use fluent_types::{FileType, GuidanceDoc, Member, MemberType, Meta, Param, Span};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("failed to read file: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("failed to parse source: {0}")]
    Syntax(String),
    #[error("unsupported language for file: {0}")]
    UnsupportedLanguage(String),
}

impl From<std::io::Error> for ParseError {
    fn from(e: std::io::Error) -> Self {
        ParseError::Io(common_core::error::IoError(e))
    }
}

pub struct AstParser {
    zig: tree_sitter::Parser,
    python: tree_sitter::Parser,
    rust: tree_sitter::Parser,
}

impl Default for AstParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AstParser {
    pub fn new() -> Self {
        let mut zig = tree_sitter::Parser::new();
        zig.set_language(&tree_sitter_zig::LANGUAGE.into())
            .expect("tree-sitter-zig language should be valid");

        let mut python = tree_sitter::Parser::new();
        python
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .expect("tree-sitter-python language should be valid");

        let mut rust = tree_sitter::Parser::new();
        rust.set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("tree-sitter-rust language should be valid");

        Self { zig, python, rust }
    }

    /// Map file extension to language name.
    fn language_from_ext(ext: &str) -> Option<&'static str> {
        match ext {
            "zig" | "zon" => Some("zig"),
            "py" => Some("python"),
            "rs" => Some("rust"),
            _ => None,
        }
    }

    pub fn parse_file(&mut self, path: &Path, source: &str) -> Result<GuidanceDoc, ParseError> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let language = Self::language_from_ext(ext)
            .ok_or_else(|| ParseError::UnsupportedLanguage(path.display().to_string()))?;

        let tree = match language {
            "zig" => self
                .zig
                .parse(source, None)
                .ok_or_else(|| ParseError::Syntax("tree-sitter returned no tree".into()))?,
            "python" => self
                .python
                .parse(source, None)
                .ok_or_else(|| ParseError::Syntax("tree-sitter returned no tree".into()))?,
            "rust" => self
                .rust
                .parse(source, None)
                .ok_or_else(|| ParseError::Syntax("tree-sitter returned no tree".into()))?,
            _ => unreachable!(),
        };

        let source_path = path.to_string_lossy().to_string();
        let _module_name = source_path
            .strip_suffix(&format!(".{ext}"))
            .unwrap_or(&source_path)
            .replace(['/', '\\'], ".");

        let root = tree.root_node();
        let cursor = &mut tree.walk();

        let module_comment = extract_module_comment(&root, source);
        let members = extract_members(&root, source, cursor, language);

        let meta = Meta {
            module: source_path.as_str().into(),
            source: source_path.as_str().into(),
            language: language.into(),
        };

        Ok(GuidanceDoc {
            meta,
            comment: module_comment.map(|c| c.as_str().into()),
            members,
            ..GuidanceDoc::default()
        })
    }

    pub fn has_errors(&mut self, source: &str, language: &str) -> bool {
        let parser = match language {
            "zig" => &mut self.zig,
            "python" => &mut self.python,
            "rust" => &mut self.rust,
            _ => return true,
        };
        let Some(tree) = parser.parse(source, None) else {
            return true;
        };
        tree.root_node().has_error()
    }
}

fn extract_module_comment(root: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut doc_comments = Vec::new();
    let mut child = root.walk();
    for node in root.children(&mut child) {
        if node.kind() == "comment" || node.kind() == "doc_comment" {
            let text = node.utf8_text(source.as_bytes()).ok()?;
            let trimmed = trim_doc_prefix(text);
            doc_comments.push(trimmed);
        } else {
            break;
        }
    }
    if doc_comments.is_empty() {
        None
    } else {
        Some(doc_comments.join("\n"))
    }
}

fn extract_members(
    root: &tree_sitter::Node,
    source: &str,
    cursor: &mut tree_sitter::TreeCursor,
    language: &str,
) -> Vec<Member> {
    let mut members = Vec::new();
    let mut impl_methods: Vec<(String, Vec<Member>)> = Vec::new();
    let mut child = root.walk();
    for node in root.children(&mut child) {
        if node.kind() == "impl_item" {
            if let Some((type_name, methods)) = collect_impl_methods(&node, source, cursor) {
                impl_methods.push((type_name, methods));
            }
        } else if let Some(member) = extract_member(&node, source, cursor, language) {
            members.push(member);
        }
    }
    for (type_name, methods) in &impl_methods {
        if let Some(target) = members
            .iter_mut()
            .find(|m| m.name.as_str() == type_name.as_str())
        {
            target.members.extend(methods.iter().cloned());
        }
    }
    members
}

fn extract_member(
    node: &tree_sitter::Node,
    source: &str,
    cursor: &mut tree_sitter::TreeCursor,
    language: &str,
) -> Option<Member> {
    match node.kind() {
        "function_declaration" | "function_definition" => {
            extract_function(node, source, cursor, language)
        }
        "function_item" => extract_rust_function_or_test(node, source, cursor),
        "struct_declaration" | "class_definition" | "struct_item" | "trait_item" => {
            extract_struct_or_class(node, source, MemberType::Struct, language, true, None)
        }
        "enum_declaration" | "enum_item" => {
            extract_struct_or_class(node, source, MemberType::Enum, language, true, None)
        }
        "union_declaration" => {
            extract_struct_or_class(node, source, MemberType::Union, language, true, None)
        }
        "type_item" => extract_rust_type_alias(node, source),
        "macro_definition" => extract_rust_macro(node, source),
        "test_declaration" => extract_test(node, source),
        "variable_declaration" => extract_var_decl(node, source, cursor),
        "comptime_expression" => Some(extract_comptime_block(node, source, cursor, language)),
        _ => None,
    }
}

fn extract_comptime_block(
    node: &tree_sitter::Node,
    source: &str,
    cursor: &mut tree_sitter::TreeCursor,
    language: &str,
) -> Member {
    let line = node.start_position().row + 1;
    // Walk children of the comptime block to find inner declarations
    let mut child_members = Vec::new();
    let mut child = node.walk();
    for child_node in node.children(&mut child) {
        if let Some(inner) = extract_member(&child_node, source, cursor, language) {
            child_members.push(inner);
        }
    }

    let name = node
        .child_by_field_name("name")
        .or_else(|| {
            // For inline comptime { ... } blocks without name, use first child's name
            node.children(&mut node.walk())
                .find_map(|c| c.child_by_field_name("name"))
        })
        .and_then(|n| n.utf8_text(source.as_bytes()).ok())
        .unwrap_or("comptime");

    Member {
        type_name: MemberType::ComptimeBlock,
        name: name.into(),
        line: Some(line as u32),
        members: child_members,
        ..Member::default()
    }
}

fn extract_var_decl(
    node: &tree_sitter::Node,
    source: &str,
    _cursor: &mut tree_sitter::TreeCursor,
) -> Option<Member> {
    let name = node
        .children(&mut node.walk())
        .find(|c| c.kind() == "identifier")
        .and_then(|c| c.utf8_text(source.as_bytes()).ok())?;
    let is_pub = check_visibility(node, source, "zig");

    let mut child = node.walk();
    for child_node in node.children(&mut child) {
        match child_node.kind() {
            "struct_declaration" => {
                return extract_struct_or_class(
                    &child_node,
                    source,
                    MemberType::Struct,
                    "zig",
                    is_pub,
                    Some(name),
                );
            }
            "enum_declaration" => {
                return extract_struct_or_class(
                    &child_node,
                    source,
                    MemberType::Enum,
                    "zig",
                    is_pub,
                    Some(name),
                );
            }
            "union_declaration" => {
                return extract_struct_or_class(
                    &child_node,
                    source,
                    MemberType::Union,
                    "zig",
                    is_pub,
                    Some(name),
                );
            }
            _ => {}
        }
    }

    None
}

fn extract_function(
    node: &tree_sitter::Node,
    source: &str,
    _cursor: &mut tree_sitter::TreeCursor,
    language: &str,
) -> Option<Member> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source.as_bytes()).ok()?;

    let doc_comment = extract_preceding_doc_comment(node, source);

    let params = match language {
        "zig" => extract_zig_params(node, source),
        "python" => extract_python_params(node, source)?,
        "rust" => extract_rust_params(node, source).unwrap_or_default(),
        _ => Vec::new(),
    };

    let parameters = params
        .iter()
        .map(|(n, t)| Param {
            name: n.as_str().into(),
            type_name: t.as_ref().map(|s| s.as_str().into()),
            default: None,
        })
        .collect();

    let return_type = match language {
        "zig" => extract_zig_return_type(node, source),
        "python" => extract_python_return_type(node, source),
        "rust" => extract_rust_return_type(node, source),
        _ => None,
    };

    let sig = build_signature(
        if language == "python" { "def" } else { "fn" },
        name,
        &params,
        return_type.as_deref(),
    );

    let is_pub = check_visibility(node, source, language);
    let line = node.start_position().row + 1;

    Some(Member {
        type_name: if is_pub {
            MemberType::FnDecl
        } else {
            MemberType::FnPrivate
        },
        name: name.into(),
        signature: Some(sig.as_str().into()),
        params: parameters,
        returns: return_type.map(|r| r.as_str().into()),
        comment: doc_comment.map(|c| c.as_str().into()),
        is_pub,
        line: Some(line as u32),
        ..Member::default()
    })
}

fn extract_zig_params(node: &tree_sitter::Node, source: &str) -> Vec<(String, Option<String>)> {
    let mut params = Vec::new();
    let mut child = node.walk();
    for child_node in node.children(&mut child) {
        if child_node.kind() == "parameter" {
            let param_name = child_node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                .map_or_else(|| "_".into(), String::from);

            let param_type = child_node
                .child_by_field_name("type")
                .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                .map(String::from);

            params.push((param_name, param_type));
        }
    }
    params
}

fn extract_zig_return_type(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut child = node.walk();
    for child_node in node.children(&mut child) {
        for sub in child_node.children(&mut child_node.walk()) {
            if sub.kind() == "return_type" {
                return sub.utf8_text(source.as_bytes()).ok().map(String::from);
            }
        }
    }
    None
}

fn extract_python_params(
    node: &tree_sitter::Node,
    source: &str,
) -> Option<Vec<(String, Option<String>)>> {
    let mut params = Vec::new();
    let parameters = node.child_by_field_name("parameters")?;
    let mut child = parameters.walk();
    for child_node in parameters.children(&mut child) {
        let param_name = child_node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .map(String::from);

        let param_type = child_node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())
            .map(String::from);

        if let Some(ref name) = param_name {
            if name != "self" && name != "cls" {
                params.push((param_name.unwrap(), param_type));
            }
        }
    }
    Some(params)
}

fn extract_python_return_type(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let return_type = node.child_by_field_name("return_type")?;
    return_type
        .utf8_text(source.as_bytes())
        .ok()
        .map(String::from)
}

fn extract_rust_function_or_test(
    node: &tree_sitter::Node,
    source: &str,
    cursor: &mut tree_sitter::TreeCursor,
) -> Option<Member> {
    if has_rust_test_attribute(node, source) {
        let name = node
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(source.as_bytes()).ok())?;
        let line = node.start_position().row + 1;
        return Some(Member {
            type_name: MemberType::TestDecl,
            name: name.into(),
            line: Some(line as u32),
            ..Member::default()
        });
    }
    extract_function(node, source, cursor, "rust")
}

fn has_rust_test_attribute(node: &tree_sitter::Node, source: &str) -> bool {
    let mut child = node.walk();
    for child_node in node.children(&mut child) {
        if child_node.kind() == "attribute_item" {
            let Ok(text) = child_node.utf8_text(source.as_bytes()) else {
                continue;
            };
            if text.contains("#[test]") || text.contains("#[test(") {
                return true;
            }
        }
    }
    false
}

fn collect_impl_methods(
    node: &tree_sitter::Node,
    source: &str,
    _cursor: &mut tree_sitter::TreeCursor,
) -> Option<(String, Vec<Member>)> {
    let type_name = node
        .child_by_field_name("type")
        .and_then(|n| n.utf8_text(source.as_bytes()).ok())?;
    let child_members = extract_body_members(node, source, "rust").unwrap_or_default();
    Some((type_name.into(), child_members))
}

fn extract_rust_type_alias(node: &tree_sitter::Node, source: &str) -> Option<Member> {
    let name = node
        .child_by_field_name("name")
        .and_then(|n| n.utf8_text(source.as_bytes()).ok())?;
    let line = node.start_position().row + 1;
    Some(Member {
        type_name: MemberType::Struct,
        name: name.into(),
        line: Some(line as u32),
        ..Member::default()
    })
}

fn extract_rust_macro(node: &tree_sitter::Node, source: &str) -> Option<Member> {
    let name = node
        .child_by_field_name("name")
        .and_then(|n| n.utf8_text(source.as_bytes()).ok())?;
    let line = node.start_position().row + 1;
    Some(Member {
        type_name: MemberType::FnDecl,
        name: name.into(),
        line: Some(line as u32),
        ..Member::default()
    })
}

fn extract_rust_params(
    node: &tree_sitter::Node,
    source: &str,
) -> Option<Vec<(String, Option<String>)>> {
    let parameters = node.child_by_field_name("parameters")?;
    let mut params = Vec::new();
    let mut child = parameters.walk();
    for child_node in parameters.children(&mut child) {
        if child_node.kind() == "parameter" {
            let param_name = child_node
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                .map(String::from);
            let param_type = child_node
                .child_by_field_name("type")
                .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                .map(String::from);
            if let Some(name) = param_name {
                if name != "self" && name != "&self" && name != "&mut self" && name != "mut self" {
                    params.push((name, param_type));
                }
            }
        }
    }
    Some(params)
}

fn extract_rust_return_type(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let return_type = node.child_by_field_name("return_type")?;
    return_type
        .utf8_text(source.as_bytes())
        .ok()
        .map(String::from)
}

fn extract_struct_or_class(
    node: &tree_sitter::Node,
    source: &str,
    member_type: MemberType,
    language: &str,
    is_pub: bool,
    override_name: Option<&str>,
) -> Option<Member> {
    let name = override_name.map_or_else(
        || {
            node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(source.as_bytes()).ok())
                .map(String::from)
        },
        |n| Some(n.to_string()),
    )?;
    let doc_comment = extract_preceding_doc_comment(node, source);

    let child_members = extract_body_members(node, source, language)?;

    let sig = format!(
        "{} {} {{ ... }}",
        match member_type {
            MemberType::Struct => "struct",
            MemberType::Enum => "enum",
            MemberType::Union => "union",
            _ => "class",
        },
        name
    );

    let line = node.start_position().row + 1;

    Some(Member {
        type_name: member_type,
        name: name.into(),
        signature: Some(sig.as_str().into()),
        comment: doc_comment.map(|c| c.as_str().into()),
        is_pub,
        members: child_members,
        line: Some(line as u32),
        ..Member::default()
    })
}

fn extract_body_members(
    node: &tree_sitter::Node,
    source: &str,
    language: &str,
) -> Option<Vec<Member>> {
    let mut members = Vec::new();

    let children_iter: Vec<tree_sitter::Node> = if let Some(body) = node.child_by_field_name("body")
    {
        let mut child = body.walk();
        body.children(&mut child).collect()
    } else {
        let mut child = node.walk();
        node.children(&mut child).collect()
    };

    for child_node in children_iter {
        match child_node.kind() {
            "function_declaration" | "function_definition" | "function_item" => {
                let name_node = child_node.child_by_field_name("name")?;
                let name = name_node.utf8_text(source.as_bytes()).ok()?;
                let line = child_node.start_position().row + 1;
                let dc = extract_preceding_doc_comment(&child_node, source);
                let is_pub = check_visibility(&child_node, source, language);
                let params = match language {
                    "rust" => extract_rust_params(&child_node, source).unwrap_or_default(),
                    _ => Vec::new(),
                };
                let return_type = match language {
                    "rust" => extract_rust_return_type(&child_node, source),
                    _ => None,
                };
                let sig = build_signature("fn", name, &params, return_type.as_deref());
                members.push(Member {
                    type_name: if is_pub {
                        MemberType::Method
                    } else {
                        MemberType::MethodPrivate
                    },
                    name: name.into(),
                    signature: Some(sig.as_str().into()),
                    comment: dc.map(|c| c.as_str().into()),
                    is_pub,
                    line: Some(line as u32),
                    ..Member::default()
                });
            }
            "macro_definition" => {
                let name_node = child_node.child_by_field_name("name")?;
                let name = name_node.utf8_text(source.as_bytes()).ok()?;
                let line = child_node.start_position().row + 1;
                members.push(Member {
                    type_name: MemberType::Method,
                    name: name.into(),
                    line: Some(line as u32),
                    ..Member::default()
                });
            }
            _ => {}
        }
    }
    Some(members)
}

fn extract_test(node: &tree_sitter::Node, source: &str) -> Option<Member> {
    // Try field name first
    if let Some(name_node) = node.child_by_field_name("name") {
        let raw_name = name_node.utf8_text(source.as_bytes()).ok()?;
        let clean_name = raw_name
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(raw_name);
        let line = node.start_position().row + 1;
        return Some(Member {
            type_name: MemberType::TestDecl,
            name: clean_name.into(),
            line: Some(line as u32),
            ..Member::default()
        });
    }

    // Fallback: find string child among direct children
    let mut child = node.walk();
    let name_node = node
        .children(&mut child)
        .find(|c| c.kind() == "string_literal" || c.kind() == "string")?;
    let raw_name = name_node.utf8_text(source.as_bytes()).ok()?;
    let clean_name = raw_name
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(raw_name);
    // clean_name already set above

    let line = node.start_position().row + 1;

    Some(Member {
        type_name: MemberType::TestDecl,
        name: clean_name.into(),
        line: Some(line as u32),
        ..Member::default()
    })
}

fn extract_preceding_doc_comment(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let prev_node = node.prev_sibling()?;
    if prev_node.kind() == "doc_comment" || prev_node.kind() == "comment" {
        let text = prev_node.utf8_text(source.as_bytes()).ok()?;
        Some(trim_doc_prefix(text))
    } else {
        None
    }
}

fn check_visibility(node: &tree_sitter::Node, source: &str, language: &str) -> bool {
    if language == "python" {
        let Some(name_node) = node.child_by_field_name("name") else {
            return true;
        };
        if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
            return !name.starts_with('_');
        }
        return true;
    }

    let mut child = node.walk();
    for child_node in node.children(&mut child) {
        let kind = child_node.kind();
        if kind == "pub" || kind == "visibility_modifier" {
            return true;
        }
    }
    false
}

fn trim_doc_prefix(text: &str) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    for line in &mut lines {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("///") {
            *line = rest.strip_prefix(' ').unwrap_or(rest);
        } else if let Some(rest) = trimmed.strip_prefix("//!") {
            *line = rest.strip_prefix(' ').unwrap_or(rest);
        } else if let Some(rest) = trimmed.strip_prefix('#') {
            *line = rest.strip_prefix(' ').unwrap_or(rest);
        }
    }
    lines.join("\n")
}

fn build_signature(
    kind: &str,
    name: &str,
    params: &[(String, Option<String>)],
    return_type: Option<&str>,
) -> String {
    let mut sig = String::new();
    sig.push_str(kind);
    sig.push(' ');
    sig.push_str(name);
    sig.push('(');
    for (i, (pname, ptype)) in params.iter().enumerate() {
        if i > 0 {
            sig.push_str(", ");
        }
        sig.push_str(pname);
        if let Some(t) = ptype {
            sig.push_str(": ");
            sig.push_str(t);
        }
    }
    sig.push(')');
    if let Some(r) = return_type {
        sig.push_str(" -> ");
        sig.push_str(r);
    }
    sig
}

pub fn file_type_from_extension(ext: &str) -> FileType {
    if ext.starts_with('.') {
        FileType::from_extension(ext)
    } else {
        FileType::from_extension(&format!(".{ext}"))
    }
}

pub fn resolve_span(path: &Path, member_name: &str, member_type: MemberType) -> Option<Span> {
    let source = common_core::io::read_to_string_err(path).ok()?;
    let ext = path.extension().and_then(|e| e.to_str())?;
    let mut parser = AstParser::new();
    let language = AstParser::language_from_ext(ext)?;
    let tree = match language {
        "zig" => parser.zig.parse(&source, None)?,
        "python" => parser.python.parse(&source, None)?,
        "rust" => parser.rust.parse(&source, None)?,
        _ => return None,
    };
    let root = tree.root_node();
    find_member_span(root, &source, member_name, member_type)
}

fn find_member_span(
    root: tree_sitter::Node,
    source: &str,
    member_name: &str,
    member_type: MemberType,
) -> Option<Span> {
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if let Some(span) = try_match_node_span(&node, source, member_name, member_type) {
            return Some(span);
        }
    }
    None
}

fn try_match_node_span(
    node: &tree_sitter::Node,
    source: &str,
    member_name: &str,
    member_type: MemberType,
) -> Option<Span> {
    let name_node = node.child_by_field_name("name");
    let node_name = name_node
        .as_ref()
        .and_then(|n| n.utf8_text(source.as_bytes()).ok())?;

    let matches_type = match node.kind() {
        "function_declaration" | "function_definition" | "function_item" => {
            matches!(
                member_type,
                MemberType::FnDecl
                    | MemberType::FnPrivate
                    | MemberType::Method
                    | MemberType::MethodPrivate
            )
        }
        "struct_declaration" | "struct_item" | "class_definition" => {
            member_type == MemberType::Struct
        }
        "enum_declaration" | "enum_item" => member_type == MemberType::Enum,
        "union_declaration" => member_type == MemberType::Union,
        "test_declaration" => member_type == MemberType::TestDecl,
        _ => false,
    };

    if matches_type && node_name == member_name {
        return Some(node_span(node, source));
    }

    if node.kind() == "impl_item" {
        let type_name = name_node?;
        let type_text = type_name.utf8_text(source.as_bytes()).ok()?;
        if type_text == member_name
            && matches!(
                member_type,
                MemberType::Struct | MemberType::FnDecl | MemberType::Method
            )
        {
            return Some(node_span(node, source));
        }
        let mut child_cursor = node.walk();
        for child in node.children(&mut child_cursor) {
            if let Some(span) = try_match_node_span(&child, source, member_name, member_type) {
                return Some(span);
            }
        }
    }

    if node.kind() == "declaration_list" || node.kind() == "block" {
        let mut child_cursor = node.walk();
        for child in node.children(&mut child_cursor) {
            if let Some(span) = try_match_node_span(&child, source, member_name, member_type) {
                return Some(span);
            }
        }
    }

    None
}

fn node_span(node: &tree_sitter::Node, source: &str) -> Span {
    let start_line = find_doc_comment_start(node, source);
    let end_line = node.end_position().row as u32 + 1;
    Span {
        start_line: start_line + 1,
        end_line,
    }
}

fn find_doc_comment_start(node: &tree_sitter::Node, source: &str) -> u32 {
    let mut current = *node;
    let mut candidate_start = node.start_position().row as u32;

    while let Some(prev) = current.prev_sibling() {
        let kind = prev.kind();
        if kind == "doc_comment" || kind == "comment" {
            let text = prev.utf8_text(source.as_bytes()).unwrap_or("");
            if text.starts_with("///") || text.starts_with("//!") || text.starts_with("#[") {
                candidate_start = prev.start_position().row as u32;
                current = prev;
                continue;
            }
        }
        if kind == "attribute_item" || kind == "outer_attribute" {
            candidate_start = prev.start_position().row as u32;
            current = prev;
            continue;
        }
        break;
    }
    candidate_start
}

/// Extracts a code snippet for a member from its source file using tree-sitter AST.
///
/// Given a source path (relative to workspace), member name, member type, and
/// workspace root, reads the source file, parses it with tree-sitter to find
/// the exact span, and returns the code lines along with 1-based line bounds.
///
/// Follows the Zig `cmdExplain` pattern: AST-driven line extraction, not
/// heuristic brace counting. Uses the tree-sitter span end line for exact
/// boundaries (no arbitrary truncation).
///
/// Returns `(snippet, start_line, end_line)` where lines are 1-based.
pub fn extract_code_snippet(
    source_path: &str,
    member_name: &str,
    member_type: MemberType,
    workspace: &std::path::Path,
    max_lines: usize,
) -> Option<(String, u32, u32)> {
    let abs_path = workspace.join(source_path);
    let source = common_core::io::read_to_string_err(&abs_path).ok()?;
    let span = resolve_span(&abs_path, member_name, member_type)?;

    let start = (span.start_line as usize).saturating_sub(1);
    let span_end = span.end_line as usize;
    let limited_end = start + max_lines;
    let end = span_end.min(limited_end);
    let lines: Vec<&str> = source.lines().collect();
    if start >= lines.len() {
        return None;
    }
    let end = end.min(lines.len());
    let snippet: String = lines[start..end].join("\n");
    if snippet.trim().is_empty() {
        None
    } else {
        Some((snippet, span.start_line, span.end_line))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parse_helper(source: &str, filename: &str) -> GuidanceDoc {
        let mut parser = AstParser::new();
        parser
            .parse_file(Path::new(filename), source)
            .expect("should parse")
    }

    #[test]
    fn test_parse_zig_file() {
        let source = r#"/// Sample Zig file for AST parsing tests
const std = @import("std");

pub fn greet(name: []const u8) []const u8 {
    return "Hello, " ++ name;
}

pub const Config = struct {
    port: u16 = 8080,
    host: []const u8 = "localhost",
};
"#;

        let doc = test_parse_helper(source, "main.zig");
        assert_eq!(doc.meta.language.as_str(), "zig");
        assert!(
            !doc.members.is_empty(),
            "should have at least one member, got 0"
        );
        for m in &doc.members {
            eprintln!("member: name={:?} type={:?}", m.name, m.type_name);
        }
        let greet = doc.members.iter().find(|m| m.name == "greet");
        assert!(greet.is_some(), "should find greet function");
        let config = doc.members.iter().find(|m| m.name == "Config");
        assert!(config.is_some(), "should find Config struct");
    }

    #[test]
    fn test_parse_python_file() {
        let source = r#"def hello(name: str) -> str:
    return f"Hello, {name}"

class MyClass:
    def method(self, x: int) -> int:
        return x * 2
"#;

        let doc = test_parse_helper(source, "main.py");
        assert_eq!(doc.meta.language.as_str(), "python");
        assert!(doc.members.len() >= 2);
        let hello = doc.members.iter().find(|m| m.name == "hello");
        assert!(hello.is_some(), "should find hello function");
        let cls = doc.members.iter().find(|m| m.name == "MyClass");
        assert!(cls.is_some(), "should find MyClass");
    }

    #[test]
    fn test_extract_zig_doc_comment() {
        let doc = test_parse_helper(
            "/// This is a test function\npub fn test_fn() void {}\n",
            "test.zig",
        );
        let func = doc.members.first().expect("should have a function");
        assert_eq!(
            func.comment.as_ref().map(smol_str::SmolStr::as_str),
            Some("This is a test function")
        );
    }

    #[test]
    fn test_extract_line_numbers() {
        let doc = test_parse_helper("pub fn foo() void {}\n\npub fn bar() void {}", "foo.zig");
        assert!(!doc.members.is_empty());
    }

    #[test]
    fn test_unsupported_language() {
        let mut parser = AstParser::new();
        let result = parser.parse_file(Path::new("main.c"), "int main() {}");
        assert!(result.is_err());
        match result {
            Err(ParseError::UnsupportedLanguage(_)) => {}
            _ => panic!("expected UnsupportedLanguage error"),
        }
    }

    #[test]
    fn test_extract_comptime_block() {
        let source = r#"const std = @import("std");

pub fn main() void {
    comptime {
        const x: u32 = 42;
        const y: u32 = x + 1;
    }
}
"#;
        let doc = test_parse_helper(source, "comptime.zig");
        assert_eq!(doc.meta.language.as_str(), "zig");
        let main_fn = doc.members.iter().find(|m| m.name == "main");
        assert!(main_fn.is_some(), "should find main function");
    }

    #[test]
    fn test_extract_test_decl() {
        let source = r#"test "hello test" {
    try std.testing.expectEqual(1, 1);
}
"#;
        let doc = test_parse_helper(source, "test.zig");
        let test_decl = doc.members.iter().find(|m| m.name == "hello test");
        assert!(test_decl.is_some(), "should find test declaration");
        if let Some(t) = test_decl {
            assert_eq!(t.type_name, MemberType::TestDecl, "should be TestDecl type");
        }
    }

    #[test]
    fn test_file_type_from_extension() {
        assert_eq!(file_type_from_extension("zig"), FileType::Source);
        assert_eq!(file_type_from_extension("py"), FileType::Source);
        assert_eq!(file_type_from_extension(".zig"), FileType::Source);
        assert_eq!(file_type_from_extension("md"), FileType::Markdown);
        assert_eq!(file_type_from_extension("json"), FileType::Config);
        assert_eq!(file_type_from_extension("xyz"), FileType::Unknown);
    }
}
