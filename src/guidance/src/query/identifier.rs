use common_core::string::{contains_ignore_case, detect_identifier_kind};
use fluent_types::GuidanceDoc;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct IdentifierPattern {
    pub name: String,
    pub kind: IdentifierKind,
}

/// Identifier case-style classification. Re-exported from
/// `common_core::string` (M3.4/M7.2 promotion) so existing guidance
/// consumers keep the same path.
pub use common_core::string::IdentifierKind;

pub fn detect_identifier_pattern(query: &str) -> Option<IdentifierPattern> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }

    let kind = detect_identifier_kind(trimmed)?;

    let valid_identifier = Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")
        .ok()?
        .is_match(trimmed);

    if valid_identifier || kind != IdentifierKind::Other {
        Some(IdentifierPattern {
            name: trimmed.to_string(),
            kind,
        })
    } else {
        None
    }
}

pub fn find_members_by_name<'a>(doc: &'a GuidanceDoc, name: &str) -> Vec<&'a str> {
    let lower_name = name.to_lowercase();
    doc.members
        .iter()
        .filter(|m| m.name.as_str().to_lowercase() == lower_name)
        .map(|m| m.name.as_str())
        .collect()
}

pub fn find_members_by_signature<'a>(doc: &'a GuidanceDoc, query: &str) -> Vec<&'a str> {
    doc.members
        .iter()
        .filter_map(|m| {
            m.signature
                .as_ref()
                .filter(|sig| contains_ignore_case(sig.as_str(), query))
                .map(smol_str::SmolStr::as_str)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_camel_case() {
        let result = detect_identifier_pattern("helloWorld").expect("detected");
        assert_eq!(result.name, "helloWorld");
    }

    #[test]
    fn test_detect_snake_case() {
        let result = detect_identifier_pattern("hello_world").expect("detected");
        assert_eq!(result.kind, IdentifierKind::SnakeCase);
    }

    #[test]
    fn test_detect_pascal_case() {
        let result = detect_identifier_pattern("HelloWorld").expect("detected");
        assert_eq!(result.kind, IdentifierKind::PascalCase);
    }

    #[test]
    fn test_detect_dotted_path() {
        let result = detect_identifier_pattern("foo.bar.baz").expect("detected");
        assert_eq!(result.kind, IdentifierKind::DottedPath);
    }

    #[test]
    fn test_not_an_identifier() {
        assert!(detect_identifier_pattern("").is_none());
        assert!(detect_identifier_pattern("hello world").is_none());
    }

    #[test]
    fn test_find_members_by_name() {
        use fluent_types::{GuidanceDoc, Member, MemberType, Meta};

        let doc = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "test.zig".into(),
                language: "zig".into(),
            },
            members: vec![
                Member {
                    type_name: MemberType::FnDecl,
                    name: "hello".into(),
                    signature: Some("fn hello() void".into()),
                    ..Member::default()
                },
                Member {
                    type_name: MemberType::FnDecl,
                    name: "world".into(),
                    signature: Some("fn world() i32".into()),
                    ..Member::default()
                },
            ],
            ..GuidanceDoc::default()
        };

        let found = find_members_by_name(&doc, "HELLO");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "hello");
    }

    #[test]
    fn test_find_members_by_signature() {
        use fluent_types::{GuidanceDoc, Member, MemberType, Meta};

        let doc = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "test.zig".into(),
                language: "zig".into(),
            },
            members: vec![Member {
                type_name: MemberType::FnDecl,
                name: "add".into(),
                signature: Some("fn add(a: i32, b: i32) i32".into()),
                ..Member::default()
            }],
            ..GuidanceDoc::default()
        };

        let found = find_members_by_signature(&doc, "i32");
        assert_eq!(found.len(), 1);
    }
}
