use std::path::{Path, PathBuf};

use fluent_types::{CapabilityEval, GuidanceDoc, Member, MemberType, Meta, Param, Skill};
use smol_str::SmolStr;
use thiserror::Error;

use crate::walk;

#[derive(Error, Debug)]
pub enum JsonError {
    #[error("IO error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("JSON parse error: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("missing required field: {0}")]
    MissingField(String),
}

impl From<std::io::Error> for JsonError {
    fn from(e: std::io::Error) -> Self {
        JsonError::Io(common_core::error::IoError(e))
    }
}

fn parse_member_type(s: &str) -> Option<MemberType> {
    Some(match s {
        "fn_decl" => MemberType::FnDecl,
        "fn_private" => MemberType::FnPrivate,
        "struct" => MemberType::Struct,
        "enum" => MemberType::Enum,
        "union" => MemberType::Union,
        "enum_field" => MemberType::EnumField,
        "test_decl" => MemberType::TestDecl,
        "comptime_block" => MemberType::ComptimeBlock,
        "method" => MemberType::Method,
        "method_private" => MemberType::MethodPrivate,
        _ => return None,
    })
}

fn parse_param(v: &serde_json::Value) -> Option<Param> {
    let obj = v.as_object()?;
    Some(Param {
        name: obj.get("name")?.as_str()?.into(),
        type_name: obj.get("type").and_then(|v| v.as_str()).map(SmolStr::from),
        default: obj
            .get("default")
            .and_then(|v| v.as_str())
            .map(SmolStr::from),
    })
}

fn parse_member(v: &serde_json::Value) -> Option<Member> {
    let obj = v.as_object()?;
    let member_type = parse_member_type(obj.get("type")?.as_str()?)?;
    let name = obj.get("name")?.as_str()?;

    let params = obj
        .get("params")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(parse_param).collect())
        .unwrap_or_default();

    let members = obj
        .get("members")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(parse_member).collect())
        .unwrap_or_default();

    let equivalents = obj
        .get("equivalents")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(SmolStr::from)
                .collect()
        })
        .unwrap_or_default();

    let tags = obj
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(SmolStr::from)
                .collect()
        })
        .unwrap_or_default();

    Some(Member {
        type_name: member_type,
        name: name.into(),
        match_hash: obj
            .get("match_hash")
            .and_then(|v| v.as_str())
            .map(SmolStr::from),
        signature: obj
            .get("signature")
            .and_then(|v| v.as_str())
            .map(SmolStr::from),
        params,
        returns: obj
            .get("returns")
            .and_then(|v| v.as_str())
            .map(SmolStr::from),
        comment: obj
            .get("comment")
            .and_then(|v| v.as_str())
            .map(SmolStr::from),
        tags,
        is_pub: obj
            .get("is_pub")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        members,
        equivalents,
        line: obj
            .get("line")
            .and_then(serde_json::Value::as_u64)
            .map(|l| l as u32),
        comment_generated: obj
            .get("comment_generated")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    })
}

fn load_guidance_from_value(v: &serde_json::Value) -> Option<GuidanceDoc> {
    let root = v.as_object()?;
    let meta_obj = root.get("meta")?.as_object()?;
    let module = meta_obj.get("module")?.as_str()?;
    let source = meta_obj.get("source")?.as_str()?;

    let members = root
        .get("members")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(parse_member).collect())
        .unwrap_or_default();

    let skills = root
        .get("skills")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|sv| {
                    let so = sv.as_object()?;
                    Some(Skill {
                        ref_path: so.get("ref")?.as_str()?.into(),
                        context: so
                            .get("context")
                            .and_then(|v| v.as_str())
                            .map(SmolStr::from),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let parse_str_vec = |key: &str| -> Vec<SmolStr> {
        root.get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(SmolStr::from)
                    .collect()
            })
            .unwrap_or_default()
    };

    let capability_eval = root.get("capability_eval").and_then(|v| {
        let o = v.as_object()?;
        Some(CapabilityEval {
            capability_name: o.get("capability_name")?.as_str()?.into(),
            confidence: o.get("confidence")?.as_f64()? as f32,
            evaluated_at_hash: o.get("evaluated_at_hash")?.as_str()?.into(),
        })
    });

    Some(GuidanceDoc {
        meta: Meta {
            module: module.into(),
            source: source.into(),
            language: meta_obj
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("zig")
                .into(),
        },
        comment: root
            .get("comment")
            .and_then(|v| v.as_str())
            .map(SmolStr::from),
        detail: root
            .get("detail")
            .and_then(|v| v.as_str())
            .map(SmolStr::from),
        keywords: parse_str_vec("keywords"),
        skills,
        capabilities: parse_str_vec("capabilities"),
        hashtags: parse_str_vec("hashtags"),
        used_by: parse_str_vec("used_by"),
        members,
        equivalents: parse_str_vec("equivalents"),
        capability_eval,
    })
}

pub fn load_guidance(path: &Path) -> Result<Option<GuidanceDoc>, JsonError> {
    let content = common_core::io::read_to_string_err(path)?;
    if content.trim().is_empty() {
        return Ok(None);
    }
    let v: serde_json::Value = serde_json::from_str(&content)?;
    Ok(load_guidance_from_value(&v))
}

/// Walk a directory recursively for `.json` guidance files and yield `(path, doc)` tuples.
pub fn walk_guidance_docs(dir: &Path) -> impl Iterator<Item = (PathBuf, GuidanceDoc)> {
    let mut entries: Vec<(PathBuf, GuidanceDoc)> = Vec::new();
    walk::walk_files(dir, &["json"], |path| {
        if let Ok(Some(doc)) = load_guidance(path) {
            entries.push((path.to_path_buf(), doc));
        }
    });
    entries.into_iter()
}

/// Merge existing member data into a new member when match_hash is unchanged.
/// Preserves user-created content: comment, tags, equivalents, comment_generated flag.
pub fn merge_member(existing: &Member, new: &mut Member) {
    let hash_match = match (&existing.match_hash, &new.match_hash) {
        (Some(eh), Some(nh)) => eh == nh,
        _ => false,
    };
    if !hash_match {
        return;
    }
    // Preserve existing comment if the new one is empty/generated
    if new.comment.is_none() || new.comment_generated {
        if let Some(ref comment) = existing.comment {
            new.comment = Some(comment.clone());
            new.comment_generated = existing.comment_generated;
        }
    }
    // Merge tags: combine existing and new tags (unique)
    let mut merged_tags = new.tags.clone();
    for tag in &existing.tags {
        if !merged_tags.contains(tag) {
            merged_tags.push(tag.clone());
        }
    }
    new.tags = merged_tags;
    // Preserve equivalents
    let mut merged_eqs = new.equivalents.clone();
    for eq in &existing.equivalents {
        if !merged_eqs.contains(eq) {
            merged_eqs.push(eq.clone());
        }
    }
    new.equivalents = merged_eqs;
}

/// Merge existing doc-level metadata into a new doc, preserving user annotations.
pub fn merge_doc(existing: &GuidanceDoc, new: &mut GuidanceDoc) {
    // Preserve doc comment if new one is empty
    if new.comment.is_none() && existing.comment.is_some() {
        new.comment.clone_from(&existing.comment);
    }
    // Merge keywords
    for kw in &existing.keywords {
        if !new.keywords.contains(kw) {
            new.keywords.push(kw.clone());
        }
    }
    // Merge skills
    for skill in &existing.skills {
        if !new.skills.iter().any(|s| s.ref_path == skill.ref_path) {
            new.skills.push(skill.clone());
        }
    }
    // Preserve capability_eval if new doesn't have one
    if new.capability_eval.is_none() {
        new.capability_eval.clone_from(&existing.capability_eval);
    }
    // Merge members by match_hash
    for new_member in &mut new.members {
        if let Some(ref new_hash) = new_member.match_hash {
            if let Some(existing_member) = existing
                .members
                .iter()
                .find(|em| em.match_hash.as_ref() == Some(new_hash))
            {
                merge_member(existing_member, new_member);
            }
        }
    }
}

/// Save guidance doc, preserving existing metadata when hashes are unchanged.
pub fn save_guidance(path: &Path, doc: &GuidanceDoc) -> Result<(), JsonError> {
    if let Some(parent) = path.parent() {
        common_core::io::ensure_dir(parent)?;
    }
    // Load existing to preserve user annotations on hash-match
    let mut doc_to_save = doc.clone();
    if let Ok(Some(existing)) = load_guidance(path) {
        merge_doc(&existing, &mut doc_to_save);
    }
    let json_str = super::json_writer::doc_to_json_string(&doc_to_save);
    common_core::io::write_atomic(path, json_str.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_types::{GuidanceDoc, Member, MemberType, Meta};
    use fluent_wvr_testutil::tempdir;

    fn make_test_member(name: &str, sig: &str, hash: &str, comment: Option<&str>) -> Member {
        Member {
            type_name: MemberType::FnDecl,
            name: name.into(),
            signature: Some(sig.into()),
            match_hash: Some(hash.into()),
            comment: comment.map(SmolStr::from),
            ..Member::default()
        }
    }

    #[test]
    fn test_merge_member_preserves_comment() {
        let existing = make_test_member("foo", "fn foo() void", "abc123", Some("Original comment"));
        let mut new = make_test_member("foo", "fn foo() void", "abc123", None);
        merge_member(&existing, &mut new);
        assert_eq!(
            new.comment.as_ref().map(SmolStr::as_str),
            Some("Original comment")
        );
    }

    #[test]
    fn test_merge_member_different_hash() {
        let existing = make_test_member("foo", "fn foo() void", "abc123", Some("Original"));
        let mut new = make_test_member("foo", "fn foo(x: i32) void", "def456", None);
        merge_member(&existing, &mut new);
        assert!(new.comment.is_none(), "should not merge when hash differs");
    }

    #[test]
    fn test_merge_member_merges_tags() {
        let mut existing = make_test_member("foo", "fn foo() void", "abc123", Some("Original"));
        existing.tags = vec!["api".into(), "public".into()];
        let mut new = make_test_member("foo", "fn foo() void", "abc123", None);
        new.tags = vec!["api".into(), "new_tag".into()];
        merge_member(&existing, &mut new);
        assert!(new.tags.iter().any(|t| t.as_str() == "public"));
        assert!(new.tags.iter().any(|t| t.as_str() == "new_tag"));
        assert_eq!(new.tags.len(), 3);
    }

    #[test]
    fn test_save_guidance_preserves_comment_on_round_trip() {
        let dir = tempdir();
        let path = dir.path().join("test.json");

        // First save: member with comment and hash
        let doc1 = GuidanceDoc {
            meta: Meta {
                module: "roundtrip".into(),
                source: "src/rt.zig".into(),
                language: "zig".into(),
            },
            comment: Some("Module comment".into()),
            members: vec![make_test_member(
                "foo",
                "fn foo() void",
                "hash1",
                Some("User comment"),
            )],
            ..GuidanceDoc::default()
        };
        save_guidance(&path, &doc1).expect("first save");

        // Second save: same member, same hash, no comment
        let doc2 = GuidanceDoc {
            meta: Meta {
                module: "roundtrip".into(),
                source: "src/rt.zig".into(),
                language: "zig".into(),
            },
            comment: None,
            members: vec![make_test_member("foo", "fn foo() void", "hash1", None)],
            ..GuidanceDoc::default()
        };
        save_guidance(&path, &doc2).expect("second save");

        // Reload and verify comment preserved
        let loaded = load_guidance(&path).expect("load").expect("should exist");
        assert_eq!(
            loaded.members[0].comment.as_ref().map(SmolStr::as_str),
            Some("User comment"),
            "comment should be preserved on round-trip"
        );
        assert_eq!(
            loaded.comment.as_ref().map(SmolStr::as_str),
            Some("Module comment"),
            "doc comment should be preserved"
        );
    }

    #[test]
    fn test_save_guidance_overwrites_when_hash_differs() {
        let dir = tempdir();
        let path = dir.path().join("test.json");

        let doc1 = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "src/t.zig".into(),
                language: "zig".into(),
            },
            members: vec![make_test_member(
                "old_fn",
                "fn old_fn() void",
                "hash_old",
                Some("Old comment"),
            )],
            ..GuidanceDoc::default()
        };
        save_guidance(&path, &doc1).expect("first save");

        // Different hash = signature changed = don't preserve
        let doc2 = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "src/t.zig".into(),
                language: "zig".into(),
            },
            members: vec![make_test_member(
                "old_fn",
                "fn old_fn(x: i32) void",
                "hash_new",
                None,
            )],
            ..GuidanceDoc::default()
        };
        save_guidance(&path, &doc2).expect("second save");

        let loaded = load_guidance(&path).expect("load").expect("should exist");
        assert!(
            loaded.members[0].comment.is_none(),
            "comment should NOT be preserved when hash differs"
        );
    }

    #[test]
    fn test_load_guidance_minimal() {
        let json = r#"{
            "meta": {
                "module": "test",
                "source": "src/test.zig",
                "language": "zig"
            },
            "comment": "A test module"
        }"#;
        let v: serde_json::Value = serde_json::from_str(json).expect("valid json");
        let doc = load_guidance_from_value(&v).expect("should load");
        assert_eq!(doc.meta.module.as_str(), "test");
        assert_eq!(
            doc.comment.as_ref().map(smol_str::SmolStr::as_str),
            Some("A test module")
        );
    }

    #[test]
    fn test_save_and_load_round_trip() {
        let dir = tempdir();
        let path = dir.path().join("test.json");

        let doc = GuidanceDoc {
            meta: Meta {
                module: "roundtrip".into(),
                source: "src/rt.zig".into(),
                language: "zig".into(),
            },
            comment: Some("Round trip test".into()),
            members: vec![Member {
                type_name: MemberType::FnDecl,
                name: "foo".into(),
                signature: Some("fn foo() void".into()),
                is_pub: true,
                ..Member::default()
            }],
            ..GuidanceDoc::default()
        };

        save_guidance(&path, &doc).expect("should save");
        let loaded = load_guidance(&path)
            .expect("should load")
            .expect("should be Some");
        assert_eq!(loaded.meta.module.as_str(), "roundtrip");
        assert_eq!(loaded.members.len(), 1);
        assert_eq!(loaded.members[0].name.as_str(), "foo");
    }
}
