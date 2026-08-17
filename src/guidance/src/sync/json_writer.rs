use fluent_types::{GuidanceDoc, Member};
use serde_json::{json, Value};

fn member_to_json(member: &Member) -> Value {
    let mut obj = json!({
        "type": member_type_str(member.type_name),
        "name": member.name.as_str(),
    });

    if let Some(h) = &member.match_hash {
        obj["match_hash"] = json!(h.as_str());
    }
    if let Some(sig) = &member.signature {
        obj["signature"] = json!(sig.as_str());
    }
    if let Some(ret) = &member.returns {
        obj["returns"] = json!(ret.as_str());
    }
    if let Some(c) = &member.comment {
        obj["comment"] = json!(c.as_str());
    }
    if !member.params.is_empty() {
        let params: Vec<Value> = member
            .params
            .iter()
            .map(|p| {
                let mut po = json!({ "name": p.name.as_str() });
                if let Some(t) = &p.type_name {
                    po["type"] = json!(t.as_str());
                }
                if let Some(d) = &p.default {
                    po["default"] = json!(d.as_str());
                }
                po
            })
            .collect();
        obj["params"] = json!(params);
    }
    if !member.tags.is_empty() {
        let tags: Vec<Value> = member.tags.iter().map(|t| json!(t.as_str())).collect();
        obj["tags"] = json!(tags);
    }
    if member.is_pub {
        obj["is_pub"] = json!(true);
    }
    if !member.members.is_empty() {
        let child: Vec<Value> = member.members.iter().map(member_to_json).collect();
        obj["members"] = json!(child);
    }
    if member.comment_generated {
        obj["comment_generated"] = json!(true);
    }
    if !member.equivalents.is_empty() {
        let eqs: Vec<Value> = member
            .equivalents
            .iter()
            .map(|e| json!(e.as_str()))
            .collect();
        obj["equivalents"] = json!(eqs);
    }

    obj
}

fn member_type_str(t: fluent_types::MemberType) -> &'static str {
    use fluent_types::MemberType::{
        ComptimeBlock, Enum, EnumField, FnDecl, FnPrivate, Method, MethodPrivate, Struct, TestDecl,
        Union,
    };
    match t {
        FnDecl => "fn_decl",
        FnPrivate => "fn_private",
        Struct => "struct",
        Enum => "enum",
        Union => "union",
        EnumField => "enum_field",
        TestDecl => "test_decl",
        ComptimeBlock => "comptime_block",
        Method => "method",
        MethodPrivate => "method_private",
    }
}

pub fn doc_to_json(doc: &GuidanceDoc) -> Value {
    let mut obj = json!({
        "meta": {
            "module": doc.meta.module.as_str(),
            "source": doc.meta.source.as_str(),
            "language": doc.meta.language.as_str(),
        }
    });

    if let Some(c) = &doc.comment {
        obj["comment"] = json!(c.as_str());
    }
    if let Some(d) = &doc.detail {
        obj["detail"] = json!(d.as_str());
    }
    if !doc.keywords.is_empty() {
        let kws: Vec<Value> = doc.keywords.iter().map(|k| json!(k.as_str())).collect();
        obj["keywords"] = json!(kws);
    }
    if !doc.skills.is_empty() {
        let skills: Vec<Value> = doc
            .skills
            .iter()
            .map(|s| {
                let mut so = json!({ "ref": s.ref_path.as_str() });
                if let Some(ctx) = &s.context {
                    so["context"] = json!(ctx.as_str());
                }
                so
            })
            .collect();
        obj["skills"] = json!(skills);
    }
    if !doc.capabilities.is_empty() {
        let caps: Vec<Value> = doc.capabilities.iter().map(|c| json!(c.as_str())).collect();
        obj["capabilities"] = json!(caps);
    }
    if !doc.hashtags.is_empty() {
        let tags: Vec<Value> = doc.hashtags.iter().map(|t| json!(t.as_str())).collect();
        obj["hashtags"] = json!(tags);
    }
    if !doc.used_by.is_empty() {
        let ub: Vec<Value> = doc.used_by.iter().map(|u| json!(u.as_str())).collect();
        obj["used_by"] = json!(ub);
    }
    if !doc.members.is_empty() {
        let members: Vec<Value> = doc.members.iter().map(member_to_json).collect();
        obj["members"] = json!(members);
    }
    if !doc.equivalents.is_empty() {
        let eqs: Vec<Value> = doc.equivalents.iter().map(|e| json!(e.as_str())).collect();
        obj["equivalents"] = json!(eqs);
    }
    if let Some(ce) = &doc.capability_eval {
        obj["capability_eval"] = json!({
            "capability_name": ce.capability_name.as_str(),
            "confidence": ce.confidence,
            "evaluated_at_hash": ce.evaluated_at_hash.as_str(),
        });
    }

    obj
}

pub fn doc_to_json_string(doc: &GuidanceDoc) -> String {
    serde_json::to_string_pretty(&doc_to_json(doc)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::make_test_doc_hello;

    #[test]
    fn test_json_round_trip() {
        let doc = make_test_doc_hello();
        let json_str = doc_to_json_string(&doc);
        assert!(json_str.contains("test_module"));
        assert!(json_str.contains("hello"));
        assert!(json_str.contains("fn_decl"));
    }

    #[test]
    fn test_json_sorted_keys() {
        let doc = make_test_doc_hello();
        let json_str = doc_to_json_string(&doc);
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("should parse");
        assert_eq!(parsed["meta"]["module"], "test_module");
        assert_eq!(parsed["meta"]["source"], "src/test.zig");
    }
}
