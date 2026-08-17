use fluent_types::{GuidanceDoc, Member, MemberType, StageKind};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Stage {
    pub kind: StageKind,
    pub content: String,
    pub source: String,
    pub line: Option<u32>,
    #[serde(skip)]
    pub end_line: Option<u32>,
    #[serde(skip)]
    pub member_name: Option<String>,
    #[serde(skip)]
    pub member_type: Option<MemberType>,
}

pub struct Synthesizer;

impl Stage {
    pub fn prose(content: &str, source: &str) -> Self {
        Self {
            kind: StageKind::Prose,
            content: content.to_string(),
            source: source.to_string(),
            line: None,
            end_line: None,
            member_name: None,
            member_type: None,
        }
    }

    pub fn code(member: &Member, source: &str) -> Self {
        let content = member.signature.clone().map_or_else(
            || member.name.as_str().to_string(),
            |s| s.as_str().to_string(),
        );
        Self {
            kind: StageKind::Code,
            content,
            source: source.to_string(),
            line: member.line,
            end_line: None,
            member_name: Some(member.name.as_str().to_string()),
            member_type: Some(member.type_name),
        }
    }

    pub fn member_comment(member: &Member, source: &str) -> Self {
        let content = member
            .comment
            .as_ref()
            .map(|c| c.as_str().to_string())
            .unwrap_or_default();
        Self {
            kind: StageKind::Prose,
            content,
            source: source.to_string(),
            line: member.line,
            end_line: None,
            member_name: Some(member.name.as_str().to_string()),
            member_type: Some(member.type_name),
        }
    }

    pub fn not_found(query: &str, doc: &GuidanceDoc) -> Self {
        Self {
            kind: StageKind::NotFound,
            content: format!("No results found for query: {query}"),
            source: doc.meta.source.as_str().to_string(),
            line: None,
            end_line: None,
            member_name: None,
            member_type: None,
        }
    }
}

impl Synthesizer {
    pub fn synthesize(query: &str, doc: &GuidanceDoc, matched_names: &[String]) -> Vec<Stage> {
        let mut stages = Vec::new();

        if let Some(ref comment) = doc.comment {
            stages.push(Stage::prose(comment.as_str(), doc.meta.source.as_str()));
        }

        for name in matched_names {
            if let Some(member) = doc.members.iter().find(|m| m.name.as_str() == name) {
                stages.push(Stage::code(member, doc.meta.source.as_str()));

                if member.comment.is_some() {
                    stages.push(Stage::member_comment(member, doc.meta.source.as_str()));
                }
            }
        }

        if stages.is_empty() {
            stages.push(Stage {
                kind: StageKind::NotFound,
                content: format!("No results found for query: {query}"),
                source: String::new(),
                line: None,
                end_line: None,
                member_name: None,
                member_type: None,
            });
        }

        stages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::make_test_doc;
    use fluent_types::{GuidanceDoc, Member, MemberType};

    #[test]
    fn test_synthesize_with_matches() {
        let doc = GuidanceDoc {
            members: vec![Member {
                type_name: MemberType::FnDecl,
                name: "hello".into(),
                signature: Some("fn hello() void".into()),
                comment: Some("Says hello.".into()),
                line: Some(1),
                ..Member::default()
            }],
            ..make_test_doc()
        };

        let stages = Synthesizer::synthesize("hello", &doc, &["hello".to_string()]);
        assert!(!stages.is_empty());

        let has_code = stages.iter().any(|s| matches!(s.kind, StageKind::Code));
        assert!(has_code, "should have code stage");

        let has_prose = stages.iter().any(|s| matches!(s.kind, StageKind::Prose));
        assert!(has_prose, "should have prose stage");
    }

    #[test]
    fn test_synthesize_no_matches() {
        let doc = GuidanceDoc::default();
        let stages = Synthesizer::synthesize("nothing", &doc, &[]);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].kind, StageKind::NotFound);
        assert!(stages[0].content.contains("nothing"));
    }
}
