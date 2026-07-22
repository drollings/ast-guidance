use common_core::string::contains_ignore_case;
use fluent_types::GuidanceDoc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlmFilterError {
    #[error("LLM client error: {0}")]
    Client(String),
    #[error("no LLM client configured")]
    NoClient,
}

#[derive(Debug, Clone)]
pub struct RelevanceScore {
    pub member_name: String,
    pub score: f32,
    pub reasoning: String,
}

pub trait LlmFilterBackend: Send + Sync {
    fn score_relevance(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<RelevanceScore>, LlmFilterError>;
}

pub struct LlmFilter {
    backend: Option<Box<dyn LlmFilterBackend>>,
}

impl LlmFilter {
    pub fn new(backend: Option<Box<dyn LlmFilterBackend>>) -> Self {
        Self { backend }
    }

    pub fn filter_candidates(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        max_results: usize,
    ) -> Result<Vec<RelevanceScore>, LlmFilterError> {
        let backend = self.backend.as_ref().ok_or(LlmFilterError::NoClient)?;

        let candidate_names: Vec<&str> = doc
            .members
            .iter()
            .filter_map(|m| {
                m.signature
                    .as_ref()
                    .or(Some(&m.name))
                    .map(smol_str::SmolStr::as_str)
            })
            .take(20)
            .collect();

        if candidate_names.is_empty() {
            return Ok(Vec::new());
        }

        let scores = backend.score_relevance(query, &candidate_names)?;

        let mut scores = scores;
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(max_results);

        Ok(scores)
    }
}

pub struct NoopLlmFilter;

impl LlmFilterBackend for NoopLlmFilter {
    fn score_relevance(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<RelevanceScore>, LlmFilterError> {
        Ok(candidates
            .iter()
            .map(|c| {
                let score = if contains_ignore_case(c, query) {
                    0.9
                } else {
                    0.1
                };
                RelevanceScore {
                    member_name: c.to_string(),
                    score,
                    reasoning: "keyword match".into(),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_types::{GuidanceDoc, Member, MemberType, Meta};

    #[test]
    fn test_noop_filter_basic() {
        let filter = LlmFilter::new(Some(Box::new(NoopLlmFilter)));

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
                    name: "greet".into(),
                    signature: Some("fn greet(name: []const u8) void".into()),
                    ..Member::default()
                },
            ],
            ..GuidanceDoc::default()
        };

        let results = filter.filter_candidates("hello", &doc, 5).expect("filter");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_filter_no_client_error() {
        let filter = LlmFilter::new(None);
        let doc = GuidanceDoc::default();
        let result = filter.filter_candidates("test", &doc, 5);
        assert!(result.is_err());
    }
}
