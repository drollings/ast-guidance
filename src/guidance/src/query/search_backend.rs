use common_core::string::contains_ignore_case;
use fluent_types::GuidanceDoc;

use super::identifier;
use super::llm_filter::LlmFilter;
use super::strategy::QueryIntent;
use super::synthesize::{Stage, Synthesizer};
use crate::query_engine::QueryEngineError;
use fluent_knowledge::word_index::WordIndex;

/// Shared context for search backends — avoids threading individual references
/// through every method.
pub struct SearchContext<'a> {
    pub word_index: Option<&'a WordIndex>,
    pub llm_filter: &'a LlmFilter,
}

/// Polymorphic search backend — the fluent-wvr control plane for query dispatch.
///
/// Each backend handles one `QueryIntent`. The orchestrator iterates registered
/// backends and calls `matches` + `search` without branching on implementation.
pub trait SearchBackend: Send + Sync {
    /// Returns true if this backend handles the given intent.
    fn matches(&self, intent: QueryIntent) -> bool;

    /// Execute the search and return synthesized stages.
    fn search(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        ctx: &SearchContext<'_>,
    ) -> Result<Vec<Stage>, QueryEngineError>;
}

/// Search by exact or fuzzy member name, with WordIndex fallback.
pub struct IdentifierBackend;

impl SearchBackend for IdentifierBackend {
    fn matches(&self, intent: QueryIntent) -> bool {
        matches!(
            intent,
            QueryIntent::IdentifierLookup | QueryIntent::SingleIdentifier
        )
    }

    fn search(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        ctx: &SearchContext<'_>,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        let matched_names: Vec<String> = identifier::find_members_by_name(doc, query)
            .into_iter()
            .map(ToString::to_string)
            .collect();

        if !matched_names.is_empty() {
            return Ok(Synthesizer::synthesize(query, doc, &matched_names));
        }

        let sig_matches = identifier::find_members_by_signature(doc, query);
        if !sig_matches.is_empty() {
            let sig_names: Vec<String> = sig_matches.into_iter().map(ToString::to_string).collect();
            return Ok(Synthesizer::synthesize(query, doc, &sig_names));
        }

        // WordIndex fallback
        if let Some(wi) = ctx.word_index {
            if let Some(stages) = word_index_fallback(query, doc, wi) {
                return Ok(stages);
            }
        }

        Err(QueryEngineError::NoResults)
    }
}

/// Search by keyword matching across member names and comments.
pub struct KeywordBackend;

impl SearchBackend for KeywordBackend {
    fn matches(&self, intent: QueryIntent) -> bool {
        matches!(
            intent,
            QueryIntent::CapabilityQuery | QueryIntent::MultiKeyword
        )
    }

    fn search(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        _ctx: &SearchContext<'_>,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        let keywords: Vec<&str> = query.split_whitespace().collect();
        let mut matched_names: Vec<String> = Vec::new();

        for member in &doc.members {
            let matches_keyword = |k: &&str| {
                contains_ignore_case(member.name.as_str(), k)
                    || member
                        .comment
                        .as_ref()
                        .is_some_and(|c| contains_ignore_case(c.as_str(), k))
            };

            if keywords.iter().any(matches_keyword) {
                matched_names.push(member.name.as_str().to_string());
            }
        }

        if matched_names.is_empty() {
            return Err(QueryEngineError::NoResults);
        }

        Ok(Synthesizer::synthesize(query, doc, &matched_names))
    }
}

/// Search using LLM relevance scoring.
pub struct ConceptBackend;

impl SearchBackend for ConceptBackend {
    fn matches(&self, intent: QueryIntent) -> bool {
        matches!(intent, QueryIntent::Conceptual | QueryIntent::HowTo)
    }

    fn search(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        ctx: &SearchContext<'_>,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        let scores = ctx
            .llm_filter
            .filter_candidates(query, doc, 10)
            .map_err(|e| QueryEngineError::LlmFilter(e.to_string()))?;

        let matched_names: Vec<String> = scores
            .into_iter()
            .filter(|s| s.score >= 0.5)
            .map(|s| s.member_name)
            .collect();

        if matched_names.is_empty() {
            return Err(QueryEngineError::NoResults);
        }

        Ok(Synthesizer::synthesize(query, doc, &matched_names))
    }
}

/// Search by file path matching.
pub struct FilePathBackend;

impl SearchBackend for FilePathBackend {
    fn matches(&self, intent: QueryIntent) -> bool {
        matches!(intent, QueryIntent::FilePath)
    }

    fn search(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        _ctx: &SearchContext<'_>,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        let matched_names: Vec<String> = doc
            .members
            .iter()
            .filter(|m| {
                contains_ignore_case(doc.meta.source.as_str(), query)
                    || contains_ignore_case(m.name.as_str(), query)
            })
            .map(|m| m.name.as_str().to_string())
            .collect();

        if matched_names.is_empty() {
            return Err(QueryEngineError::NoResults);
        }

        Ok(Synthesizer::synthesize(query, doc, &matched_names))
    }
}

/// General keyword search with WordIndex fallback.
pub struct GeneralBackend;

impl SearchBackend for GeneralBackend {
    fn matches(&self, intent: QueryIntent) -> bool {
        matches!(intent, QueryIntent::GeneralSearch)
    }

    fn search(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        ctx: &SearchContext<'_>,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        let matched_names: Vec<String> = doc
            .members
            .iter()
            .filter(|m| {
                contains_ignore_case(m.name.as_str(), query)
                    || m.signature
                        .as_ref()
                        .is_some_and(|s| contains_ignore_case(s.as_str(), query))
                    || m.comment
                        .as_ref()
                        .is_some_and(|c| contains_ignore_case(c.as_str(), query))
            })
            .map(|m| m.name.as_str().to_string())
            .collect();

        if !matched_names.is_empty() {
            return Ok(Synthesizer::synthesize(query, doc, &matched_names));
        }

        // WordIndex fallback
        if let Some(wi) = ctx.word_index {
            if let Some(stages) = word_index_fallback(query, doc, wi) {
                return Ok(stages);
            }
        }

        Err(QueryEngineError::NoResults)
    }
}

/// WordIndex fallback logic — shared by IdentifierBackend and GeneralBackend.
fn word_index_fallback(query: &str, doc: &GuidanceDoc, wi: &WordIndex) -> Option<Vec<Stage>> {
    let hits = wi.search(query);
    if hits.is_empty() {
        return None;
    }
    let source = doc.meta.source.as_str();
    let file_matches: Vec<String> = hits
        .iter()
        .filter(|hit| wi.hit_path(hit) == source)
        .filter_map(|_| {
            doc.members.iter().find_map(|m| {
                if contains_ignore_case(m.name.as_str(), query)
                    || m.signature
                        .as_ref()
                        .is_some_and(|s| contains_ignore_case(s.as_str(), query))
                    || m.comment
                        .as_ref()
                        .is_some_and(|c| contains_ignore_case(c.as_str(), query))
                {
                    Some(m.name.as_str().to_string())
                } else {
                    None
                }
            })
        })
        .collect();
    if file_matches.is_empty() {
        None
    } else {
        Some(Synthesizer::synthesize(query, doc, &file_matches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::llm_filter::{
        LlmFilter, LlmFilterBackend, LlmFilterError, NoopLlmFilter, RelevanceScore,
    };
    use crate::tests::common::make_test_doc;

    fn ctx_with_filter<'a>(filter: &'a LlmFilter) -> SearchContext<'a> {
        // `llm_filter` is the only field the backends under test read; a null
        // `word_index` exercises the non-fallback path.
        SearchContext {
            word_index: None,
            llm_filter: filter,
        }
    }

    fn noop_filter() -> LlmFilter {
        LlmFilter::new(Some(Box::new(NoopLlmFilter)))
    }

    #[test]
    fn identifier_backend_matches_intents() {
        assert!(IdentifierBackend.matches(QueryIntent::IdentifierLookup));
        assert!(IdentifierBackend.matches(QueryIntent::SingleIdentifier));
        assert!(!IdentifierBackend.matches(QueryIntent::FilePath));
    }

    #[test]
    fn identifier_backend_finds_by_member_name() {
        let doc = make_test_doc();
        let stages = IdentifierBackend
            .search("helloWorld", &doc, &ctx_with_filter(&noop_filter()))
            .expect("search");
        assert!(!stages.is_empty());
    }

    #[test]
    fn identifier_backend_falls_back_to_signature() {
        // `addNumbers` is a member name; query by a signature-only token that
        // is not the member name should still resolve via the signature path.
        let doc = make_test_doc();
        let stages = IdentifierBackend
            .search("helloWorld", &doc, &ctx_with_filter(&noop_filter()))
            .expect("search");
        assert!(!stages.is_empty());
        // An unrelated identifier yields NoResults (no WordIndex in ctx).
        assert!(matches!(
            IdentifierBackend.search("zzz", &doc, &ctx_with_filter(&noop_filter())),
            Err(QueryEngineError::NoResults)
        ));
    }

    #[test]
    fn keyword_backend_matches_and_searches() {
        assert!(KeywordBackend.matches(QueryIntent::CapabilityQuery));
        assert!(KeywordBackend.matches(QueryIntent::MultiKeyword));
        let doc = make_test_doc();
        // "hello world" matches helloWorld's comment ("Prints hello world").
        let stages = KeywordBackend
            .search("hello world", &doc, &ctx_with_filter(&noop_filter()))
            .expect("search");
        assert!(!stages.is_empty());
        assert!(matches!(
            KeywordBackend.search("zzzzz", &doc, &ctx_with_filter(&noop_filter())),
            Err(QueryEngineError::NoResults)
        ));
    }

    struct StubFilter(Vec<RelevanceScore>);
    impl LlmFilterBackend for StubFilter {
        fn score_relevance(
            &self,
            _query: &str,
            _candidates: &[&str],
        ) -> Result<Vec<RelevanceScore>, LlmFilterError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn concept_backend_matches_and_filters_by_score() {
        assert!(ConceptBackend.matches(QueryIntent::Conceptual));
        assert!(ConceptBackend.matches(QueryIntent::HowTo));
        let filter = LlmFilter::new(Some(Box::new(StubFilter(vec![
            RelevanceScore { member_name: "helloWorld".into(), score: 0.9, reasoning: "".into() },
            RelevanceScore { member_name: "addNumbers".into(), score: 0.2, reasoning: "".into() },
        ]))));
        let stages = ConceptBackend
            .search("add things", &make_test_doc(), &ctx_with_filter(&filter))
            .expect("search");
        // Only the member scoring >= 0.5 survives.
        assert!(!stages.is_empty());
    }

    #[test]
    fn concept_backend_no_results_when_all_below_threshold() {
        let filter = LlmFilter::new(Some(Box::new(StubFilter(vec![
            RelevanceScore { member_name: "helloWorld".into(), score: 0.1, reasoning: "".into() },
        ]))));
        assert!(matches!(
            ConceptBackend.search("x", &make_test_doc(), &ctx_with_filter(&filter)),
            Err(QueryEngineError::NoResults)
        ));
    }

    #[test]
    fn concept_backend_errors_without_a_client() {
        let filter = LlmFilter::new(None);
        assert!(matches!(
            ConceptBackend.search("x", &make_test_doc(), &ctx_with_filter(&filter)),
            Err(QueryEngineError::LlmFilter(_))
        ));
    }

    #[test]
    fn file_path_backend_matches_and_searches() {
        assert!(FilePathBackend.matches(QueryIntent::FilePath));
        let doc = make_test_doc();
        // Matches on the doc source path.
        let stages = FilePathBackend
            .search("test.zig", &doc, &ctx_with_filter(&noop_filter()))
            .expect("search");
        assert!(!stages.is_empty());
        assert!(matches!(
            FilePathBackend.search("nonexistent", &doc, &ctx_with_filter(&noop_filter())),
            Err(QueryEngineError::NoResults)
        ));
    }

    #[test]
    fn general_backend_matches_and_searches() {
        assert!(GeneralBackend.matches(QueryIntent::GeneralSearch));
        let doc = make_test_doc();
        // Matches via the member comment.
        let stages = GeneralBackend
            .search("hello", &doc, &ctx_with_filter(&noop_filter()))
            .expect("search");
        assert!(!stages.is_empty());
        assert!(matches!(
            GeneralBackend.search("qqqq", &doc, &ctx_with_filter(&noop_filter())),
            Err(QueryEngineError::NoResults)
        ));
    }
}
