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
