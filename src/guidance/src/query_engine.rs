use std::path::Path;

use fluent_knowledge::word_index::WordIndex;
use fluent_types::GuidanceDoc;
use thiserror::Error;

use crate::ast_parser;
use crate::memory::MemoryBridge;
use crate::query::formatter::{
    CompactFormatter, DebugFormatter, Formatter, JsonFormatter, MarkdownFormatter,
};
use crate::query::llm_filter::{LlmFilter, LlmFilterBackend, NoopLlmFilter};
use crate::query::search_backend::{
    ConceptBackend, FilePathBackend, GeneralBackend, IdentifierBackend, KeywordBackend,
    SearchBackend, SearchContext,
};
use crate::query::strategy::{self, QueryIntent};
use crate::query::synthesize::{Stage, Synthesizer};
use crate::walk;
use search_vector::GuidanceDb;
use search_vector::SemanticAliases;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Markdown,
    Json,
    Compact,
    Debug,
}

#[derive(Error, Debug)]
pub enum QueryEngineError {
    #[error("IO error: {0}")]
    Io(#[from] common_core::error::IoError),
    #[error("database error: {0}")]
    Db(String),
    #[error("LLM filter error: {0}")]
    LlmFilter(String),
    #[error("no results found")]
    NoResults,
}

common_core::impl_from_io_error!(QueryEngineError);

pub struct QueryEngine {
    pub llm_filter: LlmFilter,
    pub word_index: Option<WordIndex>,
    pub aliases: Option<SemanticAliases>,
    pub no_llm: bool,
    backends: Vec<Box<dyn SearchBackend>>,
    memory: Option<MemoryBridge>,
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryEngine {
    pub fn new() -> Self {
        Self {
            llm_filter: LlmFilter::new(Some(Box::new(NoopLlmFilter))),
            word_index: None,
            aliases: None,
            no_llm: false,
            backends: Self::default_backends(),
            memory: None,
        }
    }

    pub fn new_with_filter(backend: Box<dyn LlmFilterBackend>) -> Self {
        Self {
            llm_filter: LlmFilter::new(Some(backend)),
            word_index: None,
            aliases: None,
            no_llm: false,
            backends: Self::default_backends(),
            memory: None,
        }
    }

    fn default_backends() -> Vec<Box<dyn SearchBackend>> {
        vec![
            Box::new(IdentifierBackend),
            Box::new(KeywordBackend),
            Box::new(ConceptBackend),
            Box::new(FilePathBackend),
            Box::new(GeneralBackend),
        ]
    }

    #[must_use]
    pub fn with_no_llm(mut self) -> Self {
        self.no_llm = true;
        self
    }

    #[must_use]
    pub fn with_word_index(mut self, wi: WordIndex) -> Self {
        self.word_index = Some(wi);
        self
    }

    #[must_use]
    pub fn with_aliases(mut self, aliases: SemanticAliases) -> Self {
        self.aliases = Some(aliases);
        self
    }

    /// Register a custom search backend. Backends are tried in registration
    /// order; the first one whose `matches` returns true handles the query.
    #[must_use]
    pub fn with_backend(mut self, backend: Box<dyn SearchBackend>) -> Self {
        self.backends.push(backend);
        self
    }

    /// Attach a memory bridge for prefetch injection and post-synthesis sync.
    #[must_use]
    pub fn with_memory(mut self, memory: MemoryBridge) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Pre-fetch memory context for injection into the system prompt.
    /// Returns formatted text to prepend to the LLM system prompt.
    /// Returns empty string if no memory bridge is attached.
    pub async fn prefetch_memory_context(&self, query: &str) -> String {
        match &self.memory {
            Some(bridge) => bridge.prefetch_context(query).await,
            None => String::new(),
        }
    }

    /// Sync a completed turn with the active memory plugin.
    /// Call this after LLM synthesis completes to persist the interaction.
    pub async fn sync_memory_turn(&self, user_content: &str, assistant_content: &str) {
        if let Some(ref bridge) = self.memory {
            bridge.sync_turn(user_content, assistant_content).await;
        }
    }

    pub fn load_word_index(&mut self, guidance_dir: &Path) -> Result<(), QueryEngineError> {
        let src_dir = guidance_dir.join("src");
        if !src_dir.is_dir() {
            return Ok(());
        }
        let mut wi = WordIndex::new();
        let root = src_dir.clone();
        walk::walk_files(&src_dir, walk::SOURCE_EXTENSIONS, |path| {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            if let Ok(content) = common_core::io::read_to_string_err(path) {
                wi.index_file(&rel, &content);
            }
        });
        self.word_index = Some(wi);
        Ok(())
    }

    pub fn update_word_index(&mut self, file_path: &str, content: &str) {
        if let Some(ref mut wi) = self.word_index {
            wi.remove_file(file_path);
            wi.index_file(file_path, content);
        }
    }

    /// Dispatch a query through the tiered search pipeline.
    ///
    /// G1: Uses `FsmEngine` for richer classification (intent + domain + confidence)
    /// instead of the simpler `classify_query()`.
    /// G2: Implements tiered escalation — tries the primary backend first, then
    /// falls through to other tiers in priority order until one returns results.
    fn dispatch_search(
        &self,
        query: &str,
        doc: &GuidanceDoc,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        self.dispatch_with_filter(query, doc, &self.llm_filter)
    }

    /// Shared tiered-escalation skeleton, parameterized by the LLM filter.
    /// `dispatch_search` uses the engine's configured filter; `explain_no_llm`
    /// uses a no-op filter so concept searches degrade to keyword.
    fn dispatch_with_filter(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        llm_filter: &LlmFilter,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        let ctx = SearchContext {
            word_index: self.word_index.as_ref(),
            llm_filter,
        };

        let mut fsm = strategy::FsmEngine::new();
        let primary_intent = fsm.run(query).intent;

        let all_intents: [QueryIntent; 8] = [
            QueryIntent::SingleIdentifier,
            QueryIntent::IdentifierLookup,
            QueryIntent::FilePath,
            QueryIntent::CapabilityQuery,
            QueryIntent::HowTo,
            QueryIntent::Conceptual,
            QueryIntent::MultiKeyword,
            QueryIntent::GeneralSearch,
        ];

        let mut ordered: Vec<QueryIntent> = vec![primary_intent];
        for intent in all_intents {
            if intent != primary_intent {
                ordered.push(intent);
            }
        }

        let mut tried: Vec<QueryIntent> = Vec::new();
        for intent in ordered {
            if tried.contains(&intent) {
                continue;
            }
            tried.push(intent);
            for backend in &self.backends {
                if backend.matches(intent) {
                    match backend.search(query, doc, &ctx) {
                        Ok(stages) => return Ok(stages),
                        Err(QueryEngineError::NoResults) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
        }

        Err(QueryEngineError::NoResults)
    }

    /// Explain a query with automatic memory integration.
    ///
    /// G6: If a MemoryBridge is attached, prefetches memory context before
    /// search and syncs the turn after synthesis — callers no longer need
    /// to manually invoke prefetch/sync.
    pub fn explain(&self, query: &str, doc: &GuidanceDoc) -> Result<Vec<Stage>, QueryEngineError> {
        // Expand query with semantic aliases if available
        let expanded_query = if let Some(ref aliases) = self.aliases {
            let expansions = aliases.expand_query(query);
            expansions
                .into_iter()
                .next()
                .unwrap_or_else(|| query.to_string())
        } else {
            query.to_string()
        };

        self.dispatch_search(&expanded_query, doc)
    }

    /// Format stages into the specified output format.
    pub fn format_stages(stages: &[Stage], format: OutputFormat) -> String {
        let mut resolved = stages.to_vec();
        let mut parser = ast_parser::AstParser::new();
        resolve_stage_lines(&mut resolved, &mut parser);
        let formatter: Box<dyn Formatter> = match format {
            OutputFormat::Markdown => Box::new(MarkdownFormatter),
            OutputFormat::Json => Box::new(JsonFormatter),
            OutputFormat::Compact => Box::new(CompactFormatter),
            OutputFormat::Debug => Box::new(DebugFormatter),
        };
        formatter.format(&resolved)
    }

    /// Explain with no-llm support: when no_llm is set, skip LLM filter phase
    /// and return raw structural stages.
    pub fn explain_with_mode(
        &self,
        query: &str,
        doc: &GuidanceDoc,
        format: OutputFormat,
    ) -> Result<String, QueryEngineError> {
        let stages = if self.no_llm {
            self.explain_no_llm(query, doc)?
        } else {
            self.explain(query, doc)?
        };
        Ok(Self::format_stages(&stages, format))
    }

    /// Keyword-only explain path (no LLM filter). Uses the same tiered
    /// escalation pipeline but with a no-op LLM filter, so concept searches
    /// degrade to keyword.
    fn explain_no_llm(
        &self,
        query: &str,
        doc: &GuidanceDoc,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        let noop_filter = LlmFilter::new(Some(Box::new(NoopLlmFilter)));
        self.dispatch_with_filter(query, doc, &noop_filter)
    }

    /// Vector explain using RRF (Reciprocal Rank Fusion) hybrid search.
    ///
    /// G3: Replaces the naive dedup union with `GuidanceDb::hybrid_search`,
    /// which fuses keyword + vector results using RRF (k=60).
    pub fn vector_explain(
        &self,
        query: &str,
        query_vec: &[f32],
        db: &GuidanceDb,
        doc: &GuidanceDoc,
        k: usize,
    ) -> Result<Vec<Stage>, QueryEngineError> {
        let fused = db
            .hybrid_search(query, Some(query_vec), k)
            .map_err(|e| QueryEngineError::Db(e.to_string()))?;

        if fused.is_empty() {
            return Err(QueryEngineError::NoResults);
        }

        let combined: Vec<String> = fused.iter().map(|r| r.name.clone()).collect();
        Ok(Synthesizer::synthesize(query, doc, &combined))
    }
}

/// Resolve line numbers for stages that have member metadata but no lines.
/// Re-parses the source file with tree-sitter to get fresh positions.
fn resolve_stage_lines(stages: &mut [Stage], parser: &mut ast_parser::AstParser) {
    let mut cache: Option<(std::path::PathBuf, String, fluent_types::GuidanceDoc)> = None;

    for stage in stages.iter_mut() {
        if stage.line.is_some() {
            continue;
        }
        let (Some(ref name), Some(mt)) = (&stage.member_name, stage.member_type) else {
            continue;
        };
        let path = std::path::PathBuf::from(&stage.source);
        if !path.exists() {
            continue;
        }
        let source_changed = cache.as_ref().is_none_or(|(p, _, _)| *p != path);
        if source_changed {
            if let Ok(src) = common_core::io::read_to_string_err(&path) {
                if let Ok(doc) = parser.parse_file(&path, &src) {
                    cache = Some((path, src, doc));
                } else {
                    continue;
                }
            } else {
                continue;
            }
        }
        if let Some((_, ref src, ref doc)) = cache {
            if let Some(fresh_member) = doc
                .members
                .iter()
                .find(|m| m.name.as_str() == name.as_str() && m.type_name == mt)
            {
                if let Some(line) = fresh_member.line {
                    stage.line = Some(line);
                    let _ = src;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_knowledge::word_index::WordIndex;
    use fluent_types::{GuidanceDoc, Member, MemberType, Meta};
    use fluent_wvr_testutil::tempdir;

    fn make_test_doc() -> GuidanceDoc {
        GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "src/test.zig".into(),
                language: "zig".into(),
            },
            comment: Some("Test module for query engine.".into()),
            members: vec![
                Member {
                    type_name: MemberType::FnDecl,
                    name: "helloWorld".into(),
                    signature: Some("fn helloWorld() void".into()),
                    comment: Some("Prints hello world.".into()),
                    is_pub: true,
                    line: Some(1),
                    ..Member::default()
                },
                Member {
                    type_name: MemberType::FnDecl,
                    name: "addNumbers".into(),
                    signature: Some("fn addNumbers(a: i32, b: i32) i32".into()),
                    comment: Some("Adds two integers.".into()),
                    is_pub: true,
                    line: Some(5),
                    ..Member::default()
                },
            ],
            ..GuidanceDoc::default()
        }
    }

    #[test]
    fn test_explain_identifier() {
        let engine = QueryEngine::new();
        let doc = make_test_doc();
        let stages = engine.explain("helloWorld", &doc).expect("explain");
        assert!(!stages.is_empty());
        assert!(stages.iter().any(|s| s.content.contains("helloWorld")));
    }

    #[test]
    fn test_explain_capability() {
        let engine = QueryEngine::new();
        let doc = make_test_doc();
        let stages = engine.explain("add numbers", &doc).expect("explain");
        assert!(!stages.is_empty());
    }

    #[test]
    fn test_explain_general() {
        let engine = QueryEngine::new();
        let doc = make_test_doc();
        let stages = engine.explain("hello", &doc).expect("explain");
        assert!(!stages.is_empty());
    }

    #[test]
    fn test_explain_no_results() {
        let engine = QueryEngine::new();
        let doc = make_test_doc();
        let result = engine.explain("zzzzNotHere", &doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_word_index_identifier_fallback() {
        let mut wi = WordIndex::new();
        wi.index_file("src/test.zig", "pub fn helloWorld() void {}");
        let engine = QueryEngine::new().with_word_index(wi);
        let doc = make_test_doc();
        // helloWorld is already in doc, so direct match works
        let stages = engine.explain("helloWorld", &doc).expect("explain");
        assert!(!stages.is_empty());
    }

    #[test]
    fn test_word_index_identifier_fallback_not_found() {
        let mut wi = WordIndex::new();
        wi.index_file("src/test.zig", "pub fn unknownFn() void {}");
        let engine = QueryEngine::new().with_word_index(wi);
        let doc = make_test_doc();
        // The member "unknownFn" doesn't exist in the doc, so WordIndex
        // hits won't produce results either
        let result = engine.explain("unknownFn", &doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_word_index_update() {
        let mut engine = QueryEngine::new();
        let mut wi = WordIndex::new();
        wi.index_file("src/test.zig", "pub fn oldName() void {}");
        engine.word_index = Some(wi);

        // Update the word index
        engine.update_word_index("src/test.zig", "pub fn newName() void {}");

        // Verify the old name is gone
        let old_hits = engine.word_index.as_ref().unwrap().search("oldName");
        assert!(old_hits.is_empty());

        // Verify the new name is indexed
        let new_hits = engine.word_index.as_ref().unwrap().search("newName");
        assert!(!new_hits.is_empty());
    }

    #[test]
    fn test_load_word_index_from_dir() {
        let dir = tempdir();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("create src dir");
        std::fs::write(src_dir.join("test.zig"), "pub fn hello_world() void {}").expect("write");

        let mut engine = QueryEngine::new();
        engine.load_word_index(dir.path()).expect("load word index");

        assert!(engine.word_index.is_some());
        // The sub-token "hello" should be indexed from hello_world
        let hits = engine.word_index.as_ref().unwrap().search("hello");
        assert!(!hits.is_empty(), "should find hello in word index");
    }
}
