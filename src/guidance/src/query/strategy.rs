use fluent_types::GuidanceDoc;

use super::identifier;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryIntent {
    IdentifierLookup,
    SingleIdentifier,
    CapabilityQuery,
    FilePath,
    HowTo,
    Conceptual,
    MultiKeyword,
    GeneralSearch,
}

impl QueryIntent {
    pub fn priority(&self) -> u8 {
        match self {
            Self::IdentifierLookup | Self::SingleIdentifier => 0,
            Self::FilePath => 1,
            Self::CapabilityQuery => 2,
            Self::HowTo | Self::Conceptual => 4,
            Self::MultiKeyword => 5,
            Self::GeneralSearch => 6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryMatch {
    pub intent: QueryIntent,
    pub priority: u8,
    pub matched: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FsmState {
    Intake,
    Classify,
    Route,
    Validate,
}

#[derive(Debug, Clone)]
pub struct QueryClass {
    pub intent: QueryIntent,
    pub domain: String,
    pub confidence: f32,
    pub tokens: Vec<String>,
    pub detected_file_paths: Vec<String>,
    pub detected_identifiers: Vec<String>,
}

pub struct FsmEngine {
    pub state: FsmState,
}

impl Default for FsmEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FsmEngine {
    pub fn new() -> Self {
        Self {
            state: FsmState::Intake,
        }
    }

    /// INTAKE state: tokenize, detect file paths, detect identifiers
    pub fn intake(&mut self, query: &str) -> QueryClass {
        let trimmed = query.trim().to_string();
        let tokens: Vec<String> = trimmed
            .split_whitespace()
            .map(ToString::to_string)
            .collect();

        let detected_identifiers: Vec<String> = tokens
            .iter()
            .filter(|t| identifier::detect_identifier_pattern(t).is_some())
            .cloned()
            .collect();

        let detected_file_paths: Vec<String> = tokens
            .iter()
            .filter(|t| {
                t.contains('/')
                    || t.contains('\\')
                    || (t.contains('.') && t.chars().any(|c| c.is_ascii_uppercase()))
            })
            .cloned()
            .collect();

        self.state = FsmState::Classify;
        QueryClass {
            intent: QueryIntent::GeneralSearch,
            domain: String::new(),
            confidence: 0.0,
            tokens,
            detected_file_paths,
            detected_identifiers,
        }
    }

    /// CLASSIFY state: determine intent + domain + confidence
    pub fn classify(&self, qc: &QueryClass) -> QueryClass {
        let trimmed: String = qc.tokens.join(" ");
        let trimmed = trimmed.trim();

        let mut result = qc.clone();

        if trimmed.is_empty() {
            result.intent = QueryIntent::GeneralSearch;
            result.domain = "empty".into();
            result.confidence = 1.0;
            return result;
        }

        // File path detection
        if !qc.detected_file_paths.is_empty() {
            result.intent = QueryIntent::FilePath;
            result.domain = "file_path".into();
            result.confidence = 0.95;
            return result;
        }

        // Single identifier
        if qc.tokens.len() == 1 && !qc.detected_identifiers.is_empty() {
            result.intent = QueryIntent::SingleIdentifier;
            result.domain = "identifier".into();
            result.confidence = 0.9;
            return result;
        }

        // How-to questions
        let how_to_words = [
            "how", "what", "why", "when", "where", "which", "can", "does", "do", "is", "are",
        ];
        let first_word = qc
            .tokens
            .first()
            .map(|s| s.to_lowercase())
            .unwrap_or_default();
        if how_to_words.contains(&first_word.as_str()) {
            result.intent = QueryIntent::HowTo;
            result.domain = "how_to".into();
            result.confidence = 0.85;
            return result;
        }

        let word_count = qc.tokens.len();

        // Multi-keyword (short phrases, 2-4 words)
        if (2..=4).contains(&word_count) {
            result.intent = QueryIntent::CapabilityQuery;
            result.domain = "capability".into();
            result.confidence = 0.75;
            return result;
        }

        // Conceptual (5+ words)
        if word_count >= 5 {
            result.intent = QueryIntent::Conceptual;
            result.domain = "conceptual".into();
            result.confidence = 0.7;
            return result;
        }

        // Multi-keyword fallback (2+ words shorter than capability)
        if word_count >= 2 {
            result.intent = QueryIntent::MultiKeyword;
            result.domain = "multi_keyword".into();
            result.confidence = 0.6;
        }

        result
    }

    /// Route classifier — determines which search primitive to use
    pub fn route(&self, qc: &QueryClass) -> &'static str {
        match qc.intent {
            QueryIntent::SingleIdentifier | QueryIntent::IdentifierLookup => "word_index",
            QueryIntent::FilePath => "anchor_lookup",
            QueryIntent::CapabilityQuery | QueryIntent::MultiKeyword => "fts",
            QueryIntent::HowTo => "hybrid",
            QueryIntent::Conceptual => "vector",
            QueryIntent::GeneralSearch => "keyword",
        }
    }

    /// VALIDATE state: check result quality
    pub fn validate(&self, qc: &QueryClass) -> bool {
        // Basic validation: confidence threshold
        qc.confidence >= 0.5
    }

    /// Run the full FSM pipeline for a query
    pub fn run(&mut self, query: &str) -> QueryClass {
        let qc = self.intake(query);
        let qc = self.classify(&qc);
        self.state = FsmState::Route;
        let _route = self.route(&qc);
        self.state = FsmState::Validate;
        let _valid = self.validate(&qc);
        qc
    }
}

pub fn classify_query(query: &str) -> QueryIntent {
    let trimmed = query.trim();

    if trimmed.is_empty() {
        return QueryIntent::GeneralSearch;
    }

    if identifier::detect_identifier_pattern(trimmed).is_some() {
        return QueryIntent::IdentifierLookup;
    }

    let word_count = trimmed.split_whitespace().count();

    if (2..=4).contains(&word_count) {
        QueryIntent::CapabilityQuery
    } else if word_count >= 5 {
        QueryIntent::Conceptual
    } else {
        QueryIntent::GeneralSearch
    }
}

pub fn matches(query: &str, _db: &GuidanceDoc) -> QueryMatch {
    let intent = classify_query(query);

    QueryMatch {
        intent,
        priority: intent.priority(),
        matched: !matches!(intent, QueryIntent::GeneralSearch),
    }
}

pub fn query_strategy_priority(intent: QueryIntent) -> u8 {
    intent.priority()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Original classify tests ─────────────────────────────────────────────────

    #[test]
    fn test_classify_identifier() {
        assert_eq!(classify_query("helloWorld"), QueryIntent::IdentifierLookup);
        assert_eq!(
            classify_query("snake_case_fn"),
            QueryIntent::IdentifierLookup
        );
        assert_eq!(
            classify_query("PascalCaseType"),
            QueryIntent::IdentifierLookup
        );
    }

    #[test]
    fn test_classify_capability() {
        assert_eq!(
            classify_query("target registry"),
            QueryIntent::CapabilityQuery
        );
        assert_eq!(
            classify_query("ast parsing zig"),
            QueryIntent::CapabilityQuery
        );
    }

    #[test]
    fn test_classify_concept() {
        assert_eq!(
            classify_query("how do I parse a zig file"),
            QueryIntent::Conceptual
        );
        assert_eq!(
            classify_query("what is the target registry used for"),
            QueryIntent::Conceptual
        );
    }

    #[test]
    fn test_classify_empty() {
        assert_eq!(classify_query(""), QueryIntent::GeneralSearch);
        assert_eq!(classify_query("   "), QueryIntent::GeneralSearch);
    }

    #[test]
    fn test_matches() {
        let doc = GuidanceDoc::default();
        let m = matches("helloWorld", &doc);
        assert!(m.matched);
        assert_eq!(m.intent, QueryIntent::IdentifierLookup);
    }

    #[test]
    fn test_priority_order() {
        assert!(
            query_strategy_priority(QueryIntent::IdentifierLookup)
                < query_strategy_priority(QueryIntent::CapabilityQuery)
        );
        assert!(
            query_strategy_priority(QueryIntent::CapabilityQuery)
                < query_strategy_priority(QueryIntent::Conceptual)
        );
    }

    // ── FSM Engine tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_fsm_intake_tokenizes() {
        let mut engine = FsmEngine::new();
        let qc = engine.intake("hello world test");
        assert_eq!(qc.tokens.len(), 3);
        assert_eq!(engine.state, FsmState::Classify);
    }

    #[test]
    fn test_fsm_intake_detects_identifiers() {
        let mut engine = FsmEngine::new();
        let qc = engine.intake("helloWorld foo_bar");
        assert!(!qc.detected_identifiers.is_empty());
        assert!(qc.detected_identifiers.contains(&"helloWorld".to_string()));
    }

    #[test]
    fn test_fsm_intake_detects_file_paths() {
        let mut engine = FsmEngine::new();
        let qc = engine.intake("src/main.zig");
        assert!(!qc.detected_file_paths.is_empty());
    }

    #[test]
    fn test_fsm_classify_single_identifier() {
        let mut engine = FsmEngine::new();
        let qc = engine.intake("helloWorld");
        let qc = engine.classify(&qc);
        assert_eq!(qc.intent, QueryIntent::SingleIdentifier);
    }

    #[test]
    fn test_fsm_classify_how_to() {
        let mut engine = FsmEngine::new();
        let qc = engine.intake("how do I parse a file");
        let qc = engine.classify(&qc);
        assert_eq!(qc.intent, QueryIntent::HowTo);
    }

    #[test]
    fn test_fsm_classify_capability() {
        let mut engine = FsmEngine::new();
        let qc = engine.intake("ast parsing");
        let qc = engine.classify(&qc);
        assert_eq!(qc.intent, QueryIntent::CapabilityQuery);
    }

    #[test]
    fn test_fsm_classify_conceptual() {
        let mut engine = FsmEngine::new();
        let qc = engine.intake("the target registry resolves dependencies across modules");
        let qc = engine.classify(&qc);
        assert_eq!(qc.intent, QueryIntent::Conceptual);
    }

    #[test]
    fn test_fsm_classify_file_path() {
        let mut engine = FsmEngine::new();
        let qc = engine.intake("src/main.rs");
        let qc = engine.classify(&qc);
        assert_eq!(qc.intent, QueryIntent::FilePath);
    }

    #[test]
    fn test_fsm_route_returns_strategy() {
        let mut engine = FsmEngine::new();

        let qc1 = engine.intake("helloWorld");
        let qc1 = engine.classify(&qc1);
        assert_eq!(engine.route(&qc1), "word_index");

        let qc2 = engine.intake("what is a target used for");
        let qc2 = engine.classify(&qc2);
        assert_eq!(qc2.intent, QueryIntent::HowTo);
        assert_eq!(engine.route(&qc2), "hybrid");

        let qc3 = engine.intake("ast parsing zig");
        let qc3 = engine.classify(&qc3);
        assert_eq!(engine.route(&qc3), "fts");
    }

    #[test]
    fn test_fsm_full_run() {
        let mut engine = FsmEngine::new();
        let qc = engine.run("helloWorld");
        assert_eq!(qc.intent, QueryIntent::SingleIdentifier);
        assert!(qc.confidence >= 0.5);
    }

    #[test]
    fn test_fsm_confidence_threshold() {
        let mut engine = FsmEngine::new();
        let qc = engine.run("");
        assert_eq!(qc.intent, QueryIntent::GeneralSearch);
        assert!(engine.validate(&qc));
    }
}
