use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::transforms::{rewrite_text_messages, TransformError, TransformStrategy};
use crate::types::RouterRequest;

/// Session-scoped, reversible codeword anonymizer (MOA_ROUTER_SPEC §4).
///
/// Uses the filter engine's `RegexMatch` results to perform consistent
/// substitution: every occurrence of the same identifier gets the same
/// codeword within a session.
#[derive(Debug)]
pub struct CodewordAnonymizer {
    mapping: Mutex<HashMap<String, String>>,
    counters: Mutex<HashMap<String, usize>>,
    /// Whether to skip reverse substitution on transform output.
    skip_reverse: bool,
}

/// Serializable form of a regex match for metadata round-tripping.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MatchEntry {
    pattern_name: String,
    matched_text: String,
    start: usize,
    end: usize,
    action: String,
}

impl CodewordAnonymizer {
    pub fn new() -> Self {
        Self {
            mapping: Mutex::new(HashMap::new()),
            counters: Mutex::new(HashMap::new()),
            skip_reverse: false,
        }
    }

    #[must_use]
    pub fn with_skip_reverse(mut self) -> Self {
        self.skip_reverse = true;
        self
    }

    pub fn into_map(self) -> HashMap<String, String> {
        self.mapping.into_inner().unwrap_or_default()
    }

    /// Reverse-substitute codewords back to real values.
    pub fn reverse(&self, text: &str) -> String {
        let map = self.mapping.lock().unwrap();
        let mut result = text.to_string();
        for (codeword, real) in map.iter() {
            result = result.replace(codeword, real);
        }
        result
    }

    /// Extract regex matches from request metadata (set by DeterministicPreFilter).
    fn extract_matches_from_metadata(request: &RouterRequest) -> Vec<MatchEntry> {
        let Some(pii_filter) = request.metadata.get("pii_filter") else {
            return Vec::new();
        };
        let Some(matches_val) = pii_filter.get("matches") else {
            return Vec::new();
        };
        let Some(matches_arr) = matches_val.as_array() else {
            return Vec::new();
        };
        matches_arr
            .iter()
            .filter_map(|v| serde_json::from_value::<MatchEntry>(v.clone()).ok())
            .collect()
    }

    /// Generate a stable codeword for a given pattern type.
    /// Uses an external dedup cache to avoid duplicate codewords for the same text
    /// within a single `apply_substitution` batch, preventing lock recursion.
    fn codeword_for(
        &self,
        pattern_name: &str,
        text: &str,
        dedup: &mut HashMap<String, String>,
    ) -> String {
        let type_name = pattern_name.to_uppercase();

        // Check dedup cache first
        if let Some(codeword) = dedup.get(text) {
            return codeword.clone();
        }

        // Check shared mapping
        {
            let map = self.mapping.lock().unwrap();
            if let Some((k, _)) = map.iter().find(|(_, v)| v.as_str() == text) {
                let codeword = k.clone();
                dedup.insert(text.to_string(), codeword.clone());
                return codeword;
            }
        }

        let mut counters = self.counters.lock().unwrap();
        let count = counters.entry(type_name).or_insert(0);
        *count += 1;
        let codeword = format!("CODEWORD_{}_{}", pattern_name.to_uppercase(), count);
        dedup.insert(text.to_string(), codeword.clone());
        codeword
    }

    /// Perform consistent substitution using regex matches.
    /// Matches are sorted rightmost-first to preserve indices during replacement.
    fn apply_substitution(&self, text: &str, matches: &[MatchEntry]) -> String {
        let mut result = text.to_string();
        let mut dedup: HashMap<String, String> = HashMap::new();

        // Collect (start, end, codeword, matched_text) tuples and sort by position descending
        let mut substitutions: Vec<(usize, usize, String, String)> = matches
            .iter()
            .map(|m| {
                let codeword = self.codeword_for(&m.pattern_name, &m.matched_text, &mut dedup);
                (m.start, m.end, codeword, m.matched_text.clone())
            })
            .collect();
        substitutions.sort_by_key(|k| std::cmp::Reverse(k.0));

        // Store codeword → original-value mappings
        {
            let mut map = self.mapping.lock().unwrap();
            for (_, _, ref codeword, ref original) in &substitutions {
                map.entry(codeword.clone())
                    .or_insert_with(|| original.clone());
            }
        }

        // Apply substitutions rightmost-first
        for (start, end, codeword, _) in &substitutions {
            result.replace_range(*start..*end, codeword);
        }

        result
    }
}

impl Default for CodewordAnonymizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformStrategy for CodewordAnonymizer {
    fn name(&self) -> &str {
        "codeword_anonymize"
    }

    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError> {
        let matches = Self::extract_matches_from_metadata(request);

        if matches.is_empty() {
            return Ok(request.clone());
        }

        let mut transformed = rewrite_text_messages(request, |content| {
            Ok(self.apply_substitution(content, &matches))
        })?;

        // Record the codeword map in metadata for downstream reverse-substitution
        let map = self.mapping.lock().unwrap();
        if !map.is_empty() {
            let map_obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect();
            transformed
                .metadata
                .insert("codeword_map".into(), serde_json::Value::Object(map_obj));
        }

        Ok(transformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RouterMessage, RouterMessageContent, RouterRequest};

    fn make_request_with_matches(text: &str, matches: &[MatchEntry]) -> RouterRequest {
        let mut req = RouterRequest {
            model: "test-model".into(),
            messages: vec![RouterMessage {
                role: "user".into(),
                content: RouterMessageContent::Text(text.into()),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
            session_id: None,
            agent_id: None,
            adapter: None,
            metadata: Default::default(),
        };
        req.metadata.insert(
            "pii_filter".into(),
            serde_json::json!({
                "matches": matches,
            }),
        );
        req
    }

    fn make_match(pattern: &str, text: &str, start: usize, end: usize) -> MatchEntry {
        MatchEntry {
            pattern_name: pattern.to_string(),
            matched_text: text.to_string(),
            start,
            end,
            action: "Anonymize".to_string(),
        }
    }

    #[test]
    fn same_email_gets_same_codeword() {
        let text = "Contact user@example.com or admin@example.com";
        let matches = vec![
            make_match("email", "user@example.com", 8, 23),
            make_match("email", "admin@example.com", 27, 44),
        ];

        let anon = CodewordAnonymizer::new();
        let request = make_request_with_matches(text, &matches);
        let result = anon.transform(&request).unwrap();
        let output = result.messages[0].content.to_string_lossy();

        // The first email gets CODEWORD_EMAIL_1, second gets CODEWORD_EMAIL_2
        assert!(
            output.contains("CODEWORD_EMAIL_1"),
            "first email should become CODEWORD_EMAIL_1, got: {output}"
        );
        assert!(
            output.contains("CODEWORD_EMAIL_2"),
            "second email should become CODEWORD_EMAIL_2, got: {output}"
        );
        assert!(
            !output.contains("user@example.com"),
            "original email should be replaced"
        );
    }

    #[test]
    fn same_text_gets_same_codeword() {
        let text = "email1@test.com and email1@test.com again";
        let matches = vec![
            make_match("email", "email1@test.com", 0, 15),
            make_match("email", "email1@test.com", 20, 35),
        ];

        let anon = CodewordAnonymizer::new();
        let request = make_request_with_matches(text, &matches);
        let result = anon.transform(&request).unwrap();
        let output = result.messages[0].content.to_string_lossy();

        // Both occurrences of "email1@test.com" should become CODEWORD_EMAIL_1
        let count = output.matches("CODEWORD_EMAIL_1").count();
        assert_eq!(
            count, 2,
            "same email should map to same codeword (CODEWORD_EMAIL_1) appearing twice, got: {output}"
        );
    }

    #[test]
    fn reverse_substitution_restores_original() {
        let text = "My email is user@example.com";
        let matches = vec![make_match("email", "user@example.com", 12, 28)];

        let anon = CodewordAnonymizer::new();
        let request = make_request_with_matches(text, &matches);
        let result = anon.transform(&request).unwrap();
        let output = result.messages[0].content.to_string_lossy();

        assert!(
            output.contains("CODEWORD_EMAIL_1"),
            "should contain codeword"
        );

        let reversed = anon.reverse(&output);
        assert!(
            reversed.contains("user@example.com"),
            "reverse should restore original, got: {reversed}"
        );
    }

    #[test]
    fn no_matches_passes_unchanged() {
        let text = "What is the capital of France?";
        let request = make_request_with_matches(text, &[]);
        let anon = CodewordAnonymizer::new();
        let result = anon.transform(&request).unwrap();
        let output = result.messages[0].content.to_string_lossy();
        assert_eq!(output, "What is the capital of France?");
    }

    #[test]
    fn stores_codeword_map_in_metadata() {
        let text = "email is user@example.com";
        let matches = vec![make_match("email", "user@example.com", 9, 25)];

        let anon = CodewordAnonymizer::new();
        let request = make_request_with_matches(text, &matches);
        let result = anon.transform(&request).unwrap();

        let map = result.metadata.get("codeword_map");
        assert!(map.is_some(), "codeword_map should be present in metadata");
    }

    #[test]
    fn name_returns_correct_value() {
        let anon = CodewordAnonymizer::new();
        assert_eq!(anon.name(), "codeword_anonymize");
    }
}
