use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::transforms::{
    insert_string_map, rewrite_text_messages, TransformError, TransformStrategy,
};
use crate::types::RouterRequest;

/// Session-scoped, reversible codeword anonymizer (MOA_ROUTER_SPEC §4).
///
/// Uses the filter engine's `RegexMatch` results to perform consistent
/// substitution: every occurrence of the same identifier gets the same
/// codeword within a session.
///
/// M6: this type owns NO pattern table by design — matches arrive via
/// request metadata from `DeterministicPreFilter`, whose builtin engine is
/// single-sourced from `fluent_llm::pii_patterns` (see
/// `stages::deterministic::builtin_filter_engine`). The `CODEWORD_<TYPE>_N`
/// vocabulary itself is intentionally separate from `anonymize`'s `[TYPE]`
/// placeholders (reversible session scope vs one-way request scrub — do not
/// merge; each is locked by its own goldens).
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
        let map = self.mapping.lock().unwrap().clone();
        insert_string_map(&mut transformed, "codeword_map", map);

        Ok(transformed)
    }
}
#[cfg(test)]
#[path = "../../tests/transforms_codeword_anonymize.rs"]
mod tests;
