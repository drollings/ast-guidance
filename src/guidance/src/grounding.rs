//! Grounding enforcement — ensures LLM synthesis is grounded in source excerpts.
//!
//! Implements the VISION's "no synthesis without source" invariant:
//! - Pre-synthesis: verifies Code stages have source + line numbers
//! - Post-synthesis: parses LLM output for file:line citations and verifies
//!   each citation exists in the provided stages

use std::collections::HashSet;

use crate::query::synthesize::Stage;

/// Result of grounding verification on LLM synthesis output.
#[derive(Debug, Clone)]
pub struct GroundingResult {
    /// True if all citations in the output trace to provided stages.
    pub verified: bool,
    /// Citations found in the output that match provided stages.
    pub verified_citations: Vec<String>,
    /// Citations found in the output that do NOT match any provided stage.
    pub unverified_citations: Vec<String>,
}

impl GroundingResult {
    pub fn is_grounded(&self) -> bool {
        self.verified && self.unverified_citations.is_empty()
    }
}

/// Check whether the stages provide sufficient grounding for LLM synthesis.
///
/// Returns true if at least one Code stage has both a non-empty source
/// and a line number — this is the minimum requirement for grounded synthesis.
pub fn can_synthesize(stages: &[Stage]) -> bool {
    stages.iter().any(|s| {
        s.kind == fluent_types::StageKind::Code && !s.source.is_empty() && s.line.is_some()
    })
}

/// Extract all `file:line` citations from LLM output text.
///
/// Matches patterns like `src/foo.rs:42`, `src/foo.rs:42:10`, or `path/file.zig:7`.
fn extract_citations(text: &str) -> Vec<String> {
    let mut citations = Vec::new();
    for line in text.lines() {
        let mut search_start = 0;
        while search_start < line.len() {
            if let Some(colon_pos) = line[search_start..].find(':') {
                let abs_pos = search_start + colon_pos;
                if let Some(citation) = extract_citation_at(line, abs_pos) {
                    citations.push(citation.clone());
                    search_start = abs_pos + citation.len();
                } else {
                    search_start = abs_pos + 1;
                }
            } else {
                break;
            }
        }
    }
    citations
}

/// Try to extract a `file:line` citation at the given colon position.
fn extract_citation_at(text: &str, colon_pos: usize) -> Option<String> {
    let before = &text[..colon_pos];
    let after = &text[colon_pos + 1..];

    let file_start = before
        .rfind(|c: char| {
            !c.is_alphanumeric() && c != '/' && c != '\\' && c != '.' && c != '_' && c != '-'
        })
        .map_or(0, |p| p + 1);

    let file_part = &before[file_start..];
    if file_part.is_empty() || !file_part.contains('.') {
        return None;
    }

    let mut digits_end = 0;
    for (i, c) in after.char_indices() {
        if c.is_ascii_digit() {
            digits_end = i + c.len_utf8();
        } else {
            break;
        }
    }

    if digits_end == 0 {
        return None;
    }

    let line_number = &after[..digits_end];
    Some(format!("{file_part}:{line_number}"))
}

/// Verify that all citations in LLM output trace to provided stages.
///
/// Builds a set of grounded `file:line` references from Code stages,
/// then checks each citation in the output against this set.
pub fn verify_citations(output: &str, stages: &[Stage]) -> GroundingResult {
    let grounded_refs: HashSet<String> = stages
        .iter()
        .filter(|s| s.kind == fluent_types::StageKind::Code && s.line.is_some())
        .map(|s| {
            let file = std::path::Path::new(&s.source)
                .file_name()
                .map_or_else(|| s.source.clone(), |f| f.to_string_lossy().to_string());
            format!("{file}:{}", s.line.unwrap())
        })
        .collect();

    let citations = extract_citations(output);

    let mut verified = Vec::new();
    let mut unverified = Vec::new();

    for citation in &citations {
        let citation_file = std::path::Path::new(citation.split(':').next().unwrap_or(""))
            .file_name()
            .map_or_else(|| citation.clone(), |f| f.to_string_lossy().to_string());
        let is_grounded = grounded_refs.iter().any(|ref_str| {
            let ref_file = ref_str.split(':').next().unwrap_or("");
            ref_file == citation_file
        });

        if is_grounded {
            verified.push(citation.clone());
        } else {
            unverified.push(citation.clone());
        }
    }

    let all_verified = unverified.is_empty();
    GroundingResult {
        verified: all_verified,
        verified_citations: verified,
        unverified_citations: unverified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_types::StageKind;

    fn make_code_stage(source: &str, line: u32) -> Stage {
        Stage {
            kind: StageKind::Code,
            content: "fn test() {}".to_string(),
            source: source.to_string(),
            line: Some(line),
            end_line: None,
            member_name: Some("test".to_string()),
            member_type: None,
        }
    }

    fn make_prose_stage(content: &str) -> Stage {
        Stage {
            kind: StageKind::Prose,
            content: content.to_string(),
            source: String::new(),
            line: None,
            end_line: None,
            member_name: None,
            member_type: None,
        }
    }

    #[test]
    fn test_can_synthesize_with_grounded_code() {
        let stages = vec![
            make_prose_stage("Module description"),
            make_code_stage("src/main.rs", 42),
        ];
        assert!(can_synthesize(&stages));
    }

    #[test]
    fn test_can_synthesize_without_code() {
        let stages = vec![make_prose_stage("Only prose, no code")];
        assert!(!can_synthesize(&stages));
    }

    #[test]
    fn test_can_synthesize_with_code_no_line() {
        let stages = vec![Stage {
            kind: StageKind::Code,
            content: "fn test() {}".to_string(),
            source: "src/main.rs".to_string(),
            line: None,
            end_line: None,
            member_name: None,
            member_type: None,
        }];
        assert!(!can_synthesize(&stages));
    }

    #[test]
    fn test_can_synthesize_empty() {
        assert!(!can_synthesize(&[]));
    }

    #[test]
    fn test_extract_citations_basic() {
        let text = "See src/main.rs:42 for details";
        let cites = extract_citations(text);
        assert_eq!(cites, vec!["src/main.rs:42"]);
    }

    #[test]
    fn test_extract_citations_multiple() {
        let text = "Implemented in foo.rs:10 and bar.rs:20";
        let cites = extract_citations(text);
        assert!(cites.contains(&"foo.rs:10".to_string()));
        assert!(cites.contains(&"bar.rs:20".to_string()));
    }

    #[test]
    fn test_extract_citations_no_file_extension() {
        let text = "See Makefile:42";
        let cites = extract_citations(text);
        assert!(
            cites.is_empty(),
            "citations without file extension should be ignored"
        );
    }

    #[test]
    fn test_verify_citations_all_grounded() {
        let stages = vec![make_code_stage("src/main.rs", 42)];
        let output = "The function is at main.rs:42 and does X";
        let result = verify_citations(output, &stages);
        assert!(result.is_grounded());
        assert!(result.unverified_citations.is_empty());
    }

    #[test]
    fn test_verify_citations_unverified() {
        let stages = vec![make_code_stage("src/main.rs", 42)];
        let output = "Related to main.rs:42 and also other.rs:99";
        let result = verify_citations(output, &stages);
        assert!(!result.is_grounded());
        assert!(result
            .unverified_citations
            .contains(&"other.rs:99".to_string()));
    }

    #[test]
    fn test_verify_citations_no_citations() {
        let stages = vec![make_code_stage("src/main.rs", 42)];
        let output = "This function does something interesting.";
        let result = verify_citations(output, &stages);
        assert!(result.is_grounded());
        assert!(result.verified_citations.is_empty());
    }

    #[test]
    fn test_verify_citations_empty_output() {
        let stages = vec![make_code_stage("src/main.rs", 42)];
        let result = verify_citations("", &stages);
        assert!(result.is_grounded());
    }
}
