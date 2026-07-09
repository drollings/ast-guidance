//! Commit message generation — LLM-powered with guidance JSON context enrichment.
//!
//! Parses staged diffs, loads guidance context for changed files, and generates
//! commit messages via the configured LLM.

use std::path::Path;

use guidance_core::config::ProjectConfig;
use guidance_core::sync::json_store::load_guidance;
use guidance_llm::ChatMessage;

/// Maximum total characters of diff context sent to the LLM.
const TOTAL_CONTEXT_CAP: usize = 12_000;

/// Number of context lines around a hunk range for member inclusion.
const MEMBER_CONTEXT_LINES: u32 = 15;

// ---------------------------------------------------------------------------
// Diff parsing
// ---------------------------------------------------------------------------

/// Extracts file paths from `diff --git a/... b/...` lines.
#[allow(dead_code)]
pub fn parse_changed_files(diff: &str) -> Vec<String> {
    let prefix = "diff --git a/";
    diff.lines()
        .filter_map(|line| {
            let rest = line.strip_prefix(prefix)?;
            let space = rest.find(' ')?;
            Some(rest[..space].to_string())
        })
        .collect()
}

/// A hunk range `[start, end)` extracted from `@@ ... @@` headers.
type HunkRange = (u32, u32);

/// Parses `@@ -a,b +c,d @@` hunk headers into new-file line ranges.
fn parse_hunk_ranges(chunk: &str) -> Vec<HunkRange> {
    let mut ranges = Vec::new();
    for line in chunk.lines() {
        if !line.starts_with("@@ ") {
            continue;
        }
        let Some(plus_pos) = line.find(" +") else {
            continue;
        };
        let after_plus = &line[plus_pos + 2..];
        let space_pos = after_plus.find(' ').unwrap_or(after_plus.len());
        let range_part = &after_plus[..space_pos];
        let (start_str, count_str) = match range_part.find(',') {
            Some(c) => (&range_part[..c], &range_part[c + 1..]),
            None => (range_part, "1"),
        };
        let start = start_str.parse::<u32>().unwrap_or(0);
        let count = count_str.parse::<u32>().unwrap_or(1);
        if count > 0 {
            ranges.push((start, start + count));
        }
    }
    ranges
}

/// Returns true if `line_num` falls within any hunk range (with context padding).
fn line_in_ranges(line_num: u32, ranges: &[HunkRange], context: u32) -> bool {
    for &(lo, hi) in ranges {
        let padded_lo = lo.saturating_sub(context);
        let padded_hi = hi + context;
        if line_num >= padded_lo && line_num <= padded_hi {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Guidance context loading
// ---------------------------------------------------------------------------

/// A member relevant to a staged change.
struct ChangedMember {
    name: String,
    line: Option<u32>,
    comment: String,
    signature: String,
}

/// Loads the guidance JSON for a source file and returns members within hunk ranges.
fn members_in_hunks(
    guidance_dir: &Path,
    rel_path: &str,
    hunk_ranges: &[HunkRange],
) -> Vec<ChangedMember> {
    let json_path = guidance_dir.join("src").join(format!("{rel_path}.json"));
    let doc = match load_guidance(&json_path) {
        Ok(Some(doc)) => doc,
        _ => return Vec::new(),
    };

    let mut result = Vec::new();
    collect_members(&doc.members, hunk_ranges, &mut result);
    result
}

fn collect_members(
    members: &[guidance_types::Member],
    hunk_ranges: &[HunkRange],
    out: &mut Vec<ChangedMember>,
) {
    for member in members {
        let include = hunk_ranges.is_empty()
            || member.line.is_none()
            || member
                .line
                .is_some_and(|ln| line_in_ranges(ln, hunk_ranges, MEMBER_CONTEXT_LINES));

        if include {
            out.push(ChangedMember {
                name: member.name.to_string(),
                line: member.line,
                comment: member.comment.as_deref().unwrap_or("").to_string(),
                signature: member.signature.as_deref().unwrap_or("").to_string(),
            });
        }

        // Recurse into nested members (structs, enums).
        collect_members(&member.members, hunk_ranges, out);
    }
}

/// Builds the guidance context string for all changed files.
///
/// For each code file, loads the corresponding `.guidance/src/*.json`, parses
/// hunk ranges from the diff, and extracts member names, line numbers,
/// comments, and signatures.
pub fn load_guidance_context(diff: &str, guidance_dir: &Path) -> String {
    let guidance_prefix = format!("{}/", guidance_dir.display());

    // Split diff into per-file chunks.
    let chunks = split_diff_by_file(diff);

    let mut context = String::new();
    let mut code_chunks_processed = 0usize;

    for chunk in &chunks {
        let rel_path = extract_file_path(chunk);
        if rel_path.is_empty() {
            continue;
        }

        // Skip guidance JSON files themselves.
        if rel_path.starts_with(&guidance_prefix) && rel_path.ends_with(".json") {
            continue;
        }

        let hunk_ranges = parse_hunk_ranges(chunk);
        let members = members_in_hunks(guidance_dir, &rel_path, &hunk_ranges);

        if !members.is_empty() {
            context.push_str(&format!("### Functions in {rel_path}:\n"));
            for m in &members {
                if let Some(ln) = m.line {
                    context.push_str(&format!("- {} (line {ln})\n", m.name));
                } else {
                    context.push_str(&format!("- {}\n", m.name));
                }
                if !m.comment.is_empty() {
                    let end = m.comment.find('.').unwrap_or(m.comment.len());
                    let snippet_len = std::cmp::min(end + 1, 120);
                    let snippet = safe_truncate(&m.comment, snippet_len);
                    context.push_str(&format!(": {snippet}\n"));
                } else if !m.signature.is_empty() {
                    let snippet = safe_truncate(&m.signature, 80);
                    context.push_str(&format!(": `{snippet}`\n"));
                }
            }
            context.push('\n');
        }

        // Append a budget-limited excerpt of the raw diff.
        let budget = std::cmp::min(
            chunk.len(),
            TOTAL_CONTEXT_CAP / std::cmp::max(1, chunks.len()),
        );
        context.push_str(safe_truncate(chunk, budget));
        context.push('\n');

        code_chunks_processed += 1;
        if code_chunks_processed >= 4 {
            break;
        }
    }

    context
}

/// Truncates `s` to at most `max_bytes` bytes, rounding down to the nearest
/// char boundary so the result is always valid UTF-8.
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    let end = s.floor_char_boundary(std::cmp::min(max_bytes, s.len()));
    &s[..end]
}

/// Splits a full diff into per-file chunks on `diff --git` boundaries.
fn split_diff_by_file(diff: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0usize;

    for (pos, line) in diff.lines().enumerate() {
        if line.starts_with("diff --git ") && pos > 0 {
            let chunk_end = diff[..]
                .lines()
                .take(pos)
                .map(|l| l.len() + 1)
                .sum::<usize>();
            if start < chunk_end {
                chunks.push(diff[start..chunk_end].to_string());
            }
            start = chunk_end;
        }
    }

    if start < diff.len() {
        chunks.push(diff[start..].to_string());
    }

    chunks
}

/// Extracts the file path from a diff chunk's first line.
fn extract_file_path(chunk: &str) -> String {
    let prefix = "diff --git a/";
    let first_line = chunk.lines().next().unwrap_or("");
    let Some(rest) = first_line.strip_prefix(prefix) else {
        return String::new();
    };
    let space = rest.find(' ').unwrap_or(rest.len());
    rest[..space].to_string()
}

// ---------------------------------------------------------------------------
// LLM commit message generation
// ---------------------------------------------------------------------------

/// Resolves the commit model from project config.
///
/// Looks at `models.commit`, falls back to `models.default`.
/// Returns `(api_url, model_name)`.
pub fn resolve_commit_model(config: &ProjectConfig) -> (String, String) {
    let model_ref = guidance_core::config::resolve_model_ref(config, "default");
    let (api_url, model_name, _) = guidance_core::config::resolve_model_url(config, &model_ref);
    let api_url = if api_url.is_empty() {
        "http://localhost:11434/v1".to_string()
    } else {
        api_url
    };
    (api_url, model_name)
}

/// Generates a commit message via the LLM.
///
/// Sends the diff, guidance context, and an improved prompt that asks for the
/// **reason** behind changes, not just the mechanics.
pub fn generate_commit_message(
    diff: &str,
    context: &str,
    api_url: &str,
    model: &str,
    debug: bool,
) -> Result<String, guidance_llm::LlmError> {
    let system_prompt = "You are a concise git commit message writer. \
        Output only a bullet list. No code fences, no explanations, no headings.";

    let user_prompt = format!(
        "TASK: Write a git commit message as a bullet list documenting WHY changes were made.\n\
         \n\
         Rules:\n\
         - One bullet per distinct logical change.\n\
         - Each bullet: \"* <FunctionOrFileName>: <past-tense description of what changed and WHY>\"\n\
         - Focus on the reason and intent, not just the mechanics.\n\
         - Be specific and concise. Use the function descriptions above when available.\n\
         - Output ONLY the bullet list. No code. No explanations. No headings.\n\
         \n\
         Example:\n\
         * cmdExplain: added --staged flag to enable filtering relevance results by git staging state\n\
         * searchWithAliases: expanded stop-word list to reduce noise in short queries\n\
         \n\
         CONTEXT (guidance JSON members in changed files):\n\
         {context}\n\
         \n\
         DIFF:\n\
         {diff}\n\
         \n\
         Now write the bullet list (each line starts with \"* \"):"
    );

    if debug {
        eprintln!(
            "[commit] prompt ({} chars context, {} chars diff):",
            context.len(),
            diff.len()
        );
    }

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: system_prompt.to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: user_prompt,
        },
    ];

    // Use the direct HTTP path — avoids LlmClient's DefaultQueue which creates
    // a nested tokio runtime that panics when called from #[tokio::main].
    let raw = guidance_llm::chat_complete_http(api_url, &messages, model, None, debug, false)?;

    // Parse bullet lines from the response — strip prefix, then re-add
    // uniformly so the output is consistent regardless of what the LLM used.
    let bullets: Vec<String> = raw
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start_matches([' ', '\t', '\r']);
            let text = if let Some(t) = trimmed.strip_prefix("* ") {
                t.trim()
            } else {
                let t = trimmed.strip_prefix("- ")?;
                t.trim()
            };
            if !text.is_empty() {
                Some(text.to_string())
            } else {
                None
            }
        })
        .collect();

    if bullets.is_empty() {
        return Ok("* Update codebase".to_string());
    }

    let result: String = bullets
        .iter()
        .map(|b| format!("* {b}"))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_changed_files() {
        let diff = "\
diff --git a/src/main.rs b/src/main.rs
index 1234567..abcdefg 100644
--- a/src/main.rs
+++ b/src/main.rs
diff --git a/src/lib.rs b/src/lib.rs
index aaa1111..bbb2222 100644
--- a/src/lib.rs
+++ b/src/lib.rs
";
        let files = parse_changed_files(diff);
        assert_eq!(files, vec!["src/main.rs", "src/lib.rs"]);
    }

    #[test]
    fn test_parse_hunk_ranges() {
        let chunk = "\
diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -10,6 +10,8 @@ fn foo() {
 old line
+new line
@@ -50,3 +52,5 @@ fn bar() {
";
        let ranges = parse_hunk_ranges(chunk);
        assert_eq!(ranges, vec![(10, 18), (52, 57)]);
    }

    #[test]
    fn test_line_in_ranges() {
        let ranges = vec![(10, 20), (50, 60)];
        assert!(line_in_ranges(15, &ranges, 5));
        assert!(line_in_ranges(5, &ranges, 5)); // within context
        assert!(!line_in_ranges(30, &ranges, 5));
    }

    #[test]
    fn test_extract_file_path() {
        let chunk = "diff --git a/src/main.rs b/src/main.rs\n--- a/src/main.rs\n";
        assert_eq!(extract_file_path(chunk), "src/main.rs");
    }

    #[test]
    fn test_split_diff_by_file() {
        let diff = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1 +1 @@
-a
+b
diff --git a/b.rs b/b.rs
--- a/b.rs
+++ b/b.rs
@@ -1 +1 @@
-x
+y
";
        let chunks = split_diff_by_file(diff);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_safe_truncate_ascii() {
        let s = "hello world";
        assert_eq!(safe_truncate(s, 5), "hello");
        assert_eq!(safe_truncate(s, 20), "hello world");
        assert_eq!(safe_truncate(s, 0), "");
    }

    #[test]
    fn test_safe_truncate_multibyte_no_split() {
        // '─' is 3 bytes in UTF-8 (E2 94 80)
        let s = "foo ─── bar";
        // "foo " is 4 bytes, "─" starts at byte 4
        assert_eq!(safe_truncate(s, 4), "foo ");
        // byte 5 lands mid-'─' — must round down to byte 4
        assert_eq!(safe_truncate(s, 5), "foo ");
        assert_eq!(safe_truncate(s, 6), "foo ");
        // byte 7 is past '─' (4 + 3 = 7)
        assert_eq!(safe_truncate(s, 7), "foo ─");
    }

    #[test]
    fn test_safe_truncate_cjk_chars() {
        // Each CJK char is 3 bytes
        let s = "你好世界";
        assert_eq!(s.len(), 12);
        assert_eq!(safe_truncate(s, 3), "你");
        assert_eq!(safe_truncate(s, 5), "你"); // byte 5 mid-char
        assert_eq!(safe_truncate(s, 6), "你好");
        assert_eq!(safe_truncate(s, 12), "你好世界");
        assert_eq!(safe_truncate(s, 100), "你好世界");
    }

    #[test]
    fn test_safe_truncate_emoji() {
        // '😀' is 4 bytes
        let s = "a😀b";
        assert_eq!(safe_truncate(s, 1), "a");
        assert_eq!(safe_truncate(s, 2), "a"); // byte 2 mid-emoji
        assert_eq!(safe_truncate(s, 4), "a"); // byte 4 mid-emoji
        assert_eq!(safe_truncate(s, 5), "a😀");
        assert_eq!(safe_truncate(s, 6), "a😀b");
    }

    #[test]
    fn test_generate_commit_message_e2e() {
        use std::io::{BufRead, BufReader, Write};

        let llm_response = "* Added staged diff parser for multi-file commit context\n\
                             * Integrated guidance JSON context to enrich commit messages";

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let reader = BufReader::new(stream.try_clone().unwrap());
                let mut content_length: usize = 0;
                for line in reader.lines().map_while(Result::ok) {
                    if line.is_empty() {
                        break;
                    }
                    if let Some(val) = line.strip_prefix("Content-Length:") {
                        content_length = val.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0u8; content_length];
                std::io::Read::read_exact(&mut stream, &mut body).ok();

                let json = serde_json::json!({
                    "choices": [{
                        "message": { "content": llm_response }
                    }]
                });
                let resp_body = serde_json::to_string(&json).unwrap();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{resp_body}",
                    resp_body.len()
                );
                stream.write_all(resp.as_bytes()).ok();
            }
        });

        let api_url = format!("http://{addr}");
        let diff = "diff --git a/src/main.rs b/src/main.rs\n\
                     --- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-old\n+new\n";
        let context =
            "### Functions in src/main.rs:\n- cmdCommit (line 10)\n: Handles git commits\n";

        let result = generate_commit_message(diff, context, &api_url, "test-model", false).unwrap();

        assert!(result.contains("staged diff parser"));
        assert!(result.contains("guidance JSON context"));
        // Each line should be a bullet.
        for line in result.lines() {
            assert!(line.starts_with("* "), "expected bullet line, got: {line}");
        }
    }

    #[test]
    fn test_generate_commit_message_multibyte_e2e() {
        use std::io::{BufRead, BufReader, Write};

        // Response containing multi-byte UTF-8 characters (─ box-drawing)
        let llm_response = "* cmdCommit: added ── separator for commit message display\n\
                             * loadContext: parsed ─── unicode boundaries safely";

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let reader = BufReader::new(stream.try_clone().unwrap());
                let mut content_length: usize = 0;
                for line in reader.lines().map_while(Result::ok) {
                    if line.is_empty() {
                        break;
                    }
                    if let Some(val) = line.strip_prefix("Content-Length:") {
                        content_length = val.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0u8; content_length];
                std::io::Read::read_exact(&mut stream, &mut body).ok();

                let json = serde_json::json!({
                    "choices": [{
                        "message": { "content": llm_response }
                    }]
                });
                let resp_body = serde_json::to_string(&json).unwrap();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{resp_body}",
                    resp_body.len()
                );
                stream.write_all(resp.as_bytes()).ok();
            }
        });

        let api_url = format!("http://{addr}");
        let diff = "diff --git a/src/main.rs b/src/main.rs\n\
                     --- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-old\n+new\n";
        let context = "### Functions in src/main.rs:\n- cmdCommit (line 10)\n";

        let result = generate_commit_message(diff, context, &api_url, "test-model", false).unwrap();

        // Verify multi-byte chars survived the round-trip.
        assert!(result.contains("──"));
        assert!(result.contains("───"));
        for line in result.lines() {
            assert!(line.starts_with("* "), "expected bullet: {line}");
        }
    }

    #[test]
    fn test_generate_commit_message_llm_unavailable_fallback() {
        // Point at a port that nothing is listening on.
        let result = generate_commit_message(
            "diff --git a/x b/x\n",
            "context",
            "http://127.0.0.1:1",
            "model",
            false,
        );
        // Should fall through to the caller's error handler — not panic.
        assert!(result.is_err());
    }
}
