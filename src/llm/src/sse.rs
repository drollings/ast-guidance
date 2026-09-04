//! SSE line framing — the single owner (ROADMAP_20260903_LLM M2).
//!
//! Moved verbatim from `common_core::string::drain_sse_lines`. It lives with
//! its sole protocol consumer (`parse_openai_stream_delta` in
//! [`crate::openai`], which re-exports it) and has zero cross-crate
//! dependencies.
//!
//! M11 deleted the `common-core::string` byte-identical shim copy (kept
//! through M10 under `#[deprecated]`); the owner goldens in `tests/sse.rs`
//! are the lasting contract.

/// Deferred-UTF-8-decode SSE line drainer.
///
/// Appends `chunk` to `buffer`, drains every complete newline-terminated line,
/// decodes it via `String::from_utf8_lossy`, trims trailing whitespace, and
/// returns the lines.  Any unterminated tail is left in `buffer` for the next
/// call.
///
/// Because `\n` (0x0A) is never a UTF-8 lead or continuation byte, splitting on
/// it cannot cut a codepoint, so every drained line is a whole number of
/// codepoints and decodes losslessly — safe for CJK, emoji, etc.
///
/// Framing is transport-completeness (roadmap §1): it is neither a
/// confidence nor a correctness signal — a complete frame can carry a wrong
/// answer, and an incomplete frame must never be treated as a verdict.
pub fn drain_sse_lines(buffer: &mut Vec<u8>, chunk: &[u8]) -> Vec<String> {
    buffer.extend_from_slice(chunk);
    let mut lines = Vec::new();
    while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = buffer.drain(..=pos).collect();
        lines.push(String::from_utf8_lossy(&line).trim_end().to_string());
    }
    lines
}
