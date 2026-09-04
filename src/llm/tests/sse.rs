//! ROADMAP_20260903_LLM M2.3 — SSE framing goldens (moved, not copied).
//!
//! Canonical home for the `drain_sse_lines` framing goldens: the CJK
//! split-across-chunks + partial-tail cases moved from
//! `src/common-core/tests/string.rs:5-32`, plus the must-hold controls
//! (no-`\n` drains zero and preserves bytes; `\n` never splits a
//! codepoint — exhaustive split-point property test over CJK/emoji
//! payloads). Behavior is byte-identical to the removed
//! `common_core::string` shim (M11 deleted it with `parity_new_eq_old`).
//!
//! Framing is transport-completeness (§1): a complete frame can carry a
//! wrong answer and an incomplete frame is never a verdict.

use fluent_llm::openai::drain_sse_lines;

// ── Moved from common-core/tests/string.rs ───────────────────────────────

#[test]
fn drain_sse_lines_cjk_split_across_chunks() {
    let mut buf = Vec::new();
    // "data: 안녕\n" — Korean "annyeong" — UTF-8: EC 95 88 EB 85 95
    // Split at byte 7, which falls inside the "안" character (EC 95 88).
    let chunk1 = b"data: \xEC\x95";
    let chunk2 = b"\x88\xEB\x85\x95\n";
    let lines = drain_sse_lines(&mut buf, chunk1);
    assert!(lines.is_empty(), "no complete line yet");
    let lines = drain_sse_lines(&mut buf, chunk2);
    assert_eq!(lines.len(), 1);
    let decoded = &lines[0];
    assert!(
        !decoded.contains('\u{FFFD}'),
        "got replacement character in: {decoded:?}"
    );
    assert!(decoded.contains("안녕"), "expected 안녕, got: {decoded:?}");
    assert!(buf.is_empty(), "buffer should be drained");
}

#[test]
fn drain_sse_lines_partial_tail_reassembly() {
    let mut buf = Vec::new();
    let lines1 = drain_sse_lines(&mut buf, b"event: ping\r\ndata: {}\npartial");
    assert_eq!(lines1.len(), 2);
    assert_eq!(lines1[0], "event: ping");
    assert_eq!(lines1[1], "data: {}");
    assert_eq!(&buf, b"partial", "tail should remain in buffer");
    let lines2 = drain_sse_lines(&mut buf, b" tail\n");
    assert_eq!(lines2.len(), 1);
    assert_eq!(lines2[0], "partial tail");
    assert!(buf.is_empty(), "buffer should be drained");
}

// ── Controls: must hold for any framing implementation ────────────────────

#[test]
fn control_no_newline_drains_zero_and_preserves_bytes() {
    let mut buf = Vec::new();
    let lines = drain_sse_lines(&mut buf, "partial tail — no newline 安".as_bytes());
    assert!(lines.is_empty(), "no complete line without \\n");
    assert_eq!(
        buf,
        "partial tail — no newline 安".as_bytes(),
        "bytes must be preserved verbatim for the next call"
    );
    // And the preserved bytes reassemble losslessly once terminated.
    let lines = drain_sse_lines(&mut buf, b"\n");
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0], "partial tail — no newline 安");
    assert!(buf.is_empty());
}

#[test]
fn control_empty_chunk_drains_nothing() {
    let mut buf = Vec::new();
    assert!(drain_sse_lines(&mut buf, b"").is_empty());
    assert!(buf.is_empty());
    let lines = drain_sse_lines(&mut buf, b"data: x\n");
    assert_eq!(lines, vec!["data: x"]);
}

#[test]
fn property_newline_never_splits_codepoint() {
    // Exhaustive: every byte offset of CJK/emoji payloads is a split point.
    // Every drained line must decode losslessly (no U+FFFD) and the
    // concatenation of drained lines must equal the input lines.
    let payloads = [
        "data: 안녕하세요 🌍🌏🌎\n".to_string(),
        "data: {\"delta\": \"日本語テスト😀\"}\n".to_string(),
        "data: Héllo Wörld —Zażółć — emoji 🎉🎊\n".to_string(),
        "event: ping\r\ndata: 混合mix😀done\n".to_string(),
    ];
    for payload in &payloads {
        let bytes = payload.as_bytes();
        let want: Vec<String> = payload
            .lines()
            .map(|l| l.trim_end().to_string())
            .collect();
        for split in 0..=bytes.len() {
            let mut buf = Vec::new();
            let first = drain_sse_lines(&mut buf, &bytes[..split]);
            let second = drain_sse_lines(&mut buf, &bytes[split..]);
            let mut got = first;
            got.extend(second);
            assert_eq!(got, want, "split at byte {split} of {payload:?}");
            for line in &got {
                assert!(
                    !line.contains('\u{FFFD}'),
                    "replacement char at split {split} of {payload:?}"
                );
            }
            assert!(buf.is_empty(), "buffer drained at split {split}");
        }
    }
}
