//! ROADMAP_20260903_LLM M1.3 — think-block stripping goldens (moved, not copied).
//!
//! Canonical home for every think-block golden: the free-function assertions
//! moved from `src/router/tests/streaming.rs:141-238` plus the
//! `StreamingThinkFilter` cross-chunk cases moved from
//! `src/common-core/tests/string.rs`, plus the must-NOT-strip precision
//! control group. Behavior is byte-identical to the removed
//! `common_core::string` shims (M11 deleted them with `parity_new_eq_old`).

use fluent_llm::thinking::{
    strip_think_block, strip_thinking_blocks, StreamingThinkFilter,
};

// ── Moved from router/tests/streaming.rs:141-156 ──────────────────────────

#[test]
fn strip_thinking_blocks_free_function() {
    assert_eq!(strip_thinking_blocks("Hello  world"), "Hello  world");
    assert_eq!(
        strip_thinking_blocks("Hello <thinking>reason</thinking> world"),
        "Hello  world"
    );
    assert_eq!(
        strip_thinking_blocks("<thinking>a</thinking>B<thinking>c</thinking>D"),
        "BD"
    );
    assert_eq!(strip_thinking_blocks("start<thinking>unclosed"), "start");
    assert_eq!(
        strip_thinking_blocks("<thinking>only thinking</thinking>"),
        ""
    );
}

#[test]
fn strip_ollama_thinking_blocks() {
    assert_eq!(
        strip_thinking_blocks("<think>reason</think>result"),
        "result"
    );
    assert_eq!(
        strip_thinking_blocks("before <think>reason</think> after"),
        "before  after"
    );
    assert_eq!(strip_thinking_blocks("<think>unclosed"), "");
    assert_eq!(
        strip_thinking_blocks("A<think>B</think>C<think>D</think>E"),
        "ACE"
    );
    assert_eq!(strip_thinking_blocks("no tags here"), "no tags here");
}

#[test]
fn strip_ollama_thinking_at_any_position() {
    assert_eq!(
        strip_thinking_blocks("prefix <think> middle stuff </think> suffix"),
        "prefix  suffix"
    );
}

#[test]
fn strip_thinking_blocks_multiple_formats() {
    assert_eq!(
        strip_thinking_blocks(
            "<think>ollama</think>plain<thinking>xml</thinking>end"
        ),
        "plainend"
    );
}

#[test]
fn strip_plain_thinking_blocks() {
    assert_eq!(
        strip_thinking_blocks(" thinking let me check response\nThe answer is 4"),
        "The answer is 4"
    );
    assert_eq!(
        strip_thinking_blocks("Hello  thinking let me think response\n result"),
        "Hello  result"
    );
    assert_eq!(strip_thinking_blocks(" thinking unclosed"), "");
    assert_eq!(
        strip_thinking_blocks("normal text response here"),
        "normal text response here"
    );
    assert_eq!(
        strip_thinking_blocks(" thinking a response\nB thinking c response\nD"),
        "BD"
    );
    assert_eq!(
        strip_thinking_blocks(" thinking only thinking response\n"),
        ""
    );
}

#[test]
fn strip_plain_thinking_multiple_blocks() {
    assert_eq!(
        strip_thinking_blocks(" thinking a\n response\nB thinking c\n response\nD"),
        "BD"
    );
}

#[test]
fn strip_plain_thinking_only_thinking_response() {
    assert_eq!(
        strip_thinking_blocks(" thinking only thinking response\n"),
        ""
    );
}

#[test]
fn strip_plain_thinking_respects_word_boundary() {
    assert_eq!(
        strip_thinking_blocks("rethinking a plan"),
        "rethinking a plan"
    );
}

// `strip_think_block` is the single-tag-pair spelling over the same engine.
#[test]
fn strip_think_block_delegates_to_all_formats() {
    assert_eq!(strip_think_block("<think>hidden</think>visible"), "visible");
    assert_eq!(strip_think_block("[THINK]hidden[/THINK]visible"), "visible");
    assert_eq!(strip_think_block("no tags here"), "no tags here");
    assert_eq!(
        strip_think_block("Hello <thinking>reason</thinking> world"),
        "Hello  world"
    );
}

// ── Moved from common-core/tests/string.rs (StreamingThinkFilter) ─────────

#[test]
fn streaming_filter_passthrough() {
    let mut f = StreamingThinkFilter::new();
    assert_eq!(f.push("Hello world"), "Hello world");
    assert_eq!(f.finish(), "");
}

#[test]
fn streaming_filter_open_tag_split_across_chunks() {
    let mut f = StreamingThinkFilter::new();
    assert_eq!(f.push("Hello <thi"), "Hello ");
    assert_eq!(
        f.push("nk>secret reasoning</think>the answer"),
        "the answer"
    );
    assert_eq!(f.finish(), "");
}

#[test]
fn streaming_filter_close_tag_split_across_chunks() {
    let mut f = StreamingThinkFilter::new();
    assert_eq!(f.push("A <think>secret</thi"), "A ");
    assert_eq!(f.push("nk>B"), "B");
    assert_eq!(f.finish(), "");
}

#[test]
fn streaming_filter_incomplete_tag_prefix_not_emitted_partial() {
    let mut f = StreamingThinkFilter::new();
    assert_eq!(f.push("value <"), "value ");
    assert_eq!(f.push(""), "");
    assert_eq!(f.finish(), "<", "the incomplete prefix is held back");
    assert_eq!(f.push("input"), "<input");
    assert_eq!(f.finish(), "");
}

#[test]
fn streaming_filter_multiple_blocks() {
    let mut f = StreamingThinkFilter::new();
    assert_eq!(f.push("A"), "A");
    assert_eq!(f.push("<thinking>skip</thinking>"), "");
    assert_eq!(f.push("B"), "B");
    assert_eq!(f.push("<thinking>skip2</thinking>"), "");
    assert_eq!(f.push("C"), "C");
}

#[test]
fn streaming_filter_thinking_at_start_and_end() {
    let mut f = StreamingThinkFilter::new();
    assert_eq!(f.push("<thinking>reasoning</thinking>"), "");
    assert_eq!(f.push("result"), "result");
    assert_eq!(f.push("<think>more</think>"), "");
}

#[test]
fn streaming_filter_unclosed_thinking_discards() {
    let mut f = StreamingThinkFilter::new();
    assert_eq!(f.push("A "), "A ");
    assert_eq!(f.push("<thinking>unclosed"), "");
    assert_eq!(f.finish(), "", "unclosed think content is discarded");
}

// ── Precision control group: must NEVER strip (false-positive rate 0) ─────

#[test]
fn control_group_must_not_strip() {
    // Word-boundary guard: "rethinking" must not match " thinking".
    assert_eq!(strip_thinking_blocks("rethinking a plan"), "rethinking a plan");
    assert_eq!(strip_think_block("rethinking a plan"), "rethinking a plan");
    // Plain prose with no tags.
    assert_eq!(
        strip_thinking_blocks("normal text response here"),
        "normal text response here"
    );
    // Double-space prose (router filtered_content golden shape).
    assert_eq!(strip_thinking_blocks("Hello  world"), "Hello  world");
    // Unclosed text with no think marker at all.
    assert_eq!(
        strip_thinking_blocks("unclosed-without-marker text"),
        "unclosed-without-marker text"
    );
    // Angle prose that is not a think tag.
    assert_eq!(
        strip_thinking_blocks("use <div> tags in html"),
        "use <div> tags in html"
    );
}
