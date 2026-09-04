use super::*;
use crate::types::{RouterChoice, RouterMessage, RouterMessageContent};
// NOTE (ROADMAP_20260903_LLM M1): free-function think-block goldens moved to
// `fluent-llm --test thinking`; this file keeps the handler-level behavior
// (`filtered_content` through `StreamingHandler`).

#[test]
fn stream_answer_finalizes_content() {
    let answer = StreamAnswer::new();
    assert_eq!(answer.get(), None);
    answer.finalize("assembled".into());
    assert_eq!(answer.get().as_deref(), Some("assembled"));
}

#[tokio::test]
async fn stream_answer_wait_returns_after_finalize() {
    let answer = StreamAnswer::new();
    let waiter = answer.clone();
    let task = tokio::spawn(async move {
        waiter
            .wait(std::time::Duration::from_millis(2000))
            .await
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    answer.finalize("done".into());
    assert_eq!(task.await.expect("waiter completes").as_deref(), Some("done"));
}

#[tokio::test]
async fn stream_answer_wait_times_out_without_finalize() {
    let answer = StreamAnswer::new();
    let content = answer
        .wait(std::time::Duration::from_millis(50))
        .await;
    assert_eq!(content, None);
}

#[test]
fn format_single_chunk() {
    let mut h = StreamingHandler::new("req-1", "test-model");
    let line = h.format_chunk("hello", None);
    assert!(line.starts_with("data: "));
    assert!(line.ends_with("\n\n"));
    assert!(line.contains("\"delta\""));
    assert!(line.contains("\"content\":\"hello\""));
    assert_eq!(h.chunk_count(), 1);
}

#[test]
fn format_chunk_with_finish_reason() {
    let mut h = StreamingHandler::new("req-2", "gpt-4");
    let line = h.format_chunk("world", Some("stop"));
    assert!(line.contains("\"finish_reason\":\"stop\""));
    assert!(line.contains("\"content\":\"world\""));
}

#[test]
fn format_choice_chunk() {
    let mut h = StreamingHandler::new("req-3", "gpt-4");
    let choice = RouterChoice {
        index: 0,
        message: RouterMessage {
            role: "assistant".into(),
            content: RouterMessageContent::Text("done".into()),
            tool_calls: None,
            tool_call_id: None,
        },
        finish_reason: "stop".into(),
    };
    let line = h.format_choice_chunk(&choice);
    assert!(line.contains("\"content\":\"done\""));
    assert!(line.contains("\"finish_reason\":\"stop\""));
}

#[test]
fn format_done_marker() {
    let h = StreamingHandler::new("req-1", "test");
    assert_eq!(h.format_done(), "data: [DONE]\n\n");
}

#[test]
fn accumulated_content() {
    let mut h = StreamingHandler::new("req-1", "test");
    h.format_chunk("hello", None);
    h.format_chunk(" world", None);
    assert_eq!(h.accumulated_content(), "hello world");
}

#[test]
fn filtered_content_strips_thinking_blocks() {
    let mut h = StreamingHandler::new("req-1", "test");
    h.format_chunk("Hello ", None);
    h.format_chunk("<thinking>let me think", None);
    h.format_chunk(" carefully</thinking>", None);
    h.format_chunk(" world", None);
    assert_eq!(h.filtered_content(), "Hello  world");
}

#[test]
fn filtered_content_handles_unclosed_thinking() {
    let mut h = StreamingHandler::new("req-1", "test");
    h.format_chunk("A ", None);
    h.format_chunk("<thinking>unclosed", None);
    assert_eq!(h.filtered_content(), "A ");
}

#[test]
fn filtered_content_no_thinking_noop() {
    let mut h = StreamingHandler::new("req-1", "test");
    h.format_chunk("hello", None);
    h.format_chunk(" world", None);
    assert_eq!(h.filtered_content(), "hello world");
}

#[test]
fn filtered_content_multiple_thinking_blocks() {
    let mut h = StreamingHandler::new("req-1", "test");
    h.format_chunk("A", None);
    h.format_chunk("<thinking>skip</thinking>", None);
    h.format_chunk("B", None);
    h.format_chunk("<thinking>skip2</thinking>", None);
    h.format_chunk("C", None);
    assert_eq!(h.filtered_content(), "ABC");
}

#[test]
fn filtered_content_thinking_at_start() {
    let mut h = StreamingHandler::new("req-1", "test");
    h.format_chunk("<thinking>reasoning</thinking>", None);
    h.format_chunk("result", None);
    assert_eq!(h.filtered_content(), "result");
}

#[test]
fn filtered_content_thinking_at_end() {
    let mut h = StreamingHandler::new("req-1", "test");
    h.format_chunk("result", None);
    h.format_chunk("<thinking>reasoning</thinking>", None);
    assert_eq!(h.filtered_content(), "result");
}
