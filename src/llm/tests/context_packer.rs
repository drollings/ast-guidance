use super::*;

#[test]
fn test_no_truncation_when_within_budget() {
    let packer = ContextPacker::new(100);
    let text = "short text";
    let result = packer.budget.truncate_to_budget(text);
    assert_eq!(result, "short text");
}

#[test]
fn test_truncation_when_over_budget() {
    let packer = ContextPacker::new(2);
    let text = "this is a longer text that exceeds the budget";
    let result = packer.budget.truncate_to_budget(text);
    assert!(result.ends_with("..."));
    assert!(result.len() < text.len());
}

#[test]
fn test_pack_context() {
    let packer = ContextPacker::new(100);
    let messages = packer.pack_context(
        "You are a helpful assistant.",
        "Some context here.",
        "What is this?",
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[1].role, "user");
    assert!(messages[1].content.contains("Some context here."));
    assert!(messages[1].content.contains("What is this?"));
}

#[test]
fn test_estimate_tokens() {
    assert_eq!(ContextPacker::estimate_tokens("hello"), 1);
    assert_eq!(ContextPacker::estimate_tokens(""), 0);
    assert_eq!(ContextPacker::estimate_tokens(&"a".repeat(12)), 3);
}

#[test]
fn test_select_lod_by_distance() {
    assert_eq!(ContextPacker::select_lod_by_distance(0.5, 2.0), 0);
    // effective = 1.5 / (1 + 2/3) = 1.5 / 1.667 ≈ 0.9 → lod 0
    assert_eq!(ContextPacker::select_lod_by_distance(1.5, 2.0), 0);
    // effective = 5.0 / 1.667 ≈ 3.0 → lod 3
    assert_eq!(ContextPacker::select_lod_by_distance(5.0, 2.0), 3);
}

#[test]
fn test_ffd_pack_respects_budget() {
    let packer = ContextPacker::new(10);
    let items = vec![("aaaa", "first"), ("bb", "second"), ("cc", "third")];
    let packed = packer.ffd_pack(&items);
    // Budget 10 tokens (generous) — all three fit
    assert_eq!(packed.len(), 3);
}

#[test]
fn test_ffd_pack_respects_order_by_size() {
    let packer = ContextPacker::new(1);
    let first = 1;
    let second = 2;
    let third = 3;
    let items = vec![("aaaa", &first), ("bb", &second), ("cc", &third)];
    let packed = packer.ffd_pack(&items);
    // Only the smallest single item fits within a 1-token budget
    assert_eq!(packed.len(), 1);
}
