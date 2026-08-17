//! Crate-typed test fixtures shared by guidance-core Tier-1 suites.
//!
//! **Why Tier 1, not `tests/common/mod.rs`:** the ≥8 `GuidanceDoc` fixture
//! builders live inside inline `#[cfg(test)] mod tests` in `src/`, which per
//! Rust's integration-test rule can only import crate-internal items — a
//! Tier-2 `tests/common` is a separate crate linked against the public API.
//! So the shared fixtures live here (`src/tests/common.rs`, Tier 1) and the
//! e2e-only SyncEngine setup lives in `tests/common/mod.rs` (Tier 2).
//! See `ROADMAP_20260816_TESTS.md` §2.1.

use fluent_types::{GuidanceDoc, Member, MemberType, Meta, Param};

/// A `GuidanceDoc` for a hypothetical `src/test.zig` module with two public
/// functions (`helloWorld`, `addNumbers`). The canonical shape — every suite
/// that needs a different member set or module name overrides via
/// struct-update on top of this.
pub fn make_test_doc() -> GuidanceDoc {
    GuidanceDoc {
        meta: Meta {
            module: "test".into(),
            source: "src/test.zig".into(),
            language: "zig".into(),
        },
        comment: Some("Test module for query engine.".into()),
        members: vec![
            Member {
                type_name: MemberType::FnDecl,
                name: "helloWorld".into(),
                signature: Some("fn helloWorld() void".into()),
                comment: Some("Prints hello world.".into()),
                is_pub: true,
                line: Some(1),
                ..Member::default()
            },
            Member {
                type_name: MemberType::FnDecl,
                name: "addNumbers".into(),
                signature: Some("fn addNumbers(a: i32, b: i32) i32".into()),
                comment: Some("Adds two integers.".into()),
                is_pub: true,
                line: Some(5),
                ..Member::default()
            },
        ],
        ..GuidanceDoc::default()
    }
}

/// A `Member` with a `match_hash` — used by the json-store round-trip
/// suites (formerly duplicated as `json_store.rs::make_test_member`).
pub fn make_test_member(name: &str, sig: &str, hash: &str, comment: Option<&str>) -> Member {
    Member {
        type_name: MemberType::FnDecl,
        name: name.into(),
        signature: Some(sig.into()),
        match_hash: Some(hash.into()),
        comment: comment.map(Into::into),
        ..Member::default()
    }
}

/// A `GuidanceDoc` for a `src/test.zig` module with a single `hello`
/// function carrying one `name` parameter — the shape the json-writer
/// round-trip suites assert on.
pub fn make_test_doc_hello() -> GuidanceDoc {
    GuidanceDoc {
        meta: Meta {
            module: "test_module".into(),
            source: "src/test.zig".into(),
            language: "zig".into(),
        },
        comment: Some("A test module.".into()),
        members: vec![Member {
            type_name: MemberType::FnDecl,
            name: "hello".into(),
            signature: Some("fn hello(name: []const u8) -> []const u8".into()),
            params: vec![Param {
                name: "name".into(),
                type_name: Some("[]const u8".into()),
                default: None,
            }],
            returns: Some("[]const u8".into()),
            is_pub: true,
            line: Some(3),
            ..Member::default()
        }],
        ..GuidanceDoc::default()
    }
}