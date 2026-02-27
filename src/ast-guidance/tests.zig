/// Unit tests for src/guidance — json_store merge logic, query engine leaks.
///
/// Run with: zig build test-guidance
const std = @import("std");
const types = @import("types.zig");
const json_store = @import("json_store.zig");
const query = @import("query.zig");
const main = @import("main.zig");
const sync_mod = @import("sync.zig");
const config_mod = @import("config.zig");

// ---------------------------------------------------------------------------
// parseHunkRanges
// ---------------------------------------------------------------------------

test "parseHunkRanges returns empty for non-hunk input" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const chunk = "diff --git a/foo.zig b/foo.zig\nindex abc..def 100644\n--- a/foo.zig\n+++ b/foo.zig\n";
    const ranges = try main.parseHunkRangesPub(allocator, chunk);
    defer allocator.free(ranges);
    try std.testing.expect(ranges.len == 0);
}

test "parseHunkRanges parses single @@ header" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const chunk = "diff --git a/foo.zig b/foo.zig\n@@ -10,6 +15,8 @@ fn foo() {\n+added line\n";
    const ranges = try main.parseHunkRangesPub(allocator, chunk);
    defer allocator.free(ranges);

    try std.testing.expect(ranges.len == 1);
    try std.testing.expectEqual(@as(u32, 15), ranges[0][0]); // start
    try std.testing.expectEqual(@as(u32, 23), ranges[0][1]); // 15 + 8
}

test "parseHunkRanges parses multiple @@ headers" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const chunk =
        \\diff --git a/foo.zig b/foo.zig
        \\@@ -1,3 +1,4 @@ fn first() {
        \\ unchanged
        \\+added
        \\@@ -50,2 +51,3 @@ fn second() {
        \\ other
        \\+also added
    ;
    const ranges = try main.parseHunkRangesPub(allocator, chunk);
    defer allocator.free(ranges);

    try std.testing.expect(ranges.len == 2);
    try std.testing.expectEqual(@as(u32, 1), ranges[0][0]);
    try std.testing.expectEqual(@as(u32, 5), ranges[0][1]); // 1 + 4
    try std.testing.expectEqual(@as(u32, 51), ranges[1][0]);
    try std.testing.expectEqual(@as(u32, 54), ranges[1][1]); // 51 + 3
}

test "parseHunkRanges handles @@ without count (implicit 1)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Some diffs omit the count when it is 1: @@ -5 +7 @@
    const chunk = "@@ -5 +7 @@ fn single() {}\n+line\n";
    const ranges = try main.parseHunkRangesPub(allocator, chunk);
    defer allocator.free(ranges);

    try std.testing.expect(ranges.len == 1);
    try std.testing.expectEqual(@as(u32, 7), ranges[0][0]);
    try std.testing.expectEqual(@as(u32, 8), ranges[0][1]); // 7 + 1
}

// ---------------------------------------------------------------------------
// loadChangedMembers
// ---------------------------------------------------------------------------

test "loadChangedMembers returns empty for missing JSON file" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const members = try main.loadChangedMembersPub(allocator, tmp_path, "src/nonexistent.zig", &.{});
    defer {
        for (members) |m| m.deinit(allocator);
        allocator.free(members);
    }
    try std.testing.expect(members.len == 0);
}

test "loadChangedMembers returns all members when hunk_ranges is empty" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    // Write guidance JSON at {tmp_path}/src/foo.zig.json
    try tmp.dir.makePath("src");
    const json_content =
        \\{
        \\  "meta": {"module": "foo", "source": "src/foo.zig", "language": "zig"},
        \\  "comment": "Foo module.",
        \\  "members": [
        \\    {"type": "fn_decl", "name": "alpha", "line": 10, "is_pub": true,
        \\     "comment": "Does alpha.", "signature": "fn alpha() void",
        \\     "params": [], "tags": [], "patterns": [], "members": []},
        \\    {"type": "fn_decl", "name": "beta",  "line": 30, "is_pub": true,
        \\     "comment": "Does beta.",  "signature": "fn beta() void",
        \\     "params": [], "tags": [], "patterns": [], "members": []}
        \\  ]
        \\}
    ;
    const f = try tmp.dir.createFile("src/foo.zig.json", .{});
    try f.writeAll(json_content);
    f.close();

    // Empty hunk_ranges → all members returned.
    const members = try main.loadChangedMembersPub(allocator, tmp_path, "src/foo.zig", &.{});
    defer {
        for (members) |m| m.deinit(allocator);
        allocator.free(members);
    }

    try std.testing.expect(members.len == 2);
    try std.testing.expectEqualStrings("alpha", members[0].name);
    try std.testing.expectEqualStrings("Does alpha.", members[0].comment);
    try std.testing.expectEqual(@as(?u32, 10), members[0].line);
    try std.testing.expectEqualStrings("beta", members[1].name);
}

test "loadChangedMembers filters by hunk range with context window" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    try tmp.dir.makePath("src");
    const json_content =
        \\{
        \\  "meta": {"module": "bar", "source": "src/bar.zig", "language": "zig"},
        \\  "comment": "",
        \\  "members": [
        \\    {"type": "fn_decl", "name": "near",  "line": 20, "is_pub": true,
        \\     "comment": "Near the hunk.", "signature": "fn near() void",
        \\     "params": [], "tags": [], "patterns": [], "members": []},
        \\    {"type": "fn_decl", "name": "far",   "line": 200, "is_pub": true,
        \\     "comment": "Far from hunk.", "signature": "fn far() void",
        \\     "params": [], "tags": [], "patterns": [], "members": []}
        \\  ]
        \\}
    ;
    const f = try tmp.dir.createFile("src/bar.zig.json", .{});
    try f.writeAll(json_content);
    f.close();

    // Hunk touches new-file lines 25–35 → "near" (line 20) is within ±15 context, "far" (200) is not.
    const hunk_ranges = [_][2]u32{.{ 25, 35 }};
    const members = try main.loadChangedMembersPub(allocator, tmp_path, "src/bar.zig", &hunk_ranges);
    defer {
        for (members) |m| m.deinit(allocator);
        allocator.free(members);
    }

    try std.testing.expect(members.len == 1);
    try std.testing.expectEqualStrings("near", members[0].name);
}

// ---------------------------------------------------------------------------
// Diary local timezone tests (M4)
// ---------------------------------------------------------------------------

test "getLocalTime returns valid ranges" {
    // Use a known UTC timestamp: 2025-01-15 12:30:00 UTC = 1736944200
    const ts: i64 = 1736944200;
    const lt = main.getLocalTime(ts) catch {
        // If localtime_r is unavailable, skip the test gracefully.
        return;
    };
    // year field is years since 1900; 2024 = 124, 2026 = 126.
    try std.testing.expect(lt.year >= 124); // year >= 2024
    // month is 0-based [0–11]
    try std.testing.expect(lt.month >= 0 and lt.month <= 11);
    // mday [1–31]
    try std.testing.expect(lt.mday >= 1 and lt.mday <= 31);
    // hour [0–23]
    try std.testing.expect(lt.hour >= 0 and lt.hour <= 23);
    // minute [0–59]
    try std.testing.expect(lt.minute >= 0 and lt.minute <= 59);
}

// ---------------------------------------------------------------------------
// M5: learn skills path consistency
// ---------------------------------------------------------------------------

test "classifyBulletKeyword returns path under skills_dir" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a temp dir structure with .opencode/skills/zig-current/SKILL.md
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    // Create the skills directory and a SKILL.md file.
    try tmp.dir.makePath(".opencode/skills/zig-current");
    const skill_path = try std.fs.path.join(allocator, &.{ tmp_path, ".opencode", "skills", "zig-current", "SKILL.md" });
    defer allocator.free(skill_path);
    const sf = try std.fs.createFileAbsolute(skill_path, .{});
    sf.close();

    const skills_dir = try std.fs.path.join(allocator, &.{ tmp_path, ".opencode", "skills" });
    defer allocator.free(skills_dir);

    const skill_names = [_][]const u8{"zig-current"};
    const result = try main.classifyBulletKeywordPub(allocator, "use zig comptime for type reflection", &skill_names, skills_dir);
    defer if (result) |r| allocator.free(r);

    // Path must exist and end with SKILL.md
    try std.testing.expect(result != null);
    try std.testing.expect(std.mem.startsWith(u8, result.?, tmp_path));
    try std.testing.expect(std.mem.endsWith(u8, result.?, "SKILL.md"));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn makeMember(name: []const u8, hash: ?[]const u8, doc: ?[]const u8) types.Member {
    return .{
        .type = .fn_decl,
        .name = name,
        .match_hash = hash,
        .comment = doc,
    };
}

// ---------------------------------------------------------------------------
// dupeMember: every field is independently owned
// ---------------------------------------------------------------------------

test "dupeMember produces independent copies" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const orig = types.Member{
        .type = .fn_decl,
        .name = "hello",
        .match_hash = "abc123",
        .signature = "fn hello() void",
        .comment = "Say hello.",
        .returns = "void",
        .is_pub = true,
        .line = 10,
    };

    const copy = try store.dupeMember(orig);
    defer store.freeMember(copy);

    // Verify values match.
    try std.testing.expectEqualStrings(orig.name, copy.name);
    try std.testing.expectEqualStrings(orig.match_hash.?, copy.match_hash.?);
    try std.testing.expectEqualStrings(orig.signature.?, copy.signature.?);
    try std.testing.expectEqualStrings(orig.comment.?, copy.comment.?);
    try std.testing.expectEqualStrings(orig.returns.?, copy.returns.?);
    try std.testing.expect(copy.is_pub == orig.is_pub);
    try std.testing.expect(copy.line.? == orig.line.?);

    // Verify that pointers are different (truly independent).
    try std.testing.expect(copy.name.ptr != orig.name.ptr);
}

test "dupeMember with params" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const params = [_]types.Param{
        .{ .name = "x", .type = "u32", .default = null },
        .{ .name = "y", .type = null, .default = "0" },
    };
    const orig = types.Member{
        .type = .fn_decl,
        .name = "add",
        .params = &params,
    };

    const copy = try store.dupeMember(orig);
    defer store.freeMember(copy);

    try std.testing.expect(copy.params.len == 2);
    try std.testing.expectEqualStrings("x", copy.params[0].name);
    try std.testing.expectEqualStrings("u32", copy.params[0].type.?);
    try std.testing.expectEqualStrings("y", copy.params[1].name);
    try std.testing.expectEqualStrings("0", copy.params[1].default.?);

    // Pointers must differ from the stack-allocated original slice.
    try std.testing.expect(copy.params.ptr != orig.params.ptr);
}

test "dupeMember with nested members" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const nested = [_]types.Member{
        .{ .type = .method, .name = "init" },
        .{ .type = .method, .name = "deinit" },
    };
    const orig = types.Member{
        .type = .@"struct",
        .name = "Foo",
        .members = &nested,
    };

    const copy = try store.dupeMember(orig);
    defer store.freeMember(copy);

    try std.testing.expect(copy.members.len == 2);
    try std.testing.expectEqualStrings("init", copy.members[0].name);
    try std.testing.expectEqualStrings("deinit", copy.members[1].name);
    try std.testing.expect(copy.members.ptr != orig.members.ptr);
}

// ---------------------------------------------------------------------------
// mergeMembers: ownership and correctness
// ---------------------------------------------------------------------------

test "mergeMembers with no existing produces all new members" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const source = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "h1" },
        .{ .type = .fn_decl, .name = "bar", .match_hash = "h2" },
    };

    const result = try store.mergeMembers(&source, &.{}, true);
    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    try std.testing.expect(result.members.len == 2);
    try std.testing.expect(result.members_added == 2);
    try std.testing.expect(result.members_removed == 0);
    try std.testing.expect(result.has_changes == true);
    try std.testing.expectEqualStrings("foo", result.members[0].name);
    try std.testing.expectEqualStrings("bar", result.members[1].name);
}

test "mergeMembers preserves comment when hash unchanged" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const source = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "same_hash", .comment = null },
    };
    const existing = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "same_hash", .comment = "Hand-written doc." },
    };

    const result = try store.mergeMembers(&source, &existing, true);
    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    try std.testing.expect(result.members.len == 1);
    try std.testing.expectEqualStrings("Hand-written doc.", result.members[0].comment.?);
    // Hash unchanged — no update counted.
    try std.testing.expect(result.members_updated == 0);
    try std.testing.expect(result.has_changes == false);
}

test "mergeMembers counts update when hash changed" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const source = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "new_hash" },
    };
    const existing = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "old_hash" },
    };

    const result = try store.mergeMembers(&source, &existing, true);
    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    try std.testing.expect(result.members_updated == 1);
    try std.testing.expect(result.has_changes == true);
}

test "mergeMembers counts removed members" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const source = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo" },
    };
    const existing = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo" },
        .{ .type = .fn_decl, .name = "old_func" },
    };

    const result = try store.mergeMembers(&source, &existing, true);
    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    // "old_func" not in source → removed.
    try std.testing.expect(result.members_removed == 1);
    try std.testing.expect(result.has_changes == true);
    // Only "foo" in result; old_func is dropped.
    try std.testing.expect(result.members.len == 1);
}

test "mergeMembers no changes when source matches existing exactly" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const source = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "h1" },
    };
    const existing = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "h1" },
    };

    const result = try store.mergeMembers(&source, &existing, true);
    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    try std.testing.expect(result.has_changes == false);
    try std.testing.expect(result.members_added == 0);
    try std.testing.expect(result.members_updated == 0);
    try std.testing.expect(result.members_removed == 0);
}

test "mergeMembers clears stale comment when hash changes" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    // Source has no doc comment (return type changed → new hash).
    const source = [_]types.Member{
        .{ .type = .fn_decl, .name = "doSomething", .match_hash = "new_hash", .comment = null },
    };
    // Existing has a comment that is now stale.
    const existing = [_]types.Member{
        .{ .type = .fn_decl, .name = "doSomething", .match_hash = "old_hash", .comment = "Old stale description." },
    };

    const result = try store.mergeMembers(&source, &existing, true);
    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    // Comment must be null (blanked for infill), not the stale old one.
    try std.testing.expect(result.members[0].comment == null);
    try std.testing.expect(result.members_stale == 1);
    try std.testing.expect(result.has_changes == true);
}

test "mergeMembers preserves tags when hash unchanged" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const tags = [_][]const u8{ "important", "public-api" };
    const source = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "h1", .tags = &.{} },
    };
    const existing = [_]types.Member{
        .{ .type = .fn_decl, .name = "foo", .match_hash = "h1", .tags = &tags },
    };

    const result = try store.mergeMembers(&source, &existing, true);
    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    try std.testing.expect(result.members[0].tags.len == 2);
    try std.testing.expectEqualStrings("important", result.members[0].tags[0]);
}

// ---------------------------------------------------------------------------
// freeGuidanceDoc: smoke test (no double-free with GPA)
// ---------------------------------------------------------------------------

test "freeGuidanceDoc frees all fields without double-free" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const doc = types.GuidanceDoc{
        .meta = .{
            .module = try allocator.dupe(u8, "my.module"),
            .source = try allocator.dupe(u8, "src/my.zig"),
        },
        .comment = try allocator.dupe(u8, "Module docs."),
        .skills = try store.dupeSkills(&.{
            .{ .ref = "zig-current", .context = "relevant" },
        }),
        .hashtags = try store.dupeStrings(&.{"#zig"}),
    };

    // Should not crash, no leaks reported by GPA.
    store.freeGuidanceDoc(doc);
}

// ---------------------------------------------------------------------------
// dupeSkills
// ---------------------------------------------------------------------------

test "dupeSkills produces independent copies" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const orig = [_]types.Skill{
        .{ .ref = "zig-current", .context = "API changes" },
        .{ .ref = "gof-patterns", .context = null },
    };

    const copy = try store.dupeSkills(&orig);
    defer {
        for (copy) |s| {
            allocator.free(s.ref);
            if (s.context) |c| allocator.free(c);
        }
        allocator.free(copy);
    }

    try std.testing.expect(copy.len == 2);
    try std.testing.expectEqualStrings("zig-current", copy[0].ref);
    try std.testing.expectEqualStrings("API changes", copy[0].context.?);
    try std.testing.expectEqualStrings("gof-patterns", copy[1].ref);
    try std.testing.expect(copy[1].context == null);
    try std.testing.expect(copy[0].ref.ptr != orig[0].ref.ptr);
}

// ---------------------------------------------------------------------------
// Round-trip: mergeMembers does not alias source / existing strings
// ---------------------------------------------------------------------------

test "mergeMembers result is independent after source freed" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    // Build source members on the heap so we can free them independently.
    const src_name = try allocator.dupe(u8, "myFunc");
    const src_hash = try allocator.dupe(u8, "deadbeef");
    const src_sig = try allocator.dupe(u8, "fn myFunc() void");

    const src_members = try allocator.alloc(types.Member, 1);
    src_members[0] = .{
        .type = .fn_decl,
        .name = src_name,
        .match_hash = src_hash,
        .signature = src_sig,
    };

    const result = try store.mergeMembers(src_members, &.{}, true);

    // Free source members; result must remain valid.
    store.freeMember(src_members[0]);
    allocator.free(src_members);

    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    try std.testing.expect(result.members.len == 1);
    try std.testing.expectEqualStrings("myFunc", result.members[0].name);
    try std.testing.expectEqualStrings("fn myFunc() void", result.members[0].signature.?);
}

test "mergeMembers result is independent after existing freed" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var store = json_store.JsonStore.init(allocator);

    const ex_name = try allocator.dupe(u8, "stableFunc");
    const ex_hash = try allocator.dupe(u8, "cafebabe");
    const ex_doc = try allocator.dupe(u8, "My doc string.");

    const ex_members = try allocator.alloc(types.Member, 1);
    ex_members[0] = .{
        .type = .fn_decl,
        .name = ex_name,
        .match_hash = ex_hash,
        .comment = ex_doc,
    };

    const src_members = [_]types.Member{
        .{ .type = .fn_decl, .name = "stableFunc", .match_hash = "cafebabe" },
    };

    const result = try store.mergeMembers(&src_members, ex_members, true);

    // Free existing members; result must remain valid.
    store.freeMember(ex_members[0]);
    allocator.free(ex_members);

    defer {
        for (result.members) |m| store.freeMember(m);
        allocator.free(result.members);
    }

    try std.testing.expect(result.members.len == 1);
    // Hash matched — comment from existing should be preserved.
    try std.testing.expectEqualStrings("My doc string.", result.members[0].comment.?);
}

// ---------------------------------------------------------------------------
// Query engine: leak detection tests
//
// Each test creates an isolated temp directory, writes a minimal guidance JSON
// into it, runs QueryEngine.execute(), calls freeQueryResult, and lets GPA
// detect any unreleased memory.
// ---------------------------------------------------------------------------

/// Write a minimal valid guidance JSON to a file and return the path (owned).
fn writeTempGuidance(allocator: std.mem.Allocator, dir_path: []const u8, filename: []const u8, module: []const u8) ![]u8 {
    const path = try std.fs.path.join(allocator, &.{ dir_path, filename });
    errdefer allocator.free(path);

    const json = try std.fmt.allocPrint(allocator,
        \\{{
        \\  "meta": {{ "module": "{s}", "source": "src/fake.zig", "language": "zig" }},
        \\  "comment": "A test module.",
        \\  "skills": [{{"ref": "zig-current"}}],
        \\  "hashtags": ["#test"],
        \\  "members": [
        \\    {{"type": "fn_decl", "name": "doThing", "match_hash": "abc", "is_pub": true, "line": 1,
        \\      "signature": "fn doThing() void", "comment": "Does a thing.", "params": [], "tags": [], "patterns": [], "members": []}}
        \\  ]
        \\}}
    , .{module});
    defer allocator.free(json);

    const file = try std.fs.createFileAbsolute(path, .{});
    defer file.close();
    try file.writeAll(json);

    return path;
}

test "QueryEngine.execute no leaks with empty query results" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        // Use a temp dir that won't match any real files — query produces empty results.
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        const cfg = try config_mod.loadConfig(allocator, tmp_path);
        var engine = query.QueryEngine.init(allocator, "nonexistent_xyz_query", tmp_path, false, false, cfg);
        defer engine.deinit();

        const result = try engine.execute();
        defer query.freeQueryResult(allocator, &engine.store, result);

        try std.testing.expect(result.file_matches.len == 0);
        try std.testing.expect(result.guidance_files.len == 0);
    }

    // All allocations must be freed before this check.
    try std.testing.expectEqual(.ok, gpa.deinit());
}

// ---------------------------------------------------------------------------
// M8: infillJsonFile / infillAllJson — cross-language infill sweep
// ---------------------------------------------------------------------------

/// Write a minimal guidance JSON with an optional module comment.
fn writeGuidanceJson(dir: std.fs.Dir, filename: []const u8, comment: ?[]const u8, has_member: bool) !void {
    var buf: [2048]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const w = fbs.writer();
    try w.writeAll("{\"meta\":{\"module\":\"test\",\"source\":\"src/test.zig\"}");
    if (comment) |c| {
        try w.print(",\"comment\":\"{s}\"", .{c});
    }
    if (has_member) {
        try w.writeAll(",\"members\":[{\"type\":\"fn_decl\",\"name\":\"doThing\",\"line\":1,\"is_pub\":true}]");
    } else {
        try w.writeAll(",\"members\":[]");
    }
    try w.writeByte('}');
    const content = fbs.getWritten();
    const f = try dir.createFile(filename, .{});
    defer f.close();
    try f.writeAll(content);
}

test "infillJsonFile returns false when no enhancer configured" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        try writeGuidanceJson(tmp.dir, "a.zig.json", null, false);
        const json_path = try std.fs.path.join(allocator, &.{ tmp_path, "a.zig.json" });
        defer allocator.free(json_path);

        var processor = sync_mod.SyncProcessor.init(allocator, tmp_path, tmp_path, false, false);
        defer processor.deinit();
        processor.infill_comments = true;
        // enhancer is null → must return false without crashing.

        const changed = try processor.infillJsonFile(json_path);
        try std.testing.expect(!changed);
    }
    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "infillJsonFile returns false when no infill/regen flag set" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        try writeGuidanceJson(tmp.dir, "b.zig.json", null, false);
        const json_path = try std.fs.path.join(allocator, &.{ tmp_path, "b.zig.json" });
        defer allocator.free(json_path);

        var processor = sync_mod.SyncProcessor.init(allocator, tmp_path, tmp_path, false, false);
        defer processor.deinit();
        // Neither infill_comments nor regen_comments — must short-circuit.

        const changed = try processor.infillJsonFile(json_path);
        try std.testing.expect(!changed);
    }
    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "infillJsonFile returns false for nonexistent path" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        const json_path = try std.fs.path.join(allocator, &.{ tmp_path, "does_not_exist.json" });
        defer allocator.free(json_path);

        var processor = sync_mod.SyncProcessor.init(allocator, tmp_path, tmp_path, false, false);
        defer processor.deinit();
        processor.infill_comments = true;
        // No enhancer — safe to call; returns false without error.

        const changed = try processor.infillJsonFile(json_path);
        try std.testing.expect(!changed);
    }
    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "infillAllJson returns 0 when no enhancer configured" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        try writeGuidanceJson(tmp.dir, "c.zig.json", null, true);

        var processor = sync_mod.SyncProcessor.init(allocator, tmp_path, tmp_path, false, false);
        defer processor.deinit();
        processor.infill_comments = true;

        var skip: std.StringHashMapUnmanaged(void) = .{};
        defer skip.deinit(allocator);

        const count = try processor.infillAllJson(tmp_path, &skip);
        try std.testing.expectEqual(@as(usize, 0), count);
    }
    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "infillAllJson returns 0 when cross-language flags not set" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        try writeGuidanceJson(tmp.dir, "d.zig.json", null, true);

        var processor = sync_mod.SyncProcessor.init(allocator, tmp_path, tmp_path, false, false);
        defer processor.deinit();
        // Neither infill_comments nor regen_comments set.

        var skip: std.StringHashMapUnmanaged(void) = .{};
        defer skip.deinit(allocator);

        const count = try processor.infillAllJson(tmp_path, &skip);
        try std.testing.expectEqual(@as(usize, 0), count);
    }
    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "infillAllJson skips files in skip_paths" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        try writeGuidanceJson(tmp.dir, "e.zig.json", null, true);
        const skip_file = try std.fs.path.join(allocator, &.{ tmp_path, "e.zig.json" });
        defer allocator.free(skip_file);

        var processor = sync_mod.SyncProcessor.init(allocator, tmp_path, tmp_path, false, false);
        defer processor.deinit();
        processor.infill_comments = true;

        var skip: std.StringHashMapUnmanaged(void) = .{};
        defer skip.deinit(allocator);
        try skip.put(allocator, skip_file, {});

        // File is in skip_paths; no enhancer → count 0, no crash.
        const count = try processor.infillAllJson(tmp_path, &skip);
        try std.testing.expectEqual(@as(usize, 0), count);
    }
    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "infillAllJson ignores non-json files" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        const f = try tmp.dir.createFile("README.md", .{});
        f.close();
        const g = try tmp.dir.createFile("notes.txt", .{});
        g.close();

        var processor = sync_mod.SyncProcessor.init(allocator, tmp_path, tmp_path, false, false);
        defer processor.deinit();
        processor.infill_comments = true;

        var skip: std.StringHashMapUnmanaged(void) = .{};
        defer skip.deinit(allocator);

        const count = try processor.infillAllJson(tmp_path, &skip);
        try std.testing.expectEqual(@as(usize, 0), count);
    }
    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "infillAllJson processes .py.json files alongside .zig.json files" {
    // Verifies that the walk covers both extension types without crashing.
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        try writeGuidanceJson(tmp.dir, "module.zig.json", "existing", false);
        try writeGuidanceJson(tmp.dir, "script.py.json", null, false);

        var processor = sync_mod.SyncProcessor.init(allocator, tmp_path, tmp_path, false, false);
        defer processor.deinit();
        processor.infill_comments = true;
        // No enhancer → returns 0, but both files are visited without error.

        var skip: std.StringHashMapUnmanaged(void) = .{};
        defer skip.deinit(allocator);

        const count = try processor.infillAllJson(tmp_path, &skip);
        try std.testing.expectEqual(@as(usize, 0), count);
    }
    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "freeQueryResult handles all empty slices" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        var store = json_store.JsonStore.init(allocator);
        const r = types.QueryResult{
            .query = "test",
            .file_matches = try allocator.alloc(types.FileMatch, 0),
            .guidance_files = try allocator.alloc(types.GuidanceInfo, 0),
            .ast_analysis = try allocator.alloc(types.ASTAnalysis, 0),
            .related_skills = try allocator.alloc([]const u8, 0),
            .suggested_actions = try allocator.alloc([]const u8, 0),
            .insights = try allocator.alloc([]const u8, 0),
            .recent_capabilities = try allocator.alloc([]const u8, 0),
        };
        query.freeQueryResult(allocator, &store, r);
    }

    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "freeQueryResult frees FileMatch strings" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        var store = json_store.JsonStore.init(allocator);

        const matches = try allocator.alloc(types.FileMatch, 2);
        matches[0] = .{
            .filename = try allocator.dupe(u8, "foo.zig"),
            .filepath = try allocator.dupe(u8, "/tmp/foo.zig"),
            .description = try allocator.dupe(u8, "source file"),
            .line_context = try allocator.dupe(u8, "foo.zig  # main module"),
        };
        matches[1] = .{
            .filename = try allocator.dupe(u8, "bar.zig"),
            .filepath = try allocator.dupe(u8, "/tmp/bar.zig"),
            .description = try allocator.dupe(u8, ""),
            .line_context = try allocator.dupe(u8, ""),
        };

        const r = types.QueryResult{
            .query = "foo",
            .file_matches = matches,
            .guidance_files = try allocator.alloc(types.GuidanceInfo, 0),
            .ast_analysis = try allocator.alloc(types.ASTAnalysis, 0),
            .related_skills = try allocator.alloc([]const u8, 0),
            .suggested_actions = try allocator.alloc([]const u8, 0),
            .insights = try allocator.alloc([]const u8, 0),
            .recent_capabilities = try allocator.alloc([]const u8, 0),
        };
        query.freeQueryResult(allocator, &store, r);
    }

    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "freeQueryResult frees GuidanceInfo strings and slices" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        var store = json_store.JsonStore.init(allocator);

        const g_infos = try allocator.alloc(types.GuidanceInfo, 1);
        const skill_slice = try allocator.alloc([]const u8, 1);
        skill_slice[0] = try allocator.dupe(u8, "zig-current");
        const tag_slice = try allocator.alloc([]const u8, 1);
        tag_slice[0] = try allocator.dupe(u8, "#test");
        g_infos[0] = .{
            .path = try allocator.dupe(u8, "/tmp/.guidance/src/foo.zig.json"),
            .comment = try allocator.dupe(u8, "Module comment."),
            .functions = try allocator.alloc(types.Member, 0),
            .classes = try allocator.alloc(types.Member, 0),
            .skills = skill_slice,
            .tags = tag_slice,
        };

        const r = types.QueryResult{
            .query = "foo",
            .file_matches = try allocator.alloc(types.FileMatch, 0),
            .guidance_files = g_infos,
            .ast_analysis = try allocator.alloc(types.ASTAnalysis, 0),
            .related_skills = try allocator.alloc([]const u8, 0),
            .suggested_actions = try allocator.alloc([]const u8, 0),
            .insights = try allocator.alloc([]const u8, 0),
            .recent_capabilities = try allocator.alloc([]const u8, 0),
        };
        query.freeQueryResult(allocator, &store, r);
    }

    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "QueryEngine deinit with no execute is safe" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        const cfg2 = try config_mod.loadConfig(allocator, tmp_path);
        var engine = query.QueryEngine.init(allocator, "whatever", tmp_path, false, false, cfg2);
        engine.deinit();
    }

    try std.testing.expectEqual(.ok, gpa.deinit());
}

// ---------------------------------------------------------------------------
// config.zig: loadConfig — defaults and JSON parsing
// ---------------------------------------------------------------------------

test "loadConfig falls back to built-in defaults when no config file exists" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    // No config file in tmp_path — must use built-in defaults.
    var cfg = try config_mod.loadConfig(allocator, tmp_path);
    defer cfg.deinit();

    // guidance_root = {tmp_path}/.ast-guidance
    const expected_root = try std.fs.path.join(allocator, &.{ tmp_path, ".ast-guidance" });
    defer allocator.free(expected_root);
    try std.testing.expectEqualStrings(expected_root, cfg.guidance_root);

    // json_base == guidance_root
    try std.testing.expectEqualStrings(cfg.guidance_root, cfg.json_base);

    // skills_dir = {guidance_root}/.skills
    const expected_skills = try std.fs.path.join(allocator, &.{ expected_root, ".skills" });
    defer allocator.free(expected_skills);
    try std.testing.expectEqualStrings(expected_skills, cfg.skills_dir);

    // inbox_dir = {guidance_root}/.doc/inbox
    const expected_inbox = try std.fs.path.join(allocator, &.{ expected_root, ".doc", "inbox" });
    defer allocator.free(expected_inbox);
    try std.testing.expectEqualStrings(expected_inbox, cfg.inbox_dir);

    // Default src_dirs = ["src"]
    try std.testing.expect(cfg.src_dirs.len == 1);
    try std.testing.expectEqualStrings("src", cfg.src_dirs[0]);

    // Default model
    try std.testing.expectEqualStrings(config_mod.DEFAULT_MODEL, cfg.model);

    // Default api_url
    try std.testing.expectEqualStrings(config_mod.DEFAULT_API_URL, cfg.api_url);
}

test "loadConfig deinit releases all memory (no leaks)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        var cfg = try config_mod.loadConfig(allocator, tmp_path);
        cfg.deinit();
    }

    try std.testing.expectEqual(.ok, gpa.deinit());
}

test "loadConfig reads guidance_dir from project config JSON" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    // Write a config JSON with a custom guidance_dir.
    try tmp.dir.makePath(".ast-guidance");
    const cfg_json =
        \\{"guidance_dir": "custom-guidance", "models": {}, "ollama": {}}
    ;
    const cfg_file = try tmp.dir.createFile(".ast-guidance/ast-guidance-config.json", .{});
    try cfg_file.writeAll(cfg_json);
    cfg_file.close();

    var cfg = try config_mod.loadConfig(allocator, tmp_path);
    defer cfg.deinit();

    const expected_root = try std.fs.path.join(allocator, &.{ tmp_path, "custom-guidance" });
    defer allocator.free(expected_root);
    try std.testing.expectEqualStrings(expected_root, cfg.guidance_root);
}

test "loadConfig reads src_dirs array from JSON" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    try tmp.dir.makePath(".ast-guidance");
    const cfg_json =
        \\{"src_dirs": ["src", "lib", "tools"]}
    ;
    const cfg_file = try tmp.dir.createFile(".ast-guidance/ast-guidance-config.json", .{});
    try cfg_file.writeAll(cfg_json);
    cfg_file.close();

    var cfg = try config_mod.loadConfig(allocator, tmp_path);
    defer cfg.deinit();

    try std.testing.expect(cfg.src_dirs.len == 3);
    try std.testing.expectEqualStrings("src", cfg.src_dirs[0]);
    try std.testing.expectEqualStrings("lib", cfg.src_dirs[1]);
    try std.testing.expectEqualStrings("tools", cfg.src_dirs[2]);
}

test "loadConfig reads models.infill as model" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    try tmp.dir.makePath(".ast-guidance");
    const cfg_json =
        \\{"models": {"infill": "mymodel:v2", "default": "other:latest"}}
    ;
    const cfg_file = try tmp.dir.createFile(".ast-guidance/ast-guidance-config.json", .{});
    try cfg_file.writeAll(cfg_json);
    cfg_file.close();

    var cfg = try config_mod.loadConfig(allocator, tmp_path);
    defer cfg.deinit();

    // infill takes priority over default
    try std.testing.expectEqualStrings("mymodel:v2", cfg.model);
}

test "loadConfig falls back to models.default when infill absent" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    try tmp.dir.makePath(".ast-guidance");
    const cfg_json =
        \\{"models": {"default": "default-model:latest"}}
    ;
    const cfg_file = try tmp.dir.createFile(".ast-guidance/ast-guidance-config.json", .{});
    try cfg_file.writeAll(cfg_json);
    cfg_file.close();

    var cfg = try config_mod.loadConfig(allocator, tmp_path);
    defer cfg.deinit();

    try std.testing.expectEqualStrings("default-model:latest", cfg.model);
}

test "loadConfig constructs api_url from ollama base_url and chat_endpoint" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    try tmp.dir.makePath(".ast-guidance");
    const cfg_json =
        \\{"ollama": {"base_url": "http://myhost:9999", "chat_endpoint": "/api/chat"}}
    ;
    const cfg_file = try tmp.dir.createFile(".ast-guidance/ast-guidance-config.json", .{});
    try cfg_file.writeAll(cfg_json);
    cfg_file.close();

    var cfg = try config_mod.loadConfig(allocator, tmp_path);
    defer cfg.deinit();

    try std.testing.expectEqualStrings("http://myhost:9999/api/chat", cfg.api_url);
}

test "loadConfig with invalid JSON falls back to defaults" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    try tmp.dir.makePath(".ast-guidance");
    const cfg_file = try tmp.dir.createFile(".ast-guidance/ast-guidance-config.json", .{});
    try cfg_file.writeAll("not valid json {{{{");
    cfg_file.close();

    // Must not return an error — falls back to built-in defaults.
    var cfg = try config_mod.loadConfig(allocator, tmp_path);
    defer cfg.deinit();

    try std.testing.expectEqualStrings(config_mod.DEFAULT_MODEL, cfg.model);
}

// ---------------------------------------------------------------------------
// query.zig: readInboxBullets — bullet scoring
// ---------------------------------------------------------------------------

/// Helper: write an inbox markdown file and return the absolute path (owned).
fn writeInboxFile(allocator: std.mem.Allocator, dir: std.fs.Dir, dir_path: []const u8, filename: []const u8, content: []const u8) ![]u8 {
    const path = try std.fs.path.join(allocator, &.{ dir_path, filename });
    errdefer allocator.free(path);
    const f = try dir.createFile(filename, .{});
    defer f.close();
    try f.writeAll(content);
    return path;
}

test "readInboxBullets returns empty slice for missing file" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const cfg = try config_mod.loadConfig(allocator, tmp_path);
    var engine = query.QueryEngine.init(allocator, "sync", tmp_path, false, false, cfg);
    defer engine.deinit();

    const nonexistent = try std.fs.path.join(allocator, &.{ tmp_path, "no_such_file.md" });
    defer allocator.free(nonexistent);

    const bullets = try engine.readInboxBulletsTest(nonexistent);
    defer {
        for (bullets) |b| allocator.free(b);
        allocator.free(bullets);
    }

    try std.testing.expect(bullets.len == 0);
}

test "readInboxBullets returns matching bullets and skips non-matching" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const inbox_content =
        \\# Insights
        \\- sync guidance files on every build
        \\- unrelated thing about database migrations
        \\- sync writes JSON to the guidance directory
        \\not a bullet
        \\  - indented non-bullet
    ;
    const inbox_path = try writeInboxFile(allocator, tmp.dir, tmp_path, "INSIGHTS.md", inbox_content);
    defer allocator.free(inbox_path);

    const cfg = try config_mod.loadConfig(allocator, tmp_path);
    var engine = query.QueryEngine.init(allocator, "sync", tmp_path, false, false, cfg);
    defer engine.deinit();

    const bullets = try engine.readInboxBulletsTest(inbox_path);
    defer {
        for (bullets) |b| allocator.free(b);
        allocator.free(bullets);
    }

    // Two bullets contain "sync"; "database migrations" and non-bullets are excluded.
    try std.testing.expect(bullets.len == 2);
    // Each returned bullet is the text after "- "
    for (bullets) |b| {
        try std.testing.expect(std.mem.indexOf(u8, b, "sync") != null);
    }
}

test "readInboxBullets skips heading lines" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const inbox_content =
        \\## sync section
        \\# sync heading
        \\- sync is important
    ;
    const inbox_path = try writeInboxFile(allocator, tmp.dir, tmp_path, "CAP.md", inbox_content);
    defer allocator.free(inbox_path);

    const cfg = try config_mod.loadConfig(allocator, tmp_path);
    var engine = query.QueryEngine.init(allocator, "sync", tmp_path, false, false, cfg);
    defer engine.deinit();

    const bullets = try engine.readInboxBulletsTest(inbox_path);
    defer {
        for (bullets) |b| allocator.free(b);
        allocator.free(bullets);
    }

    // Heading lines must not be returned, only the bullet.
    try std.testing.expect(bullets.len == 1);
    try std.testing.expectEqualStrings("sync is important", bullets[0]);
}

// ---------------------------------------------------------------------------
// QueryEngine.execute: happy-path — guidance JSON found for a matching file
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// chunkIsIgnored / chunkFilePath / splitDiffByFile
// ---------------------------------------------------------------------------
//
// These tests guard against the class of bug where filter logic uses the
// wrong prefix (e.g. "guidance/" instead of ".ast-guidance/"), causing
// guidance JSON diffs to leak into the commit prompt or source diffs to
// be silently dropped.

test "chunkIsIgnored: .ast-guidance/ prefix is filtered" {
    const chunk =
        \\diff --git a/.ast-guidance/src/foo.zig.json b/.ast-guidance/src/foo.zig.json
        \\index 000..111 100644
        \\--- a/.ast-guidance/src/foo.zig.json
        \\+++ b/.ast-guidance/src/foo.zig.json
        \\@@ -1,3 +1,3 @@
    ;
    try std.testing.expect(main.chunkIsIgnoredPub(chunk));
}

test "chunkIsIgnored: guidance/ prefix is NOT filtered (regression guard)" {
    // Old code filtered "guidance/". That prefix no longer exists in the repo;
    // filtering it would silently drop any future file with that name.
    const chunk =
        \\diff --git a/guidance/README.md b/guidance/README.md
        \\index 000..111 100644
        \\--- a/guidance/README.md
        \\+++ b/guidance/README.md
        \\@@ -1 +1 @@
    ;
    try std.testing.expect(!main.chunkIsIgnoredPub(chunk));
}

test "chunkIsIgnored: regular source files are not filtered" {
    const src_chunk =
        \\diff --git a/src/main.zig b/src/main.zig
        \\index 000..111 100644
        \\--- a/src/main.zig
        \\+++ b/src/main.zig
        \\@@ -10,5 +10,6 @@ fn foo() void {
    ;
    try std.testing.expect(!main.chunkIsIgnoredPub(src_chunk));

    const bin_chunk =
        \\diff --git a/bin/ast-guidance-py b/bin/ast-guidance-py
        \\index 000..111 100755
        \\--- a/bin/ast-guidance-py
        \\+++ b/bin/ast-guidance-py
        \\@@ -1,2 +1,3 @@
    ;
    try std.testing.expect(!main.chunkIsIgnoredPub(bin_chunk));
}

test "chunkFilePath: extracts path from diff --git header" {
    const chunk =
        \\diff --git a/src/ast-guidance/sync.zig b/src/ast-guidance/sync.zig
        \\index abc..def 100644
    ;
    const path = main.chunkFilePathPub(chunk);
    try std.testing.expectEqualStrings("src/ast-guidance/sync.zig", path);
}

test "chunkFilePath: returns empty string for malformed chunk" {
    const path = main.chunkFilePathPub("not a diff header\n+added line\n");
    try std.testing.expectEqualStrings("", path);
}

test "splitDiffByFile: single file diff produces one chunk" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const diff =
        \\diff --git a/src/foo.zig b/src/foo.zig
        \\index 000..111 100644
        \\--- a/src/foo.zig
        \\+++ b/src/foo.zig
        \\@@ -1,3 +1,4 @@
        \\ fn foo() void {}
        \\+fn bar() void {}
    ;
    var chunks: std.ArrayList([]const u8) = .{};
    defer chunks.deinit(allocator);
    try main.splitDiffByFilePub(diff, &chunks, allocator);

    try std.testing.expectEqual(@as(usize, 1), chunks.items.len);
    try std.testing.expectEqualStrings("src/foo.zig", main.chunkFilePathPub(chunks.items[0]));
}

test "splitDiffByFile: multi-file diff splits into correct chunks" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const diff =
        \\diff --git a/src/foo.zig b/src/foo.zig
        \\index 000..111 100644
        \\--- a/src/foo.zig
        \\+++ b/src/foo.zig
        \\@@ -1,2 +1,3 @@
        \\+added to foo
        \\diff --git a/src/bar.zig b/src/bar.zig
        \\index 222..333 100644
        \\--- a/src/bar.zig
        \\+++ b/src/bar.zig
        \\@@ -5,2 +5,3 @@
        \\+added to bar
    ;
    var chunks: std.ArrayList([]const u8) = .{};
    defer chunks.deinit(allocator);
    try main.splitDiffByFilePub(diff, &chunks, allocator);

    try std.testing.expectEqual(@as(usize, 2), chunks.items.len);
    try std.testing.expectEqualStrings("src/foo.zig", main.chunkFilePathPub(chunks.items[0]));
    try std.testing.expectEqualStrings("src/bar.zig", main.chunkFilePathPub(chunks.items[1]));
}

test "splitDiffByFile: .ast-guidance/ chunks split correctly and are identifiable" {
    // The filter (chunkIsIgnored) runs after splitting, so we verify that
    // guidance chunks split cleanly and are correctly tagged as ignored.
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const diff =
        \\diff --git a/src/main.zig b/src/main.zig
        \\index 000..111 100644
        \\@@ -1,2 +1,3 @@
        \\+fn new() void {}
        \\diff --git a/.ast-guidance/src/main.zig.json b/.ast-guidance/src/main.zig.json
        \\index 222..333 100644
        \\@@ -1,3 +1,4 @@
        \\+  "comment": "updated"
    ;
    var chunks: std.ArrayList([]const u8) = .{};
    defer chunks.deinit(allocator);
    try main.splitDiffByFilePub(diff, &chunks, allocator);

    try std.testing.expectEqual(@as(usize, 2), chunks.items.len);
    try std.testing.expect(!main.chunkIsIgnoredPub(chunks.items[0])); // src/main.zig — keep
    try std.testing.expect(main.chunkIsIgnoredPub(chunks.items[1])); // .ast-guidance/ — ignore
}

// ---------------------------------------------------------------------------
// QueryEngine.execute finds guidance JSON matching query
// ---------------------------------------------------------------------------

test "QueryEngine.execute finds guidance JSON matching query" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
        defer allocator.free(tmp_path);

        // Create src/ directory with a source file whose name matches the query.
        try tmp.dir.makePath("src");
        const src_file = try tmp.dir.createFile("src/syncer.zig", .{});
        src_file.close();

        // Write a guidance JSON at the path the engine will derive:
        // json_base/{rel}.json = .ast-guidance/src/syncer.zig.json
        try tmp.dir.makePath(".ast-guidance/src");
        const guidance_json =
            \\{
            \\  "meta": {"module": "syncer", "source": "src/syncer.zig", "language": "zig"},
            \\  "comment": "Sync engine.",
            \\  "skills": [{"ref": "zig-current"}],
            \\  "hashtags": ["#sync"],
            \\  "members": [
            \\    {"type": "fn_decl", "name": "runSync", "is_pub": true, "line": 5,
            \\     "match_hash": "aabbcc", "signature": "fn runSync() void",
            \\     "comment": "Run the sync loop.", "params": [], "tags": [], "patterns": [], "members": []}
            \\  ]
            \\}
        ;
        const gj = try tmp.dir.createFile(".ast-guidance/src/syncer.zig.json", .{});
        try gj.writeAll(guidance_json);
        gj.close();

        const cfg = try config_mod.loadConfig(allocator, tmp_path);
        var engine = query.QueryEngine.init(allocator, "syncer", tmp_path, false, false, cfg);
        defer engine.deinit();

        const result = try engine.execute();
        defer query.freeQueryResult(allocator, &engine.store, result);

        // At minimum: syncer.zig is found as a file match.
        try std.testing.expect(result.file_matches.len > 0);

        // The guidance JSON must be loaded.
        try std.testing.expect(result.guidance_files.len > 0);
        try std.testing.expectEqualStrings("Sync engine.", result.guidance_files[0].comment);
        try std.testing.expect(result.guidance_files[0].functions.len == 1);
        try std.testing.expectEqualStrings("runSync", result.guidance_files[0].functions[0].name);
    }

    try std.testing.expectEqual(.ok, gpa.deinit());
}
