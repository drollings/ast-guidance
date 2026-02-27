/// Unit tests for src/guidance — json_store merge logic, query engine leaks.
///
/// Run with: zig build test-guidance
const std = @import("std");
const types = @import("types.zig");
const json_store = @import("json_store.zig");
const query = @import("query.zig");
const main = @import("main.zig");
const sync_mod = @import("sync.zig");

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

test "classifyBulletKeyword returns path under .opencode/skills" {
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

        var engine = query.QueryEngine.init(allocator, "nonexistent_xyz_query", tmp_path, false, false);
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

        var engine = query.QueryEngine.init(allocator, "whatever", tmp_path, false, false);
        engine.deinit();
    }

    try std.testing.expectEqual(.ok, gpa.deinit());
}
