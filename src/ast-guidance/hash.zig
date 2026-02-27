const std = @import("std");
const types = @import("types.zig");

pub fn sha256Hex(allocator: std.mem.Allocator, data: []const u8) ![]const u8 {
    var hash_out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &hash_out, .{});
    const hex = std.fmt.bytesToHex(hash_out, .lower);
    const result = try allocator.alloc(u8, hex.len);
    @memcpy(result, &hex);
    return result;
}

pub fn apiHash(allocator: std.mem.Allocator, name: []const u8, params: []const types.Param, returns: ?[]const u8) ![]const u8 {
    var sig_buf: std.ArrayList(u8) = .{};
    defer sig_buf.deinit(allocator);
    const writer = sig_buf.writer(allocator);

    try writer.writeAll(name);
    try writer.writeByte('(');
    for (params, 0..) |param, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeAll(param.name);
        try writer.writeByte(':');
        try writer.writeAll(param.type orelse "anytype");
    }
    try writer.writeAll(")->");
    try writer.writeAll(normalizeType(returns orelse "void"));

    return sha256Hex(allocator, sig_buf.items);
}

pub fn structHash(allocator: std.mem.Allocator, name: []const u8, bases: []const []const u8) ![]const u8 {
    var sig_buf: std.ArrayList(u8) = .{};
    defer sig_buf.deinit(allocator);
    const writer = sig_buf.writer(allocator);

    try writer.writeAll(name);
    try writer.writeByte('(');
    for (bases, 0..) |base, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeAll(base);
    }
    try writer.writeByte(')');

    return sha256Hex(allocator, sig_buf.items);
}

pub fn normalizeType(type_str: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, type_str, " \t\n\r");
    if (trimmed.len == 0) return "void";
    return trimmed;
}

test "sha256Hex produces correct length" {
    const result = try sha256Hex(std.testing.allocator, "test");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqual(@as(usize, 64), result.len);
}

test "apiHash is deterministic" {
    const params1 = [_]types.Param{
        .{ .name = "x", .type = "f64" },
        .{ .name = "y", .type = "f64" },
    };
    const hash1 = try apiHash(std.testing.allocator, "add", &params1, "f64");
    defer std.testing.allocator.free(hash1);

    const hash2 = try apiHash(std.testing.allocator, "add", &params1, "f64");
    defer std.testing.allocator.free(hash2);

    try std.testing.expectEqualSlices(u8, hash1, hash2);
}

test "apiHash differs for different names" {
    const params = [_]types.Param{};
    const hash1 = try apiHash(std.testing.allocator, "foo", &params, null);
    defer std.testing.allocator.free(hash1);

    const hash2 = try apiHash(std.testing.allocator, "bar", &params, null);
    defer std.testing.allocator.free(hash2);

    try std.testing.expect(!std.mem.eql(u8, hash1, hash2));
}

test "structHash differs when field names change" {
    const fields1 = [_][]const u8{ "x", "y" };
    const hash1 = try structHash(std.testing.allocator, "Vec2", &fields1);
    defer std.testing.allocator.free(hash1);

    // Adding a field changes the hash.
    const fields2 = [_][]const u8{ "x", "y", "z" };
    const hash2 = try structHash(std.testing.allocator, "Vec2", &fields2);
    defer std.testing.allocator.free(hash2);

    try std.testing.expect(!std.mem.eql(u8, hash1, hash2));

    // Renaming a field changes the hash.
    const fields3 = [_][]const u8{ "a", "y" };
    const hash3 = try structHash(std.testing.allocator, "Vec2", &fields3);
    defer std.testing.allocator.free(hash3);

    try std.testing.expect(!std.mem.eql(u8, hash1, hash3));
}

test "structHash same fields produce same hash" {
    const fields = [_][]const u8{ "x", "y" };
    const hash1 = try structHash(std.testing.allocator, "Vec2", &fields);
    defer std.testing.allocator.free(hash1);

    const hash2 = try structHash(std.testing.allocator, "Vec2", &fields);
    defer std.testing.allocator.free(hash2);

    try std.testing.expectEqualSlices(u8, hash1, hash2);
}
