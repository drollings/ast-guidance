const std = @import("std");

pub const MemberType = enum {
    fn_decl,
    fn_private,
    @"struct",
    @"enum",
    @"union",
    enum_field,
    test_decl,
    comptime_block,
    method,
    method_private,
};

pub const PatternType = enum {
    Domain,
    GoF,
};

pub const Pattern = struct {
    name: []const u8,
    type: PatternType,
    ref: ?[]const u8 = null,
};

pub const Param = struct {
    name: []const u8,
    type: ?[]const u8 = null,
    default: ?[]const u8 = null,
};

pub const Member = struct {
    type: MemberType,
    name: []const u8,
    match_hash: ?[]const u8 = null,
    signature: ?[]const u8 = null,
    params: []const Param = &.{},
    returns: ?[]const u8 = null,
    comment: ?[]const u8 = null,
    tags: []const []const u8 = &.{},
    patterns: []const Pattern = &.{},
    is_pub: bool = false,
    members: []const Member = &.{},
    line: ?u32 = null,
};

pub const Skill = struct {
    ref: []const u8,
    context: ?[]const u8 = null,
};

pub const Meta = struct {
    module: []const u8,
    source: []const u8,
    language: []const u8 = "zig",
};

pub const GuidanceDoc = struct {
    meta: Meta,
    comment: ?[]const u8 = null,
    skills: []const Skill = &.{},
    hashtags: []const []const u8 = &.{},
    used_by: []const []const u8 = &.{},
    members: []const Member = &.{},
};

pub const FileMatch = struct {
    filename: []const u8,
    filepath: []const u8,
    description: []const u8 = "",
    line_context: []const u8 = "",
};

pub const GuidanceInfo = struct {
    path: []const u8,
    comment: []const u8 = "",
    functions: []const Member = &.{},
    classes: []const Member = &.{},
    skills: []const []const u8 = &.{},
    tags: []const []const u8 = &.{},
};

pub const ASTAnalysis = struct {
    filepath: []const u8,
    functions: []const Member = &.{},
    classes: []const Member = &.{},
    imports: []const []const u8 = &.{},
    patterns_detected: []const []const u8 = &.{},
    token_count: usize = 0,
    signature_preview: []const u8 = "",
};

pub const QueryResult = struct {
    query: []const u8,
    file_matches: []const FileMatch = &.{},
    guidance_files: []const GuidanceInfo = &.{},
    ast_analysis: []const ASTAnalysis = &.{},
    intelligent_summary: []const u8 = "",
    related_skills: []const []const u8 = &.{},
    suggested_actions: []const []const u8 = &.{},
    insights: []const []const u8 = &.{},
    recent_capabilities: []const []const u8 = &.{},
};

pub const SyncResult = struct {
    filepath: []const u8,
    members_added: usize = 0,
    members_updated: usize = 0,
    members_removed: usize = 0,
    has_changes: bool = false,
};

pub fn jsonStringifyAlloc(allocator: std.mem.Allocator, value: anytype) ![]u8 {
    var out: std.io.Writer.Allocating = .init(allocator);
    const writer = &out.writer;
    defer out.deinit();

    try std.json.Stringify.value(value, .{ .whitespace = .indent_2 }, writer);
    return try allocator.dupe(u8, out.written());
}

pub fn jsonifyMember(allocator: std.mem.Allocator, member: Member) !?[]u8 {
    var list: std.ArrayList(u8) = .{};
    errdefer list.deinit(allocator);
    const writer = list.writer(allocator);

    try writer.writeAll("{\n");
    try writer.print("  \"type\": \"{s}\",\n", .{@tagName(member.type)});
    try writeEscapedString(writer, "name", member.name);

    if (member.match_hash) |h| {
        try writer.writeAll("  \"match_hash\": \"");
        try writeEscapedValue(writer, h);
        try writer.writeAll("\",\n");
    }
    if (member.signature) |s| {
        try writeEscapedString(writer, "signature", s);
    }
    if (member.params.len > 0) {
        try writer.writeAll("  \"params\": [\n");
        for (member.params, 0..) |param, i| {
            try writer.writeAll("    { ");
            try writer.writeAll("\"name\": \"");
            try writeEscapedValue(writer, param.name);
            try writer.writeAll("\"");
            if (param.type) |t| {
                try writer.writeAll(", \"type\": \"");
                try writeEscapedValue(writer, t);
                try writer.writeAll("\"");
            }
            if (param.default) |d| {
                try writer.writeAll(", \"default\": \"");
                try writeEscapedValue(writer, d);
                try writer.writeAll("\"");
            }
            try writer.writeAll(" }");
            if (i < member.params.len - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }
        try writer.writeAll("  ],\n");
    }
    if (member.returns) |r| {
        try writeEscapedString(writer, "returns", r);
    }
    try writeFieldOrComma(writer, "comment", member.comment);
    if (member.tags.len > 0) {
        try writer.writeAll("  \"tags\": [");
        for (member.tags, 0..) |tag, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("\"{s}\"", .{tag});
        }
        try writer.writeAll("],\n");
    }
    if (member.patterns.len > 0) {
        try writer.writeAll("  \"patterns\": [\n");
        for (member.patterns, 0..) |pat, i| {
            try writer.writeAll("    { \"name\": \"");
            try writeEscapedValue(writer, pat.name);
            try writer.writeAll("\", \"type\": \"");
            try writeEscapedValue(writer, @tagName(pat.type));
            try writer.writeAll("\"");
            if (pat.ref) |ref| {
                try writer.writeAll(", \"ref\": \"");
                try writeEscapedValue(writer, ref);
                try writer.writeAll("\"");
            }
            try writer.writeAll(" }");
            if (i < member.patterns.len - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }
        try writer.writeAll("  ],\n");
    }
    try writer.print("  \"is_pub\": {},\n", .{member.is_pub});

    if (member.members.len > 0) {
        try writer.writeAll("  \"members\": [\n");
        for (member.members, 0..) |nested, i| {
            if (i > 0) try writer.writeAll(",");
            const nested_json = try jsonifyMember(allocator, nested);
            if (nested_json) |nj| {
                defer allocator.free(nj);
                try writer.writeAll("    ");
                for (nj) |c| {
                    if (c == '\n') {
                        try writer.writeAll("\n    ");
                    } else {
                        try writer.writeByte(c);
                    }
                }
                try writer.writeAll("\n");
            }
        }
        try writer.writeAll("  ]");
        if (member.line != null) {
            try writer.writeAll(",");
        }
        try writer.writeAll("\n");
    }
    if (member.line) |l| {
        try writer.print("  \"line\": {}\n", .{l});
    }
    try writer.writeAll("}");
    return @as(?[]u8, try list.toOwnedSlice(allocator));
}

fn writeFieldOrComma(writer: anytype, key: []const u8, value: ?[]const u8) !void {
    if (value) |v| {
        try writeEscapedString(writer, key, v);
    }
    // Omit field entirely when null — matches jsonifyGuidanceDoc behaviour for
    // the file-level comment and keeps JSON clean for infill-eligible members.
}

fn writeEscapedValue(writer: anytype, value: []const u8) !void {
    for (value) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(c),
        }
    }
}

fn writeEscapedString(writer: anytype, key: []const u8, value: []const u8) !void {
    try writer.print("  \"{s}\": \"", .{key});
    try writeEscapedValue(writer, value);
    try writer.writeAll("\",\n");
}

pub fn jsonifyGuidanceDoc(allocator: std.mem.Allocator, doc: GuidanceDoc) ![]u8 {
    var list: std.ArrayList(u8) = .{};
    errdefer list.deinit(allocator);
    const writer = list.writer(allocator);

    try writer.writeAll("{\n");
    try writer.writeAll("  \"meta\": {\n");
    try writer.writeAll("    \"module\": \"");
    try writeEscapedValue(writer, doc.meta.module);
    try writer.writeAll("\",\n");
    try writer.writeAll("    \"source\": \"");
    try writeEscapedValue(writer, doc.meta.source);
    try writer.writeAll("\",\n");
    try writer.writeAll("    \"language\": \"");
    try writeEscapedValue(writer, doc.meta.language);
    try writer.writeAll("\"\n");
    try writer.writeAll("  },\n");

    if (doc.comment) |d| {
        try writer.writeAll("  \"comment\": \"");
        try writeEscapedValue(writer, d);
        try writer.writeAll("\",\n");
    }

    if (doc.skills.len > 0) {
        try writer.writeAll("  \"skills\": [\n");
        for (doc.skills, 0..) |skill, i| {
            try writer.writeAll("    { \"ref\": \"");
            try writeEscapedValue(writer, skill.ref);
            try writer.writeAll("\"");
            if (skill.context) |ctx| {
                try writer.writeAll(", \"context\": \"");
                try writeEscapedValue(writer, ctx);
                try writer.writeAll("\"");
            }
            try writer.writeAll(" }");
            if (i < doc.skills.len - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }
        try writer.writeAll("  ],\n");
    }

    if (doc.hashtags.len > 0) {
        try writer.writeAll("  \"hashtags\": [");
        for (doc.hashtags, 0..) |tag, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.writeAll("\"");
            try writeEscapedValue(writer, tag);
            try writer.writeAll("\"");
        }
        try writer.writeAll("],\n");
    }

    if (doc.used_by.len > 0) {
        try writer.writeAll("  \"used_by\": [");
        for (doc.used_by, 0..) |u, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.writeAll("\"");
            try writeEscapedValue(writer, u);
            try writer.writeAll("\"");
        }
        try writer.writeAll("],\n");
    }

    if (doc.members.len > 0) {
        try writer.writeAll("  \"members\": [\n");
        for (doc.members, 0..) |member, i| {
            const member_json = try jsonifyMember(allocator, member);
            if (member_json) |mj| {
                defer allocator.free(mj);
                try writer.writeAll("    ");
                for (mj) |c| {
                    if (c == '\n') {
                        try writer.writeAll("\n    ");
                    } else {
                        try writer.writeByte(c);
                    }
                }
            }
            if (i < doc.members.len - 1) {
                try writer.writeAll(",\n");
            } else {
                try writer.writeAll("\n");
            }
        }
        try writer.writeAll("  ]\n");
    }

    try writer.writeAll("}\n");
    return list.toOwnedSlice(allocator);
}
