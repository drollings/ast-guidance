const std = @import("std");

pub const GitignoreFilter = struct {
    allocator: std.mem.Allocator,
    patterns: [][]const u8,
    negations: [][]const u8,
    project_root: []const u8,
    always_exclude: []const []const u8,

    pub fn init(allocator: std.mem.Allocator, project_root: []const u8) GitignoreFilter {
        return .{
            .allocator = allocator,
            .patterns = &.{},
            .negations = &.{},
            .project_root = project_root,
            .always_exclude = &.{ ".git", ".zig-cache", "zig-out" },
        };
    }

    pub fn deinit(self: *GitignoreFilter) void {
        for (self.patterns) |p| self.allocator.free(p);
        for (self.negations) |p| self.allocator.free(p);
        self.allocator.free(self.patterns);
        self.allocator.free(self.negations);
    }

    pub fn loadFromFile(self: *GitignoreFilter, path: []const u8) !void {
        const file = std.fs.openFileAbsolute(path, .{}) catch return;
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 512 * 1024) catch return;
        defer self.allocator.free(content);

        var patterns_list: std.ArrayList([]const u8) = .{};
        var negations_list: std.ArrayList([]const u8) = .{};
        errdefer {
            for (patterns_list.items) |p| self.allocator.free(p);
            patterns_list.deinit(self.allocator);
            for (negations_list.items) |p| self.allocator.free(p);
            negations_list.deinit(self.allocator);
        }

        var line_it = std.mem.splitScalar(u8, content, '\n');
        while (line_it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0 or trimmed[0] == '#') continue;

            if (trimmed[0] == '!') {
                try negations_list.append(self.allocator, try self.allocator.dupe(u8, trimmed[1..]));
            } else {
                try patterns_list.append(self.allocator, try self.allocator.dupe(u8, trimmed));
            }
        }

        // Free any previously loaded patterns before replacing.
        for (self.patterns) |p| self.allocator.free(p);
        self.allocator.free(self.patterns);
        for (self.negations) |p| self.allocator.free(p);
        self.allocator.free(self.negations);

        self.patterns = try patterns_list.toOwnedSlice(self.allocator);
        self.negations = try negations_list.toOwnedSlice(self.allocator);
    }

    pub fn shouldIgnore(self: *const GitignoreFilter, filepath: []const u8) bool {
        var rel_path: []const u8 = filepath;

        if (std.mem.startsWith(u8, filepath, self.project_root)) {
            rel_path = filepath[self.project_root.len..];
            if (rel_path.len > 0 and rel_path[0] == '/') {
                rel_path = rel_path[1..];
            }
        }

        for (self.always_exclude) |exclude| {
            if (std.mem.indexOf(u8, rel_path, exclude) != null) {
                return true;
            }
        }

        var is_ignored = false;
        for (self.patterns) |pattern| {
            if (matchesPattern(rel_path, pattern)) {
                is_ignored = true;
                break;
            }
        }

        if (is_ignored) {
            for (self.negations) |pattern| {
                if (matchesPattern(rel_path, pattern)) {
                    is_ignored = false;
                    break;
                }
            }
        }

        return is_ignored;
    }
};

fn matchesPattern(path: []const u8, pattern: []const u8) bool {
    if (pattern.len == 0) return false;

    const is_dir_pattern = pattern[pattern.len - 1] == '/';
    const pat = if (is_dir_pattern) pattern[0 .. pattern.len - 1] else pattern;

    if (std.mem.indexOf(u8, pat, "**")) |doublestar| {
        const prefix = pat[0..doublestar];
        const suffix = pat[doublestar + 2 ..];

        if (prefix.len > 0 and !std.mem.startsWith(u8, path, prefix)) return false;
        if (suffix.len > 0 and !std.mem.endsWith(u8, path, suffix)) return false;
        return true;
    }

    if (std.mem.startsWith(u8, pat, "/") or std.mem.indexOf(u8, pat, "/") != null) {
        const clean_pat = if (pat[0] == '/') pat[1..] else pat;
        return globMatch(path, clean_pat);
    }

    if (std.mem.indexOf(u8, path, "/")) |slash| {
        const filename = path[slash + 1 ..];
        if (globMatch(filename, pat)) return true;
    }

    return globMatch(path, pat);
}

fn globMatch(text: []const u8, pattern: []const u8) bool {
    var ti: usize = 0;
    var pi: usize = 0;
    var star_idx: ?usize = null;
    var match_idx: usize = 0;

    while (ti < text.len) {
        if (pi < pattern.len and (pattern[pi] == text[ti] or pattern[pi] == '?')) {
            ti += 1;
            pi += 1;
        } else if (pi < pattern.len and pattern[pi] == '*') {
            star_idx = pi;
            match_idx = ti;
            pi += 1;
        } else if (star_idx) |si| {
            pi = si + 1;
            match_idx += 1;
            ti = match_idx;
        } else {
            return false;
        }
    }

    while (pi < pattern.len and pattern[pi] == '*') {
        pi += 1;
    }

    return pi == pattern.len;
}

test "globMatch basic patterns" {
    try std.testing.expect(globMatch("foo.zig", "*.zig"));
    try std.testing.expect(globMatch("src/foo.zig", "*.zig"));
    try std.testing.expect(!globMatch("foo.zig", "*.py"));
    try std.testing.expect(globMatch("test_foo.zig", "test_*.zig"));
    try std.testing.expect(globMatch("foo", "foo"));
}

test "GitignoreFilter always excludes .git" {
    var filter = GitignoreFilter.init(std.testing.allocator, "/project");
    defer filter.deinit();

    try std.testing.expect(filter.shouldIgnore("/project/.git/config"));
    try std.testing.expect(filter.shouldIgnore("/project/.zig-cache/foo"));
}
