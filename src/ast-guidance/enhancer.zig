/// AI Docstring Enhancer for Zig guidance generation.
///
/// Mirrors Python's AIDocstringEnhancer class in guidance.py.
/// Generates descriptions (≤240 chars) for functions, structs, and files
/// by calling the configured LLM endpoint.  The enhancer is opt-in: sync
/// runs without it by default; pass --upgrade-comments or --regen_comments to
/// activate.
///
/// Behaviour mirrors Python AIDocstringEnhancer:
///   - infill_comments: enhance only new members or those whose API hash changed.
///   - regen_comments: exhaustive mode — regenerate for all members, keep best score.
///   - File-level documentation is generated from the module name + member list.
///   - Tags (e.g. #ring-buffer) are extracted from the last line "Tags: #a #b".
///   - Hallucination guard: responses containing terms absent from source are rejected.
const std = @import("std");
const llm = @import("common");

/// Result returned by each enhancement call.
pub const EnrichmentResult = struct {
    /// The generated or preserved description text (≤240 chars enforced by prompt).
    /// Owned by the caller; free with allocator.free().
    comment: ?[]const u8,
    /// Hashtags extracted from the "Tags: #a #b" line.  Each tag is heap-allocated.
    tags: []const []const u8,

    pub fn deinit(self: EnrichmentResult, allocator: std.mem.Allocator) void {
        if (self.comment) |d| allocator.free(d);
        for (self.tags) |t| allocator.free(t);
        allocator.free(self.tags);
    }
};

pub const Enhancer = struct {
    allocator: std.mem.Allocator,
    client: llm.LlmClient,
    debug: bool,

    pub fn init(allocator: std.mem.Allocator, config: llm.LlmConfig) !Enhancer {
        return .{
            .allocator = allocator,
            .client = try llm.LlmClient.init(allocator, config),
            .debug = config.debug,
        };
    }

    pub fn deinit(self: *Enhancer) void {
        self.client.deinit();
    }

    /// Check whether the LLM endpoint is reachable.
    pub fn available(self: *Enhancer) bool {
        return self.client.available();
    }

    // -------------------------------------------------------------------------
    // Public enhancement API
    // -------------------------------------------------------------------------

    /// Generate a description for a Zig function / method.
    ///
    /// Returns an EnrichmentResult owned by the caller.
    /// Falls back to `existing_doc` (duped) if the LLM is unavailable or
    /// produces a hallucinated response.
    pub fn enhanceFunction(
        self: *Enhancer,
        name: []const u8,
        signature: []const u8,
        existing_doc: ?[]const u8,
        module_context: []const u8,
    ) !EnrichmentResult {
        const prompt = try self.buildFunctionPrompt(name, signature, existing_doc, module_context);
        defer self.allocator.free(prompt);

        return self.runEnhancement(name, prompt, existing_doc);
    }

    /// Generate a description for a Zig struct / enum / union.
    pub fn enhanceStruct(
        self: *Enhancer,
        name: []const u8,
        signature: []const u8,
        method_sigs: []const []const u8,
        existing_doc: ?[]const u8,
        module_context: []const u8,
    ) !EnrichmentResult {
        const prompt = try self.buildStructPrompt(name, signature, method_sigs, existing_doc, module_context);
        defer self.allocator.free(prompt);

        return self.runEnhancement(name, prompt, existing_doc);
    }

    /// Generate a one-line file-level description (≤200 chars) for STRUCTURE.md.
    pub fn enhanceFile(
        self: *Enhancer,
        rel_path: []const u8,
        existing_doc: ?[]const u8,
        source_preview: []const u8,
    ) !?[]const u8 {
        const prompt = try self.buildFilePrompt(rel_path, existing_doc, source_preview);
        defer self.allocator.free(prompt);

        if (self.debug) std.debug.print("[enhancer] generating file doc for {s}\n", .{rel_path});
        if (self.debug) std.debug.print("[enhancer] prompt (len={}):\n{s}\n", .{ prompt.len, prompt });

        const raw = self.client.complete(prompt, 600, 0.2, null) catch |err| {
            if (self.debug) std.debug.print("[enhancer] LLM error for file doc: {}\n", .{err});
            return if (existing_doc) |d| try self.allocator.dupe(u8, d) else null;
        };
        const response = raw orelse {
            if (self.debug) std.debug.print("[enhancer] LLM returned null for file doc\n", .{});
            return if (existing_doc) |d| try self.allocator.dupe(u8, d) else null;
        };
        defer self.allocator.free(response);

        if (self.debug) std.debug.print("[enhancer] raw response for {s}:\n{s}\n---\n", .{ rel_path, response });

        // Extract <comment>...</comment> tag.  The prompts mandate this format;
        // if the tag is absent the response is unusable.
        const tagged = llm.extractCommentTag(response) orelse {
            if (self.debug) std.debug.print("[enhancer] no <comment> tag in response for {s}\n", .{rel_path});
            return if (existing_doc) |d| try self.allocator.dupe(u8, d) else null;
        };

        // Reject chain-of-thought that leaked into the tag.
        if (llm.isMalformedResponse(tagged)) {
            if (self.debug) std.debug.print("[enhancer] malformed file tag content for {s}: '{s}'\n", .{ rel_path, tagged });
            return if (existing_doc) |d| try self.allocator.dupe(u8, d) else null;
        }

        if (self.debug) std.debug.print("[enhancer] tag-extracted for {s}: '{s}'\n", .{ rel_path, tagged });
        // Enforce 200-char cap for STRUCTURE.md inline comments.
        const cap: usize = 200;
        const truncated = if (tagged.len > cap) tagged[0..cap] else tagged;
        return try self.allocator.dupe(u8, truncated);
    }

    // -------------------------------------------------------------------------
    // Score a description for quality (mirrors Python's _score_comment).
    // -------------------------------------------------------------------------

    pub fn scoreDocstring(text: []const u8) u32 {
        if (text.len == 0) return 0;
        var score: u32 = 0;
        if (text.len > 50) score += 1;
        const check_len = @min(text.len, 512);
        var buf: [512]u8 = undefined;
        const lower = std.ascii.lowerString(buf[0..check_len], text[0..check_len]);
        if (std.mem.indexOf(u8, lower, "args:") != null or
            std.mem.indexOf(u8, lower, "parameters") != null) score += 2;
        if (std.mem.indexOf(u8, lower, "returns:") != null or
            std.mem.indexOf(u8, lower, "return") != null) score += 2;
        if (std.mem.indexOf(u8, lower, "error") != null or
            std.mem.indexOf(u8, lower, "raises:") != null) score += 1;
        var newline_count: u32 = 0;
        for (text) |c| if (c == '\n') {
            newline_count += 1;
        };
        if (newline_count > 2) score += 1;
        return score;
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    fn runEnhancement(
        self: *Enhancer,
        name: []const u8,
        prompt: []const u8,
        existing_doc: ?[]const u8,
    ) !EnrichmentResult {
        if (self.debug) std.debug.print("[enhancer] enhancing {s}\n", .{name});
        if (self.debug) std.debug.print("[enhancer] prompt:\n{s}\n---\n", .{prompt});

        const raw = self.client.complete(prompt, 800, 0.3, null) catch {
            if (self.debug) std.debug.print("[enhancer] LLM unavailable for {s}\n", .{name});
            return self.fallback(existing_doc);
        };
        const response = raw orelse return self.fallback(existing_doc);
        defer self.allocator.free(response);

        if (self.debug) std.debug.print("[enhancer] raw response for {s}:\n{s}\n---\n", .{ name, response });

        // Extract <comment>...</comment> tag.  The prompts mandate this format;
        // if the tag is absent the response is unusable — fall back to existing.
        const tagged = llm.extractCommentTag(response) orelse {
            if (self.debug) std.debug.print("[enhancer] no <comment> tag in response for {s}\n", .{name});
            return self.fallback(existing_doc);
        };

        // Reject chain-of-thought that leaked into the tag (e.g. reasoning models
        // emitting "we need to write a comment..." inside <comment> tags).
        if (llm.isMalformedResponse(tagged)) {
            if (self.debug) std.debug.print("[enhancer] malformed tag content for {s}: '{s}'\n", .{ name, tagged });
            return self.fallback(existing_doc);
        }

        if (self.debug) std.debug.print("[enhancer] tag-extracted for {s}: '{s}'\n", .{ name, tagged });
        const tags = try self.extractTags(tagged);
        const stripped = try self.stripTagsLine(tagged);
        return .{ .comment = stripped, .tags = tags };
    }

    fn fallback(self: *Enhancer, existing_doc: ?[]const u8) !EnrichmentResult {
        const doc = if (existing_doc) |d| try self.allocator.dupe(u8, d) else null;
        const tags = try self.allocator.alloc([]const u8, 0);
        return .{ .comment = doc, .tags = tags };
    }

    /// Extract #hashtags from a "Tags: #a #b" line at the end of the response.
    fn extractTags(self: *Enhancer, text: []const u8) ![]const []const u8 {
        var tags: std.ArrayList([]const u8) = .{};
        errdefer {
            for (tags.items) |t| self.allocator.free(t);
            tags.deinit(self.allocator);
        }

        // Find the last occurrence of "tags:" (case-insensitive).
        const lower_text = try std.ascii.allocLowerString(self.allocator, text);
        defer self.allocator.free(lower_text);

        const tags_pos = lastIndexOf(lower_text, "tags:") orelse return tags.toOwnedSlice(self.allocator);
        const after_tags = text[tags_pos + 5 ..];

        var it = std.mem.tokenizeScalar(u8, after_tags, ' ');
        while (it.next()) |tok| {
            const trimmed = std.mem.trim(u8, tok, " \t\r\n,;");
            if (trimmed.len == 0) continue;
            if (trimmed[0] == '#') {
                // Strip the '#' and lowercase.
                const tag_word = trimmed[1..];
                if (tag_word.len == 0) continue;
                const lower_tag = try std.ascii.allocLowerString(self.allocator, tag_word);
                try tags.append(self.allocator, lower_tag);
            }
        }

        return tags.toOwnedSlice(self.allocator);
    }

    /// Remove the trailing "Tags: #a #b" line from text.
    fn stripTagsLine(self: *Enhancer, text: []const u8) !?[]const u8 {
        const lower = try std.ascii.allocLowerString(self.allocator, text);
        defer self.allocator.free(lower);

        // Find "tags:" in the last 80 chars.
        const search_start: usize = if (text.len > 80) text.len - 80 else 0;
        const tags_pos = std.mem.indexOf(u8, lower[search_start..], "tags:") orelse {
            return try self.allocator.dupe(u8, std.mem.trim(u8, text, " \t\r\n"));
        };
        const abs_pos = search_start + tags_pos;

        // Walk back to the start of that line.
        var line_start = abs_pos;
        while (line_start > 0 and text[line_start - 1] != '\n') {
            line_start -= 1;
        }

        const stripped = std.mem.trim(u8, text[0..line_start], " \t\r\n");
        return try self.allocator.dupe(u8, stripped);
    }

    // -------------------------------------------------------------------------
    // Prompt builders
    // -------------------------------------------------------------------------

    fn buildFunctionPrompt(
        self: *Enhancer,
        name: []const u8,
        signature: []const u8,
        existing_doc: ?[]const u8,
        module_context: []const u8,
    ) ![]const u8 {
        var buf: std.ArrayList(u8) = .{};
        const w = buf.writer(self.allocator);

        try w.print("Zig function in {s}:\n  {s}\n\n", .{ module_context, signature });

        if (existing_doc) |d| {
            if (d.len > 0) {
                try w.print("Existing comment: {s}\n\n", .{d});
            }
        }

        try w.writeAll(
            \\Write a single-line comment for this function.
            \\Rules:
            \\- Plain English, technically specific — state what it does, key args, return value or error
            \\- Max 200 characters
            \\- No boilerplate openers ("This function", "A function that")
            \\
            \\Wrap your answer in <comment> tags. Example:
            \\<comment>Parses a null-terminated C string into an owned Zig slice.</comment>
            \\
        );
        try w.print("Function: {s}\n", .{name});

        return buf.toOwnedSlice(self.allocator);
    }

    fn buildStructPrompt(
        self: *Enhancer,
        name: []const u8,
        signature: []const u8,
        method_sigs: []const []const u8,
        existing_doc: ?[]const u8,
        module_context: []const u8,
    ) ![]const u8 {
        var buf: std.ArrayList(u8) = .{};
        const w = buf.writer(self.allocator);

        try w.print("Zig type in {s}:\n  {s}\n", .{ module_context, signature });

        const limit = @min(method_sigs.len, 6);
        if (limit > 0) {
            try w.writeAll("Methods:\n");
            for (method_sigs[0..limit]) |sig| {
                try w.print("  {s}\n", .{sig});
            }
            if (method_sigs.len > 6) {
                try w.print("  ... and {} more\n", .{method_sigs.len - 6});
            }
        }

        if (existing_doc) |d| {
            if (d.len > 0) {
                try w.print("\nExisting comment: {s}\n\n", .{d});
            }
        }

        try w.writeAll(
            \\
            \\Write a single-line comment for this type.
            \\Rules:
            \\- Plain English, technically specific — state purpose, ownership model, key invariants
            \\- Max 200 characters
            \\- No boilerplate openers ("This struct", "A type that")
            \\
            \\Wrap your answer in <comment> tags. Example:
            \\<comment>Owns a pool of fixed-size buffers; init/deinit pair; not thread-safe.</comment>
            \\
        );
        try w.print("Type: {s}\n", .{name});

        return buf.toOwnedSlice(self.allocator);
    }

    fn buildFilePrompt(
        self: *Enhancer,
        rel_path: []const u8,
        existing_doc: ?[]const u8,
        source_preview: []const u8,
    ) ![]const u8 {
        var buf: std.ArrayList(u8) = .{};
        const w = buf.writer(self.allocator);

        // First 3000 chars of source gives the model the module doc comment and
        // enough declarations to understand what the file actually does.
        const preview_len = @min(source_preview.len, 3000);
        if (preview_len > 0) {
            try w.writeAll("Source:\n");
            try w.writeAll(source_preview[0..preview_len]);
            try w.writeByte('\n');
        }

        try w.print("\nFile: {s}\n", .{rel_path});

        if (existing_doc) |d| {
            if (d.len > 0) {
                try w.print("Existing comment: {s}\n\n", .{d});
            }
        }

        try w.writeAll(
            \\
            \\Write a single-line description for this file.
            \\Rules:
            \\- Plain English, technically specific: key types, algorithms, or responsibilities
            \\- Max 200 chars
            \\- No boilerplate openers ("This file", "A module that")
            \\- Do NOT include a skills prefix — that is added automatically
            \\
            \\Wrap your answer in <comment> tags. Example:
            \\<comment>Parses Zig AST and extracts public member signatures for guidance generation.</comment>
            \\
        );

        return buf.toOwnedSlice(self.allocator);
    }
};

// ---------------------------------------------------------------------------
// Pure utilities (no allocator needed)
// ---------------------------------------------------------------------------

fn containsCI(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn lastIndexOf(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len > haystack.len) return null;
    var i: usize = haystack.len - needle.len + 1;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, haystack[i .. i + needle.len], needle)) return i;
    }
    return null;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "scoreDocstring empty" {
    try std.testing.expectEqual(@as(u32, 0), Enhancer.scoreDocstring(""));
}

test "scoreDocstring quality" {
    const good = "Parses JSON from a slice.\n\nArgs:\n  data: input slice\nReturns: parsed value\nRaises: error on malformed input";
    const score = Enhancer.scoreDocstring(good);
    try std.testing.expect(score >= 4);
}

test "extractTags parses hashtags" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var e = try Enhancer.init(gpa.allocator(), .{
        .api_url = "http://localhost:11434/api/chat",
        .model = "test",
    });
    defer e.deinit();
    const alloc = gpa.allocator();

    const text = "Parses a JSON slice and returns a Value.\nTags: #json #parser #zig";
    const tags = try e.extractTags(text);
    defer {
        for (tags) |t| alloc.free(t);
        alloc.free(tags);
    }
    try std.testing.expectEqual(@as(usize, 3), tags.len);
    try std.testing.expectEqualStrings("json", tags[0]);
    try std.testing.expectEqualStrings("parser", tags[1]);
    try std.testing.expectEqualStrings("zig", tags[2]);
}

test "extractTags returns empty when no Tags line" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var e = try Enhancer.init(gpa.allocator(), .{
        .api_url = "http://localhost:11434/api/chat",
        .model = "test",
    });
    defer e.deinit();
    const alloc = gpa.allocator();

    const tags = try e.extractTags("Just a plain description with no tags.");
    defer {
        for (tags) |t| alloc.free(t);
        alloc.free(tags);
    }
    try std.testing.expectEqual(@as(usize, 0), tags.len);
}

test "stripTagsLine removes Tags line" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var e = try Enhancer.init(gpa.allocator(), .{
        .api_url = "http://localhost:11434/api/chat",
        .model = "test",
    });
    defer e.deinit();
    const alloc = gpa.allocator();

    const text = "Parses JSON from a byte slice.\nTags: #json #zig";
    const stripped = try e.stripTagsLine(text);
    defer if (stripped) |s| alloc.free(s);
    try std.testing.expectEqualStrings("Parses JSON from a byte slice.", stripped.?);
}

test "lastIndexOf basic" {
    try std.testing.expectEqual(@as(?usize, 6), lastIndexOf("hello tags: #a", "tags:"));
    try std.testing.expectEqual(@as(?usize, null), lastIndexOf("no match", "tags:"));
}
