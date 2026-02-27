const std = @import("std");
const types = @import("types.zig");
const ast_parser = @import("ast_parser.zig");
const json_store = @import("json_store.zig");
const gitignore = @import("gitignore.zig");

// ---------------------------------------------------------------------------
// Free helpers for QueryResult and its nested types.
// All slices returned by execute() must be freed via freeQueryResult.
// ---------------------------------------------------------------------------

pub fn freeFileMatch(allocator: std.mem.Allocator, m: types.FileMatch) void {
    allocator.free(m.filename);
    allocator.free(m.filepath);
    if (m.description.len > 0) allocator.free(m.description);
    if (m.line_context.len > 0) allocator.free(m.line_context);
}

pub fn freeGuidanceInfo(allocator: std.mem.Allocator, g: types.GuidanceInfo) void {
    allocator.free(g.path);
    if (g.comment.len > 0) allocator.free(g.comment);
    // functions/classes are deep-copied in loadGuidanceDoc; free each Member.
    var store = json_store.JsonStore.init(allocator);
    for (g.functions) |m| store.freeMember(m);
    allocator.free(g.functions);
    for (g.classes) |m| store.freeMember(m);
    allocator.free(g.classes);
    for (g.skills) |s| allocator.free(s);
    allocator.free(g.skills);
    for (g.tags) |t| allocator.free(t);
    allocator.free(g.tags);
}

pub fn freeASTAnalysis(allocator: std.mem.Allocator, store: *json_store.JsonStore, a: types.ASTAnalysis) void {
    allocator.free(a.filepath);
    for (a.functions) |m| store.freeMember(m);
    allocator.free(a.functions);
    for (a.classes) |m| store.freeMember(m);
    allocator.free(a.classes);
    for (a.imports) |imp| allocator.free(imp);
    allocator.free(a.imports);
    for (a.patterns_detected) |p| allocator.free(p);
    allocator.free(a.patterns_detected);
    if (a.signature_preview.len > 0) allocator.free(a.signature_preview);
}

pub fn freeQueryResult(allocator: std.mem.Allocator, store: *json_store.JsonStore, r: types.QueryResult) void {
    for (r.file_matches) |m| freeFileMatch(allocator, m);
    allocator.free(r.file_matches);
    for (r.guidance_files) |g| freeGuidanceInfo(allocator, g);
    allocator.free(r.guidance_files);
    for (r.ast_analysis) |a| freeASTAnalysis(allocator, store, a);
    allocator.free(r.ast_analysis);
    for (r.related_skills) |s| allocator.free(s);
    allocator.free(r.related_skills);
    for (r.suggested_actions) |s| allocator.free(s);
    allocator.free(r.suggested_actions);
    for (r.insights) |s| allocator.free(s);
    allocator.free(r.insights);
    for (r.recent_capabilities) |s| allocator.free(s);
    allocator.free(r.recent_capabilities);
    if (r.intelligent_summary.len > 0) allocator.free(r.intelligent_summary);
}

// ---------------------------------------------------------------------------
// QueryEngine
// ---------------------------------------------------------------------------

pub const QueryEngine = struct {
    allocator: std.mem.Allocator,
    query: []const u8,
    project_root: []const u8,
    use_ast: bool,
    debug: bool,
    file_matches: std.ArrayList(types.FileMatch),
    guidance_files: std.ArrayList(types.GuidanceInfo),
    ast_analyses: std.ArrayList(types.ASTAnalysis),
    related_skills: std.ArrayList([]const u8),
    gitignore_filter: gitignore.GitignoreFilter,
    store: json_store.JsonStore,

    pub fn init(allocator: std.mem.Allocator, query_str: []const u8, project_root: []const u8, use_ast: bool, debug: bool) QueryEngine {
        return .{
            .allocator = allocator,
            .query = query_str,
            .project_root = project_root,
            .use_ast = use_ast,
            .debug = debug,
            .file_matches = .{},
            .guidance_files = .{},
            .ast_analyses = .{},
            .related_skills = .{},
            .gitignore_filter = gitignore.GitignoreFilter.init(allocator, project_root),
            .store = json_store.JsonStore.init(allocator),
        };
    }

    pub fn deinit(self: *QueryEngine) void {
        self.gitignore_filter.deinit();

        for (self.file_matches.items) |m| freeFileMatch(self.allocator, m);
        self.file_matches.deinit(self.allocator);

        for (self.guidance_files.items) |g| freeGuidanceInfo(self.allocator, g);
        self.guidance_files.deinit(self.allocator);

        for (self.ast_analyses.items) |a| freeASTAnalysis(self.allocator, &self.store, a);
        self.ast_analyses.deinit(self.allocator);

        for (self.related_skills.items) |s| self.allocator.free(s);
        self.related_skills.deinit(self.allocator);
    }

    pub fn execute(self: *QueryEngine) !types.QueryResult {
        try self.grepStructure();
        try self.deepFileSearch();
        try self.parseGuidanceFiles();

        if (self.use_ast) {
            try self.analyzeAst();
        }

        try self.findRelatedSkills();

        const insights = try self.readInboxBullets(".ast-guidance/.doc/inbox/INSIGHTS.md");
        const capabilities = try self.readInboxBullets(".ast-guidance/.doc/inbox/CAPABILITIES.md");

        return .{
            .query = self.query,
            .file_matches = try self.file_matches.toOwnedSlice(self.allocator),
            .guidance_files = try self.guidance_files.toOwnedSlice(self.allocator),
            .ast_analysis = try self.ast_analyses.toOwnedSlice(self.allocator),
            .related_skills = try self.related_skills.toOwnedSlice(self.allocator),
            .insights = insights,
            .recent_capabilities = capabilities,
        };
    }

    /// Read markdown bullet lines from an inbox file that score > 0 against the query.
    /// Each returned string is freshly allocated and owned by the caller.
    fn readInboxBullets(self: *QueryEngine, rel_file: []const u8) ![]const []const u8 {
        const path = try std.fs.path.join(self.allocator, &.{ self.project_root, rel_file });
        defer self.allocator.free(path);

        const file = std.fs.openFileAbsolute(path, .{}) catch return &.{};
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 1024 * 1024) catch return &.{};
        defer self.allocator.free(content);

        const query_lower = try std.ascii.allocLowerString(self.allocator, self.query);
        defer self.allocator.free(query_lower);

        var matched: std.ArrayList([]const u8) = .{};
        errdefer {
            for (matched.items) |s| self.allocator.free(s);
            matched.deinit(self.allocator);
        }

        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            const stripped = std.mem.trim(u8, line, " \t\r");
            if (stripped.len == 0) continue;
            if (stripped[0] == '#') continue;
            if (!std.mem.startsWith(u8, stripped, "- ")) continue;

            const bullet = stripped[2..];
            // Score: count query terms found in the bullet (simple word split).
            const bullet_lower = try std.ascii.allocLowerString(self.allocator, bullet);
            defer self.allocator.free(bullet_lower);

            var score: u32 = 0;
            var term_iter = std.mem.splitScalar(u8, query_lower, ' ');
            while (term_iter.next()) |term| {
                if (term.len == 0) continue;
                if (std.mem.indexOf(u8, bullet_lower, term) != null) score += 1;
            }

            if (score > 0) {
                try matched.append(self.allocator, try self.allocator.dupe(u8, bullet));
            }
        }

        return matched.toOwnedSlice(self.allocator);
    }

    /// Case-insensitive substring check on a stack-allocated lowercase copy.
    fn containsIgnoreCase(allocator: std.mem.Allocator, haystack: []const u8, needle: []const u8) bool {
        const h = std.ascii.allocLowerString(allocator, haystack) catch return false;
        defer allocator.free(h);
        const n = std.ascii.allocLowerString(allocator, needle) catch return false;
        defer allocator.free(n);
        return std.mem.indexOf(u8, h, n) != null;
    }

    fn grepStructure(self: *QueryEngine) !void {
        const struct_path = try std.fs.path.join(self.allocator, &.{ self.project_root, "STRUCTURE.md" });
        defer self.allocator.free(struct_path);

        const file = std.fs.openFileAbsolute(struct_path, .{}) catch return;
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 1024 * 1024) catch return;
        defer self.allocator.free(content);

        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            if (!containsIgnoreCase(self.allocator, line, self.query)) continue;
            if (try self.parseStructureLine(line)) |match| {
                try self.file_matches.append(self.allocator, match);
            }
        }
    }

    fn parseStructureLine(self: *QueryEngine, line: []const u8) !?types.FileMatch {
        var i: usize = 0;
        while (i < line.len and (line[i] == ' ' or line[i] == '|' or line[i] == '+' or line[i] == '`' or line[i] == '-' or line[i] == 0xe2 or line[i] == 0x94 or line[i] == 0x80 or line[i] == 0x9c or line[i] == 0x94 or line[i] == 0x82 or line[i] == 0x8c or line[i] == 0x94)) {
            i += 1;
        }
        const rest = line[i..];
        if (rest.len == 0) return null;

        var name_end = rest.len;
        var comment_start: ?usize = null;
        if (std.mem.indexOf(u8, rest, "  #")) |idx| {
            name_end = idx;
            comment_start = idx + 3;
        }

        const filename = std.mem.trim(u8, rest[0..name_end], " ");
        if (filename.len == 0) return null;
        // Skip pure-separator lines
        if (std.mem.eql(u8, filename, "|") or std.mem.eql(u8, filename, "---")) return null;

        const description: []const u8 = if (comment_start) |cs|
            std.mem.trim(u8, rest[cs..], " ")
        else
            "";

        return .{
            .filename = try self.allocator.dupe(u8, filename),
            .filepath = try self.resolveFilepath(filename),
            .description = try self.allocator.dupe(u8, description),
            .line_context = try self.allocator.dupe(u8, line),
        };
    }

    fn resolveFilepath(self: *QueryEngine, filename: []const u8) ![]const u8 {
        const search_dirs = [_][]const u8{
            "src",
            "src/guidance",
            "src/sandbox",
            "",
        };

        for (search_dirs) |subdir| {
            const full_path = if (subdir.len > 0)
                try std.fs.path.join(self.allocator, &.{ self.project_root, subdir, filename })
            else
                try std.fs.path.join(self.allocator, &.{ self.project_root, filename });

            std.fs.accessAbsolute(full_path, .{}) catch {
                self.allocator.free(full_path);
                continue;
            };
            return full_path;
        }

        return try std.fs.path.join(self.allocator, &.{ self.project_root, filename });
    }

    /// Search src/ for files whose name contains the query.
    fn deepFileSearch(self: *QueryEngine) !void {
        const search_dirs = [_][]const u8{"src"};
        for (search_dirs) |subdir| {
            const dir_path = try std.fs.path.join(self.allocator, &.{ self.project_root, subdir });
            defer self.allocator.free(dir_path);

            var dir = std.fs.openDirAbsolute(dir_path, .{ .iterate = true }) catch continue;
            defer dir.close();

            var walker = dir.walk(self.allocator) catch continue;
            defer walker.deinit();

            while (try walker.next()) |entry| {
                if (entry.kind != .file) continue;
                if (!containsIgnoreCase(self.allocator, entry.basename, self.query)) continue;

                const full_path = try std.fs.path.join(self.allocator, &.{ dir_path, entry.path });

                // Dedup: skip if already in file_matches
                var exists = false;
                for (self.file_matches.items) |m| {
                    if (std.mem.eql(u8, m.filepath, full_path)) {
                        exists = true;
                        break;
                    }
                }
                if (exists) {
                    self.allocator.free(full_path);
                    continue;
                }

                try self.file_matches.append(self.allocator, .{
                    .filename = try self.allocator.dupe(u8, entry.basename),
                    .filepath = full_path,
                    .description = try self.allocator.dupe(u8, ""),
                    .line_context = try self.allocator.dupe(u8, ""),
                });
            }
        }
    }

    /// Load guidance JSON for each matched source file, and also directly for
    /// any .json files found in deepFileSearch.
    fn parseGuidanceFiles(self: *QueryEngine) !void {
        for (self.file_matches.items) |match| {
            // Direct hit on a .json guidance file
            if (std.mem.endsWith(u8, match.filepath, ".json")) {
                try self.loadGuidanceDoc(match.filepath);
                continue;
            }
            // Source .zig file — derive guidance path
            if (std.mem.endsWith(u8, match.filepath, ".zig")) {
                // Try guidance/<rel>.json
                const rel = self.relPath(match.filepath);
                const gpath = try std.fmt.allocPrint(self.allocator, "{s}/guidance/{s}.json", .{ self.project_root, rel });
                defer self.allocator.free(gpath);
                try self.loadGuidanceDoc(gpath);
            }
        }
    }

    fn relPath(self: *QueryEngine, filepath: []const u8) []const u8 {
        if (std.mem.indexOf(u8, filepath, self.project_root)) |idx| {
            const after = filepath[idx + self.project_root.len ..];
            if (after.len > 0 and after[0] == '/') return after[1..];
            return after;
        }
        return filepath;
    }

    fn loadGuidanceDoc(self: *QueryEngine, gpath: []const u8) !void {
        const doc = (try self.store.loadGuidance(gpath)) orelse return;
        // Free doc immediately after we deep-copy what we need; doc is not kept alive.
        defer self.store.freeGuidanceDoc(doc);

        var functions: std.ArrayList(types.Member) = .{};
        var classes: std.ArrayList(types.Member) = .{};
        errdefer {
            for (functions.items) |m| self.store.freeMember(m);
            functions.deinit(self.allocator);
            for (classes.items) |m| self.store.freeMember(m);
            classes.deinit(self.allocator);
        }

        for (doc.members) |member| {
            switch (member.type) {
                .fn_decl, .fn_private, .method, .method_private => {
                    // Deep-copy so GuidanceInfo owns its members independently.
                    try functions.append(self.allocator, try self.store.dupeMember(member));
                },
                .@"struct", .@"enum", .@"union" => {
                    try classes.append(self.allocator, try self.store.dupeMember(member));
                },
                else => {},
            }
        }

        // Collect skills as string refs
        var skills: std.ArrayList([]const u8) = .{};
        errdefer {
            for (skills.items) |s| self.allocator.free(s);
            skills.deinit(self.allocator);
        }
        for (doc.skills) |s| {
            try skills.append(self.allocator, try self.allocator.dupe(u8, s.ref));
        }

        // Collect hashtags
        var tags: std.ArrayList([]const u8) = .{};
        errdefer {
            for (tags.items) |t| self.allocator.free(t);
            tags.deinit(self.allocator);
        }
        for (doc.hashtags) |t| {
            try tags.append(self.allocator, try self.allocator.dupe(u8, t));
        }

        try self.guidance_files.append(self.allocator, .{
            .path = try self.allocator.dupe(u8, gpath),
            .comment = if (doc.comment) |d| try self.allocator.dupe(u8, d) else "",
            .functions = try functions.toOwnedSlice(self.allocator),
            .classes = try classes.toOwnedSlice(self.allocator),
            .skills = try skills.toOwnedSlice(self.allocator),
            .tags = try tags.toOwnedSlice(self.allocator),
        });
    }

    fn analyzeAst(self: *QueryEngine) !void {
        for (self.file_matches.items) |match| {
            if (!std.mem.endsWith(u8, match.filepath, ".zig")) continue;

            const file = std.fs.openFileAbsolute(match.filepath, .{}) catch continue;
            const source = file.readToEndAllocOptions(self.allocator, 10 * 1024 * 1024, null, .@"1", 0) catch {
                file.close();
                continue;
            };
            file.close();
            defer self.allocator.free(source);

            var real_parser = ast_parser.AstParser.init(self.allocator, source) catch continue;
            defer real_parser.deinit();

            const members = try real_parser.extractMembers();

            var functions: std.ArrayList(types.Member) = .{};
            var classes: std.ArrayList(types.Member) = .{};
            errdefer {
                for (functions.items) |m| self.store.freeMember(m);
                functions.deinit(self.allocator);
                for (classes.items) |m| self.store.freeMember(m);
                classes.deinit(self.allocator);
            }

            for (members) |member| {
                if (member.type == .fn_decl or member.type == .fn_private) {
                    try functions.append(self.allocator, member);
                } else if (member.type == .@"struct" or member.type == .@"enum" or member.type == .@"union") {
                    try classes.append(self.allocator, member);
                } else {
                    self.store.freeMember(member);
                }
            }
            self.allocator.free(members);

            try self.ast_analyses.append(self.allocator, .{
                .filepath = try self.allocator.dupe(u8, match.filepath),
                .token_count = real_parser.countTokens(),
                .functions = try functions.toOwnedSlice(self.allocator),
                .classes = try classes.toOwnedSlice(self.allocator),
            });
        }
    }

    fn findRelatedSkills(self: *QueryEngine) !void {
        const skills_dir = try std.fs.path.join(self.allocator, &.{ self.project_root, ".opencode", "skills" });
        defer self.allocator.free(skills_dir);

        var dir = std.fs.openDirAbsolute(skills_dir, .{ .iterate = true }) catch return;
        defer dir.close();

        const query_lower = try std.ascii.allocLowerString(self.allocator, self.query);
        defer self.allocator.free(query_lower);

        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind == .directory) {
                const name_lower = try std.ascii.allocLowerString(self.allocator, entry.name);
                defer self.allocator.free(name_lower);

                if (std.mem.indexOf(u8, name_lower, query_lower) != null or
                    std.mem.indexOf(u8, query_lower, name_lower) != null)
                {
                    const skill_path = try std.fmt.allocPrint(self.allocator, ".opencode/skills/{s}/SKILL.md", .{entry.name});
                    try self.related_skills.append(self.allocator, skill_path);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// formatCompact: mirrors Python OutputFormatter.compact
// ---------------------------------------------------------------------------

pub fn formatCompact(result: *const types.QueryResult, allocator: std.mem.Allocator, project_root: []const u8) ![]u8 {
    var output: std.ArrayList(u8) = .{};
    defer output.deinit(allocator);
    const w = output.writer(allocator);

    try w.print("# Query: {s}\n\n", .{result.query});

    // --- Primary Match from AST ---
    if (result.ast_analysis.len > 0) {
        const primary = result.ast_analysis[0];
        const rel = relPathFrom(primary.filepath, project_root);

        try w.print("## Primary Match\n\n", .{});
        try w.print("- **{s}** ({} tokens)\n", .{ rel, primary.token_count });

        if (primary.patterns_detected.len > 0) {
            try w.writeAll("- Patterns: ");
            for (primary.patterns_detected, 0..) |p, i| {
                if (i > 0) try w.writeAll(", ");
                try w.writeAll(p);
            }
            try w.writeAll("\n");
        }

        if (primary.functions.len > 0 or primary.classes.len > 0) {
            try w.writeAll("\n### Members\n");
            for (primary.classes) |cls| {
                try w.print("- **{s}**", .{cls.name});
                if (cls.line) |l| try w.print(" (line {})", .{l});
                try w.writeAll("\n");
                for (cls.members) |m| {
                    if (m.line) |l| {
                        try w.print("  - Line {}", .{l});
                    } else {
                        try w.writeAll("  -");
                    }
                    if (m.signature) |s| try w.print(": `{s}`", .{s});
                    if (m.comment) |d| {
                        const nl = std.mem.indexOfScalar(u8, d, '\n') orelse d.len;
                        const preview = d[0..@min(60, nl)];
                        if (preview.len > 0) try w.print(" — {s}", .{preview});
                    }
                    try w.writeAll("\n");
                }
            }
            for (primary.functions) |fn_info| {
                if (fn_info.line) |l| {
                    try w.print("- Line {}", .{l});
                } else {
                    try w.writeAll("-");
                }
                if (fn_info.signature) |s| try w.print(": `{s}`", .{s});
                if (fn_info.comment) |d| {
                    const nl = std.mem.indexOfScalar(u8, d, '\n') orelse d.len;
                    const preview = d[0..@min(60, nl)];
                    if (preview.len > 0) try w.print(" — {s}", .{preview});
                }
                try w.writeAll("\n");
            }
        }
    }

    // --- Guidance JSON members ---
    // Find the best guidance file: prefer one whose path ends in a .zig.json
    // that corresponds to the primary AST file (if any), otherwise use the first.
    var primary_guidance: ?types.GuidanceInfo = null;
    if (result.guidance_files.len > 0) {
        if (result.ast_analysis.len > 0) {
            const ast_stem = std.fs.path.basename(result.ast_analysis[0].filepath);
            for (result.guidance_files) |g| {
                const gbase = std.fs.path.basename(g.path);
                // e.g. "ast_parser.zig.json" starts with "ast_parser.zig"
                if (std.mem.startsWith(u8, gbase, ast_stem)) {
                    primary_guidance = g;
                    break;
                }
            }
        }
        if (primary_guidance == null) primary_guidance = result.guidance_files[0];
    }

    if (primary_guidance) |g| {
        // Only show guidance members section when AST is not available
        // (avoids duplicate listing).
        if (result.ast_analysis.len == 0) {
            const rel = relPathFrom(g.path, project_root);
            try w.print("\n## Primary Match\n\n- **{s}**\n", .{rel});

            if (g.comment.len > 0) {
                const nl = std.mem.indexOfScalar(u8, g.comment, '\n') orelse g.comment.len;
                try w.print("- {s}\n", .{g.comment[0..nl]});
            }

            if (g.functions.len > 0 or g.classes.len > 0) {
                try w.writeAll("\n### Members\n");
                for (g.classes) |cls| {
                    try w.print("- **{s}**", .{cls.name});
                    if (cls.line) |l| try w.print(" (line {})", .{l});
                    if (cls.comment) |d| {
                        const nl = std.mem.indexOfScalar(u8, d, '\n') orelse d.len;
                        const preview = d[0..@min(60, nl)];
                        if (preview.len > 0) try w.print(" — {s}", .{preview});
                    }
                    try w.writeAll("\n");
                    for (cls.members) |m| {
                        if (m.line) |l| {
                            try w.print("  - Line {}", .{l});
                        } else {
                            try w.writeAll("  -");
                        }
                        if (m.signature) |s| try w.print(": `{s}`", .{s});
                        if (m.comment) |d| {
                            const nl = std.mem.indexOfScalar(u8, d, '\n') orelse d.len;
                            const preview = d[0..@min(60, nl)];
                            if (preview.len > 0) try w.print(" — {s}", .{preview});
                        }
                        try w.writeAll("\n");
                    }
                }
                for (g.functions) |fn_info| {
                    if (fn_info.line) |l| {
                        try w.print("- Line {}", .{l});
                    } else {
                        try w.writeAll("-");
                    }
                    if (fn_info.signature) |s| try w.print(": `{s}`", .{s});
                    if (fn_info.comment) |d| {
                        const nl = std.mem.indexOfScalar(u8, d, '\n') orelse d.len;
                        const preview = d[0..@min(60, nl)];
                        if (preview.len > 0) try w.print(" — {s}", .{preview});
                    }
                    try w.writeAll("\n");
                }
            }
        }

        if (g.skills.len > 0) {
            try w.writeAll("\n## Knowledge Base\n\n");
            try w.writeAll("**READ BEFORE IMPLEMENTING**\n\n");
            for (g.skills) |s| try w.print("- `{s}`\n", .{s});
        }

        if (g.tags.len > 0) {
            try w.writeAll("\n## Tags\n");
            for (g.tags) |t| try w.print("- {s}\n", .{t});
        }
    }

    // --- See Also: other guidance files (not the primary) ---
    {
        var see_also_count: usize = 0;
        for (result.guidance_files) |g| {
            if (primary_guidance) |pg| {
                if (std.mem.eql(u8, g.path, pg.path)) continue;
            }
            if (see_also_count == 0) try w.writeAll("\n## See Also\n\n");
            const rel = relPathFrom(g.path, project_root);
            try w.print("- `{s}`\n", .{rel});
            see_also_count += 1;
        }
    }

    // --- Context Graph: other src/ file matches (exclude guidance JSON, show only src/ files) ---
    {
        // Collect eligible matches: src/ source files only, skip .json guidance stubs,
        // skip the primary AST file.
        var graph_items: std.ArrayList([]const u8) = .{};
        defer graph_items.deinit(allocator);

        const primary_ast_path: []const u8 = if (result.ast_analysis.len > 0) result.ast_analysis[0].filepath else "";

        for (result.file_matches) |m| {
            // Skip guidance JSON stubs
            if (std.mem.endsWith(u8, m.filepath, ".json")) continue;
            // Only include files under src/
            const rel = relPathFrom(m.filepath, project_root);
            if (!std.mem.startsWith(u8, rel, "src/")) continue;
            // Skip the primary AST file itself
            if (primary_ast_path.len > 0 and std.mem.eql(u8, m.filepath, primary_ast_path)) continue;
            try graph_items.append(allocator, rel);
        }

        if (graph_items.items.len > 0) {
            try w.print("\n## Context Graph ({} related)\n\n", .{graph_items.items.len});
            const limit: usize = @min(graph_items.items.len, 6);
            for (graph_items.items[0..limit]) |rel| {
                try w.print("- `{s}`\n", .{rel});
            }
        }
    }

    // --- Related Skills ---
    if (result.related_skills.len > 0) {
        try w.writeAll("\n## Knowledge Base\n\n");
        try w.writeAll("**READ BEFORE IMPLEMENTING**\n\n");
        for (result.related_skills) |skill| {
            try w.print("- `{s}`\n", .{skill});
        }
    }

    // --- Recent Knowledge (inbox bullets matching the query) ---
    if (result.insights.len > 0 or result.recent_capabilities.len > 0) {
        try w.writeAll("\n## Recent Knowledge\n\n");
        for (result.insights) |b| {
            try w.print("- [insight] {s}\n", .{b});
        }
        for (result.recent_capabilities) |b| {
            try w.print("- [capability] {s}\n", .{b});
        }
    }

    return output.toOwnedSlice(allocator);
}

fn relPathFrom(filepath: []const u8, project_root: []const u8) []const u8 {
    if (std.mem.indexOf(u8, filepath, project_root)) |idx| {
        const after = filepath[idx + project_root.len ..];
        if (after.len > 0 and after[0] == '/') return after[1..];
        return after;
    }
    return filepath;
}
