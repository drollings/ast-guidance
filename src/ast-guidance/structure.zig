/// STRUCTURE.md generator.
///
/// Mirrors Python's StructureGenerator class in guidance.py.
/// Walks the project directory tree (respecting .gitignore), annotates each
/// file entry with a one-line comment sourced from:
///   1. The corresponding guidance JSON `comment` field (first line).
///   2. The comment already present in the existing STRUCTURE.md (preserved).
///
/// Output format: a Markdown file whose tree lives inside a fenced block.
/// Comments are right-aligned to a common column (≥60, or longest line + 2).
const std = @import("std");
const gitignore_mod = @import("gitignore.zig");
const json_store = @import("json_store.zig");

/// A single entry in the directory tree.
const TreeEntry = union(enum) {
    /// A plain text line (directory label, ".", etc.).
    text: []const u8,
    /// A file whose annotation may come from guidance or the old STRUCTURE.md.
    file: FileEntry,
};

const FileEntry = struct {
    /// The tree prefix string (e.g. "│   ├── ").
    prefix: []const u8,
    /// The filename (basename only, e.g. "main.zig").
    name: []const u8,
    /// Full absolute path to the file.
    abs_path: []const u8,
    /// Path relative to project root (e.g. "src/foo.zig").
    rel_path: []const u8,
};

pub const StructureGenerator = struct {
    allocator: std.mem.Allocator,
    project_root: []const u8,
    guidance_dir: []const u8,
    debug: bool,
    gitignore: gitignore_mod.GitignoreFilter,
    store: json_store.JsonStore,

    pub fn init(
        allocator: std.mem.Allocator,
        project_root: []const u8,
        guidance_dir: []const u8,
        debug: bool,
    ) StructureGenerator {
        return .{
            .allocator = allocator,
            .project_root = project_root,
            .guidance_dir = guidance_dir,
            .debug = debug,
            .gitignore = gitignore_mod.GitignoreFilter.init(allocator, project_root),
            .store = json_store.JsonStore.init(allocator),
        };
    }

    pub fn deinit(self: *StructureGenerator) void {
        self.gitignore.deinit();
    }

    /// Generate (or regenerate) STRUCTURE.md in `project_root`.
    pub fn generate(self: *StructureGenerator) !void {
        // Load .gitignore if present.
        const gitignore_path = try std.fs.path.join(self.allocator, &.{ self.project_root, ".gitignore" });
        defer self.allocator.free(gitignore_path);
        self.gitignore.loadFromFile(gitignore_path) catch {};

        // Parse existing comments before overwriting.
        var old_comments = try self.parseOldComments();
        defer {
            var it = old_comments.iterator();
            while (it.next()) |kv| {
                self.allocator.free(kv.key_ptr.*);
                self.allocator.free(kv.value_ptr.*);
            }
            old_comments.deinit(self.allocator);
        }

        // Build the tree.
        var entries: std.ArrayList(TreeEntry) = .{};
        defer self.freeEntries(&entries);

        try entries.append(self.allocator, .{ .text = try self.allocator.dupe(u8, ".") });
        try self.buildTree(&entries, self.project_root, "");

        // Measure the longest tree-line to determine comment column.
        var max_len: usize = 0;
        for (entries.items) |entry| {
            switch (entry) {
                .text => |t| max_len = @max(max_len, t.len),
                .file => |f| max_len = @max(max_len, f.prefix.len + f.name.len),
            }
        }
        const comment_col: usize = @max(60, max_len + 2);

        // Annotate and write.
        const structure_path = try std.fs.path.join(self.allocator, &.{ self.project_root, "STRUCTURE.md" });
        defer self.allocator.free(structure_path);

        try self.write(structure_path, entries.items, old_comments, comment_col);
    }

    // -------------------------------------------------------------------------
    // Tree building
    // -------------------------------------------------------------------------

    fn buildTree(
        self: *StructureGenerator,
        entries: *std.ArrayList(TreeEntry),
        dir_path: []const u8,
        prefix: []const u8,
    ) !void {
        var dir = std.fs.openDirAbsolute(dir_path, .{ .iterate = true }) catch return;
        defer dir.close();

        // Collect children; filter gitignored and hidden entries.
        var children: std.ArrayList(ChildEntry) = .{};
        defer {
            for (children.items) |c| {
                self.allocator.free(c.name);
                self.allocator.free(c.abs_path);
            }
            children.deinit(self.allocator);
        }

        var it = dir.iterate();
        while (try it.next()) |entry| {
            if (entry.name[0] == '.') continue;

            const abs = try std.fs.path.join(self.allocator, &.{ dir_path, entry.name });
            if (self.gitignore.shouldIgnore(abs)) {
                self.allocator.free(abs);
                continue;
            }

            try children.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, entry.name),
                .abs_path = abs,
                .is_dir = entry.kind == .directory,
            });
        }

        // Sort: directories first, then files; each group alphabetically (case-insensitive).
        std.mem.sort(ChildEntry, children.items, {}, childLessThan);

        for (children.items, 0..) |child, i| {
            const is_last = (i == children.items.len - 1);
            const connector: []const u8 = if (is_last) "└── " else "├── ";
            const extension: []const u8 = if (is_last) "    " else "│   ";

            const child_prefix = try std.mem.concat(self.allocator, u8, &.{ prefix, connector });
            defer self.allocator.free(child_prefix);

            if (child.is_dir) {
                const line = try std.mem.concat(self.allocator, u8, &.{ prefix, connector, child.name });
                try entries.append(self.allocator, .{ .text = line });

                const next_prefix = try std.mem.concat(self.allocator, u8, &.{ prefix, extension });
                defer self.allocator.free(next_prefix);
                try self.buildTree(entries, child.abs_path, next_prefix);
            } else {
                const rel = try self.relPath(child.abs_path);
                try entries.append(self.allocator, .{ .file = .{
                    .prefix = try self.allocator.dupe(u8, child_prefix),
                    .name = try self.allocator.dupe(u8, child.name),
                    .abs_path = try self.allocator.dupe(u8, child.abs_path),
                    .rel_path = rel,
                } });
            }
        }
    }

    fn relPath(self: *StructureGenerator, abs: []const u8) ![]const u8 {
        if (std.mem.startsWith(u8, abs, self.project_root)) {
            var rel = abs[self.project_root.len..];
            if (rel.len > 0 and rel[0] == '/') rel = rel[1..];
            return self.allocator.dupe(u8, rel);
        }
        return self.allocator.dupe(u8, abs);
    }

    // -------------------------------------------------------------------------
    // Comment resolution
    // -------------------------------------------------------------------------

    /// Look up the comment for a file entry.
    /// Priority: guidance JSON documentation → old STRUCTURE.md comment → null.
    fn resolveComment(
        self: *StructureGenerator,
        entry: FileEntry,
        old_comments: std.StringHashMapUnmanaged([]const u8),
    ) !?[]const u8 {
        // Try guidance JSON for all files (not just .zig — JSON files are present
        // for .zig source files; others simply won't have a matching JSON file).
        if (try self.commentFromGuidance(entry.rel_path)) |c| return c;

        // Fall back to the preserved comment from the previous STRUCTURE.md.
        if (old_comments.get(entry.name)) |c| return try self.allocator.dupe(u8, c);

        return null;
    }

    /// Load the guidance JSON for `rel_path` and return the first line of `comment`.
    /// Returns null if no guidance file exists or the field is empty.
    fn commentFromGuidance(self: *StructureGenerator, rel_path: []const u8) !?[]const u8 {
        // Guidance JSON path: guidance_dir / rel_path + ".json"
        // e.g. rel_path = "src/foo/bar.zig"  →  .guidance/src/foo/bar.zig.json
        const json_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}.json", .{ self.guidance_dir, rel_path });
        defer self.allocator.free(json_path);

        const doc = (try self.store.loadGuidance(json_path)) orelse return null;
        defer self.store.freeGuidanceDoc(doc);

        const comment = doc.comment orelse return null;
        if (comment.len == 0) return null;

        // Take only the first line.
        const newline = std.mem.indexOfScalar(u8, comment, '\n') orelse comment.len;
        const first_line = std.mem.trim(u8, comment[0..newline], " \t");
        if (first_line.len == 0) return null;

        // Enforce 120-char cap; truncate cleanly at a word boundary when possible.
        const cap: usize = 120;
        if (first_line.len > cap) {
            return try std.fmt.allocPrint(self.allocator, "{s}...", .{first_line[0..117]});
        }
        return try self.allocator.dupe(u8, first_line);
    }

    // -------------------------------------------------------------------------
    // Parse existing STRUCTURE.md comments
    // -------------------------------------------------------------------------

    /// Extract `filename → comment` from the existing STRUCTURE.md fenced block.
    /// Pattern: `[tree-chars] filename  # comment`
    fn parseOldComments(self: *StructureGenerator) !std.StringHashMapUnmanaged([]const u8) {
        var comments: std.StringHashMapUnmanaged([]const u8) = .{};

        const structure_path = try std.fs.path.join(self.allocator, &.{ self.project_root, "STRUCTURE.md" });
        defer self.allocator.free(structure_path);

        const file = std.fs.openFileAbsolute(structure_path, .{}) catch return comments;
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 2 * 1024 * 1024) catch return comments;
        defer self.allocator.free(content);

        // Find the first fenced block (``` ... ```).
        const fence_start = std.mem.indexOf(u8, content, "```") orelse return comments;
        // Skip past the opening fence line.
        const block_start_raw = std.mem.indexOfScalarPos(u8, content, fence_start + 3, '\n') orelse return comments;
        const block_start = block_start_raw + 1;

        const fence_end = std.mem.indexOfPos(u8, content, block_start, "```") orelse return comments;

        const block = content[block_start..fence_end];

        var line_it = std.mem.splitScalar(u8, block, '\n');
        while (line_it.next()) |line| {
            // A comment line has "  #" (at least two spaces before #).
            const hash_pos = std.mem.indexOf(u8, line, "  #") orelse continue;

            // Everything before the double-space is the tree+filename portion.
            const before = std.mem.trimRight(u8, line[0..hash_pos], " ");
            // Comment is after "  # ".
            const after_hash = std.mem.trimLeft(u8, line[hash_pos + 3 ..], " ");
            if (after_hash.len == 0) continue;

            // Extract just the filename: last path component after tree chars.
            // Tree chars: │ ├ └ ─ space.  Strip them by finding the last non-tree char run.
            const filename = extractFilename(before) orelse continue;

            // Only insert if not already present (first occurrence wins).
            const gop = try comments.getOrPut(self.allocator, filename);
            if (!gop.found_existing) {
                gop.key_ptr.* = try self.allocator.dupe(u8, filename);
                gop.value_ptr.* = try self.allocator.dupe(u8, after_hash);
            }
        }

        return comments;
    }

    // -------------------------------------------------------------------------
    // Write STRUCTURE.md
    // -------------------------------------------------------------------------

    fn write(
        self: *StructureGenerator,
        path: []const u8,
        entries: []const TreeEntry,
        old_comments: std.StringHashMapUnmanaged([]const u8),
        comment_col: usize,
    ) !void {
        const file = try std.fs.createFileAbsolute(path, .{ .truncate = true });
        defer file.close();

        var buf: [16384]u8 = undefined;
        var fw = file.writer(&buf);
        const w = &fw.interface;

        try w.writeAll(HEADER);

        for (entries) |entry| {
            switch (entry) {
                .text => |t| {
                    try w.writeAll(t);
                    try w.writeByte('\n');
                },
                .file => |f| {
                    const base = try std.mem.concat(self.allocator, u8, &.{ f.prefix, f.name });
                    defer self.allocator.free(base);

                    const comment_opt = try self.resolveComment(f, old_comments);
                    defer if (comment_opt) |c| self.allocator.free(c);

                    if (comment_opt) |comment| {
                        const pad_len = if (comment_col > base.len) comment_col - base.len else 1;
                        try w.writeAll(base);
                        // Write padding spaces.
                        var pad_left = pad_len;
                        const spaces = "                                                            ";
                        while (pad_left > 0) {
                            const chunk = @min(pad_left, spaces.len);
                            try w.writeAll(spaces[0..chunk]);
                            pad_left -= chunk;
                        }
                        try w.writeAll("# ");
                        try w.writeAll(comment);
                        try w.writeByte('\n');
                    } else {
                        try w.writeAll(base);
                        try w.writeByte('\n');
                    }
                },
            }
        }

        try w.writeAll("```\n");
        try w.flush();
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    fn freeEntries(self: *StructureGenerator, entries: *std.ArrayList(TreeEntry)) void {
        for (entries.items) |entry| {
            switch (entry) {
                .text => |t| self.allocator.free(t),
                .file => |f| {
                    self.allocator.free(f.prefix);
                    self.allocator.free(f.name);
                    self.allocator.free(f.abs_path);
                    self.allocator.free(f.rel_path);
                },
            }
        }
        entries.deinit(self.allocator);
    }
};

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

const ChildEntry = struct {
    name: []const u8,
    abs_path: []const u8,
    is_dir: bool,
};

fn childLessThan(_: void, a: ChildEntry, b: ChildEntry) bool {
    // Dirs before files.
    if (a.is_dir != b.is_dir) return a.is_dir;
    // Within each group: case-insensitive alpha.
    return std.ascii.lessThanIgnoreCase(a.name, b.name);
}

/// Extract the filename from a tree line like "│   ├── main.zig".
/// Returns a slice into the input — not heap-allocated.
fn extractFilename(line: []const u8) ?[]const u8 {
    // Walk from the end of the string backwards past any trailing spaces,
    // then find the start of the filename token.
    const trimmed = std.mem.trimRight(u8, line, " \t");
    if (trimmed.len == 0) return null;

    // The filename starts after the last space in the tree prefix.
    var i: usize = trimmed.len;
    while (i > 0) {
        i -= 1;
        const c = trimmed[i];
        // Tree box-drawing chars and their ASCII equivalents used in prefixes.
        if (c == ' ' or c == '\t') {
            return trimmed[i + 1 ..];
        }
        // Unicode box-drawing characters occupy 3 bytes in UTF-8.
        // If we hit a non-ASCII byte, look for the space that precedes it.
    }
    return trimmed; // whole line is the filename (e.g. ".")
}

// ---------------------------------------------------------------------------
// Static header — matches the existing STRUCTURE.md preamble exactly.
// ---------------------------------------------------------------------------

const HEADER =
    \\# AST-Guidance Project Structure
    \\
    \\A fast, lightweight code navigation and orchestration framework friendly to
    \\human and human-in-the-loop LLM agentic software engineering.  It is based
    \\on enriched AST, and uses optional AI for documentation which is cached,
    \\idempotent, and upcycled for lightweight searches and local agentic
    \\intelligence.
    \\
    \\## Quick Navigation (Coding Assistants)
    \\
    \\| Purpose | File | Use When |
    \\|---------|------|----------|
    \\| **Find related code** | `make query QUERY="search terms"` | Searching for code |
    \\| **Check Implementation** | `make explore QUERY="search terms"` | Before implementing anything |
    \\| **Understand patterns** | `doc/capabilities/*.md` | Implementation examples + patterns |
    \\| **Find existing code** | `mcp_grep` or `mcp_lsp_find_references` | Searching for implementations |
    \\
    \\## **Attention**: Skills needed to understand files
    \\
    \\Skills are referenced per-file in comments below.  The lookup path for the skills is: 
    \\`.ast-guidance/.skills/{skill}/SKILL.md`
    \\
    \\So if you find a file you're looking for named file.zig:
    \\`file.zig      # [zig-current, gof-patterns] Summary of files' contents` , 
    \\Then you you must read
    \\
    \\```
    \\.ast-guidance/.skills/zig-current/SKILL.md
    \\.ast-guidance/.skills/gof-patterns/SKILL.md
    \\```
    \\
    \\---
    \\
    \\## Directory Tree (Git-Tracked Files Only)
    \\
    \\```
    \\
;
