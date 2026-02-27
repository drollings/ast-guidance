const std = @import("std");
const types = @import("types.zig");
const ast_parser = @import("ast_parser.zig");
const sync = @import("sync.zig");
const query = @import("query.zig");
const deps = @import("deps.zig");
const structure_mod = @import("structure.zig");
const triage_mod = @import("triage.zig");
const llm = @import("common");
const enhancer_mod = @import("enhancer.zig");
const config_mod = @import("config.zig");

pub const version = "0.2.0";

const Command = enum {
    sync,
    query,
    explore,
    explain,
    clean,
    structure,
    diary,
    commit,
    learn,
    deps,
    triage,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printHelp();
        return;
    }

    // Handle global flags before subcommand.
    if (args.len >= 2 and (std.mem.eql(u8, args[1], "-h") or std.mem.eql(u8, args[1], "--help"))) {
        try printHelp();
        return;
    }

    const subcmd = std.meta.stringToEnum(Command, args[1]) orelse {
        std.debug.print("Unknown subcommand: {s}\n\n", .{args[1]});
        try printHelp();
        return;
    };

    switch (subcmd) {
        .clean => try cmdClean(allocator, args[2..]),
        .commit => try cmdCommit(allocator, args[2..]),
        .deps => try cmdDeps(allocator, args[2..]),
        .diary => try cmdDiary(allocator, args[2..]),
        .explain => try cmdExplain(allocator, args[2..]),
        .explore => try cmdExplore(allocator, args[2..]),
        .learn => try cmdLearn(allocator, args[2..]),
        .query => try cmdQuery(allocator, args[2..]),
        .structure => try cmdStructure(allocator, args[2..]),
        .sync => try cmdSync(allocator, args[2..]),
        .triage => try cmdTriage(allocator, args[2..]),
    }
}

fn printHelp() !void {
    var out_buf: [8192]u8 = undefined;
    var fw = std.fs.File.stdout().writer(&out_buf);
    const stdout = &fw.interface;

    try stdout.writeAll(
        \\ast-guidance-zig v0.2.0: Zig-primary guidance system for agentic coding loops
        \\
        \\Usage:
        \\  ast-guidance-zig <subcommand> [options]
        \\  ast-guidance-zig --help
        \\
        \\Subcommands:
        \\  clean      Strip ephemeral fields from a guidance JSON file (stdout)
        \\  commit     Generate AI git commit message and open editor
        \\  deps       Generate Makefile .depend file from Zig imports
        \\  diary      Append a timestamped diary entry
        \\  explore    Comprehensive exploration of a module (primary for 'make explain')
        \\  explain    Explain a module with AI summary (like bin/guidance.py explain)
        \\  learn      Drain INSIGHTS.md and CAPABILITIES.md into structured knowledge
        \\  query      Query codebase context (AST on by default)
        \\  structure  Regenerate STRUCTURE.md from all guidance JSON
        \\  sync       Generate/sync guidance JSON from Zig source files
        \\  triage     Triage a TODO work item → generate TRIAGE.md
        \\
        \\Common LLM options (sync, explore, structure, commit, learn):
        \\  --api-url URL     LLM API endpoint (default: http://localhost:11434/api/chat)
        \\  -m, --model NAME  Model name (default: fast:latest)
        \\
        \\Sync options:
        \\  --file FILE       Process a single .zig file
        \\  --scan DIR        Scan all .zig files under DIR
        \\  --output DIR      Output directory for guidance JSON (required)
        \\  --infill          LLM-fill blank comment fields
        \\  --regen           LLM-regenerate all comments; keep the better score
        \\  --structure       Create guidance JSON for new files (no existing JSON)
        \\  --cross-language  Also infill .py.json files after sync (default: on)
        \\  --dry-run         Show what would change without writing
        \\  --verbose         Print LLM prompts and raw responses (alias: --debug)
        \\
        \\Explore options:
        \\  <query>           Search query; may use "[term] focus question" bracket syntax
        \\  --guidance DIR    Guidance JSON directory (required)
        \\  --no-ai           Skip LLM decomposition, retry loop, and AI synthesis
        \\  --src-dir DIR     Override source directory for excerpt extraction
        \\  --format FMT      Output format: markdown (default), compact, json
        \\  --api-url URL     LLM endpoint (default: http://localhost:11434/api/chat)
        \\  -m, --model NAME  Model name (default: fast:latest)
        \\
        \\Explain options:
        \\  <query>           Module or concept to explain
        \\  --guidance DIR    Guidance JSON directory (required)
        \\  --no-ai           Skip LLM AI synthesis
        \\  --api-url URL     LLM endpoint (default: http://localhost:11434/api/chat)
        \\  -m, --model NAME  Model name (default: fast:latest)
        \\
        \\Structure options:
        \\  --guidance DIR    Guidance JSON directory (required)
        \\  --no-ai           Skip AI infill pre-pass
        \\  --api-url URL     LLM endpoint for AI infill (default: http://localhost:11434/api/chat)
        \\  -m, --model NAME  Model for AI infill (default: fast:latest)
        \\
        \\Diary options:
        \\  <note>            Text to append (required)
        \\  --guidance DIR    Guidance directory (required)
        \\
        \\Commit options:
        \\  --dry-run         Print generated message; do not open editor
        \\  --debug           Print LLM prompts and responses; show generated message before editor
        \\
        \\Learn options:
        \\  --guidance DIR    Guidance directory (required)
        \\  --dry-run         Show what would be written without making changes
        \\  --api-url URL     LLM endpoint (default: http://localhost:11434/api/chat)
        \\  -m, --model NAME  Model name (default: fast:latest)
        \\
        \\Triage options:
        \\  <todo_path>       Path to TODO.md (required)
        \\  --guidance DIR    Guidance directory (required)
        \\  --api-url URL     LLM endpoint for step generation
        \\  -m, --model NAME  Model name
        \\
        \\Learn options:
        \\  --dry-run         Show what would be promoted without writing
        \\
        \\Examples:
        \\  ast-guidance-zig sync --scan src --output guidance
        \\  ast-guidance-zig sync --file src/main.zig --infill -m fast:latest
        \\  ast-guidance-zig explore "ast_parser"
        \\  ast-guidance-zig explore --no-ai --format json "sync"
        \\  ast-guidance-zig explore "[SyncProcessor] what flags does it support?"
        \\  ast-guidance-zig structure
        \\  ast-guidance-zig diary "Fixed the ring buffer race condition"
        \\  ast-guidance-zig commit
        \\  ast-guidance-zig learn --dry-run
        \\  ast-guidance-zig query "hash"
        \\  ast-guidance-zig deps --src src | sed ... > .zig.depend
        \\
    );
    try stdout.flush();
}

// =============================================================================
// sync
// =============================================================================

const SyncArgs = struct {
    scan: ?[]const u8 = null,
    file: ?[]const u8 = null,
    output: ?[]const u8 = null,
    dry_run: bool = false,
    debug: bool = false,
    api_url: []const u8 = "http://localhost:11434/api/chat",
    model: []const u8 = "fast:latest",
    infill_comments: bool = false,
    regen_comments: bool = false,
    regen_structure: bool = false,
    /// After sync, walk all guidance JSON (including .py.json) and infill blank comments.
    cross_language: bool = true,
};

fn cmdSync(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var sync_args: SyncArgs = .{};
    var i: usize = 0;

    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--scan")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Missing argument for --scan\n", .{});
                return;
            }
            sync_args.scan = args[i];
        } else if (std.mem.eql(u8, arg, "--file")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Missing argument for --file\n", .{});
                return;
            }
            sync_args.file = args[i];
        } else if (std.mem.eql(u8, arg, "--output")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Missing argument for --output\n", .{});
                return;
            }
            sync_args.output = args[i];
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            sync_args.dry_run = true;
        } else if (std.mem.eql(u8, arg, "--debug") or std.mem.eql(u8, arg, "--verbose")) {
            sync_args.debug = true;
        } else if (std.mem.eql(u8, arg, "--infill")) {
            sync_args.infill_comments = true;
        } else if (std.mem.eql(u8, arg, "--regen")) {
            sync_args.regen_comments = true;
        } else if (std.mem.eql(u8, arg, "--structure")) {
            sync_args.regen_structure = true;
        } else if (std.mem.eql(u8, arg, "--no-cross-language")) {
            sync_args.cross_language = false;
        } else if (std.mem.eql(u8, arg, "--cross-language")) {
            sync_args.cross_language = true;
        } else if (std.mem.eql(u8, arg, "--api-url")) {
            i += 1;
            if (i >= args.len) return;
            sync_args.api_url = args[i];
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return;
            sync_args.model = args[i];
        }
    }

    if (sync_args.output == null) {
        std.debug.print("❌ Error: --output is required. Usage: ast-guidance sync --file <file> --output <dir> [options]\n", .{});
        return;
    }

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    const output_path = if (std.fs.path.isAbsolute(sync_args.output.?))
        try allocator.dupe(u8, sync_args.output.?)
    else
        try std.fs.path.join(allocator, &.{ cwd, sync_args.output.? });
    defer allocator.free(output_path);

    var processor = sync.SyncProcessor.init(allocator, cwd, output_path, sync_args.dry_run, sync_args.debug);
    defer processor.deinit();

    if (sync_args.infill_comments or sync_args.regen_comments or sync_args.regen_structure) {
        const config: llm.LlmConfig = .{
            .api_url = sync_args.api_url,
            .model = sync_args.model,
            .debug = sync_args.debug,
        };
        processor.enhancer = enhancer_mod.Enhancer.init(allocator, config) catch |err| blk: {
            std.debug.print("Warning: could not init LLM enhancer: {}\n", .{err});
            break :blk null;
        };
        processor.infill_comments = sync_args.infill_comments;
        processor.regen_comments = sync_args.regen_comments;
        processor.regen_structure = sync_args.regen_structure;
    }

    if (sync_args.file) |file_path| {
        const full_path = if (std.fs.path.isAbsolute(file_path))
            try allocator.dupe(u8, file_path)
        else
            try std.fs.path.join(allocator, &.{ cwd, file_path });
        defer allocator.free(full_path);

        _ = processor.processFile(full_path) catch |err| {
            std.debug.print("Error processing {s}: {}\n", .{ full_path, err });
        };

        // Cross-language infill: targeted to the single corresponding JSON file so
        // that per-file Makefile targets (make sync --file src/foo.zig --infill)
        // only touch that file's guidance JSON — not the whole directory.
        if (sync_args.cross_language and (sync_args.infill_comments or sync_args.regen_comments)) {
            const rel = blk: {
                const p = full_path;
                if (std.mem.startsWith(u8, p, cwd)) {
                    var r = p[cwd.len..];
                    if (r.len > 0 and r[0] == '/') r = r[1..];
                    break :blk r;
                }
                break :blk p;
            };
            const json_path = try std.fmt.allocPrint(allocator, "{s}/{s}.json", .{ output_path, rel });
            defer allocator.free(json_path);
            _ = processor.infillJsonFile(json_path) catch |err| {
                std.debug.print("warning: infill failed for {s}: {}\n", .{ json_path, err });
            };
        }
    } else if (sync_args.scan) |scan_dir| {
        const full_path = if (std.fs.path.isAbsolute(scan_dir))
            try allocator.dupe(u8, scan_dir)
        else
            try std.fs.path.join(allocator, &.{ cwd, scan_dir });
        defer allocator.free(full_path);

        const count = try processor.processDirectory(full_path);
        std.debug.print("Processed {} files\n", .{count});

        // Cross-language infill: walk ALL guidance JSON (including .py.json)
        // and fill any remaining blank comments after the directory sweep.
        if (sync_args.cross_language and (sync_args.infill_comments or sync_args.regen_comments)) {
            var skip_paths: std.StringHashMapUnmanaged(void) = .{};
            defer skip_paths.deinit(allocator);
            const infilled = processor.infillAllJson(output_path, &skip_paths) catch 0;
            if (infilled > 0) {
                std.debug.print("Cross-language infill: {} file(s) updated\n", .{infilled});
            }
        }
    } else {
        std.debug.print("Error: --scan or --file required\n", .{});
        try printHelp();
        return;
    }

    if (sync_args.dry_run) {
        std.debug.print("(dry-run complete — no files written)\n", .{});
    }
}

// =============================================================================
// query
// =============================================================================

const QueryArgs = struct {
    query_str: ?[]const u8 = null,
    use_ast: bool = true,
    use_smart: bool = false,
    format: []const u8 = "compact",
    debug: bool = false,
    api_url: []const u8 = "http://localhost:11434/api/chat",
    model: []const u8 = "fast:latest",
};

fn cmdQuery(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var query_args: QueryArgs = .{};
    var i: usize = 0;

    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--no-ast")) {
            query_args.use_ast = false;
        } else if (std.mem.eql(u8, arg, "--ast")) {
            query_args.use_ast = true;
        } else if (std.mem.eql(u8, arg, "--smart")) {
            query_args.use_smart = true;
        } else if (std.mem.eql(u8, arg, "--format")) {
            i += 1;
            if (i >= args.len) return;
            query_args.format = args[i];
        } else if (std.mem.eql(u8, arg, "--debug")) {
            query_args.debug = true;
        } else if (std.mem.eql(u8, arg, "--api-url")) {
            i += 1;
            if (i >= args.len) return;
            query_args.api_url = args[i];
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return;
            query_args.model = args[i];
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            query_args.query_str = arg;
        }
    }

    const query_text = query_args.query_str orelse {
        std.debug.print("Error: query string required\n", .{});
        return;
    };

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    // Load config (ownership transferred to engine — do NOT free cfg separately).
    const cfg = try config_mod.loadConfig(allocator, cwd);
    var engine = query.QueryEngine.init(allocator, query_text, cwd, query_args.use_ast, query_args.debug, cfg);
    defer engine.deinit();

    const result = try engine.execute();
    defer query.freeQueryResult(allocator, &engine.store, result);

    const output = if (std.mem.eql(u8, query_args.format, "json"))
        try types.jsonStringifyAlloc(allocator, result)
    else
        try query.formatCompact(&result, allocator, cwd);

    defer allocator.free(output);
    std.debug.print("{s}", .{output});
}

// =============================================================================
// explore  (primary 'make explain' target)
// =============================================================================

// =============================================================================
// explore helpers
// =============================================================================

/// A collected guidance document for the explore command.
const ExploreDoc = struct {
    full_path: []const u8, // owned
    source: []const u8, // owned dupe of doc.meta.source
    module: []const u8, // owned dupe
    language: []const u8, // owned dupe
    comment: ?[]const u8, // owned dupe or null
    skills: []types.Skill, // owned dupe slice
    hashtags: [][]const u8, // owned dupe slice
    used_by: [][]const u8, // owned dupe slice
    members: []types.Member, // owned dupe slice (from store.dupeMember)
    is_primary: bool,

    fn deinit(self: ExploreDoc, allocator: std.mem.Allocator, store: *@import("json_store.zig").JsonStore) void {
        allocator.free(self.full_path);
        allocator.free(self.source);
        allocator.free(self.module);
        allocator.free(self.language);
        if (self.comment) |c| allocator.free(c);
        for (self.skills) |sk| {
            allocator.free(sk.ref);
            if (sk.context) |ctx| allocator.free(ctx);
        }
        allocator.free(self.skills);
        for (self.hashtags) |h| allocator.free(h);
        allocator.free(self.hashtags);
        for (self.used_by) |u| allocator.free(u);
        allocator.free(self.used_by);
        for (self.members) |m| store.freeMember(m);
        allocator.free(self.members);
    }
};

/// Dupe a GuidanceDoc into an ExploreDoc (caller owns result).
fn dupeExploreDoc(
    allocator: std.mem.Allocator,
    store: *@import("json_store.zig").JsonStore,
    doc: types.GuidanceDoc,
    full_path: []const u8,
    is_primary: bool,
) !ExploreDoc {
    const skills = try allocator.alloc(types.Skill, doc.skills.len);
    for (doc.skills, 0..) |sk, idx| {
        skills[idx] = .{
            .ref = try allocator.dupe(u8, sk.ref),
            .context = if (sk.context) |c| try allocator.dupe(u8, c) else null,
        };
    }
    const hashtags = try allocator.alloc([]const u8, doc.hashtags.len);
    for (doc.hashtags, 0..) |h, idx| hashtags[idx] = try allocator.dupe(u8, h);
    const used_by = try allocator.alloc([]const u8, doc.used_by.len);
    for (doc.used_by, 0..) |u, idx| used_by[idx] = try allocator.dupe(u8, u);
    const members = try allocator.alloc(types.Member, doc.members.len);
    for (doc.members, 0..) |m, idx| members[idx] = try store.dupeMember(m);
    return .{
        .full_path = try allocator.dupe(u8, full_path),
        .source = try allocator.dupe(u8, doc.meta.source),
        .module = try allocator.dupe(u8, doc.meta.module),
        .language = try allocator.dupe(u8, doc.meta.language),
        .comment = if (doc.comment) |c| try allocator.dupe(u8, c) else null,
        .skills = skills,
        .hashtags = hashtags,
        .used_by = used_by,
        .members = members,
        .is_primary = is_primary,
    };
}

/// Check whether any search term matches a guidance doc (filename, module, source, or member names).
fn docMatchesTerms(
    allocator: std.mem.Allocator,
    doc: types.GuidanceDoc,
    terms: []const []const u8,
    satisfied: []bool, // parallel to terms; updated in-place for newly matched terms
) !bool {
    const module_lower = try std.ascii.allocLowerString(allocator, doc.meta.module);
    defer allocator.free(module_lower);
    const source_lower = try std.ascii.allocLowerString(allocator, doc.meta.source);
    defer allocator.free(source_lower);

    var any = false;
    for (terms, 0..) |term, ti| {
        const matched_here = std.mem.indexOf(u8, module_lower, term) != null or
            std.mem.indexOf(u8, source_lower, term) != null;
        if (matched_here) {
            any = true;
            satisfied[ti] = true;
        }
    }
    for (doc.members) |m| {
        const mname_lower = try std.ascii.allocLowerString(allocator, m.name);
        defer allocator.free(mname_lower);
        for (terms, 0..) |term, ti| {
            if (std.mem.indexOf(u8, mname_lower, term) != null) {
                any = true;
                satisfied[ti] = true;
            }
        }
        for (m.members) |mm| {
            const mmname_lower = try std.ascii.allocLowerString(allocator, mm.name);
            defer allocator.free(mmname_lower);
            for (terms, 0..) |term, ti| {
                if (std.mem.indexOf(u8, mmname_lower, term) != null) {
                    any = true;
                    satisfied[ti] = true;
                }
            }
        }
    }
    return any;
}

/// Return up to `max_lines` source lines starting at `start_line` (1-based),
/// stopping at the next top-level pub/fn declaration or 80 lines, whichever comes first.
fn extractSourceExcerpt(
    allocator: std.mem.Allocator,
    src_content: []const u8,
    start_line: usize,
    max_lines: usize,
) ![]const u8 {
    var lines_iter = std.mem.splitScalar(u8, src_content, '\n');
    var line_idx: usize = 0;
    // Advance to start_line (1-based → 0-based index = start_line - 1).
    const target_start = if (start_line > 0) start_line - 1 else 0;
    while (line_idx < target_start) : (line_idx += 1) {
        _ = lines_iter.next() orelse return allocator.dupe(u8, "");
    }

    var buf: std.ArrayList(u8) = .{};
    var captured: usize = 0;
    var first = true;
    var saw_close_brace = false; // tracks whether we've emitted a col-0 closing }
    while (lines_iter.next()) |line| {
        if (captured >= max_lines) break;
        const trimmed_line = std.mem.trimRight(u8, line, "\r");
        // Stop at next top-level declaration at column 0 (not on the very first line).
        if (!first and trimmed_line.len > 0 and trimmed_line[0] != ' ' and trimmed_line[0] != '\t') {
            if (std.mem.startsWith(u8, trimmed_line, "pub ") or
                std.mem.startsWith(u8, trimmed_line, "fn ") or
                std.mem.startsWith(u8, trimmed_line, "const ") or
                std.mem.startsWith(u8, trimmed_line, "var ") or
                std.mem.startsWith(u8, trimmed_line, "// =") or
                std.mem.startsWith(u8, trimmed_line, "// -") or
                // A col-0 doc-comment or plain comment after we've seen the closing brace
                (saw_close_brace and std.mem.startsWith(u8, trimmed_line, "//")) or
                (saw_close_brace and trimmed_line.len == 0))
            {
                break;
            }
        }
        // Track col-0 closing brace (end of top-level block).
        if (!first and std.mem.eql(u8, std.mem.trim(u8, trimmed_line, " \t"), "};")) {
            saw_close_brace = true;
        }
        // Skip separator banners.
        if (std.mem.startsWith(u8, std.mem.trimLeft(u8, trimmed_line, " \t"), "// ---")) {
            captured += 1;
            first = false;
            continue;
        }
        try buf.appendSlice(allocator, line);
        try buf.append(allocator, '\n');
        captured += 1;
        first = false;
    }

    // Strip trailing blank lines.
    const raw = buf.toOwnedSlice(allocator) catch return allocator.dupe(u8, "");
    const trimmed = std.mem.trimRight(u8, raw, " \t\r\n");
    defer allocator.free(raw);
    return allocator.dupe(u8, trimmed);
}

/// Extract source lines [start_line, end_line] (1-based, inclusive), capped at 80 lines.
/// Prunes trailing blank lines then trailing comment-only lines in a stable loop,
/// mirroring guidance.py _extract_source_excerpts trimming behaviour.
/// Returns an owned allocation; caller must free.
fn extractSourceExcerptPruned(
    allocator: std.mem.Allocator,
    src_content: []const u8,
    start_line: u32,
    end_line: u32,
) ![]const u8 {
    const MAX_LINES: u32 = 80;
    const actual_end = @min(end_line, start_line + MAX_LINES - 1);

    // Collect the raw lines in the window as slices into src_content (not owned).
    var lines: std.ArrayList([]const u8) = .{};
    defer lines.deinit(allocator);

    var iter = std.mem.splitScalar(u8, src_content, '\n');
    var line_no: u32 = 0;
    while (iter.next()) |raw_line| {
        line_no += 1;
        if (line_no < start_line) continue;
        if (line_no > actual_end) break;
        try lines.append(allocator, std.mem.trimRight(u8, raw_line, "\r"));
    }

    if (lines.items.len == 0) return allocator.dupe(u8, "");

    // Stable prune: repeat until no more changes.
    // Step 1: remove trailing blank lines.
    // Step 2: remove trailing comment-only lines (// or ///).
    // Both steps repeat until neither removes anything (stable loop).
    var changed = true;
    while (changed) {
        changed = false;
        while (lines.items.len > 0) {
            if (std.mem.trim(u8, lines.items[lines.items.len - 1], " \t\r").len == 0) {
                _ = lines.pop();
                changed = true;
            } else break;
        }
        while (lines.items.len > 0) {
            const lstripped = std.mem.trimLeft(u8, lines.items[lines.items.len - 1], " \t");
            if (std.mem.startsWith(u8, lstripped, "//")) {
                _ = lines.pop();
                changed = true;
            } else break;
        }
    }

    if (lines.items.len == 0) return allocator.dupe(u8, "");

    // Join into a single owned string.
    var buf: std.ArrayList(u8) = .{};
    errdefer buf.deinit(allocator);
    for (lines.items, 0..) |line, idx| {
        if (idx > 0) try buf.append(allocator, '\n');
        try buf.appendSlice(allocator, line);
    }
    return try buf.toOwnedSlice(allocator);
}

/// A single grep match result.
const GrepMatch = struct {
    line_no: usize,
    line: []const u8, // owned
};

/// Grep a file for any of the search terms (case-insensitive substring).
/// Returns up to `max_results` matches, skipping pure comment lines.
fn grepFile(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    terms: []const []const u8,
    max_results: usize,
) ![]GrepMatch {
    const file = std.fs.openFileAbsolute(file_path, .{}) catch return &.{};
    defer file.close();
    const content = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return &.{};
    defer allocator.free(content);

    var results: std.ArrayList(GrepMatch) = .{};
    var lines_iter = std.mem.splitScalar(u8, content, '\n');
    var line_no: usize = 0;
    while (lines_iter.next()) |line| {
        line_no += 1;
        if (results.items.len >= max_results) break;
        const trimmed = std.mem.trimLeft(u8, line, " \t");
        // Skip pure comment lines.
        if (std.mem.startsWith(u8, trimmed, "//") or std.mem.startsWith(u8, trimmed, "#")) continue;
        const line_lower = try std.ascii.allocLowerString(allocator, line);
        defer allocator.free(line_lower);
        for (terms) |term| {
            if (std.mem.indexOf(u8, line_lower, term) != null) {
                try results.append(allocator, .{
                    .line_no = line_no,
                    .line = try allocator.dupe(u8, std.mem.trimRight(u8, line, "\r")),
                });
                break;
            }
        }
    }
    return results.toOwnedSlice(allocator);
}

// =============================================================================
// explore / explain
// =============================================================================

fn cmdExplore(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const common = llm.parseCommonArgs(args);

    var query_str: ?[]const u8 = null;
    var guidance_dir_arg: ?[]const u8 = null;
    var src_dir_override: ?[]const u8 = null;
    var format: enum { markdown, compact, json } = .markdown;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--guidance")) {
            i += 1;
            if (i >= args.len) return;
            guidance_dir_arg = args[i];
        } else if (std.mem.eql(u8, arg, "--src-dir")) {
            i += 1;
            if (i >= args.len) return;
            src_dir_override = args[i];
        } else if (std.mem.eql(u8, arg, "--format")) {
            i += 1;
            if (i >= args.len) return;
            const fmt_str = args[i];
            if (std.mem.eql(u8, fmt_str, "compact")) {
                format = .compact;
            } else if (std.mem.eql(u8, fmt_str, "json")) {
                format = .json;
            } else {
                format = .markdown;
            }
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            query_str = arg;
        }
    }

    if (guidance_dir_arg == null) {
        std.debug.print("❌ Error: --guidance is required. Usage: ast-guidance explore <query> --guidance <dir> [options]\n", .{});
        return;
    }

    const debug = common.debug;
    const no_ai = common.no_ai;
    const api_url = common.api_url;
    const model = common.model;

    const q = query_str orelse {
        std.debug.print("Error: query string required\n", .{});
        return;
    };

    if (debug) std.debug.print("explore: query={s} api_url={s} model={s} no_ai={}\n", .{ q, api_url, model, no_ai });

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    // -------------------------------------------------------------------------
    // M2.2 — bracket syntax: "[search_term] focus_question"
    // -------------------------------------------------------------------------
    var focus_question: ?[]const u8 = null;
    var raw_search_query: []const u8 = q;
    if (q.len > 0 and q[0] == '[') {
        if (std.mem.indexOfScalar(u8, q, ']')) |bracket_end| {
            raw_search_query = q[1..bracket_end];
            const after = std.mem.trim(u8, q[bracket_end + 1 ..], " \t");
            if (after.len > 0) focus_question = after;
        }
    }

    // -------------------------------------------------------------------------
    // LLM client (may be unavailable)
    // -------------------------------------------------------------------------
    var llm_client_opt: ?llm.LlmClient = if (!no_ai)
        llm.LlmClient.init(allocator, .{ .api_url = api_url, .model = model, .debug = debug }) catch null
    else
        null;
    defer if (llm_client_opt) |*c| c.deinit();
    const llm_available = if (llm_client_opt) |*c| c.available() else false;

    // -------------------------------------------------------------------------
    // M2.3 — Multi-word query decomposition via LLM
    // -------------------------------------------------------------------------
    var search_terms: std.ArrayList([]const u8) = .{};
    defer {
        for (search_terms.items) |t| allocator.free(t);
        search_terms.deinit(allocator);
    }

    const has_spaces = std.mem.indexOfScalar(u8, raw_search_query, ' ') != null;
    if (has_spaces and llm_available and !no_ai) {
        if (llm_client_opt) |*client| {
            const decompose_prompt = try std.fmt.allocPrint(
                allocator,
                "Extract the 1-3 most important code identifiers from this query.\nQuery: {s}\nReturn ONLY identifiers, one per line, most important first.",
                .{raw_search_query},
            );
            defer allocator.free(decompose_prompt);

            if (client.complete(decompose_prompt, 60, 0.1, null) catch null) |resp| {
                defer allocator.free(resp);
                var resp_lines = std.mem.splitScalar(u8, resp, '\n');
                var count: usize = 0;
                while (resp_lines.next()) |line| {
                    if (count >= 3) break;
                    const trimmed_id = std.mem.trim(u8, line, " \t\r-.");
                    if (trimmed_id.len == 0 or trimmed_id.len > 80) continue;
                    const lower_id = try std.ascii.allocLowerString(allocator, trimmed_id);
                    try search_terms.append(allocator, lower_id);
                    count += 1;
                }
                // Use the raw_search_query as focus_question if not already set.
                if (focus_question == null) focus_question = raw_search_query;
            }
        }
    }

    // Fallback: single term from raw_search_query.
    if (search_terms.items.len == 0) {
        const lower_q = try std.ascii.allocLowerString(allocator, raw_search_query);
        try search_terms.append(allocator, lower_q);
    }

    if (debug) {
        std.debug.print("explore: search_terms=", .{});
        for (search_terms.items) |t| std.debug.print("'{s}' ", .{t});
        std.debug.print("\n", .{});
    }

    // -------------------------------------------------------------------------
    // Guidance directory setup
    // -------------------------------------------------------------------------
    const guidance_dir = if (std.fs.path.isAbsolute(guidance_dir_arg.?))
        try allocator.dupe(u8, guidance_dir_arg.?)
    else
        try std.fs.path.join(allocator, &.{ cwd, guidance_dir_arg.? });
    defer allocator.free(guidance_dir);

    var gdir = std.fs.openDirAbsolute(guidance_dir, .{ .iterate = true }) catch {
        std.debug.print("No guidance directory found at {s}\n", .{guidance_dir});
        return;
    };
    defer gdir.close();

    var store = @import("json_store.zig").JsonStore.init(allocator);

    // -------------------------------------------------------------------------
    // Gather function: collect all JSON paths in the guidance dir.
    // -------------------------------------------------------------------------
    var all_json_paths: std.ArrayList([]const u8) = .{};
    defer {
        for (all_json_paths.items) |p| allocator.free(p);
        all_json_paths.deinit(allocator);
    }
    {
        var walker_scan = try gdir.walk(allocator);
        defer walker_scan.deinit();
        while (try walker_scan.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".json")) continue;
            const fp = try std.fs.path.join(allocator, &.{ guidance_dir, entry.path });
            try all_json_paths.append(allocator, fp);
        }
    }

    // -------------------------------------------------------------------------
    // M2.1 — Multi-doc gather with satisfied-terms tracking.
    // Primary: filename/module/source matches; up to 3.
    // Secondary: member-name matches for unsatisfied terms; up to 3.
    // -------------------------------------------------------------------------

    // Helper: run a single gather pass over all_json_paths.
    // Returns collected ExploreDoc slice (caller owns).
    const GatherResult = struct {
        primary: []ExploreDoc,
        secondary: []ExploreDoc,
        satisfied: []bool,
    };

    const runGather = struct {
        fn call(
            alloc: std.mem.Allocator,
            st: *@import("json_store.zig").JsonStore,
            json_paths: []const []const u8,
            terms: []const []const u8,
        ) !GatherResult {
            var primary_docs: std.ArrayList(ExploreDoc) = .{};
            errdefer {
                for (primary_docs.items) |*ed| ed.deinit(alloc, st);
                primary_docs.deinit(alloc);
            }
            var secondary_docs: std.ArrayList(ExploreDoc) = .{};
            errdefer {
                for (secondary_docs.items) |*ed| ed.deinit(alloc, st);
                secondary_docs.deinit(alloc);
            }

            const satisfied = try alloc.alloc(bool, terms.len);
            @memset(satisfied, false);

            // Track which JSON paths and source paths we've already added.
            var seen_json: std.StringHashMapUnmanaged(void) = .{};
            defer seen_json.deinit(alloc);
            var seen_sources: std.StringHashMapUnmanaged(void) = .{};
            defer seen_sources.deinit(alloc);

            // Primary pass: filename/module/source match.
            for (json_paths) |fp| {
                if (primary_docs.items.len >= 3) break;
                const doc = (try st.loadGuidance(fp)) orelse continue;
                defer st.freeGuidanceDoc(doc);

                // Deduplicate by source file (two JSON files for the same source → skip second).
                if (seen_sources.contains(doc.meta.source)) continue;

                const module_lower = try std.ascii.allocLowerString(alloc, doc.meta.module);
                defer alloc.free(module_lower);
                const source_lower = try std.ascii.allocLowerString(alloc, doc.meta.source);
                defer alloc.free(source_lower);
                const fp_lower = try std.ascii.allocLowerString(alloc, fp);
                defer alloc.free(fp_lower);

                var filename_match = false;
                for (terms) |term| {
                    if (std.mem.indexOf(u8, module_lower, term) != null or
                        std.mem.indexOf(u8, source_lower, term) != null or
                        std.mem.indexOf(u8, fp_lower, term) != null)
                    {
                        filename_match = true;
                        break;
                    }
                }
                if (!filename_match) continue;

                // Update satisfied for all terms matched in this doc.
                _ = try docMatchesTerms(alloc, doc, terms, satisfied);
                const ed = try dupeExploreDoc(alloc, st, doc, fp, true);
                try primary_docs.append(alloc, ed);
                try seen_json.put(alloc, fp, {});
                // Store the owned source from the duped ExploreDoc as the dedup key.
                try seen_sources.put(alloc, ed.source, {});
            }

            // Secondary pass: member-name match for unsatisfied terms.
            for (json_paths) |fp| {
                if (secondary_docs.items.len >= 3) break;
                if (seen_json.contains(fp)) continue;
                const doc = (try st.loadGuidance(fp)) orelse continue;
                defer st.freeGuidanceDoc(doc);

                // Also deduplicate by source.
                if (seen_sources.contains(doc.meta.source)) continue;

                // Only pick up if it matches an unsatisfied term via member names.
                var member_matched_unsatisfied = false;
                for (doc.members) |m| {
                    const mname_lower = try std.ascii.allocLowerString(alloc, m.name);
                    defer alloc.free(mname_lower);
                    for (terms, 0..) |term, ti| {
                        if (!satisfied[ti] and std.mem.indexOf(u8, mname_lower, term) != null) {
                            member_matched_unsatisfied = true;
                            satisfied[ti] = true;
                        }
                    }
                }
                if (!member_matched_unsatisfied) continue;

                const ed = try dupeExploreDoc(alloc, st, doc, fp, false);
                try secondary_docs.append(alloc, ed);
                try seen_json.put(alloc, fp, {});
                try seen_sources.put(alloc, ed.source, {});
            }

            return .{
                .primary = try primary_docs.toOwnedSlice(alloc),
                .secondary = try secondary_docs.toOwnedSlice(alloc),
                .satisfied = satisfied,
            };
        }
    }.call;

    var gather = try runGather(allocator, &store, all_json_paths.items, search_terms.items);
    var primary_docs = gather.primary;
    var secondary_docs = gather.secondary;
    defer {
        for (primary_docs) |*ed| ed.deinit(allocator, &store);
        allocator.free(primary_docs);
        for (secondary_docs) |*ed| ed.deinit(allocator, &store);
        allocator.free(secondary_docs);
        allocator.free(gather.satisfied);
    }

    // -------------------------------------------------------------------------
    // M2.4 — LLM-driven retry loop when no results found.
    // -------------------------------------------------------------------------
    if (primary_docs.len == 0 and secondary_docs.len == 0 and llm_available and !no_ai) {
        var tried_terms: std.StringHashMapUnmanaged(void) = .{};
        defer tried_terms.deinit(allocator);
        for (search_terms.items) |t| try tried_terms.put(allocator, t, {});

        var retry: usize = 0;
        while (retry < 5 and primary_docs.len == 0 and secondary_docs.len == 0) : (retry += 1) {
            if (llm_client_opt) |*client| {
                const retry_prompt = try std.fmt.allocPrint(
                    allocator,
                    "Given query '{s}' returned no results. Generate 3-5 alternative search terms (function names, type names) one per line.",
                    .{q},
                );
                defer allocator.free(retry_prompt);

                const resp = client.complete(retry_prompt, 80, 0.3, null) catch break orelse break;
                defer allocator.free(resp);

                var new_terms: std.ArrayList([]const u8) = .{};
                defer {
                    for (new_terms.items) |t| allocator.free(t);
                    new_terms.deinit(allocator);
                }

                var resp_lines = std.mem.splitScalar(u8, resp, '\n');
                var added: usize = 0;
                while (resp_lines.next()) |line| {
                    if (added >= 2 or tried_terms.count() >= 12) break;
                    const trimmed_id = std.mem.trim(u8, line, " \t\r-.");
                    if (trimmed_id.len == 0 or trimmed_id.len > 80) continue;
                    const lower_id = try std.ascii.allocLowerString(allocator, trimmed_id);
                    if (tried_terms.contains(lower_id)) {
                        allocator.free(lower_id);
                        continue;
                    }
                    try tried_terms.put(allocator, lower_id, {});
                    try new_terms.append(allocator, lower_id);
                    added += 1;
                }

                if (new_terms.items.len == 0) break;

                // Add new terms to search_terms and re-gather.
                for (new_terms.items) |t| {
                    try search_terms.append(allocator, try allocator.dupe(u8, t));
                }

                const retry_gather = try runGather(allocator, &store, all_json_paths.items, search_terms.items);
                // Replace primary/secondary docs.
                for (primary_docs) |*ed| ed.deinit(allocator, &store);
                allocator.free(primary_docs);
                for (secondary_docs) |*ed| ed.deinit(allocator, &store);
                allocator.free(secondary_docs);
                allocator.free(gather.satisfied);
                gather = retry_gather;
                primary_docs = gather.primary;
                secondary_docs = gather.secondary;
            }
        }
    }

    const found = primary_docs.len > 0 or secondary_docs.len > 0;
    if (!found) {
        std.debug.print("No guidance JSON matched '{s}'.\n", .{q});
        std.debug.print("Try: grep -ri '{s}' src/\n", .{q});
        return;
    }

    // -------------------------------------------------------------------------
    // M2.5 — Source excerpt extraction per matched member.
    // -------------------------------------------------------------------------

    // Collect source excerpts for all docs: source_path → list of (member_line, excerpt).
    // We'll store them per ExploreDoc for rendering.
    const DocExcerpt = struct {
        member_name: []const u8,
        excerpt: []const u8,
    };
    const DocExcerpts = struct {
        source: []const u8,
        items: []DocExcerpt,
    };

    var all_excerpts: std.ArrayList(DocExcerpts) = .{};
    defer {
        for (all_excerpts.items) |de| {
            for (de.items) |it| {
                allocator.free(it.member_name);
                allocator.free(it.excerpt);
            }
            allocator.free(de.items);
        }
        all_excerpts.deinit(allocator);
    }

    const all_docs_combined: [2][]const ExploreDoc = .{ primary_docs, secondary_docs };
    for (all_docs_combined) |doc_slice| {
        for (doc_slice) |ed| {
            const src_abs = try std.fs.path.join(allocator, &.{ cwd, ed.source });
            defer allocator.free(src_abs);

            const src_content_opt: ?[]const u8 = blk: {
                const sf = std.fs.openFileAbsolute(src_abs, .{}) catch break :blk null;
                defer sf.close();
                break :blk sf.readToEndAlloc(allocator, 10 * 1024 * 1024) catch null;
            };
            defer if (src_content_opt) |sc| allocator.free(sc);

            var exc_list: std.ArrayList(DocExcerpt) = .{};
            var shown: usize = 0;
            for (ed.members) |m| {
                if (shown >= 3) break;
                // Only emit excerpt for members matching a search term.
                const mname_lower = try std.ascii.allocLowerString(allocator, m.name);
                defer allocator.free(mname_lower);
                var term_match = false;
                for (search_terms.items) |term| {
                    if (std.mem.indexOf(u8, mname_lower, term) != null) {
                        term_match = true;
                        break;
                    }
                }
                if (!term_match) continue;

                const excerpt = if (src_content_opt) |src_content|
                    try extractSourceExcerpt(allocator, src_content, m.line orelse 0, 80)
                else
                    try allocator.dupe(u8, "");

                try exc_list.append(allocator, .{
                    .member_name = try allocator.dupe(u8, m.name),
                    .excerpt = excerpt,
                });
                shown += 1;
            }
            try all_excerpts.append(allocator, .{
                .source = ed.source,
                .items = try exc_list.toOwnedSlice(allocator),
            });
        }
    }

    // -------------------------------------------------------------------------
    // M2.6 — Grep phase.
    // -------------------------------------------------------------------------
    const GrepFileResult = struct {
        path: []const u8, // NOT owned (points into ed.source via cwd join)
        matches: []GrepMatch,
    };
    var grep_results: std.ArrayList(GrepFileResult) = .{};
    defer {
        for (grep_results.items) |gr| {
            for (gr.matches) |gm| allocator.free(gm.line);
            allocator.free(gr.matches);
        }
        grep_results.deinit(allocator);
    }

    {
        // Collect unique source paths to grep.
        var src_paths_seen: std.StringHashMapUnmanaged(void) = .{};
        defer src_paths_seen.deinit(allocator);

        for (all_docs_combined) |doc_slice| {
            for (doc_slice) |ed| {
                // Use the relative source path as the dedup key (stable pointer, owned by ed).
                if (src_paths_seen.contains(ed.source)) continue;
                try src_paths_seen.put(allocator, ed.source, {});

                const abs = try std.fs.path.join(allocator, &.{ cwd, ed.source });
                defer allocator.free(abs);
                const matches = try grepFile(allocator, abs, search_terms.items, 5);
                if (matches.len > 0) {
                    try grep_results.append(allocator, .{ .path = ed.source, .matches = matches });
                } else {
                    allocator.free(matches);
                }
            }
        }
        // Sort by match count descending; cap at 3 files.
        if (grep_results.items.len > 3) {
            // Simple bubble sort for small count.
            var gi: usize = 0;
            while (gi < grep_results.items.len) : (gi += 1) {
                var gj = gi + 1;
                while (gj < grep_results.items.len) : (gj += 1) {
                    if (grep_results.items[gj].matches.len > grep_results.items[gi].matches.len) {
                        const tmp = grep_results.items[gi];
                        grep_results.items[gi] = grep_results.items[gj];
                        grep_results.items[gj] = tmp;
                    }
                }
            }
            // Free and remove items beyond index 2.
            var gi2: usize = 3;
            while (gi2 < grep_results.items.len) : (gi2 += 1) {
                for (grep_results.items[gi2].matches) |gm| allocator.free(gm.line);
                allocator.free(grep_results.items[gi2].matches);
            }
            grep_results.shrinkRetainingCapacity(3);
        }
    }

    // -------------------------------------------------------------------------
    // M2.7 — Skill excerpt loading.
    // -------------------------------------------------------------------------
    const SkillExcerpt = struct {
        name: []const u8,
        excerpt: []const u8,
    };
    var skill_excerpts: std.ArrayList(SkillExcerpt) = .{};
    defer {
        for (skill_excerpts.items) |se| {
            allocator.free(se.name);
            allocator.free(se.excerpt);
        }
        skill_excerpts.deinit(allocator);
    }
    {
        var seen_skills: std.StringHashMapUnmanaged(void) = .{};
        defer seen_skills.deinit(allocator);
        for (all_docs_combined) |doc_slice| {
            for (doc_slice) |ed| {
                for (ed.skills) |sk| {
                    if (seen_skills.contains(sk.ref)) continue;
                    try seen_skills.put(allocator, sk.ref, {});

                    // Derive skill short name.
                    const slash = std.mem.lastIndexOfScalar(u8, sk.ref, '/') orelse 0;
                    const after_slash = if (slash > 0) sk.ref[slash + 1 ..] else sk.ref;
                    // If ref ends in /SKILL.md, get parent dir name.
                    const skill_name = if (std.mem.eql(u8, after_slash, "SKILL.md")) blk: {
                        const ref_trimmed = sk.ref[0..slash];
                        const parent_slash = std.mem.lastIndexOfScalar(u8, ref_trimmed, '/') orelse 0;
                        break :blk if (parent_slash > 0) ref_trimmed[parent_slash + 1 ..] else ref_trimmed;
                    } else after_slash;

                    // Try {guidance_dir}/.skills/<name>/SKILL.md first, then doc/skills/.
                    const path1 = try std.fs.path.join(allocator, &.{ guidance_dir, ".skills", skill_name, "SKILL.md" });
                    defer allocator.free(path1);
                    const path2 = try std.fs.path.join(allocator, &.{ cwd, "doc", "skills", skill_name, "SKILL.md" });
                    defer allocator.free(path2);

                    const skill_path = if (std.fs.openFileAbsolute(path1, .{})) |sf| blk: {
                        sf.close();
                        break :blk path1;
                    } else |_| path2;

                    const sf = std.fs.openFileAbsolute(skill_path, .{}) catch continue;
                    defer sf.close();
                    const content = sf.readToEndAlloc(allocator, 512 * 1024) catch continue;
                    defer allocator.free(content);

                    // Extract first paragraph (up to first blank line, max 500 chars).
                    const para_end = std.mem.indexOf(u8, content, "\n\n") orelse content.len;
                    const para = content[0..@min(para_end, 500)];
                    try skill_excerpts.append(allocator, .{
                        .name = try allocator.dupe(u8, skill_name),
                        .excerpt = try allocator.dupe(u8, para),
                    });
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // M2.8 — AI synthesis pass.
    // -------------------------------------------------------------------------
    var ai_summary: ?[]const u8 = null;
    defer if (ai_summary) |s| allocator.free(s);

    if (llm_available and !no_ai) {
        // Build knowledge block.
        var kb: std.ArrayList(u8) = .{};
        defer kb.deinit(allocator);
        const kbw = kb.writer(allocator);

        for (all_docs_combined) |doc_slice| {
            for (doc_slice) |ed| {
                try kbw.print("## Module: {s}\n", .{ed.module});
                if (ed.comment) |c| try kbw.print("{s}\n\n", .{c});
                for (ed.members) |m| {
                    const mname_lower = try std.ascii.allocLowerString(allocator, m.name);
                    defer allocator.free(mname_lower);
                    var term_match = false;
                    for (search_terms.items) |term| {
                        if (std.mem.indexOf(u8, mname_lower, term) != null) {
                            term_match = true;
                            break;
                        }
                    }
                    if (!term_match) continue;
                    try kbw.print("### {s}", .{m.name});
                    if (m.signature) |sig| try kbw.print("\n`{s}`", .{sig});
                    if (m.comment) |c| try kbw.print("\n{s}", .{c});
                    try kbw.print("\n\n", .{});
                }
            }
        }

        // Source excerpts.
        for (all_excerpts.items) |de| {
            for (de.items) |it| {
                if (it.excerpt.len == 0) continue;
                try kbw.print("### Source: {s} / {s}\n```zig\n{s}\n```\n\n", .{ de.source, it.member_name, it.excerpt });
            }
        }

        // Grep matches.
        if (grep_results.items.len > 0) {
            try kbw.print("### Grep matches\n", .{});
            for (grep_results.items) |gr| {
                try kbw.print("File: {s}\n", .{gr.path});
                for (gr.matches) |gm| {
                    try kbw.print("  L{}: {s}\n", .{ gm.line_no, gm.line });
                }
            }
            try kbw.print("\n", .{});
        }

        // Skill excerpts.
        if (skill_excerpts.items.len > 0) {
            try kbw.print("### Skill excerpts\n", .{});
            for (skill_excerpts.items) |se| {
                try kbw.print("**{s}**: {s}\n\n", .{ se.name, se.excerpt });
            }
        }

        // Inbox bullets.
        const inbox_names = [_][]const u8{ "INSIGHTS.md", "CAPABILITIES.md" };
        for (inbox_names) |inbox_name| {
            const inbox_path = try std.fs.path.join(allocator, &.{ guidance_dir, ".doc", "inbox", inbox_name });
            defer allocator.free(inbox_path);
            const inbox_file = std.fs.openFileAbsolute(inbox_path, .{}) catch continue;
            defer inbox_file.close();
            const content = inbox_file.readToEndAlloc(allocator, 1024 * 1024) catch continue;
            defer allocator.free(content);
            var lines_it = std.mem.splitScalar(u8, content, '\n');
            while (lines_it.next()) |line| {
                const stripped = std.mem.trim(u8, line, " \t\r");
                if (!std.mem.startsWith(u8, stripped, "- ")) continue;
                for (search_terms.items) |term| {
                    const stripped_lower = try std.ascii.allocLowerString(allocator, stripped);
                    defer allocator.free(stripped_lower);
                    if (std.mem.indexOf(u8, stripped_lower, term) != null) {
                        try kbw.print("{s}\n", .{stripped});
                        break;
                    }
                }
            }
        }

        // Build the final LLM prompt.
        const kb_text = kb.items;
        const primary_term = search_terms.items[0];
        const prompt = if (focus_question) |fq|
            try std.fmt.allocPrint(allocator, "{s}\n\nQuestion: {s}\nAnswer concisely using only information from the knowledge block above.", .{ kb_text, fq })
        else
            try std.fmt.allocPrint(allocator, "{s}\n\nSummarise what {s} is, what it does, and when to use it.\nBe concise (3-5 sentences). Do not claim absence of features not shown.", .{ kb_text, primary_term });
        defer allocator.free(prompt);

        if (llm_client_opt) |*client| {
            if (client.complete(prompt, 400, 0.3, null) catch null) |raw_summary| {
                // Post-process: strip absence-claim sentences.
                var summary_buf: std.ArrayList(u8) = .{};
                // No defer deinit here — toOwnedSlice transfers ownership to ai_summary.
                const swb = summary_buf.writer(allocator);
                var sentence_iter = std.mem.splitScalar(u8, raw_summary, '\n');
                while (sentence_iter.next()) |sentence| {
                    const s_lower = try std.ascii.allocLowerString(allocator, sentence);
                    defer allocator.free(s_lower);
                    const is_absence = std.mem.indexOf(u8, s_lower, "no other") != null or
                        std.mem.indexOf(u8, s_lower, "not present") != null or
                        std.mem.indexOf(u8, s_lower, "only has") != null or
                        std.mem.indexOf(u8, s_lower, "does not contain") != null;
                    if (!is_absence) {
                        try swb.print("{s}\n", .{sentence});
                    }
                }
                allocator.free(raw_summary);
                ai_summary = try summary_buf.toOwnedSlice(allocator);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Render output in requested format.
    // -------------------------------------------------------------------------
    var out_buf: [65536]u8 = undefined;
    var fw_explore = std.fs.File.stdout().writer(&out_buf);
    const stdout = &fw_explore.interface;

    switch (format) {

        // -----------------------------------------------------------------------
        // compact: one line per matched member
        // -----------------------------------------------------------------------
        .compact => {
            for (all_docs_combined) |doc_slice| {
                for (doc_slice) |ed| {
                    for (ed.members) |m| {
                        const mname_lower = try std.ascii.allocLowerString(allocator, m.name);
                        defer allocator.free(mname_lower);
                        var term_match = false;
                        for (search_terms.items) |term| {
                            if (std.mem.indexOf(u8, mname_lower, term) != null) {
                                term_match = true;
                                break;
                            }
                        }
                        if (!term_match) continue;
                        try stdout.print("{s}:{} {s}", .{
                            ed.source,
                            m.line orelse 0,
                            m.name,
                        });
                        if (m.comment) |c| {
                            const nl = std.mem.indexOfScalar(u8, c, '\n') orelse c.len;
                            try stdout.print(" — {s}", .{c[0..nl]});
                        }
                        try stdout.print("\n", .{});
                    }
                }
            }
        },

        // -----------------------------------------------------------------------
        // json: machine-readable JSON object
        // -----------------------------------------------------------------------
        .json => {
            try stdout.print("{{\n", .{});
            // search_terms
            try stdout.print("  \"search_terms\": [", .{});
            for (search_terms.items, 0..) |t, ti| {
                if (ti > 0) try stdout.print(", ", .{});
                try stdout.print("\"", .{});
                try stdout.print("{s}", .{t});
                try stdout.print("\"", .{});
            }
            try stdout.print("],\n", .{});
            // docs
            try stdout.print("  \"docs\": [\n", .{});
            var doc_printed_any = false;
            for (all_docs_combined) |doc_slice| {
                for (doc_slice) |ed| {
                    if (doc_printed_any) try stdout.print(",\n", .{});
                    try stdout.print("    {{\"source\":\"{s}\",\"module\":\"{s}\",\"is_primary\":{},", .{
                        ed.source, ed.module, ed.is_primary,
                    });
                    try stdout.print("\"comment\":", .{});
                    if (ed.comment) |c| {
                        try stdout.print("\"", .{});
                        for (c) |byte| {
                            if (byte == '"') try stdout.print("\\\"", .{}) else if (byte == '\\') try stdout.print("\\\\", .{}) else if (byte == '\n') try stdout.print("\\n", .{}) else try stdout.writeByte(byte);
                        }
                        try stdout.print("\",", .{});
                    } else {
                        try stdout.print("null,", .{});
                    }
                    // matched_members
                    try stdout.print("\"matched_members\":[", .{});
                    var first_mm = true;
                    for (ed.members) |m| {
                        const mname_lower = try std.ascii.allocLowerString(allocator, m.name);
                        defer allocator.free(mname_lower);
                        var term_match = false;
                        for (search_terms.items) |term| {
                            if (std.mem.indexOf(u8, mname_lower, term) != null) {
                                term_match = true;
                                break;
                            }
                        }
                        if (!term_match) continue;
                        if (!first_mm) try stdout.print(",", .{});
                        first_mm = false;
                        try stdout.print("{{\"name\":\"{s}\"", .{m.name});
                        if (m.line) |l| try stdout.print(",\"line\":{}", .{l});
                        try stdout.print("}}", .{});
                    }
                    try stdout.print("]}}", .{});
                    doc_printed_any = true;
                }
            }
            try stdout.print("\n  ],\n", .{});
            // ai_summary
            try stdout.print("  \"ai_summary\":", .{});
            if (ai_summary) |s| {
                try stdout.print("\"", .{});
                for (s) |byte| {
                    if (byte == '"') try stdout.print("\\\"", .{}) else if (byte == '\\') try stdout.print("\\\\", .{}) else if (byte == '\n') try stdout.print("\\n", .{}) else try stdout.writeByte(byte);
                }
                try stdout.print("\"\n", .{});
            } else {
                try stdout.print("null\n", .{});
            }
            try stdout.print("}}\n", .{});
        },

        // -----------------------------------------------------------------------
        // markdown (default)
        // -----------------------------------------------------------------------
        .markdown => {
            // AI summary prepended.
            if (ai_summary) |s| {
                try stdout.print("\n## Summary\n\n{s}\n\n---\n", .{s});
            }

            for (all_docs_combined) |doc_slice| {
                for (doc_slice, 0..) |ed, ei| {
                    const pri_label: []const u8 = if (ed.is_primary) "" else " (secondary)";
                    try stdout.print("\n# {s}{s}: {s}\n\n", .{ if (ei == 0 and ed.is_primary) "Explore" else "Related", pri_label, q });
                    try stdout.print("## Module: {s}\n", .{ed.module});
                    try stdout.print("## Source: {s} [language: {s}]\n", .{ ed.source, ed.language });

                    if (ed.comment) |d| {
                        try stdout.print("\n{s}\n", .{d});
                    }

                    if (ed.skills.len > 0) {
                        try stdout.print("\n## Skills\n", .{});
                        for (ed.skills) |skill| {
                            try stdout.print("- `{s}`", .{skill.ref});
                            if (skill.context) |ctx| try stdout.print(" — {s}", .{ctx});
                            try stdout.print("\n", .{});
                        }
                    }

                    if (ed.hashtags.len > 0) {
                        try stdout.print("\n## Hashtags\n", .{});
                        for (ed.hashtags) |tag| try stdout.print("- {s}\n", .{tag});
                    }

                    if (ed.used_by.len > 0) {
                        try stdout.print("\n## Used By\n", .{});
                        for (ed.used_by) |u| try stdout.print("- {s}\n", .{u});
                    }

                    if (ed.members.len > 0) {
                        try stdout.print("\n### Members\n", .{});
                        for (ed.members) |m| {
                            const pub_str: []const u8 = if (m.is_pub) "pub " else "";
                            try stdout.print("- `{s}{s}` ({s})", .{ pub_str, m.name, @tagName(m.type) });
                            if (m.line) |l| try stdout.print(" (line {})", .{l});
                            if (m.comment) |d| {
                                const nl = std.mem.indexOfScalar(u8, d, '\n') orelse d.len;
                                try stdout.print(" — {s}", .{d[0..nl]});
                            }
                            try stdout.print("\n", .{});
                            if (m.signature) |sig| try stdout.print("  `{s}`\n", .{sig});
                            if (m.patterns.len > 0) {
                                try stdout.print("  Patterns:", .{});
                                for (m.patterns) |p| try stdout.print(" {s}", .{p.name});
                                try stdout.print("\n", .{});
                            }
                            for (m.members) |mm| {
                                const mm_pub: []const u8 = if (mm.is_pub) "pub " else "";
                                try stdout.print("  - `{s}{s}` ({s})", .{ mm_pub, mm.name, @tagName(mm.type) });
                                if (mm.line) |l| try stdout.print(" (line {})", .{l});
                                if (mm.comment) |d| {
                                    const nl = std.mem.indexOfScalar(u8, d, '\n') orelse d.len;
                                    try stdout.print(" — {s}", .{d[0..nl]});
                                }
                                try stdout.print("\n", .{});
                                if (mm.signature) |sig| try stdout.print("    `{s}`\n", .{sig});
                            }
                        }
                    }

                    // Source excerpts for this doc.
                    for (all_excerpts.items) |de| {
                        if (!std.mem.eql(u8, de.source, ed.source)) continue;
                        if (de.items.len == 0) continue;
                        try stdout.print("\n### Source Excerpts\n", .{});
                        for (de.items) |it| {
                            if (it.excerpt.len == 0) continue;
                            try stdout.print("\n**{s}**\n```zig\n{s}\n```\n", .{ it.member_name, it.excerpt });
                        }
                        const extra = ed.members.len -| 3;
                        if (extra > 0) try stdout.print("({} more members)\n", .{extra});
                    }
                }
            }

            // Grep matches section.
            if (grep_results.items.len > 0) {
                try stdout.print("\n### Grep matches\n", .{});
                for (grep_results.items) |gr| {
                    try stdout.print("\n**{s}**\n", .{gr.path});
                    for (gr.matches) |gm| {
                        try stdout.print("  L{d}: {s}\n", .{ gm.line_no, gm.line });
                    }
                }
            }

            // Skill excerpts section.
            if (skill_excerpts.items.len > 0) {
                try stdout.print("\n### Skill Excerpts\n", .{});
                for (skill_excerpts.items) |se| {
                    try stdout.print("\n**{s}**\n{s}\n", .{ se.name, se.excerpt });
                }
            }

            // Inbox bullets matching query.
            {
                const inbox_names2 = [_][]const u8{ "INSIGHTS.md", "CAPABILITIES.md" };
                var inbox_header_printed = false;
                for (inbox_names2) |inbox_name| {
                    const inbox_path = try std.fs.path.join(allocator, &.{ guidance_dir, ".doc", "inbox", inbox_name });
                    defer allocator.free(inbox_path);
                    const inbox_file = std.fs.openFileAbsolute(inbox_path, .{}) catch continue;
                    defer inbox_file.close();
                    const content = inbox_file.readToEndAlloc(allocator, 1024 * 1024) catch continue;
                    defer allocator.free(content);
                    var lines_it2 = std.mem.splitScalar(u8, content, '\n');
                    while (lines_it2.next()) |line| {
                        const stripped = std.mem.trim(u8, line, " \t\r");
                        if (!std.mem.startsWith(u8, stripped, "- ")) continue;
                        for (search_terms.items) |term| {
                            const stripped_lower = try std.ascii.allocLowerString(allocator, stripped);
                            defer allocator.free(stripped_lower);
                            if (std.mem.indexOf(u8, stripped_lower, term) != null) {
                                if (!inbox_header_printed) {
                                    try stdout.print("\n## Recent Knowledge\n", .{});
                                    inbox_header_printed = true;
                                }
                                try stdout.print("{s}\n", .{stripped});
                                break;
                            }
                        }
                    }
                }
            }

            // Source/skill ref footer.
            try stdout.print("\n---\nSources:", .{});
            for (all_docs_combined) |doc_slice| {
                for (doc_slice) |ed| try stdout.print(" {s}", .{ed.source});
            }
            try stdout.print("\n", .{});
            if (skill_excerpts.items.len > 0) {
                try stdout.print("Skills:", .{});
                for (skill_excerpts.items) |se| try stdout.print(" {s}", .{se.name});
                try stdout.print("\n", .{});
            }
        },
    }

    try stdout.flush();
}

// =============================================================================
// explain (Python-style)
// =============================================================================

fn cmdExplain(allocator: std.mem.Allocator, args: []const []const u8) !void {
    // ── Argument parsing ──────────────────────────────────────────────────────
    const common = llm.parseCommonArgs(args);
    var query_str: ?[]const u8 = null;
    var guidance_dir_arg: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--guidance")) {
            i += 1;
            if (i >= args.len) return;
            guidance_dir_arg = args[i];
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            query_str = arg;
        }
    }
    const debug = common.debug;
    const no_ai = common.no_ai;
    const api_url = common.api_url;
    const model = common.model;

    const q = query_str orelse {
        std.debug.print("Error: query string required\n", .{});
        return;
    };

    if (guidance_dir_arg == null) {
        std.debug.print("❌ Error: --guidance is required. Usage: ast-guidance explain <query> --guidance <dir> [options]\n", .{});
        return;
    }

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    // ── Bracket syntax: "[term] focus question" ──────────────────────────────
    var focus_question: ?[]const u8 = null;
    var raw_search_query: []const u8 = q;
    if (q.len > 0 and q[0] == '[') {
        if (std.mem.indexOfScalar(u8, q, ']')) |bracket_end| {
            raw_search_query = q[1..bracket_end];
            const after = std.mem.trim(u8, q[bracket_end + 1 ..], " \t");
            if (after.len > 0) focus_question = after;
        }
    }

    // LLM client
    var llm_client_opt: ?llm.LlmClient = if (!no_ai)
        llm.LlmClient.init(allocator, .{ .api_url = api_url, .model = model, .debug = debug }) catch null
    else
        null;
    defer if (llm_client_opt) |*c| c.deinit();
    const llm_available = if (llm_client_opt) |*c| c.available() else false;

    // Search terms
    var search_terms: std.ArrayList([]const u8) = .{};
    defer {
        for (search_terms.items) |t| allocator.free(t);
        search_terms.deinit(allocator);
    }

    const has_spaces = std.mem.indexOfScalar(u8, raw_search_query, ' ') != null;
    if (has_spaces and llm_available and !no_ai) {
        if (llm_client_opt) |*client| {
            const decompose_prompt = try std.fmt.allocPrint(
                allocator,
                "Extract the 1-3 most important code identifiers from this query.\nQuery: {s}\nReturn ONLY identifiers, one per line, most important first.",
                .{raw_search_query},
            );
            defer allocator.free(decompose_prompt);

            if (client.complete(decompose_prompt, 60, 0.1, null) catch null) |resp| {
                defer allocator.free(resp);
                var resp_lines = std.mem.splitScalar(u8, resp, '\n');
                var count: usize = 0;
                while (resp_lines.next()) |line| {
                    if (count >= 3) break;
                    const trimmed_id = std.mem.trim(u8, line, " \t\r-.");
                    if (trimmed_id.len == 0 or trimmed_id.len > 80) continue;
                    const lower_id = try std.ascii.allocLowerString(allocator, trimmed_id);
                    try search_terms.append(allocator, lower_id);
                    count += 1;
                }
                if (focus_question == null) focus_question = raw_search_query;
            }
        }
    }

    if (search_terms.items.len == 0) {
        const lower_q = try std.ascii.allocLowerString(allocator, raw_search_query);
        try search_terms.append(allocator, lower_q);
    }

    // ── Guidance directory scan ───────────────────────────────────────────────
    const guidance_dir = if (std.fs.path.isAbsolute(guidance_dir_arg.?))
        try allocator.dupe(u8, guidance_dir_arg.?)
    else
        try std.fs.path.join(allocator, &.{ cwd, guidance_dir_arg.? });
    defer allocator.free(guidance_dir);

    var gdir = std.fs.openDirAbsolute(guidance_dir, .{ .iterate = true }) catch {
        std.debug.print("No guidance directory found at {s}\n", .{guidance_dir});
        return;
    };
    defer gdir.close();

    var store = @import("json_store.zig").JsonStore.init(allocator);

    var all_json_paths: std.ArrayList([]const u8) = .{};
    defer {
        for (all_json_paths.items) |p| allocator.free(p);
        all_json_paths.deinit(allocator);
    }
    {
        var walker_scan = try gdir.walk(allocator);
        defer walker_scan.deinit();
        while (try walker_scan.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".json")) continue;
            const fp = try std.fs.path.join(allocator, &.{ guidance_dir, entry.path });
            try all_json_paths.append(allocator, fp);
        }
    }

    // ── Gather: primary (filename/module match) + secondary (member match) ────
    const GatherResult = struct {
        primary: []ExploreDoc,
        secondary: []ExploreDoc,
        satisfied: []bool,
    };

    const runGather = struct {
        fn call(
            alloc: std.mem.Allocator,
            st: *@import("json_store.zig").JsonStore,
            json_paths: []const []const u8,
            terms: []const []const u8,
        ) !GatherResult {
            var primary_docs: std.ArrayList(ExploreDoc) = .{};
            errdefer {
                for (primary_docs.items) |*ed| ed.deinit(alloc, st);
                primary_docs.deinit(alloc);
            }
            var secondary_docs: std.ArrayList(ExploreDoc) = .{};
            errdefer {
                for (secondary_docs.items) |*ed| ed.deinit(alloc, st);
                secondary_docs.deinit(alloc);
            }

            const satisfied = try alloc.alloc(bool, terms.len);
            @memset(satisfied, false);

            var seen_json: std.StringHashMapUnmanaged(void) = .{};
            defer seen_json.deinit(alloc);
            var seen_sources: std.StringHashMapUnmanaged(void) = .{};
            defer seen_sources.deinit(alloc);

            // Primary pass: filename / module / source path matches the query.
            for (json_paths) |fp| {
                if (primary_docs.items.len >= 3) break;
                const doc = (try st.loadGuidance(fp)) orelse continue;
                defer st.freeGuidanceDoc(doc);

                if (seen_sources.contains(doc.meta.source)) continue;

                const module_lower = try std.ascii.allocLowerString(alloc, doc.meta.module);
                defer alloc.free(module_lower);
                const source_lower = try std.ascii.allocLowerString(alloc, doc.meta.source);
                defer alloc.free(source_lower);
                const fp_lower = try std.ascii.allocLowerString(alloc, fp);
                defer alloc.free(fp_lower);

                var filename_match = false;
                for (terms) |term| {
                    if (std.mem.indexOf(u8, module_lower, term) != null or
                        std.mem.indexOf(u8, source_lower, term) != null or
                        std.mem.indexOf(u8, fp_lower, term) != null)
                    {
                        filename_match = true;
                        break;
                    }
                }
                if (!filename_match) continue;

                _ = try docMatchesTerms(alloc, doc, terms, satisfied);
                const ed = try dupeExploreDoc(alloc, st, doc, fp, true);
                try primary_docs.append(alloc, ed);
                try seen_json.put(alloc, fp, {});
                try seen_sources.put(alloc, ed.source, {});
            }

            // Secondary pass
            for (json_paths) |fp| {
                if (secondary_docs.items.len >= 3) break;
                if (seen_json.contains(fp)) continue;
                const doc = (try st.loadGuidance(fp)) orelse continue;
                defer st.freeGuidanceDoc(doc);

                if (seen_sources.contains(doc.meta.source)) continue;

                var member_matched_unsatisfied = false;
                for (doc.members) |m| {
                    const mname_lower = try std.ascii.allocLowerString(alloc, m.name);
                    defer alloc.free(mname_lower);
                    for (terms, 0..) |term, ti| {
                        if (!satisfied[ti] and std.mem.indexOf(u8, mname_lower, term) != null) {
                            member_matched_unsatisfied = true;
                            satisfied[ti] = true;
                        }
                    }
                }
                if (!member_matched_unsatisfied) continue;

                const ed = try dupeExploreDoc(alloc, st, doc, fp, false);
                try secondary_docs.append(alloc, ed);
                try seen_json.put(alloc, fp, {});
                try seen_sources.put(alloc, ed.source, {});
            }

            return .{
                .primary = try primary_docs.toOwnedSlice(alloc),
                .secondary = try secondary_docs.toOwnedSlice(alloc),
                .satisfied = satisfied,
            };
        }
    }.call;

    const gather = try runGather(allocator, &store, all_json_paths.items, search_terms.items);
    const primary_docs = gather.primary;
    const secondary_docs = gather.secondary;
    defer {
        for (primary_docs) |*ed| ed.deinit(allocator, &store);
        allocator.free(primary_docs);
        for (secondary_docs) |*ed| ed.deinit(allocator, &store);
        allocator.free(secondary_docs);
        allocator.free(gather.satisfied);
    }

    const all_docs_combined: [2][]const ExploreDoc = .{ primary_docs, secondary_docs };
    const found = primary_docs.len > 0 or secondary_docs.len > 0;

    if (!found) {
        std.debug.print("# Explain: {s}\n\n", .{q});
        if (search_terms.items.len > 1) {
            std.debug.print("Not yet indexed for terms: ", .{});
            for (search_terms.items, 0..) |t, idx| {
                if (idx > 0) std.debug.print(", ", .{});
                std.debug.print("{s}", .{t});
            }
            std.debug.print(". Search the source directly:\n", .{});
        } else {
            std.debug.print("Not yet indexed for '{s}'. Search the source directly:\n", .{raw_search_query});
        }
        std.debug.print("\n", .{});
        for (search_terms.items[0..@min(3, search_terms.items.len)]) |t| {
            std.debug.print("    grep -ri '{s}' src/ | head -n 20\n", .{t});
        }
        std.debug.print("\nRun `make guidance` after finding the file to index it.\n", .{});
        return;
    }

    // ── PHASE A: Skill loading (Proposal 2 — before synthesis) ───────────────
    // Read skills[] from each matched guidance doc, then load the first paragraph
    // of the corresponding SKILL.md file.  Also parse "[skill-name]" comment
    // prefix as a fallback when the skills[] array is empty.
    // Skills are passed to the LLM prompt so it can name the pattern in its summary.
    const SkillExcerpt = struct { name: []const u8, excerpt: []const u8 };
    var skill_excerpts: std.ArrayList(SkillExcerpt) = .{};
    defer {
        for (skill_excerpts.items) |se| {
            allocator.free(se.name);
            allocator.free(se.excerpt);
        }
        skill_excerpts.deinit(allocator);
    }
    {
        // Deduplication key = short skill name (slice into ExploreDoc or const string).
        var seen_skills: std.StringHashMapUnmanaged(void) = .{};
        defer seen_skills.deinit(allocator);

        // Helper: load first paragraph of a SKILL.md file (up to 600 chars).
        // Searches {guidance_dir}/.skills/ and doc/skills/ in order.
        // Returns an owned allocation or null if not found.
        const loadSkillPara = struct {
            fn call(alloc: std.mem.Allocator, guidance_dir_path: []const u8, cwd_path: []const u8, skill_name: []const u8) ?[]const u8 {
                const prefixes = [_]struct { base: []const u8, rel: []const u8 }{
                    .{ .base = guidance_dir_path, .rel = ".skills" },
                    .{ .base = cwd_path, .rel = "doc/skills" },
                };
                for (prefixes) |prefix_info| {
                    const path = std.fs.path.join(alloc, &.{ prefix_info.base, prefix_info.rel, skill_name, "SKILL.md" }) catch continue;
                    defer alloc.free(path);
                    const sf = std.fs.openFileAbsolute(path, .{}) catch continue;
                    defer sf.close();
                    const content = sf.readToEndAlloc(alloc, 512 * 1024) catch continue;
                    defer alloc.free(content);

                    // If file starts with YAML front matter (--- ... ---), extract
                    // the `description:` value from it and return that as the summary.
                    if (std.mem.startsWith(u8, content, "---\n")) {
                        // Find closing --- (must be on its own line after the opening)
                        const fm_close = std.mem.indexOf(u8, content[4..], "\n---\n") orelse {
                            // No closing marker — skip front matter heuristically,
                            // return first non-empty line after the opening ---
                            var lines = std.mem.splitScalar(u8, content[4..], '\n');
                            while (lines.next()) |line| {
                                const trimmed = std.mem.trim(u8, line, " \t\r");
                                if (trimmed.len > 0 and !std.mem.eql(u8, trimmed, "---")) {
                                    return alloc.dupe(u8, trimmed[0..@min(trimmed.len, 200)]) catch null;
                                }
                            }
                            return null;
                        };
                        const fm_body = content[4 .. 4 + fm_close]; // between the two ---
                        // Look for "description:" line in front matter.
                        var fm_lines = std.mem.splitScalar(u8, fm_body, '\n');
                        while (fm_lines.next()) |fm_line| {
                            if (std.mem.startsWith(u8, fm_line, "description:")) {
                                const val = std.mem.trim(u8, fm_line["description:".len..], " \t\r");
                                if (val.len > 0) {
                                    return alloc.dupe(u8, val[0..@min(val.len, 200)]) catch null;
                                }
                            }
                        }
                        // No description: — return first non-empty line after front matter.
                        const after_fm = content[4 + fm_close + 5 ..]; // skip "\n---\n"
                        var body_lines = std.mem.splitScalar(u8, after_fm, '\n');
                        while (body_lines.next()) |bl| {
                            const trimmed = std.mem.trim(u8, bl, " \t\r");
                            if (trimmed.len > 0) {
                                return alloc.dupe(u8, trimmed[0..@min(trimmed.len, 200)]) catch null;
                            }
                        }
                        return null;
                    }

                    // No front matter — return first paragraph (up to blank line), max 600 chars.
                    const para_end = std.mem.indexOf(u8, content, "\n\n") orelse content.len;
                    return alloc.dupe(u8, content[0..@min(para_end, 600)]) catch null;
                }
                return null;
            }
        }.call;

        for (all_docs_combined) |doc_slice| {
            for (doc_slice) |ed| {
                // Primary: structured skills[] array from guidance JSON.
                for (ed.skills) |sk| {
                    // Derive short name from ref path second-to-last component.
                    // e.g. "guidance/skills/gof-patterns/SKILL.md" → "gof-patterns"
                    const slash = std.mem.lastIndexOfScalar(u8, sk.ref, '/') orelse 0;
                    const after_slash = if (slash > 0) sk.ref[slash + 1 ..] else sk.ref;
                    const skill_name: []const u8 = if (std.mem.eql(u8, after_slash, "SKILL.md")) blk: {
                        const ref_trimmed = sk.ref[0..slash];
                        const parent_slash = std.mem.lastIndexOfScalar(u8, ref_trimmed, '/') orelse 0;
                        break :blk if (parent_slash > 0) ref_trimmed[parent_slash + 1 ..] else ref_trimmed;
                    } else after_slash;

                    if (seen_skills.contains(skill_name)) continue;
                    try seen_skills.put(allocator, skill_name, {});

                    if (loadSkillPara(allocator, guidance_dir, cwd, skill_name)) |para| {
                        try skill_excerpts.append(allocator, .{
                            .name = try allocator.dupe(u8, skill_name),
                            .excerpt = para,
                        });
                    }
                }

                // Fallback: parse "[tag1, tag2]" comment prefix when skills[] empty.
                if (ed.skills.len == 0) {
                    if (ed.comment) |c| {
                        if (c.len > 0 and c[0] == '[') {
                            if (std.mem.indexOfScalar(u8, c, ']')) |bracket_end| {
                                const tags_str = c[1..bracket_end];
                                var tag_iter = std.mem.splitScalar(u8, tags_str, ',');
                                while (tag_iter.next()) |tag_raw| {
                                    const tag = std.mem.trim(u8, tag_raw, " \t");
                                    if (tag.len == 0) continue;
                                    if (seen_skills.contains(tag)) continue;
                                    try seen_skills.put(allocator, tag, {});
                                    if (loadSkillPara(allocator, guidance_dir, cwd, tag)) |para| {
                                        try skill_excerpts.append(allocator, .{
                                            .name = try allocator.dupe(u8, tag),
                                            .excerpt = para,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── PHASE B: Source excerpt extraction (JSON line numbers + pruned) ───────
    // For each matched member, use its JSON line number as start and the next
    // sibling's line - 1 as end boundary (mirrors guidance.py _sibling_starts).
    // extractSourceExcerptPruned then trims trailing blank + comment lines.
    //
    // ExcerptEntry.source borrows from ExploreDoc (stable until function exit).
    // ExcerptEntry.label and .code are owned allocations freed in the defer below.
    // for_output=true → printed in the code block section of the final output.
    const ExcerptEntry = struct {
        source: []const u8, // borrowed from ExploreDoc.source
        label: []const u8, // owned: "src/foo.zig:42"
        code: []const u8, // owned: pruned source block
        for_output: bool,
    };
    var excerpts: std.ArrayList(ExcerptEntry) = .{};
    defer {
        for (excerpts.items) |e| {
            allocator.free(e.label);
            allocator.free(e.code);
        }
        excerpts.deinit(allocator);
    }

    for (all_docs_combined) |doc_slice| {
        for (doc_slice) |ed| {
            const src_abs = try std.fs.path.join(allocator, &.{ cwd, ed.source });
            defer allocator.free(src_abs);
            const src_content_opt: ?[]const u8 = blk: {
                const sf = std.fs.openFileAbsolute(src_abs, .{}) catch break :blk null;
                defer sf.close();
                break :blk sf.readToEndAlloc(allocator, 10 * 1024 * 1024) catch null;
            };
            defer if (src_content_opt) |sc| allocator.free(sc);
            const src_content = src_content_opt orelse continue;

            // Classify members: exact name == term (highest priority) vs substring.
            var exact_indices: std.ArrayList(usize) = .{};
            defer exact_indices.deinit(allocator);
            var substr_indices: std.ArrayList(usize) = .{};
            defer substr_indices.deinit(allocator);

            for (ed.members, 0..) |m, mi| {
                if (m.type == .test_decl) continue;
                const mname_lower = try std.ascii.allocLowerString(allocator, m.name);
                defer allocator.free(mname_lower);
                var is_exact = false;
                var is_substr = false;
                for (search_terms.items) |term| {
                    if (std.mem.eql(u8, mname_lower, term)) {
                        is_exact = true;
                        break;
                    }
                    if (std.mem.indexOf(u8, mname_lower, term) != null) is_substr = true;
                }
                if (is_exact) try exact_indices.append(allocator, mi);
                if (is_substr and !is_exact) try substr_indices.append(allocator, mi);
            }

            // Prefer exact matches; fall back to substring; cap at 3 per doc.
            const to_extract: []const usize = if (exact_indices.items.len > 0)
                exact_indices.items[0..@min(3, exact_indices.items.len)]
            else
                substr_indices.items[0..@min(3, substr_indices.items.len)];

            // Include in printed output for primary docs or when exact match found.
            const for_output = ed.is_primary or exact_indices.items.len > 0;

            var prev_start: u32 = 0;
            var first_excerpt = true;
            for (to_extract) |mi| {
                const m = ed.members[mi];
                const start_line = m.line orelse continue;

                // End boundary: next sibling's start line - 1, or 80-line cap.
                const end_line: u32 = if (mi + 1 < ed.members.len)
                    (ed.members[mi + 1].line orelse 0) -| 1
                else
                    start_line + 79;

                // Skip if overlapping with previous excerpt.
                if (!first_excerpt and start_line <= prev_start + 5) continue;
                prev_start = start_line;
                first_excerpt = false;

                const code = try extractSourceExcerptPruned(allocator, src_content, start_line, end_line);
                if (code.len == 0) {
                    allocator.free(code);
                    continue;
                }
                const label = try std.fmt.allocPrint(allocator, "{s}:{d}", .{ ed.source, start_line });
                try excerpts.append(allocator, .{
                    .source = ed.source,
                    .label = label,
                    .code = code,
                    .for_output = for_output,
                });
            }
        }
    }

    // ── PHASE C: Grep for files with most matches ─────────────────────────────
    var grep_files_set = std.StringHashMap(void).init(allocator);
    defer {
        var iter = grep_files_set.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        grep_files_set.deinit();
    }

    var file_grep_lines = std.StringHashMap(std.ArrayList(usize)).init(allocator);
    defer {
        var iter = file_grep_lines.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*); // free owned key
            entry.value_ptr.deinit(allocator);
        }
        file_grep_lines.deinit();
    }

    for (all_docs_combined) |doc_slice| {
        for (doc_slice) |ed| {
            if (grep_files_set.contains(ed.source)) continue;
            try grep_files_set.put(try allocator.dupe(u8, ed.source), {});

            const abs = try std.fs.path.join(allocator, &.{ cwd, ed.source });
            defer allocator.free(abs);
            const matches = try grepFile(allocator, abs, search_terms.items, 10);
            if (matches.len > 0) {
                var lines_list: std.ArrayList(usize) = .{};
                for (matches) |gm| try lines_list.append(allocator, gm.line_no);
                // key ownership transferred to file_grep_lines; freed in its defer above
                try file_grep_lines.put(try allocator.dupe(u8, ed.source), lines_list);
            }
            for (matches) |gm| allocator.free(gm.line);
            allocator.free(matches);
        }
    }

    // Sort files by match count.
    // FileMatchItem borrows path/lines from file_grep_lines — no extra allocation.
    const FileMatchItem = struct { path: []const u8, count: usize, lines: []const usize };
    var sorted_files: std.ArrayList(FileMatchItem) = .{};
    defer sorted_files.deinit(allocator); // items are borrowed; no per-item free needed
    {
        var iter = file_grep_lines.iterator();
        while (iter.next()) |entry| {
            try sorted_files.append(allocator, .{
                .path = entry.key_ptr.*, // borrowed from file_grep_lines key
                .count = entry.value_ptr.items.len,
                .lines = entry.value_ptr.items, // borrowed from ArrayList slice
            });
        }
    }
    // Sort descending by count
    std.sort.insertion(FileMatchItem, sorted_files.items, {}, struct {
        fn less(ctx: void, a: FileMatchItem, b: FileMatchItem) bool {
            _ = ctx;
            return a.count > b.count;
        }
    }.less);

    // ── PHASE D: AI synthesis (skill-contextualized, Proposal 2) ─────────────
    // Skills are injected into the knowledge block first and named in the
    // instruction so the LLM identifies and names the design pattern.
    var ai_summary: ?[]const u8 = null;
    defer if (ai_summary) |s| allocator.free(s);

    if (llm_available and !no_ai) {
        var kb: std.ArrayList(u8) = .{};
        defer kb.deinit(allocator);
        const kbw = kb.writer(allocator);

        // 1. Skill context FIRST — grounding the LLM in the pattern before code.
        if (skill_excerpts.items.len > 0) {
            try kbw.print("=== Skill patterns ===\n", .{});
            for (skill_excerpts.items[0..@min(2, skill_excerpts.items.len)]) |se| {
                try kbw.print("{s}: {s}\n\n", .{ se.name, se.excerpt });
            }
        }

        // 2. Module sections: comment, used_by, member index, source excerpts.
        for (all_docs_combined) |doc_slice| {
            for (doc_slice) |ed| {
                try kbw.print("=== {s} ===\n", .{ed.source});
                if (ed.comment) |c| {
                    const nl = std.mem.indexOfScalar(u8, c, '\n') orelse c.len;
                    try kbw.print("{s}\n", .{c[0..nl]});
                }
                if (ed.used_by.len > 0) {
                    try kbw.print("Used by: ", .{});
                    for (ed.used_by, 0..) |u, ui| {
                        if (ui > 0) try kbw.print(", ", .{});
                        try kbw.print("{s}", .{u});
                    }
                    try kbw.print("\n", .{});
                }
                try kbw.print("\nMember index:\n", .{});
                for (ed.members[0..@min(16, ed.members.len)]) |m| {
                    try kbw.print("  {s} (line {?}): ", .{ m.name, m.line });
                    if (m.signature) |sig| try kbw.print("{s}", .{sig});
                    if (m.comment) |c| {
                        const nl = std.mem.indexOfScalar(u8, c, '\n') orelse c.len;
                        if (nl > 0) try kbw.print(" — {s}", .{c[0..nl]});
                    }
                    try kbw.print("\n", .{});
                    for (m.members[0..@min(8, m.members.len)]) |mm| {
                        try kbw.print("    {s} (line {?}): ", .{ mm.name, mm.line });
                        if (mm.signature) |sig| try kbw.print("{s}", .{sig});
                        if (mm.comment) |c| {
                            const nl = std.mem.indexOfScalar(u8, c, '\n') orelse c.len;
                            if (nl > 0) try kbw.print(" — {s}", .{c[0..nl]});
                        }
                        try kbw.print("\n", .{});
                    }
                }
                // Source excerpts for this module from the pruned extraction above.
                var wrote_header = false;
                for (excerpts.items) |e| {
                    if (!std.mem.eql(u8, e.source, ed.source)) continue;
                    if (!wrote_header) {
                        try kbw.print("\nSource excerpts (from {s}):\n", .{ed.source});
                        wrote_header = true;
                    }
                    try kbw.print("// {s}\n{s}\n\n", .{ e.label, e.code });
                }
                try kbw.print("\n", .{});
            }
        }

        // 3. Inbox bullets matching search terms.
        const inbox_candidate_names = [_][]const u8{ "INSIGHTS.md", "CAPABILITIES.md" };
        for (inbox_candidate_names) |inbox_name| {
            const inbox_path = try std.fs.path.join(allocator, &.{ guidance_dir, ".doc", "inbox", inbox_name });
            defer allocator.free(inbox_path);
            const inbox_file = std.fs.openFileAbsolute(inbox_path, .{}) catch continue;
            defer inbox_file.close();
            const content = inbox_file.readToEndAlloc(allocator, 1024 * 1024) catch continue;
            defer allocator.free(content);
            var lines_it = std.mem.splitScalar(u8, content, '\n');
            while (lines_it.next()) |line| {
                const stripped = std.mem.trim(u8, line, " \t\r");
                if (!std.mem.startsWith(u8, stripped, "- ")) continue;
                for (search_terms.items) |term| {
                    const stripped_lower = try std.ascii.allocLowerString(allocator, stripped);
                    defer allocator.free(stripped_lower);
                    if (std.mem.indexOf(u8, stripped_lower, term) != null) {
                        try kbw.print("{s}\n", .{stripped});
                        break;
                    }
                }
            }
        }

        // Build skill-name string for the instruction header.
        var skill_names_buf: std.ArrayList(u8) = .{};
        defer skill_names_buf.deinit(allocator);
        for (skill_excerpts.items, 0..) |se, si| {
            if (si > 0) try skill_names_buf.appendSlice(allocator, ", ");
            try skill_names_buf.appendSlice(allocator, se.name);
        }
        const skill_names_str = skill_names_buf.items;

        // Skill instruction prefix for the prompt (core Proposal 2 change).
        const skill_instruction: []const u8 = if (skill_names_str.len > 0)
            try std.fmt.allocPrint(
                allocator,
                "SKILL PATTERNS APPLIED: {s}\nThe code implements these patterns — name the pattern(s) in your summary.\n",
                .{skill_names_str},
            )
        else
            try allocator.dupe(u8, "");
        defer allocator.free(skill_instruction);

        const kb_text = kb.items;
        const primary_term = search_terms.items[0];
        const prompt: []const u8 = if (focus_question) |fq|
            try std.fmt.allocPrint(
                allocator,
                "You are a code navigation assistant for a Zig/Python codebase. Be precise and terse.\n{s}\nQUESTION: {s}\nSUBJECT: {s}\n\nKNOWLEDGE:\n{s}\n\nAnswer using only facts from KNOWLEDGE. Cite file paths, function names, and line numbers. If the answer is partial, name the next file or symbol to look at. Return only the answer.",
                .{ skill_instruction, fq, primary_term, kb_text },
            )
        else
            try std.fmt.allocPrint(
                allocator,
                "You are a code navigation assistant for a Zig/Python codebase. Be precise and terse.\n{s}\nSummarise '{s}': what it is, what design pattern it implements (if skill patterns listed above apply), key members/functions with line numbers, and who calls or imports it. 3-5 sentences. Use only facts from KNOWLEDGE. STRICT RULE: Never write sentences about absence.\n\nKNOWLEDGE:\n{s}\n\nReturn only the summary.",
                .{ skill_instruction, primary_term, kb_text },
            );
        defer allocator.free(prompt);

        if (llm_client_opt) |*client| {
            if (client.complete(prompt, 1500, 0.15, null) catch null) |raw_summary| {
                defer allocator.free(raw_summary);
                var summary_buf: std.ArrayList(u8) = .{};
                const swb = summary_buf.writer(allocator);
                const absence_kws = [_][]const u8{
                    "no other",       "not present",  "only has", "does not contain",
                    "does not exist", "nothing else", "none are", "none were",
                };
                var sentence_iter = std.mem.splitScalar(u8, raw_summary, '\n');
                while (sentence_iter.next()) |sentence| {
                    const s_lower = try std.ascii.allocLowerString(allocator, sentence);
                    defer allocator.free(s_lower);
                    var is_absence = false;
                    for (absence_kws) |kw| {
                        if (std.mem.indexOf(u8, s_lower, kw) != null) {
                            is_absence = true;
                            break;
                        }
                    }
                    if (!is_absence) try swb.print("{s}\n", .{sentence});
                }
                ai_summary = try summary_buf.toOwnedSlice(allocator);
            }
        }
    }

    // ── PHASE E: Output ───────────────────────────────────────────────────────
    var out_buf: [65536]u8 = undefined;
    var fw_explain = std.fs.File.stdout().writer(&out_buf);
    const stdout = &fw_explain.interface;

    // Header.
    try stdout.print("# Explain: {s}\n\n", .{q});

    // AI summary.
    if (ai_summary) |s| {
        const trimmed_s = std.mem.trim(u8, s, " \t\n\r");
        if (trimmed_s.len > 0) try stdout.print("{s}\n\n", .{trimmed_s});
    }

    // Separator + Source + Pattern lines.
    try stdout.print("---\n", .{});
    const primary_ed: ?ExploreDoc = if (primary_docs.len > 0) primary_docs[0] else if (secondary_docs.len > 0) secondary_docs[0] else null;
    if (primary_ed) |ed| {
        try stdout.print("**Source**: `{s}`\n", .{ed.source});
    }
    // Pattern: skill name + first line of skill excerpt (Proposal 2 addition).
    for (skill_excerpts.items[0..@min(2, skill_excerpts.items.len)]) |se| {
        const first_nl = std.mem.indexOfScalar(u8, se.excerpt, '\n') orelse se.excerpt.len;
        const first_line = se.excerpt[0..@min(first_nl, 120)];
        try stdout.print("**Pattern**: `{s}` — {s}\n", .{ se.name, first_line });
    }
    try stdout.print("\n", .{});

    // Code blocks: only excerpts flagged for output (primary / exact match).
    for (excerpts.items) |e| {
        if (!e.for_output) continue;
        const lang: []const u8 = if (std.mem.endsWith(u8, e.source, ".zig")) "zig" else if (std.mem.endsWith(u8, e.source, ".py")) "python" else "text";
        try stdout.print("```{s}\n// {s}\n{s}\n```\n\n", .{ lang, e.label, e.code });
    }

    // Keywords: public non-test top-level members from primary doc,
    // excluding the search terms themselves.
    if (primary_ed) |ed| {
        var kw_buf: std.ArrayList(u8) = .{};
        defer kw_buf.deinit(allocator);
        const kw_writer = kw_buf.writer(allocator);
        var kw_count: usize = 0;
        for (ed.members) |m| {
            if (kw_count >= 8) break;
            if (!m.is_pub) continue;
            if (m.type == .test_decl) continue;
            if (m.name.len <= 1) continue;
            const mname_lower = try std.ascii.allocLowerString(allocator, m.name);
            defer allocator.free(mname_lower);
            var is_search_term = false;
            for (search_terms.items) |term| {
                if (std.mem.eql(u8, mname_lower, term)) {
                    is_search_term = true;
                    break;
                }
            }
            if (is_search_term) continue;
            if (kw_count > 0) try kw_buf.appendSlice(allocator, ", ");
            try kw_writer.print("`{s}`", .{m.name});
            kw_count += 1;
        }
        if (kw_count > 0) {
            try stdout.print("**Keywords**: {s}\n\n", .{kw_buf.items});
        }
    }

    // See also: used_by callers from primary doc + secondary doc source paths.
    {
        var see_buf: std.ArrayList(u8) = .{};
        defer see_buf.deinit(allocator);
        const see_writer = see_buf.writer(allocator);
        var see_count: usize = 0;
        if (primary_ed) |ed| {
            for (ed.used_by[0..@min(4, ed.used_by.len)]) |ub| {
                if (see_count > 0) try see_buf.appendSlice(allocator, ", ");
                try see_writer.print("`{s}`", .{ub});
                see_count += 1;
            }
        }
        for (secondary_docs) |ed| {
            if (see_count >= 6) break;
            if (see_count > 0) try see_buf.appendSlice(allocator, ", ");
            try see_writer.print("`{s}`", .{ed.source});
            see_count += 1;
        }
        if (see_count > 0) {
            try stdout.print("**See also**: {s}\n\n", .{see_buf.items});
        }
    }

    // Skills reference line so agents know where to follow up.
    if (skill_excerpts.items.len > 0) {
        try stdout.print("**Skills**: ", .{});
        for (skill_excerpts.items, 0..) |se, si| {
            if (si > 0) try stdout.print(", ", .{});
            try stdout.print("`{s}/.skills/{s}/SKILL.md`", .{ guidance_dir, se.name });
        }
        try stdout.print("\n\n", .{});
    }

    // Files with most grep matches.
    if (sorted_files.items.len > 0) {
        try stdout.print("### Files with most matches\n\n", .{});
        for (sorted_files.items[0..@min(3, sorted_files.items.len)]) |item| {
            try stdout.print("- `{s}` ({d} matches): lines ", .{ item.path, item.count });
            for (item.lines[0..@min(10, item.lines.len)], 0..) |ln, lni| {
                if (lni > 0) try stdout.print(", ", .{});
                try stdout.print("{d}", .{ln});
            }
            try stdout.print("\n", .{});
        }
        try stdout.print("\n", .{});
    }

    try stdout.flush();
}

// =============================================================================
// clean (stdout only)
// =============================================================================

fn cmdClean(allocator: std.mem.Allocator, args: []const []const u8) !void {
    if (args.len == 0) {
        std.debug.print("Error: JSON file path required\n", .{});
        return;
    }

    const filepath = args[0];

    const file = std.fs.openFileAbsolute(filepath, .{}) catch |err| {
        std.debug.print("Error opening {s}: {}\n", .{ filepath, err });
        return;
    };
    defer file.close();

    const content = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return;
    defer allocator.free(content);

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return;
    defer parsed.deinit();

    var out_buf: [65536]u8 = undefined;
    var fw_clean = std.fs.File.stdout().writer(&out_buf);
    const stdout = &fw_clean.interface;
    try std.json.Stringify.value(parsed.value, .{ .whitespace = .indent_2 }, stdout);
    try stdout.flush();
}

// =============================================================================
// structure
// =============================================================================

fn cmdStructure(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var guidance_dir_arg: ?[]const u8 = null;
    var no_ai = false;
    var debug_mode = false;
    var api_url: []const u8 = "http://localhost:11434/api/chat";
    var model: []const u8 = "fast:latest";
    var i: usize = 0;

    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--guidance")) {
            i += 1;
            if (i >= args.len) return;
            guidance_dir_arg = args[i];
        } else if (std.mem.eql(u8, arg, "--no-ai")) {
            no_ai = true;
        } else if (std.mem.eql(u8, arg, "--debug")) {
            debug_mode = true;
        } else if (std.mem.eql(u8, arg, "--api-url")) {
            i += 1;
            if (i >= args.len) return;
            api_url = args[i];
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return;
            model = args[i];
        }
    }

    if (guidance_dir_arg == null) {
        std.debug.print("❌ Error: --guidance is required. Usage: ast-guidance structure --guidance <dir> [options]\n", .{});
        return;
    }

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    const guidance_dir = if (std.fs.path.isAbsolute(guidance_dir_arg.?))
        try allocator.dupe(u8, guidance_dir_arg.?)
    else
        try std.fs.path.join(allocator, &.{ cwd, guidance_dir_arg.? });
    defer allocator.free(guidance_dir);

    // AI infill pre-pass: fill blank module-level comments in guidance JSON before
    // building the tree, so annotations in STRUCTURE.md are as complete as possible.
    if (!no_ai) {
        const config: llm.LlmConfig = .{ .api_url = api_url, .model = model, .debug = debug_mode };
        var enh_opt = enhancer_mod.Enhancer.init(allocator, config) catch null;
        if (enh_opt) |*enh| {
            defer enh.deinit();
            if (enh.available()) {
                var infill_count: usize = 0;
                var gdir = std.fs.openDirAbsolute(guidance_dir, .{ .iterate = true }) catch null;
                if (gdir) |*gd| {
                    defer gd.close();
                    var walker = gd.walk(allocator) catch null;
                    if (walker) |*w| {
                        defer w.deinit();
                        while (w.next() catch null) |entry| {
                            if (entry.kind != .file) continue;
                            if (!std.mem.endsWith(u8, entry.basename, ".json")) continue;
                            const full_path = try std.fs.path.join(allocator, &.{ guidance_dir, entry.path });
                            defer allocator.free(full_path);

                            // Load JSON, check if comment is empty, call enhancer.
                            const jfile = std.fs.openFileAbsolute(full_path, .{}) catch continue;
                            const jcontent = jfile.readToEndAlloc(allocator, 10 * 1024 * 1024) catch blk: {
                                jfile.close();
                                break :blk null;
                            };
                            jfile.close();
                            if (jcontent == null) continue;
                            defer allocator.free(jcontent.?);

                            var jparsed = std.json.parseFromSlice(std.json.Value, allocator, jcontent.?, .{}) catch continue;
                            defer jparsed.deinit();

                            if (jparsed.value != .object) continue;
                            // Skip only when a genuine comment is already present.
                            // A non-empty string that starts with a known LLM reasoning
                            // preamble ("we need ...", etc.) is treated as absent so the
                            // enhancer overwrites it.
                            const has_comment = blk: {
                                const cv = jparsed.value.object.get("comment");
                                if (cv) |c| {
                                    if (c != .string or c.string.len == 0) break :blk false;
                                    if (llm.isMalformedResponse(c.string)) break :blk false;
                                    break :blk true;
                                }
                                break :blk false;
                            };
                            if (has_comment) continue;

                            // Get source path for preview.
                            const src_rel = blk: {
                                if (jparsed.value.object.get("meta")) |meta| {
                                    if (meta == .object) {
                                        if (meta.object.get("source")) |src| {
                                            if (src == .string) break :blk src.string;
                                        }
                                    }
                                }
                                break :blk "";
                            };
                            const src_preview: []const u8 = blk: {
                                if (src_rel.len == 0) break :blk "";
                                const src_abs = try std.fs.path.join(allocator, &.{ cwd, src_rel });
                                defer allocator.free(src_abs);
                                const sf = std.fs.openFileAbsolute(src_abs, .{}) catch break :blk "";
                                const sc = sf.readToEndAlloc(allocator, 3000) catch blk2: {
                                    sf.close();
                                    break :blk2 "";
                                };
                                sf.close();
                                break :blk sc;
                            };
                            defer if (src_preview.len > 0) allocator.free(src_preview);

                            const new_comment = enh.enhanceFile(src_rel, null, src_preview) catch continue;
                            if (new_comment) |nc| {
                                defer allocator.free(nc);
                                if (nc.len == 0) continue;
                                // Patch the JSON object and rewrite.
                                try jparsed.value.object.put("comment", .{ .string = nc });
                                var out: std.io.Writer.Allocating = .init(allocator);
                                defer out.deinit();
                                try std.json.Stringify.value(jparsed.value, .{ .whitespace = .indent_2 }, &out.writer);
                                const out_file = try std.fs.createFileAbsolute(full_path, .{ .truncate = true });
                                defer out_file.close();
                                try out_file.writeAll(out.written());
                                infill_count += 1;
                            }
                        }
                    }
                }
                if (infill_count > 0) {
                    std.debug.print("  AI infilled {} guidance file(s).\n", .{infill_count});
                }
            }
        }
    }

    var gen = structure_mod.StructureGenerator.init(allocator, cwd, guidance_dir, debug_mode);
    defer gen.deinit();
    try gen.generate();
    std.debug.print("✓ STRUCTURE.md updated\n", .{});
}

// =============================================================================
// diary
// =============================================================================

fn cmdDiary(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var guidance_dir_arg: ?[]const u8 = null;
    var note_parts: std.ArrayList([]const u8) = .{};
    defer note_parts.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--guidance")) {
            i += 1;
            if (i >= args.len) return;
            guidance_dir_arg = args[i];
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            try note_parts.append(allocator, arg);
        }
    }

    if (guidance_dir_arg == null) {
        std.debug.print("❌ Error: --guidance is required. Usage: ast-guidance diary <note> --guidance <dir>\n", .{});
        return;
    }

    if (note_parts.items.len == 0) {
        std.debug.print("Error: diary note required\n", .{});
        return;
    }

    const note = try std.mem.join(allocator, " ", note_parts.items);
    defer allocator.free(note);

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    // Get local wall-clock time via localtime_r (POSIX).
    // Falls back to UTC with a "(UTC)" annotation when unavailable.
    const LocalTime = struct {
        year: u32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        is_local: bool,
    };
    const local_time: LocalTime = blk: {
        const ts = std.time.timestamp();
        const tm = getLocalTime(ts) catch null;
        if (tm) |t| {
            // struct tm: tm_year = years since 1900, tm_mon = 0-based, tm_mday = 1-based
            break :blk .{
                .year = @intCast(t.year + 1900),
                .month = @intCast(t.month + 1),
                .day = @intCast(t.mday),
                .hour = @intCast(t.hour),
                .minute = @intCast(t.minute),
                .is_local = true,
            };
        }
        // UTC fallback.
        const epoch_secs = std.time.epoch.EpochSeconds{ .secs = @intCast(ts) };
        const epoch_day = epoch_secs.getEpochDay();
        const year_day = epoch_day.calculateYearDay();
        const month_day = year_day.calculateMonthDay();
        const day_secs = epoch_secs.getDaySeconds();
        break :blk .{
            .year = year_day.year,
            .month = month_day.month.numeric(),
            .day = month_day.day_index + 1,
            .hour = day_secs.getHoursIntoDay(),
            .minute = day_secs.getMinutesIntoHour(),
            .is_local = false,
        };
    };

    const today = try std.fmt.allocPrint(
        allocator,
        "{d:0>4}-{d:0>2}-{d:0>2}",
        .{ local_time.year, local_time.month, local_time.day },
    );
    defer allocator.free(today);

    const hour = local_time.hour;
    const minute = local_time.minute;
    _ = local_time.is_local; // could append "(UTC)" but keep format clean

    // Get git author name (best-effort).
    var author_buf: [128]u8 = undefined;
    const author = getGitAuthor(allocator, cwd, &author_buf) catch "unknown";

    // Diary file path: {guidance_dir}/.doc/diary/<today>.md
    const guidance_dir = if (std.fs.path.isAbsolute(guidance_dir_arg.?))
        try allocator.dupe(u8, guidance_dir_arg.?)
    else
        try std.fs.path.join(allocator, &.{ cwd, guidance_dir_arg.? });
    defer allocator.free(guidance_dir);

    const diary_dir = try std.fs.path.join(allocator, &.{ guidance_dir, ".doc", "diary" });
    defer allocator.free(diary_dir);
    std.fs.makeDirAbsolute(diary_dir) catch {};

    const diary_file_path = try std.fmt.allocPrint(allocator, "{s}/{s}.md", .{ diary_dir, today });
    defer allocator.free(diary_file_path);

    // Create file with header if it doesn't exist.
    const file_existed = if (std.fs.accessAbsolute(diary_file_path, .{})) true else |_| false;

    const diary_file = try std.fs.createFileAbsolute(diary_file_path, .{ .truncate = false });
    defer diary_file.close();

    if (!file_existed) {
        try diary_file.seekTo(0);
        const header = try std.fmt.allocPrint(allocator, "# Diary: {s}\n\n", .{today});
        defer allocator.free(header);
        try diary_file.writeAll(header);
    } else {
        // Append mode: seek to end.
        try diary_file.seekFromEnd(0);
    }

    const entry = try std.fmt.allocPrint(allocator, "- [{d:0>2}:{d:0>2}] ({s}) {s}\n", .{ hour, minute, author, note });
    defer allocator.free(entry);
    try diary_file.writeAll(entry);

    std.debug.print("✓ Added to {s}\n", .{diary_file_path});
}

/// POSIX struct tm fields we care about (subset of C struct tm).
pub const PosixTm = struct {
    year: i32, // years since 1900
    month: i32, // months since January [0–11]
    mday: i32, // day of the month [1–31]
    hour: i32, // hours since midnight [0–23]
    minute: i32, // minutes after the hour [0–59]
};

/// Full POSIX struct tm (9 i32 fields).  We use this as the target for
/// localtime_r and read out the fields we need.
const CStructTm = extern struct {
    tm_sec: c_int, // 0
    tm_min: c_int, // 1
    tm_hour: c_int, // 2
    tm_mday: c_int, // 3
    tm_mon: c_int, // 4
    tm_year: c_int, // 5
    tm_wday: c_int, // 6
    tm_yday: c_int, // 7
    tm_isdst: c_int, // 8
};

extern "c" fn localtime_r(timep: *const c_long, result: *CStructTm) ?*CStructTm;

/// Call localtime_r to get local wall-clock fields.
/// Returns error if the libc call fails or the platform doesn't support it.
pub fn getLocalTime(timestamp: i64) !PosixTm {
    var ctm: CStructTm = undefined;
    const ts_c: c_long = @intCast(timestamp);
    const result = localtime_r(&ts_c, &ctm);
    if (result == null) return error.LocaltimeFailed;
    return PosixTm{
        .year = ctm.tm_year,
        .month = ctm.tm_mon,
        .mday = ctm.tm_mday,
        .hour = ctm.tm_hour,
        .minute = ctm.tm_min,
    };
}

fn getGitAuthor(allocator: std.mem.Allocator, cwd: []const u8, buf: []u8) ![]const u8 {
    var child = std.process.Child.init(&.{ "git", "config", "user.name" }, allocator);
    child.cwd = cwd;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;
    try child.spawn();
    const n = try child.stdout.?.readAll(buf);
    _ = try child.wait();
    const trimmed = std.mem.trim(u8, buf[0..n], " \t\r\n");
    if (trimmed.len == 0) return error.NoAuthor;
    return trimmed;
}

// =============================================================================
// commit
// =============================================================================

/// Read `models.commit` (preferred) or `models.default` from the project-local
/// config JSON.  Returns an owned slice; caller must free.  Returns error when
/// the config is absent or neither key is present.
fn loadCommitModel(allocator: std.mem.Allocator, cwd: []const u8) ![]const u8 {
    const path = try std.fs.path.join(allocator, &.{ cwd, config_mod.DEFAULT_GUIDANCE_DIR, "ast-guidance-config.json" });
    defer allocator.free(path);

    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 64 * 1024);
    defer allocator.free(content);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.InvalidConfig;

    if (root.object.get("models")) |models| {
        if (models == .object) {
            if (models.object.get("commit")) |m| if (m == .string and m.string.len > 0)
                return allocator.dupe(u8, m.string);
            if (models.object.get("default")) |m| if (m == .string and m.string.len > 0)
                return allocator.dupe(u8, m.string);
        }
    }
    return error.NoCommitModel;
}

fn cmdCommit(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const common = llm.parseCommonArgs(args);
    const dry_run = common.dry_run;
    const debug = common.debug;

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    // Load config — resolves guidance JSON base, api_url, and model.
    // CLI flags (--api-url, --model) override config values when provided.
    // Each variable owns its slice; a single defer per variable frees it.
    var json_base: []const u8 = try allocator.dupe(u8, "");
    defer allocator.free(json_base);
    var cfg_api_url: []const u8 = try allocator.dupe(u8, common.api_url);
    defer allocator.free(cfg_api_url);
    var cfg_model: []const u8 = try allocator.dupe(u8, common.model);
    defer allocator.free(cfg_model);

    if (config_mod.loadConfig(allocator, cwd)) |cfg_val| {
        var cfg = cfg_val;
        defer cfg.deinit();

        // Replace the initial empty-string with the config value.
        const new_jb = try allocator.dupe(u8, cfg.json_base);
        allocator.free(json_base);
        json_base = new_jb;

        // Use config api_url unless the caller passed --api-url explicitly.
        if (!common.api_url_set) {
            const new_url = try allocator.dupe(u8, cfg.api_url);
            allocator.free(cfg_api_url);
            cfg_api_url = new_url;
        }

        // Use models.commit > models.default from raw JSON; fall back to cfg.model.
        if (!common.model_set) {
            const new_model = if (loadCommitModel(allocator, cwd) catch null) |m|
                m
            else
                try allocator.dupe(u8, cfg.model);
            allocator.free(cfg_model);
            cfg_model = new_model;
        }
    } else |_| {}

    const api_url = cfg_api_url;
    const model = cfg_model;

    // Only evaluate staged changes (git diff --staged).
    const effective_diff = try gitDiff(allocator, cwd, true);
    defer allocator.free(effective_diff);

    if (effective_diff.len == 0) {
        std.debug.print("No staged changes to commit. Use 'git add' to stage files first.\n", .{});
        return;
    }

    // Extract changed files from diff header.
    var changed_files: std.ArrayList([]const u8) = .{};
    defer {
        for (changed_files.items) |f| allocator.free(f);
        changed_files.deinit(allocator);
    }
    {
        var lines = std.mem.splitScalar(u8, effective_diff, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "diff --git a/")) {
                // "diff --git a/<path> b/<path>" — extract first path.
                const after = line["diff --git a/".len..];
                const space = std.mem.indexOfScalar(u8, after, ' ') orelse after.len;
                try changed_files.append(allocator, try allocator.dupe(u8, after[0..space]));
            }
        }
    }

    // Generate commit message via LLM or deterministic fallback.
    const commit_msg = try generateCommitMessage(allocator, effective_diff, changed_files.items, json_base, api_url, model, debug);
    defer allocator.free(commit_msg);

    if (debug) {
        std.debug.print("--- Generated commit message ---\n{s}\n-------------------------------\n", .{commit_msg});
    }

    if (dry_run) {
        if (!debug) std.debug.print("--- Generated commit message ---\n{s}\n-------------------------------\n", .{commit_msg});
        return;
    }

    // Write to a temp file, open $EDITOR, then git commit.
    const tmp_path = try writeTmpCommitMsg(allocator, commit_msg);
    defer {
        std.fs.deleteFileAbsolute(tmp_path) catch {};
        allocator.free(tmp_path);
    }

    // Stat the temp file before opening the editor so we can detect whether
    // the user actually saved any changes (mtime comparison).
    const mtime_before: i128 = blk: {
        const f = std.fs.openFileAbsolute(tmp_path, .{}) catch break :blk 0;
        defer f.close();
        const stat = f.stat() catch break :blk 0;
        break :blk stat.mtime;
    };

    // Open editor.
    const editor = std.process.getEnvVarOwned(allocator, "EDITOR") catch
        std.process.getEnvVarOwned(allocator, "VISUAL") catch
        try allocator.dupe(u8, "vi");
    defer allocator.free(editor);

    var editor_child = std.process.Child.init(&.{ editor, tmp_path }, allocator);
    editor_child.cwd = cwd;
    const editor_result = try editor_child.spawnAndWait();
    if (editor_result != .Exited or editor_result.Exited != 0) {
        std.debug.print("Editor exited with non-zero status. Aborting commit.\n", .{});
        return;
    }

    // Check whether the file was modified (i.e. the user saved changes).
    const mtime_after: i128 = blk: {
        const f = std.fs.openFileAbsolute(tmp_path, .{}) catch break :blk 0;
        defer f.close();
        const stat = f.stat() catch break :blk 0;
        break :blk stat.mtime;
    };

    if (mtime_after == mtime_before) {
        std.debug.print("Commit message not saved. Aborting.\n", .{});
        return;
    }

    // Read back the edited message, strip comment lines.
    const edited = try std.fs.openFileAbsolute(tmp_path, .{});
    defer edited.close();
    const raw_msg = try edited.readToEndAlloc(allocator, 64 * 1024);
    defer allocator.free(raw_msg);

    var final_parts: std.ArrayList([]const u8) = .{};
    defer final_parts.deinit(allocator);
    var msg_lines = std.mem.splitScalar(u8, raw_msg, '\n');
    while (msg_lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "#")) {
            try final_parts.append(allocator, line);
        }
    }
    const final_msg_raw = try std.mem.join(allocator, "\n", final_parts.items);
    defer allocator.free(final_msg_raw);
    const final_msg = std.mem.trim(u8, final_msg_raw, " \t\r\n");

    if (final_msg.len == 0) {
        std.debug.print("Commit message is empty. Aborting.\n", .{});
        return;
    }

    // git commit -m "<message>"
    var commit_child = std.process.Child.init(&.{ "git", "commit", "-m", final_msg }, allocator);
    commit_child.cwd = cwd;
    const commit_result = try commit_child.spawnAndWait();
    if (commit_result == .Exited and commit_result.Exited == 0) {
        std.debug.print("✓ Committed successfully.\n", .{});
    } else {
        std.debug.print("✗ git commit failed.\n", .{});
    }
}

fn gitDiff(allocator: std.mem.Allocator, cwd: []const u8, staged: bool) ![]u8 {
    const argv: []const []const u8 = if (staged)
        &.{ "git", "diff", "--staged" }
    else
        &.{ "git", "diff" };

    var child = std.process.Child.init(argv, allocator);
    child.cwd = cwd;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;
    try child.spawn();
    const output = try child.stdout.?.readToEndAlloc(allocator, 10 * 1024 * 1024);
    _ = try child.wait();
    return output;
}

/// Split a unified diff into per-file chunks.
/// Each chunk starts at "diff --git" and runs until the next one.
/// Returns a slice of slices into `diff` (no allocation per chunk).
fn splitDiffByFile(diff: []const u8, out: *std.ArrayList([]const u8), allocator: std.mem.Allocator) !void {
    var start: usize = 0;
    var pos: usize = 0;
    while (pos < diff.len) {
        // Find next newline.
        const nl = std.mem.indexOfScalarPos(u8, diff, pos, '\n') orelse diff.len;
        const line = diff[pos..nl];
        if (std.mem.startsWith(u8, line, "diff --git") and pos > start) {
            try out.append(allocator, diff[start..pos]);
            start = pos;
        }
        pos = nl + 1;
    }
    if (start < diff.len) try out.append(allocator, diff[start..]);
}

/// Extract the file path from the first line of a diff chunk.
/// "diff --git a/<path> b/<path>" → "<path>" (the a/ side).
fn chunkFilePath(chunk: []const u8) []const u8 {
    const prefix = "diff --git a/";
    const first_nl = std.mem.indexOfScalar(u8, chunk, '\n') orelse chunk.len;
    const first_line = chunk[0..first_nl];
    if (!std.mem.startsWith(u8, first_line, prefix)) return "";
    const after = first_line[prefix.len..];
    const sp = std.mem.indexOfScalar(u8, after, ' ') orelse after.len;
    return after[0..sp];
}

/// Return true if this diff chunk belongs to a path that should be ignored
/// when generating commit messages (e.g. auto-generated .ast-guidance/ JSON files).
fn chunkIsIgnored(chunk: []const u8) bool {
    const path = chunkFilePath(chunk);
    return std.mem.startsWith(u8, path, ".ast-guidance/") or
        std.mem.startsWith(u8, path, ".ast-guidance\\");
}

/// A member extracted from a guidance JSON file for commit context.
pub const CommitMemberInfo = struct {
    name: []const u8, // owned
    line: ?u32,
    comment: []const u8, // owned (may be empty)
    signature: []const u8, // owned (may be empty)

    pub fn deinit(self: CommitMemberInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.comment);
        allocator.free(self.signature);
    }
};

/// Test accessor for parseHunkRanges.
pub fn parseHunkRangesPub(allocator: std.mem.Allocator, chunk: []const u8) ![][2]u32 {
    return parseHunkRanges(allocator, chunk);
}

/// Test accessor for chunkIsIgnored.
pub fn chunkIsIgnoredPub(chunk: []const u8) bool {
    return chunkIsIgnored(chunk);
}

/// Test accessor for chunkFilePath.
pub fn chunkFilePathPub(chunk: []const u8) []const u8 {
    return chunkFilePath(chunk);
}

/// Test accessor for splitDiffByFile.
pub fn splitDiffByFilePub(diff: []const u8, out: *std.ArrayList([]const u8), allocator: std.mem.Allocator) !void {
    return splitDiffByFile(diff, out, allocator);
}

/// Test accessor for loadChangedMembers.
pub fn loadChangedMembersPub(
    allocator: std.mem.Allocator,
    json_base: []const u8,
    rel_path: []const u8,
    hunk_ranges: []const [2]u32,
) ![]CommitMemberInfo {
    return loadChangedMembers(allocator, json_base, rel_path, hunk_ranges);
}

/// Parse `@@ -X,Y +A,B @@` hunk headers from a diff chunk.
/// Returns owned slice of [start, end) pairs in new-file coordinates.
fn parseHunkRanges(allocator: std.mem.Allocator, chunk: []const u8) ![][2]u32 {
    var ranges: std.ArrayList([2]u32) = .{};
    errdefer ranges.deinit(allocator);

    var lines = std.mem.splitScalar(u8, chunk, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "@@ ")) continue;
        // Find " +A,B " portion.
        const plus_pos = std.mem.indexOf(u8, line, " +") orelse continue;
        const after_plus = line[plus_pos + 2 ..];
        const space_pos = std.mem.indexOfScalar(u8, after_plus, ' ') orelse after_plus.len;
        const range_part = after_plus[0..space_pos]; // "A,B" or "A"
        const comma = std.mem.indexOfScalar(u8, range_part, ',');
        const start_str = if (comma) |c| range_part[0..c] else range_part;
        const count_str = if (comma) |c| range_part[c + 1 ..] else "1";
        const start = std.fmt.parseInt(u32, start_str, 10) catch continue;
        const count = std.fmt.parseInt(u32, count_str, 10) catch 1;
        try ranges.append(allocator, .{ start, start + count });
    }

    return ranges.toOwnedSlice(allocator);
}

/// Load guidance JSON for `rel_path` from `json_base` and return members
/// whose line numbers fall within or near any of the `hunk_ranges`.
/// If `hunk_ranges` is empty, all members are returned.
/// Caller owns the returned slice and must call `.deinit(allocator)` on each element.
fn loadChangedMembers(
    allocator: std.mem.Allocator,
    json_base: []const u8,
    rel_path: []const u8,
    hunk_ranges: []const [2]u32,
) ![]CommitMemberInfo {
    const json_path = try std.fmt.allocPrint(allocator, "{s}/{s}.json", .{ json_base, rel_path });
    defer allocator.free(json_path);

    const file = std.fs.openFileAbsolute(json_path, .{}) catch return &.{};
    defer file.close();
    const content = file.readToEndAlloc(allocator, 256 * 1024) catch return &.{};
    defer allocator.free(content);

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return &.{};
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return &.{};
    const members_val = root.object.get("members") orelse return &.{};
    if (members_val != .array) return &.{};

    var result: std.ArrayList(CommitMemberInfo) = .{};
    errdefer {
        for (result.items) |m| m.deinit(allocator);
        result.deinit(allocator);
    }

    const CONTEXT_LINES: u32 = 15; // include members within this many lines of a hunk

    for (members_val.array.items) |member| {
        if (member != .object) continue;
        const name_val = member.object.get("name") orelse continue;
        if (name_val != .string or name_val.string.len == 0) continue;

        const line_num: ?u32 = blk: {
            if (member.object.get("line")) |lv| {
                break :blk switch (lv) {
                    .integer => |n| if (n >= 0) @as(u32, @intCast(n)) else null,
                    .float => |f| if (f >= 0) @as(u32, @intFromFloat(f)) else null,
                    else => null,
                };
            }
            break :blk null;
        };

        // Decide whether this member overlaps a changed hunk.
        const include = if (hunk_ranges.len == 0 or line_num == null)
            true
        else blk: {
            const ln = line_num.?;
            for (hunk_ranges) |range| {
                const lo = if (range[0] > CONTEXT_LINES) range[0] - CONTEXT_LINES else 0;
                const hi = range[1] + CONTEXT_LINES;
                if (ln >= lo and ln <= hi) break :blk true;
            }
            break :blk false;
        };

        if (!include) continue;

        const comment = if (member.object.get("comment")) |cv|
            if (cv == .string) cv.string else ""
        else
            "";
        const sig = if (member.object.get("signature")) |sv|
            if (sv == .string) sv.string else ""
        else
            "";

        try result.append(allocator, .{
            .name = try allocator.dupe(u8, name_val.string),
            .line = line_num,
            .comment = try allocator.dupe(u8, comment),
            .signature = try allocator.dupe(u8, sig),
        });

        // Recurse one level into nested members (struct methods).
        if (member.object.get("members")) |nested_val| {
            if (nested_val == .array) {
                for (nested_val.array.items) |nested| {
                    if (nested != .object) continue;
                    const nn = nested.object.get("name") orelse continue;
                    if (nn != .string or nn.string.len == 0) continue;
                    const nl_num: ?u32 = blk: {
                        if (nested.object.get("line")) |lv| {
                            break :blk switch (lv) {
                                .integer => |n| if (n >= 0) @as(u32, @intCast(n)) else null,
                                else => null,
                            };
                        }
                        break :blk null;
                    };
                    const n_include = if (hunk_ranges.len == 0 or nl_num == null)
                        true
                    else blk: {
                        const ln = nl_num.?;
                        for (hunk_ranges) |range| {
                            const lo = if (range[0] > CONTEXT_LINES) range[0] - CONTEXT_LINES else 0;
                            const hi = range[1] + CONTEXT_LINES;
                            if (ln >= lo and ln <= hi) break :blk true;
                        }
                        break :blk false;
                    };
                    if (!n_include) continue;
                    const nc = if (nested.object.get("comment")) |cv| if (cv == .string) cv.string else "" else "";
                    const ns = if (nested.object.get("signature")) |sv| if (sv == .string) sv.string else "" else "";
                    try result.append(allocator, .{
                        .name = try allocator.dupe(u8, nn.string),
                        .line = nl_num,
                        .comment = try allocator.dupe(u8, nc),
                        .signature = try allocator.dupe(u8, ns),
                    });
                }
            }
        }
    }

    return result.toOwnedSlice(allocator);
}

fn generateCommitMessage(
    allocator: std.mem.Allocator,
    diff: []const u8,
    changed_files: []const []const u8,
    json_base: []const u8,
    api_url: []const u8,
    model: []const u8,
    debug: bool,
) ![]u8 {
    // Try LLM first.
    const config: llm.LlmConfig = .{ .api_url = api_url, .model = model, .debug = debug };
    if (llm.LlmClient.init(allocator, config)) |client_val| {
        var client = client_val;
        defer client.deinit();
        if (client.available()) {
            // LLM is reachable — proceed with AI-generated message.
            // Split diff into per-file chunks and discard .ast-guidance/ tree.
            var all_chunks: std.ArrayList([]const u8) = .{};
            defer all_chunks.deinit(allocator);
            try splitDiffByFile(diff, &all_chunks, allocator);

            var code_chunks: std.ArrayList([]const u8) = .{};
            defer code_chunks.deinit(allocator);
            for (all_chunks.items) |chunk| {
                if (!chunkIsIgnored(chunk)) try code_chunks.append(allocator, chunk);
            }

            if (debug) {
                std.debug.print("[commit] {} total chunk(s), {} after filtering .ast-guidance/\n", .{ all_chunks.items.len, code_chunks.items.len });
                for (code_chunks.items) |chunk| {
                    std.debug.print("[commit]   file: {s}\n", .{chunkFilePath(chunk)});
                }
            }

            if (code_chunks.items.len > 0) {
                // Build an enriched context block per file:
                //   ### Functions in <path>:
                //   - funcName (line N): description from guidance JSON
                //   <diff chunk (truncated)>
                //
                // Total cap: 12 KB.  Each file gets an equal share.
                const TOTAL_CAP: usize = 12_000;
                const per_file_cap: usize = @max(1000, TOTAL_CAP / code_chunks.items.len);

                var combined: std.ArrayList(u8) = .{};
                defer combined.deinit(allocator);
                const cw = combined.writer(allocator);

                for (code_chunks.items) |chunk| {
                    if (combined.items.len >= TOTAL_CAP) break;

                    const rel_path = chunkFilePath(chunk);

                    // --- Guidance member context ---
                    if (json_base.len > 0 and rel_path.len > 0) {
                        const hunk_ranges = parseHunkRanges(allocator, chunk) catch &.{};
                        defer allocator.free(hunk_ranges);

                        const members = loadChangedMembers(allocator, json_base, rel_path, hunk_ranges) catch &.{};
                        defer {
                            for (members) |m| m.deinit(allocator);
                            allocator.free(members);
                        }

                        if (members.len > 0) {
                            try cw.print("### Functions in {s}:\n", .{rel_path});
                            for (members) |m| {
                                if (m.line) |ln| {
                                    try cw.print("- {s} (line {})", .{ m.name, ln });
                                } else {
                                    try cw.print("- {s}", .{m.name});
                                }
                                // Prefer comment; fall back to signature if comment empty.
                                if (m.comment.len > 0) {
                                    // Truncate comment to first sentence / 120 chars.
                                    const end = std.mem.indexOfScalar(u8, m.comment, '.') orelse m.comment.len;
                                    const snippet = m.comment[0..@min(end + 1, @min(m.comment.len, 120))];
                                    try cw.print(": {s}", .{snippet});
                                } else if (m.signature.len > 0) {
                                    const snippet = m.signature[0..@min(m.signature.len, 80)];
                                    try cw.print(": `{s}`", .{snippet});
                                }
                                try cw.writeByte('\n');
                            }
                            try cw.writeByte('\n');
                        }
                    }

                    // --- Diff chunk (truncated to per-file budget) ---
                    const budget = @min(chunk.len, per_file_cap);
                    try combined.appendSlice(allocator, chunk[0..budget]);
                    try combined.append(allocator, '\n');
                }

                // Prompt instructs the LLM to use the function descriptions.
                // The diff content is sandwiched between two copies of the
                // instruction so that small models don't continue the code.
                const prompt = try std.fmt.allocPrint(
                    allocator,
                    \\TASK: Write a git commit message as a bullet list.
                    \\
                    \\Rules:
                    \\  - One bullet per distinct change.
                    \\  - Each bullet: "* <FunctionName>: <past-tense description of what changed and why>"
                    \\  - Output ONLY the bullet list. No code. No explanations. No headings.
                    \\
                    \\Example:
                    \\* loadConfig: added builtin.is_test guard so warning is suppressed during unit tests
                    \\* chunkIsIgnored: removed stale guidance/ prefix, now only filters .ast-guidance/
                    \\
                    \\CHANGED FILES:
                    \\{s}
                    \\END OF DIFF.
                    \\
                    \\Now write the bullet list (each line starts with "* "):
                ,
                    .{combined.items},
                );
                defer allocator.free(prompt);

                if (debug) std.debug.print("[commit] synthesis prompt ({} chars):\n{s}\n---\n", .{ prompt.len, prompt });

                const result = client.complete(prompt, 8192, 0.1, null) catch null;

                if (result) |raw| {
                    defer allocator.free(raw);
                    if (debug) std.debug.print("[commit] synthesis response:\n{s}\n---\n", .{raw});

                    // Collect bullet lines. Accept "* " or "- " prefix
                    // (small models often use "-" instead of "*").
                    var bullets: std.ArrayList([]const u8) = .{};
                    defer {
                        for (bullets.items) |b| allocator.free(b);
                        bullets.deinit(allocator);
                    }
                    var resp_lines = std.mem.splitScalar(u8, raw, '\n');
                    while (resp_lines.next()) |line| {
                        const trimmed = std.mem.trim(u8, line, " \t\r");
                        const is_bullet = std.mem.startsWith(u8, trimmed, "* ") or
                            std.mem.startsWith(u8, trimmed, "- ");
                        if (is_bullet) {
                            const text = std.mem.trim(u8, trimmed[2..], " \t");
                            if (text.len > 0) try bullets.append(allocator, try allocator.dupe(u8, text));
                        }
                    }

                    if (bullets.items.len > 0) {
                        var out: std.ArrayList(u8) = .{};
                        for (bullets.items) |b| {
                            try out.appendSlice(allocator, "* ");
                            try out.appendSlice(allocator, b);
                            try out.append(allocator, '\n');
                        }
                        if (out.items.len > 0 and out.items[out.items.len - 1] == '\n')
                            out.items.len -= 1;
                        return out.toOwnedSlice(allocator);
                    }
                }
            }
        }
    } else |_| {}

    // LLM was unavailable or returned no bullets — warn and fall back.
    std.debug.print("warning: LLM unavailable or returned no output ({s}); using filename fallback\n", .{api_url});

    // Deterministic fallback: list changed non-guidance files as * bullets.
    var fallback_files: std.ArrayList([]const u8) = .{};
    defer fallback_files.deinit(allocator);
    for (changed_files) |f| {
        if (std.mem.startsWith(u8, f, ".ast-guidance/") or std.mem.startsWith(u8, f, ".ast-guidance\\")) continue;
        try fallback_files.append(allocator, f);
    }
    std.mem.sort([]const u8, fallback_files.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
    var summary: std.ArrayList(u8) = .{};
    defer summary.deinit(allocator);
    var any = false;
    for (fallback_files.items) |f| {
        try summary.appendSlice(allocator, "* Update ");
        try summary.appendSlice(allocator, f);
        try summary.append(allocator, '\n');
        any = true;
    }
    if (any) {
        if (summary.items.len > 0 and summary.items[summary.items.len - 1] == '\n')
            summary.items.len -= 1;
        return summary.toOwnedSlice(allocator);
    }
    return try allocator.dupe(u8, "* Update codebase");
}

fn writeTmpCommitMsg(allocator: std.mem.Allocator, msg: []const u8) ![]u8 {
    const tmp_dir = std.fs.openDirAbsolute("/tmp", .{}) catch return error.NoTmpDir;
    _ = tmp_dir;
    const path = try std.fmt.allocPrint(allocator, "/tmp/guidance_commit_{}.txt", .{std.time.timestamp()});
    const file = try std.fs.createFileAbsolute(path, .{});
    defer file.close();
    try file.writeAll(msg);
    try file.writeAll("\n\n# Lines starting with '#' will be ignored.\n");
    try file.writeAll("# Edit the commit message above. Save and close to commit.\n");
    return path;
}

// =============================================================================
// learn  (drain INSIGHTS.md + CAPABILITIES.md into structured knowledge)
// =============================================================================

fn cmdLearn(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const common = llm.parseCommonArgs(args);

    var guidance_dir_arg: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--guidance")) {
            i += 1;
            if (i >= args.len) return;
            guidance_dir_arg = args[i];
        }
    }

    if (guidance_dir_arg == null) {
        std.debug.print("❌ Error: --guidance is required. Usage: ast-guidance learn --guidance <dir> [options]\n", .{});
        return;
    }

    const api_url = common.api_url;
    const model = common.model;
    const dry_run = common.dry_run;

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    const guidance_dir = if (std.fs.path.isAbsolute(guidance_dir_arg.?))
        try allocator.dupe(u8, guidance_dir_arg.?)
    else
        try std.fs.path.join(allocator, &.{ cwd, guidance_dir_arg.? });
    defer allocator.free(guidance_dir);

    const skills_dir = try std.fs.path.join(allocator, &.{ guidance_dir, ".skills" });
    defer allocator.free(skills_dir);

    // Collect available skill names.
    var skill_names: std.ArrayList([]const u8) = .{};
    defer {
        for (skill_names.items) |s| allocator.free(s);
        skill_names.deinit(allocator);
    }
    if (std.fs.openDirAbsolute(skills_dir, .{ .iterate = true })) |sd| {
        var skill_dir = sd;
        defer skill_dir.close();
        var it = skill_dir.iterate();
        while (try it.next()) |entry| {
            if (entry.kind == .directory) {
                try skill_names.append(allocator, try allocator.dupe(u8, entry.name));
            }
        }
    } else |err| {
        std.debug.print("warning: skills directory not found ({s}): {} — skill context will be empty\n", .{ skills_dir, err });
    }

    // Build skills list string for the LLM prompt.
    const skills_list = try std.mem.join(allocator, ", ", skill_names.items);
    defer allocator.free(skills_list);

    const insights_path = try std.fs.path.join(allocator, &.{ guidance_dir, ".doc", "inbox", "INSIGHTS.md" });
    defer allocator.free(insights_path);
    const capabilities_path = try std.fs.path.join(allocator, &.{ guidance_dir, ".doc", "inbox", "CAPABILITIES.md" });
    defer allocator.free(capabilities_path);

    const inbox_files = [_]struct { path: []const u8, kind: []const u8 }{
        .{ .path = insights_path, .kind = "insights" },
        .{ .path = capabilities_path, .kind = "capabilities" },
    };

    // Init LLM client (best-effort — works without LLM via keyword matching).
    var llm_client: ?llm.LlmClient = if (llm.LlmClient.init(allocator, .{ .api_url = api_url, .model = model })) |c| c else |_| null;
    defer if (llm_client) |*c| c.deinit();

    const llm_available = if (llm_client) |*c| c.available() else false;

    var out_buf: [8192]u8 = undefined;
    var fw_learn = std.fs.File.stdout().writer(&out_buf);
    const stdout = &fw_learn.interface;

    for (inbox_files) |inbox| {
        // inbox.path is already an absolute path (derived from guidance_dir).
        const file = std.fs.openFileAbsolute(inbox.path, .{}) catch {
            std.debug.print("No {s} found.\n", .{inbox.path});
            continue;
        };
        const content = try file.readToEndAlloc(allocator, 1024 * 1024);
        file.close();
        defer allocator.free(content);

        var remaining: std.ArrayList([]const u8) = .{};
        defer {
            for (remaining.items) |s| allocator.free(s);
            remaining.deinit(allocator);
        }
        var promoted_count: usize = 0;

        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            const stripped = std.mem.trim(u8, line, " \t\r");
            if (!std.mem.startsWith(u8, stripped, "- ")) {
                try remaining.append(allocator, try allocator.dupe(u8, line));
                continue;
            }

            const bullet = stripped[2..];
            // Classify: LLM if available, keyword-match fallback.
            const target_skill = if (llm_available)
                try classifyBulletLlm(allocator, &llm_client.?, bullet, skills_list, skills_dir)
            else
                try classifyBulletKeyword(allocator, bullet, skill_names.items, skills_dir);

            if (target_skill) |ts| {
                defer allocator.free(ts);
                if (dry_run) {
                    try stdout.print("[DRY-RUN] Would promote to {s}: {s}\n", .{ ts, bullet[0..@min(bullet.len, 60)] });
                    try remaining.append(allocator, try allocator.dupe(u8, line));
                } else {
                    // Append bullet to the target skill file.
                    const skill_file = std.fs.openFileAbsolute(ts, .{ .mode = .write_only }) catch
                        try std.fs.createFileAbsolute(ts, .{});
                    defer skill_file.close();
                    try skill_file.seekFromEnd(0);
                    const bullet_line = try std.fmt.allocPrint(allocator, "\n- {s}\n", .{bullet});
                    defer allocator.free(bullet_line);
                    try skill_file.writeAll(bullet_line);
                    promoted_count += 1;
                    try stdout.print("Promoted to {s}: {s}\n", .{ ts, bullet[0..@min(bullet.len, 60)] });
                }
            } else {
                try remaining.append(allocator, try allocator.dupe(u8, line));
            }
        }

        if (!dry_run and promoted_count > 0) {
            // Rewrite the inbox file with unpromoted lines.
            const out_file = try std.fs.createFileAbsolute(inbox.path, .{ .truncate = true });
            defer out_file.close();
            for (remaining.items) |ln| {
                const ln_line = try std.fmt.allocPrint(allocator, "{s}\n", .{ln});
                defer allocator.free(ln_line);
                try out_file.writeAll(ln_line);
            }
            try stdout.print("✓ Promoted {} {s} bullets.\n", .{ promoted_count, inbox.kind });
        }
    }
    try stdout.flush();
}

/// Classify a bullet using LLM.  Returns an owned absolute path to the skill file, or null.
///
/// The LLM is prompted to return a path like `.ast-guidance/.skills/NAME/SKILL.md`.
/// We extract the skill short-name and resolve it to `.ast-guidance/.skills/NAME/SKILL.md`
/// (preferred), then `.opencode/skills/NAME/SKILL.md`, or `doc/skills/NAME/SKILL.md` (fallback).  The returned path
/// always starts with the project root and ends with `SKILL.md`.
fn classifyBulletLlm(
    allocator: std.mem.Allocator,
    client: *llm.LlmClient,
    bullet: []const u8,
    skills_list: []const u8,
    skills_dir: []const u8,
) !?[]const u8 {
    const prompt = try std.fmt.allocPrint(
        allocator,
        "Given this knowledge bullet, which skill file should it be added to?\nBullet: {s}\nAvailable skills: {s}\n\nReturn only the relative path like: .ast-guidance/.skills/SKILL-NAME/SKILL.md\nOr return: NONE if it doesn't fit any skill.",
        .{ bullet, skills_list },
    );
    defer allocator.free(prompt);

    const resp = try client.complete(prompt, 50, 0.1, null) orelse return null;
    defer allocator.free(resp);

    const trimmed = std.mem.trim(u8, resp, " \t\r\n");
    if (std.ascii.eqlIgnoreCase(trimmed, "NONE")) return null;

    // Extract the first path-looking token from the response.
    const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse trimmed.len;
    const raw_path = trimmed[0..space];

    // Extract skill short-name: second-to-last path component.
    // e.g. ".ast-guidance/.skills/gof-patterns/SKILL.md" → "gof-patterns"
    const skill_name = blk: {
        // Find basename and parent.
        var path = raw_path;
        // strip trailing /SKILL.md or similar
        if (std.mem.endsWith(u8, path, "/SKILL.md")) {
            path = path[0 .. path.len - "/SKILL.md".len];
        }
        // The skill name is the last component.
        const slash = std.mem.lastIndexOfScalar(u8, path, '/') orelse 0;
        const name = if (slash > 0) path[slash + 1 ..] else path;
        break :blk name;
    };

    if (skill_name.len == 0) return null;

    // Derive project root from skills_dir (.ast-guidance/.skills → project root).
    const project_root = blk: {
        // skills_dir is like /project/.ast-guidance/.skills
        if (std.mem.lastIndexOf(u8, skills_dir, "/.ast-guidance/")) |pos| {
            break :blk skills_dir[0..pos];
        }
        // Fallback: use skills_dir parent's parent.
        const p1 = std.fs.path.dirname(skills_dir) orelse skills_dir;
        const p2 = std.fs.path.dirname(p1) orelse p1;
        break :blk p2;
    };

    // Try .ast-guidance/.skills/<name>/SKILL.md first (preferred).
    const primary_path = try std.fs.path.join(allocator, &.{ project_root, ".ast-guidance", ".skills", skill_name, "SKILL.md" });
    if (std.fs.accessAbsolute(primary_path, .{})) {
        if (std.debug.runtime_safety) {
            std.debug.print("[learn] LLM classified to .ast-guidance skill: {s}\n", .{primary_path});
        }
        return primary_path;
    } else |_| {
        allocator.free(primary_path);
    }

    // Fall back to doc/skills/<name>/SKILL.md.
    const fallback_path = try std.fs.path.join(allocator, &.{ project_root, "doc", "skills", skill_name, "SKILL.md" });
    if (std.fs.accessAbsolute(fallback_path, .{})) {
        if (std.debug.runtime_safety) {
            std.debug.print("[learn] LLM classified to doc skill: {s}\n", .{fallback_path});
        }
        return fallback_path;
    } else |_| {
        allocator.free(fallback_path);
    }

    return null;
}

/// Keyword-based fallback: pick the skill whose name has the most tokens in common with bullet.
pub const classifyBulletKeywordPub = classifyBulletKeyword;
fn classifyBulletKeyword(
    allocator: std.mem.Allocator,
    bullet: []const u8,
    skill_names: []const []const u8,
    skills_dir: []const u8,
) !?[]const u8 {
    const bullet_lower = try std.ascii.allocLowerString(allocator, bullet);
    defer allocator.free(bullet_lower);

    var best_skill: ?[]const u8 = null;
    var best_score: u32 = 0;

    for (skill_names) |sname| {
        const sname_lower = try std.ascii.allocLowerString(allocator, sname);
        defer allocator.free(sname_lower);
        var score: u32 = 0;
        var parts = std.mem.splitScalar(u8, sname_lower, '-');
        while (parts.next()) |part| {
            if (part.len < 3) continue;
            if (std.mem.indexOf(u8, bullet_lower, part) != null) score += 1;
        }
        if (score > best_score) {
            best_score = score;
            best_skill = sname;
        }
    }

    if (best_score == 0 or best_skill == null) return null;

    // Build absolute path: skills_dir/<name>/SKILL.md
    const skill_file = try std.fs.path.join(allocator, &.{ skills_dir, best_skill.?, "SKILL.md" });
    std.fs.accessAbsolute(skill_file, .{}) catch {
        allocator.free(skill_file);
        return null;
    };
    return skill_file;
}

// =============================================================================
// triage
// =============================================================================

fn cmdTriage(allocator: std.mem.Allocator, args: []const []const u8) !void {
    const common = llm.parseCommonArgs(args);

    var todo_path: ?[]const u8 = null;
    var guidance_dir_arg: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--guidance")) {
            i += 1;
            if (i >= args.len) return;
            guidance_dir_arg = args[i];
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            todo_path = arg;
        }
    }

    if (guidance_dir_arg == null) {
        std.debug.print("❌ Error: --guidance is required. Usage: ast-guidance triage <todo-path> --guidance <dir> [options]\n", .{});
        return;
    }

    const api_url = common.api_url;
    const model = common.model;

    const tp = todo_path orelse {
        std.debug.print("Error: todo_path required\n", .{});
        return;
    };

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    // Resolve to absolute path.
    const abs_todo = if (std.fs.path.isAbsolute(tp))
        try allocator.dupe(u8, tp)
    else
        try std.fs.path.join(allocator, &.{ cwd, tp });
    defer allocator.free(abs_todo);

    // Read TODO.md
    const todo_file = std.fs.openFileAbsolute(abs_todo, .{}) catch |err| {
        std.debug.print("Error: cannot open {s}: {}\n", .{ abs_todo, err });
        return;
    };
    const todo_content = try todo_file.readToEndAlloc(allocator, 1024 * 1024);
    todo_file.close();
    defer allocator.free(todo_content);

    // Work dir = parent of TODO.md.
    const work_dir = std.fs.path.dirname(abs_todo) orelse cwd;

    // Lifecycle check: must be in TODO state.
    const current_state = try triage_mod.getLifecycleState(allocator, work_dir);
    if (!std.mem.eql(u8, current_state, "TODO")) {
        std.debug.print("Work item is in state '{s}', not TODO. Skipping triage.\n", .{current_state});
        return;
    }

    // Extract work item name (directory basename).
    const work_item_name = std.fs.path.basename(work_dir);

    // Find affected files.
    const affected = try triage_mod.findAffectedFiles(allocator, todo_content, cwd);
    defer {
        for (affected) |f| allocator.free(f);
        allocator.free(affected);
    }

    // Risk assessment.
    const risk = triage_mod.assessRisk(todo_content, affected.len);

    // LLM steps (best-effort).
    var steps_buf: std.ArrayList(u8) = .{};
    defer steps_buf.deinit(allocator);
    const steps_w = steps_buf.writer(allocator);

    const config: llm.LlmConfig = .{ .api_url = api_url, .model = model };
    if (llm.LlmClient.init(allocator, config)) |client_val| {
        var client = client_val;
        defer client.deinit();
        if (client.available()) {
            var files_str: std.ArrayList(u8) = .{};
            defer files_str.deinit(allocator);
            for (affected[0..@min(affected.len, 5)], 0..) |f, fi| {
                if (fi > 0) try files_str.appendSlice(allocator, ", ");
                try files_str.appendSlice(allocator, f);
            }
            const prompt = try std.fmt.allocPrint(
                allocator,
                "Given this TODO work item, list 5-7 concrete implementation steps:\n\n{s}\n\nAffected files: {s}\n\nReturn a numbered list of specific, actionable steps.",
                .{ todo_content[0..@min(todo_content.len, 500)], files_str.items },
            );
            defer allocator.free(prompt);
            if (try client.complete(prompt, 400, 0.3, null)) |steps| {
                defer allocator.free(steps);
                try steps_w.writeAll(steps);
            }
        }
    } else |_| {}

    const steps = if (steps_buf.items.len > 0) steps_buf.items else triage_mod.DEFAULT_STEPS;

    // Get current timestamp.
    const ts = std.time.timestamp();
    const lt = getLocalTime(ts) catch blk: {
        // UTC fallback
        break :blk PosixTm{ .year = 124, .month = 0, .mday = 1, .hour = 0, .minute = 0 };
    };
    const timestamp_str = try std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}", .{
        lt.year + 1900, lt.month + 1, lt.mday, lt.hour, lt.minute,
    });
    defer allocator.free(timestamp_str);

    // Build TRIAGE.md content.
    var triage_buf: std.ArrayList(u8) = .{};
    defer triage_buf.deinit(allocator);
    const tw = triage_buf.writer(allocator);

    try tw.print("# TRIAGE: {s}\n", .{work_item_name});
    try tw.print("\n**Generated**: {s}\n", .{timestamp_str});
    try tw.print("\n## Source TODO\n```\n{s}\n```\n", .{todo_content[0..@min(todo_content.len, 500)]});
    try tw.writeAll("\n## Affected Files\n");
    if (affected.len > 0) {
        for (affected) |f| {
            try tw.print("- `{s}`\n", .{f});
        }
    } else {
        try tw.writeAll("- (none detected automatically)\n");
    }
    try tw.print("\n## Risk Assessment\n{s}\n", .{risk});
    try tw.print("\n## Recommended Steps\n{s}\n", .{steps});
    try tw.writeAll("\n## Lifecycle Status\n\n");
    for (triage_mod.LIFECYCLE) |state| {
        const marker: []const u8 = if (std.mem.eql(u8, state, "TODO")) "✓" else "○";
        try tw.print("- {s} {s}\n", .{ marker, state });
    }
    try tw.writeAll("\n---\n*Advance to WORK by creating a WORK.md in this directory.*\n");

    // Write TRIAGE.md.
    const triage_path = try std.fmt.allocPrint(allocator, "{s}/TRIAGE.md", .{work_dir});
    defer allocator.free(triage_path);
    const triage_file = try std.fs.createFileAbsolute(triage_path, .{ .truncate = true });
    defer triage_file.close();
    try triage_file.writeAll(triage_buf.items);

    std.debug.print("✓ TRIAGE.md created: {s}\n", .{triage_path});
}

// =============================================================================
// deps
// =============================================================================

const DepsArgs = struct {
    src_dir: []const u8 = "src",
};

fn cmdDeps(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var dep_args: DepsArgs = .{};
    var src_dir_override: ?[]const u8 = null;
    var i: usize = 0;

    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--src")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Missing argument for --src\n", .{});
                return;
            }
            src_dir_override = args[i];
        }
    }

    const cwd = try std.process.getCwdAlloc(allocator);
    defer allocator.free(cwd);

    // Use --src override if given; otherwise take first src_dir from config.
    var src_dir_owned: ?[]const u8 = null;
    defer if (src_dir_owned) |d| allocator.free(d);

    if (src_dir_override) |d| {
        dep_args.src_dir = d;
    } else if (config_mod.loadConfig(allocator, cwd)) |cfg_val| {
        var cfg = cfg_val;
        defer cfg.deinit();
        if (cfg.src_dirs.len > 0) {
            src_dir_owned = try allocator.dupe(u8, cfg.src_dirs[0]);
            dep_args.src_dir = src_dir_owned.?;
        }
    } else |_| {}

    var generator = deps.DepsGenerator.init(allocator, cwd);
    try generator.generateDependencies(dep_args.src_dir);
}

// =============================================================================
// Tests
// =============================================================================

test "main compiles" {
    try std.testing.expect(true);
}

comptime {
    _ = @import("tests.zig");
    _ = @import("json_store.zig");
    _ = @import("types.zig");
    _ = @import("enhancer.zig");
    _ = @import("triage.zig");
}
