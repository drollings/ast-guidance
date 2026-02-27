/// ast-guidance project configuration loader.
///
/// Resolves paths for the guidance system using a two-level fallback chain:
///   1. {cwd}/.ast-guidance/ast-guidance-config.json  (project-local)
///   2. ~/.config/ast-guidance/ast-guidance-config.json  (user global)
///   3. Built-in defaults
///
/// All path fields in ProjectConfig are pre-computed absolute paths so callers
/// do not need to allocate to derive them.
const std = @import("std");
const builtin = @import("builtin");

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

pub const DEFAULT_GUIDANCE_DIR = ".ast-guidance";
pub const DEFAULT_SRC_DIR = "src";
pub const DEFAULT_MODEL = "code:latest";
pub const DEFAULT_API_URL = "http://localhost:11434/api/chat";

// ---------------------------------------------------------------------------
// ProjectConfig
// ---------------------------------------------------------------------------

/// Resolved, absolute paths for a single ast-guidance project instance.
/// All strings are owned by this struct; call deinit() to free them.
pub const ProjectConfig = struct {
    allocator: std.mem.Allocator,

    /// Absolute path to the guidance root (e.g. /project/.ast-guidance).
    guidance_root: []const u8,

    /// Absolute path to the guidance JSON source tree
    /// ({guidance_root}/{src_rel_prefix}).
    /// JSON for src/foo.zig lives at {guidance_root}/src/foo.zig.json
    /// so the formula is: {guidance_root}/{rel_from_project_root}.json
    /// This field stores guidance_root for that formula (callers append the rel path).
    json_base: []const u8,

    /// Absolute path to the skills directory ({guidance_root}/.skills).
    skills_dir: []const u8,

    /// Absolute path to the inbox directory ({guidance_root}/.doc/inbox).
    inbox_dir: []const u8,

    /// Source directories to search, relative to the project root.
    /// Defaults to ["src"]; read from config JSON "src_dirs" array.
    src_dirs: []const []const u8,

    /// LLM API endpoint for infill/regen (from config or default).
    api_url: []const u8,

    /// Model name for infill/regen (from config or default).
    model: []const u8,

    pub fn deinit(self: *ProjectConfig) void {
        self.allocator.free(self.guidance_root);
        self.allocator.free(self.json_base);
        self.allocator.free(self.skills_dir);
        self.allocator.free(self.inbox_dir);
        for (self.src_dirs) |d| self.allocator.free(d);
        self.allocator.free(self.src_dirs);
        self.allocator.free(self.api_url);
        self.allocator.free(self.model);
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load project configuration with the two-level fallback chain.
/// On success the caller owns the returned ProjectConfig and must call deinit().
pub fn loadConfig(allocator: std.mem.Allocator, cwd: []const u8) !ProjectConfig {
    // 1. Project-local config.
    {
        const path = try std.fs.path.join(allocator, &.{ cwd, DEFAULT_GUIDANCE_DIR, "ast-guidance-config.json" });
        defer allocator.free(path);
        if (tryLoadFile(allocator, cwd, path)) |cfg| return cfg else |err| {
            if (err != error.FileNotFound and !builtin.is_test) {
                std.debug.print("warning: config file {s} is invalid ({}) — using defaults\n", .{ path, err });
            }
        }
    }

    // 2. User-global config (~/.config/ast-guidance/ast-guidance-config.json).
    if (std.process.getEnvVarOwned(allocator, "HOME") catch null) |home| {
        defer allocator.free(home);
        const path = try std.fs.path.join(allocator, &.{ home, ".config", "ast-guidance", "ast-guidance-config.json" });
        defer allocator.free(path);
        if (tryLoadFile(allocator, cwd, path)) |cfg| return cfg else |err| {
            if (err != error.FileNotFound and !builtin.is_test) {
                std.debug.print("warning: config file {s} is invalid ({}) — using defaults\n", .{ path, err });
            }
        }
    }

    // 3. Built-in defaults.
    return buildDefault(allocator, cwd);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Attempt to load and parse a config file.  Returns error on any failure so
/// the caller can silently fall through to the next source.
fn tryLoadFile(allocator: std.mem.Allocator, cwd: []const u8, path: []const u8) !ProjectConfig {
    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 64 * 1024);
    defer allocator.free(content);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.InvalidConfig;

    // --- guidance_dir (relative string, optional) ---
    const guidance_dir_rel: []const u8 = if (root.object.get("guidance_dir")) |gd|
        if (gd == .string) gd.string else DEFAULT_GUIDANCE_DIR
    else
        DEFAULT_GUIDANCE_DIR;

    // --- src_dirs (array of strings, optional) ---
    var src_dirs: std.ArrayList([]const u8) = .{};
    errdefer {
        for (src_dirs.items) |d| allocator.free(d);
        src_dirs.deinit(allocator);
    }
    if (root.object.get("src_dirs")) |sd| {
        if (sd == .array) {
            for (sd.array.items) |item| {
                if (item == .string) {
                    try src_dirs.append(allocator, try allocator.dupe(u8, item.string));
                }
            }
        }
    }
    if (src_dirs.items.len == 0) {
        try src_dirs.append(allocator, try allocator.dupe(u8, DEFAULT_SRC_DIR));
    }

    // --- LLM settings ---
    const model = blk: {
        if (root.object.get("models")) |models| {
            if (models == .object) {
                // Prefer "infill" model; fall back to "default".
                if (models.object.get("infill")) |m| if (m == .string) break :blk m.string;
                if (models.object.get("default")) |m| if (m == .string) break :blk m.string;
            }
        }
        break :blk DEFAULT_MODEL;
    };

    const api_url = blk: {
        if (root.object.get("ollama")) |ollama| {
            if (ollama == .object) {
                const base = if (ollama.object.get("base_url")) |u| if (u == .string) u.string else "" else "";
                const ep = if (ollama.object.get("chat_endpoint")) |e| if (e == .string) e.string else DEFAULT_API_URL else DEFAULT_API_URL;
                if (base.len > 0 and ep.len > 0) {
                    // Avoid double slash between base and endpoint.
                    const ep_clean = if (ep.len > 0 and ep[0] == '/') ep else ep;
                    break :blk try std.fmt.allocPrint(allocator, "{s}{s}", .{ base, ep_clean });
                }
            }
        }
        break :blk try allocator.dupe(u8, DEFAULT_API_URL);
    };
    errdefer allocator.free(api_url);

    return buildFromParts(
        allocator,
        cwd,
        guidance_dir_rel,
        try src_dirs.toOwnedSlice(allocator),
        try allocator.dupe(u8, model),
        api_url,
    );
}

fn buildDefault(allocator: std.mem.Allocator, cwd: []const u8) !ProjectConfig {
    var src_dirs = try allocator.alloc([]const u8, 1);
    src_dirs[0] = try allocator.dupe(u8, DEFAULT_SRC_DIR);
    return buildFromParts(
        allocator,
        cwd,
        DEFAULT_GUIDANCE_DIR,
        src_dirs,
        try allocator.dupe(u8, DEFAULT_MODEL),
        try allocator.dupe(u8, DEFAULT_API_URL),
    );
}

fn buildFromParts(
    allocator: std.mem.Allocator,
    cwd: []const u8,
    guidance_dir_rel: []const u8,
    src_dirs: []const []const u8,
    model: []const u8,
    api_url: []const u8,
) !ProjectConfig {
    const guidance_root = if (std.fs.path.isAbsolute(guidance_dir_rel))
        try allocator.dupe(u8, guidance_dir_rel)
    else
        try std.fs.path.join(allocator, &.{ cwd, guidance_dir_rel });
    errdefer allocator.free(guidance_root);

    // json_base == guidance_root: callers compute {guidance_root}/{rel}.json
    const json_base = try allocator.dupe(u8, guidance_root);
    errdefer allocator.free(json_base);

    const skills_dir = try std.fs.path.join(allocator, &.{ guidance_root, ".skills" });
    errdefer allocator.free(skills_dir);

    const inbox_dir = try std.fs.path.join(allocator, &.{ guidance_root, ".doc", "inbox" });
    errdefer allocator.free(inbox_dir);

    return .{
        .allocator = allocator,
        .guidance_root = guidance_root,
        .json_base = json_base,
        .skills_dir = skills_dir,
        .inbox_dir = inbox_dir,
        .src_dirs = src_dirs,
        .api_url = api_url,
        .model = model,
    };
}
