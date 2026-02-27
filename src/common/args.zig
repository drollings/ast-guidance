const std = @import("std");

pub const CommonArgs = struct {
    debug: bool = false,
    no_ai: bool = false,
    api_url: []const u8 = "http://localhost:11434/api/chat",
    model: []const u8 = "fast:latest",
    dry_run: bool = false,
    /// True when --api-url was explicitly provided on the command line.
    api_url_set: bool = false,
    /// True when --model / -m was explicitly provided on the command line.
    model_set: bool = false,
};

pub fn parseCommonArgs(args: []const []const u8) CommonArgs {
    var result: CommonArgs = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--debug")) {
            result.debug = true;
        } else if (std.mem.eql(u8, arg, "--no-ai")) {
            result.no_ai = true;
        } else if (std.mem.eql(u8, arg, "--api-url")) {
            i += 1;
            if (i < args.len) {
                result.api_url = args[i];
                result.api_url_set = true;
            }
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i < args.len) {
                result.model = args[i];
                result.model_set = true;
            }
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            result.dry_run = true;
        }
    }
    return result;
}
