const std = @import("std");
const args = @import("args.zig");

pub const CommonArgs = args.CommonArgs;
pub const parseCommonArgs = args.parseCommonArgs;

pub const LlmError = error{
    InvalidUrl,
    ConnectionFailed,
    TlsError,
    RequestFailed,
    ParseError,
    OutOfMemory,
};

pub const LlmConfig = struct {
    api_url: []const u8,
    model: []const u8,
    timeout_ms: u32 = 30000,
    debug: bool = false,
};

fn writeEscapedString(writer: anytype, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            '{' => try writer.writeAll("\\u007B"),
            '}' => try writer.writeAll("\\u007D"),
            else => try writer.writeByte(c),
        }
    }
}

/// Strip <think>...</think> or [THINK]...[/THINK] blocks from LLM response.
/// For unclosed tags, strips everything from the open tag to end-of-string.
pub fn stripThinkBlock(text: []const u8) []const u8 {
    // Handle <think> ... </think>
    if (std.mem.indexOf(u8, text, "<think>")) |think_start| {
        const think_end = std.mem.indexOfPos(u8, text, think_start + 7, "</think>");
        if (think_end) |te| {
            const after_think = te + 8;
            if (after_think >= text.len) return "";
            var start = after_think;
            while (start < text.len and (text[start] == ' ' or text[start] == '\n')) {
                start += 1;
            }
            return text[start..];
        } else {
            // Unclosed <think> tag — strip everything from it to end.
            return std.mem.trim(u8, text[0..think_start], " \t\r\n");
        }
    }

    // Handle [THINK] ... [/THINK] (alternative format)
    if (std.mem.indexOf(u8, text, "[THINK]")) |think_start| {
        const think_end = std.mem.indexOfPos(u8, text, think_start + 7, "[/THINK]");
        if (think_end) |te| {
            const after_think = te + 8;
            if (after_think >= text.len) return "";
            var start = after_think;
            while (start < text.len and (text[start] == ' ' or text[start] == '\n')) {
                start += 1;
            }
            return text[start..];
        } else {
            // Unclosed [THINK] tag — strip from it to end.
            return std.mem.trim(u8, text[0..think_start], " \t\r\n");
        }
    }

    return text;
}

/// Strip LLM reasoning preamble from the start of a response.
/// Preambles are phrases like "Let's", "Here's", "I'll", "To answer", etc.
/// Removes the first line if it matches one of these patterns.
pub fn stripPreamble(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return allocator.dupe(u8, trimmed);

    // Find first newline to isolate the first line.
    const nl_pos = std.mem.indexOfScalar(u8, trimmed, '\n') orelse trimmed.len;
    const first_line = trimmed[0..nl_pos];

    const preambles = [_][]const u8{
        "let's ", "let me ", "we need to ",    "here's ",    "here is ",
        "i'll ",  "i will ", "the answer is ", "to answer ", "okay, ",
        "ok, ",   "sure, ",  "alright, ",
    };

    const first_lower = try std.ascii.allocLowerString(allocator, first_line);
    defer allocator.free(first_lower);

    for (preambles) |preamble| {
        if (std.mem.startsWith(u8, first_lower, preamble)) {
            // Skip this preamble line; return the rest.
            if (nl_pos >= trimmed.len) return allocator.dupe(u8, "");
            const rest = std.mem.trim(u8, trimmed[nl_pos + 1 ..], " \t\r\n");
            return allocator.dupe(u8, rest);
        }
    }

    return allocator.dupe(u8, trimmed);
}

// ---------------------------------------------------------------------------
// LLM output validation — write-time gate for comment fields
// ---------------------------------------------------------------------------

/// Return true when an LLM response is malformed or unusable as a comment.
/// Call this after stripThinkBlock / stripPreamble.  No allocations.
///
/// Patterns detected:
///   - Empty / whitespace only
///   - Truncated output (dangling preposition/article at end)
///   - Ends with "?" (uncertain / incomplete analysis)
///   - Generic self-referential filler ("this function", "this struct", …)
///   - Single overly-generic word ("helper", "wrapper", …)
///   - LLM preamble phrases present anywhere in the response
pub fn isMalformedResponse(text: []const u8) bool {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return true;

    if (llmHasDanglingEnd(trimmed)) return true;

    const rtrimmed = std.mem.trimRight(u8, trimmed, " \t");
    if (rtrimmed.len > 0 and rtrimmed[rtrimmed.len - 1] == '?') return true;

    if (llmIsGenericSelfRef(trimmed)) return true;
    if (llmIsOverlyGeneric(trimmed)) return true;

    if (llmContainsIgnoreCase(trimmed, "here's a")) return true;
    if (llmContainsIgnoreCase(trimmed, "here is a")) return true;
    if (llmContainsIgnoreCase(trimmed, "i'll ")) return true;
    if (llmContainsIgnoreCase(trimmed, "to summarize")) return true;
    if (llmContainsIgnoreCase(trimmed, "okay,")) return true;
    if (llmContainsIgnoreCase(trimmed, "ok,")) return true;

    // Reasoning-model chain-of-thought that survived stripPreamble (multi-line monologues).
    if (llmContainsIgnoreCase(trimmed, "we need ")) return true;
    if (llmContainsIgnoreCase(trimmed, "let's think")) return true;
    if (llmContainsIgnoreCase(trimmed, "let's craft")) return true;
    if (llmContainsIgnoreCase(trimmed, "let's count")) return true;
    if (llmContainsIgnoreCase(trimmed, "let me think")) return true;
    if (llmContainsIgnoreCase(trimmed, "i need to ")) return true;

    return false;
}

fn llmHasDanglingEnd(body: []const u8) bool {
    const trimmed = std.mem.trimRight(u8, body, " \t.?");
    if (trimmed.len == 0) return false;
    var i: usize = trimmed.len;
    while (i > 0 and trimmed[i - 1] != ' ') i -= 1;
    const last_word = trimmed[i..];
    const danglers = [_][]const u8{ "of", "in", "for", "from", "with", "to", "a", "an", "the" };
    for (danglers) |d| {
        if (std.ascii.eqlIgnoreCase(last_word, d)) return true;
    }
    return false;
}

fn llmIsGenericSelfRef(body: []const u8) bool {
    const patterns = [_][]const u8{
        "this function", "this method", "this class",
        "this struct",   "this type",   "this module",
    };
    const trimmed = std.mem.trim(u8, body, " \t\r\n.");
    for (patterns) |p| {
        if (std.ascii.eqlIgnoreCase(trimmed, p)) return true;
    }
    return false;
}

fn llmIsOverlyGeneric(body: []const u8) bool {
    const generics = [_][]const u8{
        "function", "method",   "helper",  "util",           "utility",
        "handler",  "callback", "wrapper", "implementation",
    };
    const trimmed = std.mem.trim(u8, body, " \t\r\n.");
    if (trimmed.len > 20) return false;
    if (std.mem.indexOfScalar(u8, trimmed, ' ') != null) return false;
    for (generics) |g| {
        if (std.ascii.eqlIgnoreCase(trimmed, g)) return true;
    }
    return false;
}

fn llmContainsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

/// Extract the content of the first <comment>...</comment> tag in an LLM response.
/// Returns a slice into `text` (no allocation).  Returns null when no tag is found.
/// The model may emit any amount of chain-of-thought before or after the tag;
/// only the tag content is returned.
pub fn extractCommentTag(text: []const u8) ?[]const u8 {
    const open = "<comment>";
    const close = "</comment>";
    const start = std.mem.indexOf(u8, text, open) orelse return null;
    const content_start = start + open.len;
    const end = std.mem.indexOfPos(u8, text, content_start, close) orelse return null;
    const content = std.mem.trim(u8, text[content_start..end], " \t\r\n");
    if (content.len == 0) return null;
    return content;
}

pub const LlmClient = struct {
    allocator: std.mem.Allocator,
    config: LlmConfig,
    is_openai_format: bool,
    chat_url: []const u8,
    http_client: std.http.Client,

    pub fn init(allocator: std.mem.Allocator, config: LlmConfig) !LlmClient {
        const is_openai = std.mem.indexOf(u8, config.api_url, "api.openai.com") != null;

        var chat_url: []const u8 = undefined;
        if (is_openai) {
            chat_url = try allocator.dupe(u8, config.api_url);
        } else {
            const is_ollama_v1 = std.mem.indexOf(u8, config.api_url, "/v1/completions") != null;
            if (is_ollama_v1) {
                chat_url = try std.mem.replaceOwned(u8, allocator, config.api_url, "/v1/completions", "/api/chat");
            } else {
                chat_url = try allocator.dupe(u8, config.api_url);
            }
        }

        const http_client = std.http.Client{ .allocator = allocator };

        return .{
            .allocator = allocator,
            .config = config,
            .is_openai_format = is_openai,
            .chat_url = chat_url,
            .http_client = http_client,
        };
    }

    pub fn deinit(self: *LlmClient) void {
        self.allocator.free(self.chat_url);
        self.http_client.deinit();
    }

    pub fn complete(self: *LlmClient, prompt: []const u8, max_tokens: usize, temperature: f32, system: ?[]const u8) LlmError!?[]const u8 {
        var body: std.ArrayList(u8) = .{};
        defer body.deinit(self.allocator);
        const writer = body.writer(self.allocator);

        if (self.is_openai_format) {
            try writer.writeAll("{\"model\":\"");
            try writer.writeAll(self.config.model);
            try writer.writeAll("\",\"prompt\":\"");
            try writeEscapedString(writer, prompt);
            try writer.print("\",\"max_tokens\":{d},\"temperature\":{d},\"stream\":false}}", .{ max_tokens, temperature });
        } else {
            try writer.writeAll("{\"model\":\"");
            try writer.writeAll(self.config.model);
            try writer.writeAll("\",\"messages\":[");
            if (system) |sys| {
                try writer.writeAll("{\"role\":\"system\",\"content\":\"");
                try writeEscapedString(writer, sys);
                try writer.writeAll("\"},");
            }
            try writer.writeAll("{\"role\":\"user\",\"content\":\"");
            try writeEscapedString(writer, prompt);
            // "think":false is a top-level Ollama field (not inside "options").
            // It disables chain-of-thought for models that support it (DeepSeek-R1,
            // qwen3, etc.) so reasoning tokens don't consume the num_predict budget
            // before the model can emit a <comment> tag.
            try writer.writeAll("\"}],\"stream\":false,\"think\":false,\"options\":{\"temperature\":");
            // Write temperature as string to avoid format issues with braces
            if (temperature < 1.0) {
                try writer.writeAll("0.");
                try writer.print("{d}", .{@as(u32, @intFromFloat(temperature * 10))});
            } else {
                try writer.print("{d}", .{@as(u32, @intFromFloat(temperature))});
            }
            try writer.writeAll(",\"num_predict\":");
            try writer.print("{d}", .{@as(usize, @max(100, max_tokens))});
            try writer.writeAll("}}");
        }

        const url = if (self.is_openai_format) self.config.api_url else self.chat_url;

        // Use curl via shell to handle the request
        return self.completeViaShell(url, body.items);
    }

    fn completeViaShell(self: *LlmClient, url: []const u8, json_body: []const u8) LlmError!?[]const u8 {
        // Write body to temp file
        const tmp_path = "/tmp/llm_request.json";
        if (std.fs.createFileAbsolute(tmp_path, .{ .truncate = true })) |f| {
            defer f.close();
            f.writeAll(json_body) catch return LlmError.RequestFailed;
        } else |_| {
            return LlmError.RequestFailed;
        }

        // Use curl with -d @filename syntax
        const cmd = try std.fmt.allocPrint(self.allocator, "curl -s -X POST -H 'Content-Type: application/json' -d @{s} {s}", .{ tmp_path, url });
        defer self.allocator.free(cmd);

        if (self.config.debug) {
            std.debug.print("DEBUG: running curl: {s}\n", .{cmd});
        }

        var child = std.process.Child.init(&[_][]const u8{ "sh", "-c", cmd }, self.allocator);
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Ignore;

        child.spawn() catch return LlmError.RequestFailed;

        const stdout = child.stdout.?.readToEndAlloc(self.allocator, 1024 * 1024) catch return LlmError.RequestFailed;
        defer self.allocator.free(stdout);

        if (self.config.debug) {
            std.debug.print("DEBUG: curl response: {s}\n", .{stdout[0..@min(200, stdout.len)]});
        }

        const term = child.wait() catch return LlmError.RequestFailed;
        if (term != .Exited or term.Exited != 0) {
            return null;
        }

        return self.extractResponseText(stdout);
    }

    fn extractResponseText(self: *LlmClient, resp: []const u8) ?[]const u8 {
        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, resp, .{}) catch return null;
        defer parsed.deinit();

        const root = parsed.value;

        if (root.object.get("choices")) |choices| {
            if (choices.array.items.len > 0) {
                const choice = choices.array.items[0];
                if (choice.object.get("text")) |text| {
                    return self.allocator.dupe(u8, text.string) catch null;
                }
                if (choice.object.get("message")) |msg| {
                    if (msg.object.get("content")) |content| {
                        return self.allocator.dupe(u8, content.string) catch null;
                    }
                }
            }
        }

        if (root.object.get("message")) |msg| {
            // Prefer `content`; fall back to `thinking` for models that route
            // chain-of-thought output there (e.g. DeepSeek-R1 style Ollama models).
            const content_str: ?[]const u8 = blk: {
                if (msg.object.get("content")) |c| {
                    const s = switch (c) {
                        .string => |sv| sv,
                        else => break :blk null,
                    };
                    if (s.len > 0) break :blk s;
                }
                if (msg.object.get("thinking")) |t| {
                    const s = switch (t) {
                        .string => |sv| sv,
                        else => break :blk null,
                    };
                    if (s.len > 0) break :blk s;
                }
                break :blk null;
            };
            if (content_str) |s| return self.allocator.dupe(u8, s) catch null;
        }

        if (root.object.get("response")) |resp_val| {
            return self.allocator.dupe(u8, resp_val.string) catch null;
        }

        return null;
    }

    pub fn available(self: *LlmClient) bool {
        // Build the health-check URL based on format:
        // OpenAI: .../v1/models
        // Ollama /api/chat → check /api/tags
        // Ollama /v1/completions → check /api/tags
        const check_url = if (self.is_openai_format) blk: {
            // OpenAI: replace last path segment with /v1/models
            if (std.mem.indexOf(u8, self.config.api_url, "/v1/")) |pos| {
                const base = self.config.api_url[0..pos];
                break :blk std.fmt.allocPrint(self.allocator, "{s}/v1/models", .{base}) catch return false;
            }
            break :blk std.mem.replaceOwned(u8, self.allocator, self.config.api_url, "/v1/completions", "/v1/models") catch return false;
        } else blk: {
            // Ollama: derive base URL and append /api/tags
            const url = self.config.api_url;
            // Find the scheme://host:port part (up to the first path component after /)
            // e.g. "http://localhost:11434/api/chat" → "http://localhost:11434"
            const scheme_end = std.mem.indexOf(u8, url, "://") orelse 0;
            const host_start = if (scheme_end > 0) scheme_end + 3 else 0;
            const path_start = std.mem.indexOfScalarPos(u8, url, host_start, '/') orelse url.len;
            const base = url[0..path_start];
            break :blk std.fmt.allocPrint(self.allocator, "{s}/api/tags", .{base}) catch return false;
        };
        defer self.allocator.free(check_url);

        const uri = std.Uri.parse(check_url) catch return false;

        var req = self.http_client.request(.GET, uri, .{}) catch return false;
        defer req.deinit();

        req.sendBodiless() catch return false;

        var redirect_buffer: [1024]u8 = undefined;
        const response = req.receiveHead(&redirect_buffer) catch return false;

        return response.head.status == .ok;
    }
};

test "LlmClient init with api/chat URL uses it directly" {
    const allocator = std.testing.allocator;
    const config = LlmConfig{
        .api_url = "http://localhost:11434/api/chat",
        .model = "code",
        .debug = false,
    };

    var client = try LlmClient.init(allocator, config);
    defer client.deinit();

    try std.testing.expect(!client.is_openai_format);
    try std.testing.expectEqualStrings("http://localhost:11434/api/chat", client.chat_url);
}

test "LlmClient init with v1/completions URL converts to api/chat" {
    const allocator = std.testing.allocator;
    const config = LlmConfig{
        .api_url = "http://localhost:11434/v1/completions",
        .model = "code",
        .debug = false,
    };

    var client = try LlmClient.init(allocator, config);
    defer client.deinit();

    try std.testing.expect(!client.is_openai_format);
    try std.testing.expectEqualStrings("http://localhost:11434/api/chat", client.chat_url);
}

test "LlmClient OpenAI format" {
    const allocator = std.testing.allocator;
    const config = LlmConfig{
        .api_url = "https://api.openai.com/v1/completions",
        .model = "gpt-4",
        .debug = false,
    };

    var client = try LlmClient.init(allocator, config);
    defer client.deinit();

    try std.testing.expect(client.is_openai_format);
    try std.testing.expectEqualStrings("https://api.openai.com/v1/completions", client.chat_url);
}

test "stripThinkBlock removes think tags" {
    const text1 = "<think>Some thinking</think>\nActual response";
    const result1 = stripThinkBlock(text1);
    try std.testing.expectEqualStrings("Actual response", result1);

    const text2 = "No think tags here";
    const result2 = stripThinkBlock(text2);
    try std.testing.expectEqualStrings(text2, result2);

    const text3 = "<think>Only think</think>";
    const result3 = stripThinkBlock(text3);
    try std.testing.expectEqualStrings("", result3);
}

test "stripThinkBlock handles unclosed think tag" {
    const text = "<think>Reasoning that never ends";
    const result = stripThinkBlock(text);
    // Should strip everything from <think> to end.
    try std.testing.expectEqualStrings("", result);
}

test "stripThinkBlock handles [THINK] tags" {
    const text = "[THINK]reasoning here[/THINK]\nActual answer";
    const result = stripThinkBlock(text);
    try std.testing.expectEqualStrings("Actual answer", result);
}

test "stripPreamble removes leading preamble line" {
    const allocator = std.testing.allocator;

    const text1 = "Let's analyze this function.\nParses JSON from a byte slice.";
    const r1 = try stripPreamble(allocator, text1);
    defer allocator.free(r1);
    try std.testing.expectEqualStrings("Parses JSON from a byte slice.", r1);

    const text2 = "Here's the description:\nBuilds the dep graph.";
    const r2 = try stripPreamble(allocator, text2);
    defer allocator.free(r2);
    try std.testing.expectEqualStrings("Builds the dep graph.", r2);

    const text3 = "Parses JSON tokens efficiently.";
    const r3 = try stripPreamble(allocator, text3);
    defer allocator.free(r3);
    try std.testing.expectEqualStrings("Parses JSON tokens efficiently.", r3);
}

test "isMalformedResponse: empty is malformed" {
    try std.testing.expect(isMalformedResponse(""));
    try std.testing.expect(isMalformedResponse("   "));
}

test "isMalformedResponse: dangling preposition" {
    try std.testing.expect(isMalformedResponse("Parses the input from"));
    try std.testing.expect(isMalformedResponse("Returns the value of"));
}

test "isMalformedResponse: ends with question mark" {
    try std.testing.expect(isMalformedResponse("Does something?"));
}

test "isMalformedResponse: generic self-reference" {
    try std.testing.expect(isMalformedResponse("this function"));
    try std.testing.expect(isMalformedResponse("This Method"));
    try std.testing.expect(!isMalformedResponse("this function parses JSON efficiently"));
}

test "isMalformedResponse: overly generic single word" {
    try std.testing.expect(isMalformedResponse("helper"));
    try std.testing.expect(isMalformedResponse("wrapper"));
    try std.testing.expect(!isMalformedResponse("Parses"));
}

test "isMalformedResponse: LLM preamble phrases" {
    try std.testing.expect(isMalformedResponse("Here's a description of the function"));
    try std.testing.expect(isMalformedResponse("I'll explain what this does"));
    try std.testing.expect(isMalformedResponse("To summarize, this parses JSON"));
    try std.testing.expect(isMalformedResponse("Okay, so this function"));
}

test "isMalformedResponse: valid responses are NOT malformed" {
    try std.testing.expect(!isMalformedResponse("Parses Zig AST tokens and extracts public members."));
    try std.testing.expect(!isMalformedResponse("Ring buffer for streaming price data."));
    try std.testing.expect(!isMalformedResponse("Builds incremental dependency graph from @import declarations."));
}

test "isMalformedResponse: reasoning-model chain-of-thought phrases" {
    try std.testing.expect(isMalformedResponse("we need to write a comment for this type"));
    try std.testing.expect(isMalformedResponse("We Need To write a single-line comment"));
    try std.testing.expect(isMalformedResponse("we need a better approach here"));
    try std.testing.expect(isMalformedResponse("let's think about what this does"));
    try std.testing.expect(isMalformedResponse("Let's craft a comment: something like"));
    try std.testing.expect(isMalformedResponse("let's count characters: Stores and parses"));
    try std.testing.expect(isMalformedResponse("let me think about the ownership model"));
    try std.testing.expect(isMalformedResponse("i need to mention that it owns the allocator"));
}

test "extractCommentTag: returns tag content" {
    const text = "some reasoning\n<comment>Parses JSON from a byte slice.</comment>\nmore text";
    const result = extractCommentTag(text);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("Parses JSON from a byte slice.", result.?);
}

test "extractCommentTag: trims whitespace inside tag" {
    const text = "<comment>  Builds dependency graph.  </comment>";
    const result = extractCommentTag(text);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("Builds dependency graph.", result.?);
}

test "extractCommentTag: returns null when no tag present" {
    try std.testing.expect(extractCommentTag("we need to write a comment for this type") == null);
    try std.testing.expect(extractCommentTag("Parses JSON.") == null);
    try std.testing.expect(extractCommentTag("") == null);
}

test "extractCommentTag: returns null for empty tag" {
    try std.testing.expect(extractCommentTag("<comment>   </comment>") == null);
}

test "extractCommentTag: chain-of-thought before tag is ignored" {
    const text =
        \\We need to write a comment for DepsGenerator.
        \\The comment should be plain English...
        \\Let's craft: something like "Generates dependency graph".
        \\<comment>[skills: zig-current] Walks src/ and resolves @import paths to build a dep graph.</comment>
    ;
    const result = extractCommentTag(text);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("[skills: zig-current] Walks src/ and resolves @import paths to build a dep graph.", result.?);
}

test "writeEscapedString escapes properly" {
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const writer = fbs.writer();

    try writeEscapedString(writer, "Hello \"world\"\n");
    try std.testing.expectEqualStrings("Hello \\\"world\\\"\\n", fbs.getWritten());
}
