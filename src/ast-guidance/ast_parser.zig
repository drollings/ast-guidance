const std = @import("std");
const types = @import("types.zig");
const hash = @import("hash.zig");
const pattern = @import("pattern.zig");

pub const AstParser = struct {
    allocator: std.mem.Allocator,
    source: [:0]const u8,
    tree: std.zig.Ast,

    pub fn init(allocator: std.mem.Allocator, source: [:0]const u8) !AstParser {
        const tree = try std.zig.Ast.parse(allocator, source, .zig);
        return .{
            .allocator = allocator,
            .source = source,
            .tree = tree,
        };
    }

    pub fn deinit(self: *AstParser) void {
        self.tree.deinit(self.allocator);
    }

    pub fn hasErrors(self: *const AstParser) bool {
        return self.tree.errors.len > 0;
    }

    pub fn extractMembers(self: *AstParser) ![]types.Member {
        var members: std.ArrayList(types.Member) = .{};
        errdefer members.deinit(self.allocator);

        for (self.tree.rootDecls()) |node_idx| {
            if (try self.extractMember(node_idx)) |member| {
                try members.append(self.allocator, member);
            }
        }

        return try members.toOwnedSlice(self.allocator);
    }

    fn extractMember(self: *AstParser, node_idx: std.zig.Ast.Node.Index) !?types.Member {
        const tag = self.tree.nodeTag(node_idx);

        switch (tag) {
            .fn_decl => {
                return try self.extractFunction(node_idx);
            },
            .global_var_decl, .local_var_decl, .simple_var_decl, .aligned_var_decl => {
                return try self.extractVarDecl(node_idx);
            },
            .test_decl => {
                return try self.extractTest(node_idx);
            },
            else => return null,
        }
    }

    fn extractFunction(self: *AstParser, node_idx: std.zig.Ast.Node.Index) !?types.Member {
        const fn_proto, const body = self.tree.nodeData(node_idx).node_and_node;
        _ = body;

        var buf: [1]std.zig.Ast.Node.Index = undefined;
        const full_proto = self.tree.fullFnProto(&buf, fn_proto) orelse return null;

        const name = if (full_proto.name_token) |name_tok|
            self.tree.tokenSlice(name_tok)
        else
            return null;

        if (std.mem.startsWith(u8, name, "test")) return null;

        const is_pub = full_proto.visib_token != null;

        const params = try self.extractParams(full_proto);

        var return_type: ?[]const u8 = null;
        if (full_proto.ast.return_type != .none) {
            return_type = try self.nodeToType(full_proto.ast.return_type);
        }

        const signature = try buildSignature(self.allocator, name, params, return_type);

        const match_hash = try hash.apiHash(self.allocator, name, params, return_type);
        errdefer self.allocator.free(match_hash);

        const comment = try self.extractDocstring(node_idx);

        const patterns_detected = try pattern.detectPatterns(self.allocator, &self.tree, fn_proto);

        const member_type: types.MemberType = if (is_pub) .fn_decl else .fn_private;

        return types.Member{
            .type = member_type,
            .name = try self.allocator.dupe(u8, name),
            .match_hash = match_hash,
            .signature = signature,
            .params = params,
            .returns = return_type,
            .comment = comment,
            .patterns = patterns_detected,
            .is_pub = is_pub,
            .line = if (full_proto.name_token) |t| @as(u32, @intCast(self.tree.tokenLocation(0, t).line + 1)) else null,
        };
    }

    fn extractVarDecl(self: *AstParser, node_idx: std.zig.Ast.Node.Index) !?types.Member {
        const full_var = self.tree.fullVarDecl(node_idx) orelse return null;

        const name_token = full_var.ast.mut_token + 1;
        const name = self.tree.tokenSlice(name_token);
        if (std.mem.startsWith(u8, name, "_")) return null;

        const is_pub = full_var.visib_token != null;

        if (full_var.ast.init_node != .none) {
            const init_node = full_var.ast.init_node.unwrap() orelse return null;
            const init_tag = self.tree.nodeTag(init_node);
            if (init_tag == .container_decl or init_tag == .container_decl_trailing or
                init_tag == .container_decl_arg or init_tag == .container_decl_arg_trailing)
            {
                // Pass node_idx (the var_decl) so extractContainer can find the
                // "///" comment that precedes "pub const Foo = struct {".
                return try self.extractContainer(name_token, init_node, node_idx, is_pub);
            }
        }

        return null;
    }

    fn extractContainer(self: *AstParser, name_token: std.zig.Ast.TokenIndex, init_node: std.zig.Ast.Node.Index, var_decl_node: std.zig.Ast.Node.Index, is_pub: bool) !?types.Member {
        const name = self.tree.tokenSlice(name_token);

        const main_token = self.tree.nodeMainToken(init_node);
        const container_tag = self.tree.tokenTag(main_token);

        const member_type: types.MemberType = switch (container_tag) {
            .keyword_struct => .@"struct",
            .keyword_enum => .@"enum",
            .keyword_union => .@"union",
            else => return null,
        };

        // Collect public field names to include in the hash so that adding/removing
        // a field changes the match_hash and triggers re-enhancement.
        var field_names: std.ArrayList([]const u8) = .{};
        defer field_names.deinit(self.allocator);

        var buf2: [2]std.zig.Ast.Node.Index = undefined;
        if (self.tree.fullContainerDecl(&buf2, init_node)) |container_for_fields| {
            for (container_for_fields.ast.members) |member_node| {
                const child_tag = self.tree.nodeTag(member_node);
                if (child_tag == .container_field_init) {
                    const main_tok = self.tree.nodeMainToken(member_node);
                    const field_name = self.tree.tokenSlice(main_tok);
                    try field_names.append(self.allocator, field_name);
                }
            }
        }

        const match_hash = try hash.structHash(self.allocator, name, field_names.items);
        errdefer self.allocator.free(match_hash);

        // The "///" doc comment is written before "pub const Foo = struct {",
        // i.e. before the var_decl node, not before the container body (init_node).
        const comment = try self.extractDocstring(var_decl_node);

        var nested_members: std.ArrayList(types.Member) = .{};
        errdefer nested_members.deinit(self.allocator);

        var buf: [2]std.zig.Ast.Node.Index = undefined;
        const full_container = self.tree.fullContainerDecl(&buf, init_node) orelse return null;
        for (full_container.ast.members) |member_node| {
            const child_tag = self.tree.nodeTag(member_node);
            if (child_tag == .fn_decl) {
                if (try self.extractFunction(member_node)) |method| {
                    try nested_members.append(self.allocator, method);
                }
            } else if (member_type == .@"enum" and child_tag == .container_field_init) {
                // Enum field: a bare identifier (e.g. `mem`, `sqlite`, `rocksdb`)
                const main_tok = self.tree.nodeMainToken(member_node);
                const field_name = self.tree.tokenSlice(main_tok);
                const field_comment = try self.extractDocstring(member_node);
                const field_line: u32 = @intCast(self.tree.tokenLocation(0, main_tok).line + 1);
                try nested_members.append(self.allocator, types.Member{
                    .type = .enum_field,
                    .name = try self.allocator.dupe(u8, field_name),
                    .comment = field_comment,
                    .line = field_line,
                });
            }
        }

        const signature = try std.fmt.allocPrint(self.allocator, "{s} {s} {{ ... }}", .{
            @tagName(container_tag),
            name,
        });

        const patterns_detected = try pattern.detectPatterns(self.allocator, &self.tree, init_node);

        return types.Member{
            .type = member_type,
            .name = try self.allocator.dupe(u8, name),
            .match_hash = match_hash,
            .signature = signature,
            .comment = comment,
            .patterns = patterns_detected,
            .is_pub = is_pub,
            .members = try nested_members.toOwnedSlice(self.allocator),
            .line = @as(u32, @intCast(self.tree.tokenLocation(0, name_token).line + 1)),
        };
    }

    fn extractTest(self: *AstParser, node_idx: std.zig.Ast.Node.Index) !?types.Member {
        const data = self.tree.nodeData(node_idx);
        const test_name = data.opt_token_and_node[0];
        if (test_name == .none) return null;

        const name_token = test_name.unwrap() orelse return null;
        const raw_name = self.tree.tokenSlice(name_token);
        // The token for a named test is its string literal, including surrounding quotes.
        // Strip them so the stored name is e.g. `sha256Hex produces correct length`.
        const name = if (raw_name.len >= 2 and raw_name[0] == '"' and raw_name[raw_name.len - 1] == '"')
            raw_name[1 .. raw_name.len - 1]
        else
            raw_name;

        return types.Member{
            .type = .test_decl,
            .name = try self.allocator.dupe(u8, name),
            .line = @as(u32, @intCast(self.tree.tokenLocation(0, name_token).line + 1)),
        };
    }

    fn extractParams(self: *AstParser, full_proto: std.zig.Ast.full.FnProto) ![]types.Param {
        var params: std.ArrayList(types.Param) = .{};
        errdefer params.deinit(self.allocator);

        var iter = full_proto.iterate(&self.tree);
        while (iter.next()) |param| {
            const param_name = if (param.name_token) |name_tok|
                self.tree.tokenSlice(name_tok)
            else if (param.anytype_ellipsis3) |anytok|
                if (self.tree.tokenTag(anytok) == .keyword_anytype) "anytype" else "..."
            else
                "_";

            const type_str = if (param.type_expr) |type_node|
                try self.nodeToTypeFromIndex(type_node)
            else
                null;

            try params.append(self.allocator, .{
                .name = try self.allocator.dupe(u8, param_name),
                .type = type_str,
            });
        }

        return params.toOwnedSlice(self.allocator);
    }

    fn nodeToType(self: *AstParser, node: std.zig.Ast.Node.OptionalIndex) !?[]const u8 {
        if (node == .none) return null;
        return try self.nodeToTypeFromIndex(node.unwrap().?);
    }

    fn nodeToTypeFromIndex(self: *AstParser, idx: std.zig.Ast.Node.Index) ![]const u8 {
        const type_slice = self.tree.getNodeSource(idx);
        return try self.allocator.dupe(u8, type_slice);
    }

    fn buildSignature(allocator: std.mem.Allocator, name: []const u8, params: []types.Param, returns: ?[]const u8) ![]const u8 {
        var sig: std.ArrayList(u8) = .{};
        errdefer sig.deinit(allocator);
        const writer = sig.writer(allocator);

        try writer.writeAll("fn ");
        try writer.writeAll(name);
        try writer.writeByte('(');

        for (params, 0..) |param, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.writeAll(param.name);
            if (param.type) |t| {
                try writer.writeAll(": ");
                try writer.writeAll(t);
            }
        }

        try writer.writeByte(')');

        if (returns) |r| {
            try writer.writeAll(" -> ");
            try writer.writeAll(r);
        }

        return sig.toOwnedSlice(allocator);
    }

    fn extractDocstring(self: *AstParser, node_idx: std.zig.Ast.Node.Index) !?[]const u8 {
        const first_token = self.tree.firstToken(node_idx);
        const token_tags = self.tree.tokens.items(.tag);

        // Walk backwards collecting every consecutive doc_comment token.
        // We only want the FIRST (topmost) line — mirrors Python's
        // `ast.get_docstring(node).split("\n")[0]` behaviour. Tokens are
        // gathered in reverse (closest-first), so we keep updating `top_line`
        // with each successive token; the last assignment is the topmost line.
        var top_tok: ?std.zig.Ast.TokenIndex = null;
        var tok = first_token;
        while (tok > 0) {
            tok -= 1;
            if (token_tags[tok] == .doc_comment) {
                top_tok = tok; // keep walking; this may still go further up
            } else if (token_tags[tok] != .invalid and token_tags[tok] != .container_doc_comment) {
                break;
            }
        }

        const t = top_tok orelse return null;
        const slice = self.tree.tokenSlice(t);
        // Strip "///" prefix (3 chars). If the next char is a space (the
        // conventional "/// text" style), strip that too so the stored
        // comment is "text", not " text".
        const after_slashes = if (slice.len > 3) slice[3..] else "";
        const line = if (after_slashes.len > 0 and after_slashes[0] == ' ')
            after_slashes[1..]
        else
            after_slashes;
        return try self.allocator.dupe(u8, line);
    }

    pub fn extractModuleDoc(self: *AstParser) !?[]const u8 {
        const token_tags = self.tree.tokens.items(.tag);

        var doc_lines: std.ArrayList([]const u8) = .{};
        defer {
            for (doc_lines.items) |line| self.allocator.free(line);
            doc_lines.deinit(self.allocator);
        }

        for (token_tags, 0..) |tag, i| {
            if (tag == .container_doc_comment) {
                const slice = self.tree.tokenSlice(@intCast(i));
                const line = if (slice.len > 3) slice[3..] else "";
                try doc_lines.append(self.allocator, try self.allocator.dupe(u8, line));
            } else if (tag != .invalid) {
                break;
            }
        }

        if (doc_lines.items.len == 0) return null;

        var result: std.ArrayList(u8) = .{};
        for (doc_lines.items, 0..) |line, i| {
            if (i > 0) try result.append(self.allocator, '\n');
            try result.appendSlice(self.allocator, line);
        }

        return @as(?[]const u8, try result.toOwnedSlice(self.allocator));
    }

    /// Walk the token stream for @import("...") string literals and return
    /// a slice of their string values (without surrounding quotes), owned by the caller.
    pub fn extractImports(self: *AstParser) ![][]const u8 {
        var imports: std.ArrayList([]const u8) = .{};
        errdefer {
            for (imports.items) |s| self.allocator.free(s);
            imports.deinit(self.allocator);
        }

        const token_tags = self.tree.tokens.items(.tag);
        var i: u32 = 0;
        while (i < token_tags.len) : (i += 1) {
            // Look for @import builtin — its token tag is .builtin.
            if (token_tags[i] != .builtin) continue;
            const builtin_slice = self.tree.tokenSlice(i);
            if (!std.mem.eql(u8, builtin_slice, "@import")) continue;
            // Next meaningful token should be '(' then a string literal.
            if (i + 2 >= token_tags.len) continue;
            if (token_tags[i + 1] != .l_paren) continue;
            if (token_tags[i + 2] != .string_literal) continue;
            const str_tok = self.tree.tokenSlice(i + 2);
            // Strip surrounding quotes.
            if (str_tok.len < 2) continue;
            const inner = str_tok[1 .. str_tok.len - 1];
            try imports.append(self.allocator, try self.allocator.dupe(u8, inner));
            i += 2; // skip past the string literal
        }

        return imports.toOwnedSlice(self.allocator);
    }

    pub fn countTokens(self: *const AstParser) usize {
        return self.source.len / 4;
    }
};

pub fn parseFile(allocator: std.mem.Allocator, path: []const u8) !AstParser {
    const file = std.fs.openFileAbsolute(path, .{}) catch |err| return err;
    defer file.close();

    const source = file.readToEndAllocOptions(allocator, std.math.maxInt(usize), null, .@"1", 0) catch |err| return err;
    defer allocator.free(source);

    return AstParser.init(allocator, source) catch |err| {
        allocator.free(source);
        return err;
    };
}
