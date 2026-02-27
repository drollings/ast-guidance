const std = @import("std");
const types = @import("types.zig");
const hash = @import("hash.zig");

pub const JsonStore = struct {
    allocator: std.mem.Allocator,
    debug: bool = false,
    /// Set to true whenever loadGuidance discards a comment via isLeakedPrompt.
    /// Reset at the start of each loadGuidance call.
    /// Callers (processFile, infillOneFile) use this to force a write-back even
    /// when the LLM produces no replacement, so leaked text is cleared on disk.
    leaked_prompts_found: bool = false,

    pub fn init(allocator: std.mem.Allocator) JsonStore {
        return .{ .allocator = allocator };
    }

    pub fn loadGuidance(self: *JsonStore, path: []const u8) !?types.GuidanceDoc {
        self.leaked_prompts_found = false;
        const file = std.fs.openFileAbsolute(path, .{}) catch return null;
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 10 * 1024 * 1024) catch return null;
        defer self.allocator.free(content);

        return self.parseGuidance(content);
    }

    fn parseGuidance(self: *JsonStore, content: []const u8) !?types.GuidanceDoc {
        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, content, .{}) catch return null;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return null;

        const meta_obj = root.object.get("meta") orelse return null;
        if (meta_obj != .object) return null;

        const module = meta_obj.object.get("module") orelse return null;
        const source = meta_obj.object.get("source") orelse return null;

        var doc = types.GuidanceDoc{
            .meta = .{
                .module = try self.allocator.dupe(u8, module.string),
                .source = try self.allocator.dupe(u8, source.string),
            },
        };

        // Accept "comment" for the file-level description.
        // Sanitize leaked LLM prompts the same way as member comments.
        const comment_val = root.object.get("comment");
        if (comment_val) |d| {
            if (d == .string) {
                if (isLeakedPrompt(d.string)) {
                    self.leaked_prompts_found = true;
                } else {
                    doc.comment = try self.allocator.dupe(u8, d.string);
                }
            }
        }

        if (root.object.get("skills")) |skills_val| {
            if (skills_val == .array) {
                var skills: std.ArrayList(types.Skill) = .{};
                for (skills_val.array.items) |skill_val| {
                    if (skill_val == .object) {
                        const ref = skill_val.object.get("ref") orelse continue;
                        if (ref == .string) {
                            var skill: types.Skill = .{ .ref = try self.allocator.dupe(u8, ref.string) };
                            if (skill_val.object.get("context")) |ctx| {
                                if (ctx == .string) {
                                    skill.context = try self.allocator.dupe(u8, ctx.string);
                                }
                            }
                            try skills.append(self.allocator, skill);
                        }
                    }
                }
                doc.skills = try skills.toOwnedSlice(self.allocator);
            }
        }

        if (root.object.get("hashtags")) |tags_val| {
            if (tags_val == .array) {
                var tags: std.ArrayList([]const u8) = .{};
                for (tags_val.array.items) |tag_val| {
                    if (tag_val == .string) {
                        try tags.append(self.allocator, try self.allocator.dupe(u8, tag_val.string));
                    }
                }
                doc.hashtags = try tags.toOwnedSlice(self.allocator);
            }
        }

        if (root.object.get("used_by")) |used_val| {
            if (used_val == .array) {
                var used: std.ArrayList([]const u8) = .{};
                for (used_val.array.items) |u_val| {
                    if (u_val == .string) {
                        try used.append(self.allocator, try self.allocator.dupe(u8, u_val.string));
                    }
                }
                doc.used_by = try used.toOwnedSlice(self.allocator);
            }
        }

        if (root.object.get("members")) |members_val| {
            doc.members = try self.parseMembers(members_val);
        }

        return doc;
    }

    fn parseMembers(self: *JsonStore, members_val: std.json.Value) std.mem.Allocator.Error![]types.Member {
        if (members_val != .array) return &.{};

        var members: std.ArrayList(types.Member) = .{};
        for (members_val.array.items) |member_val| {
            if (member_val == .object) {
                if (try self.parseMember(member_val)) |member| {
                    try members.append(self.allocator, member);
                }
            }
        }
        return members.toOwnedSlice(self.allocator);
    }

    /// Returns true when a stored comment is a leaked LLM reasoning preamble rather
    /// than real documentation.  Such comments should be treated as absent so the
    /// Python AI infill phase can regenerate them with a proper answer.
    fn isLeakedPrompt(text: []const u8) bool {
        // Case-insensitive prefix check against known reasoning-model preambles.
        const preambles = [_][]const u8{
            "we need to write",
            "we need to look",
            "we need to read",
            "i need to write",
            "let's write",
            "let me write",
            "write a one-sentence",
        };
        const check_len = @min(text.len, 30);
        var buf: [30]u8 = undefined;
        const lower = std.ascii.lowerString(buf[0..check_len], text[0..check_len]);
        for (preambles) |p| {
            if (p.len <= lower.len and std.mem.startsWith(u8, lower, p)) return true;
        }
        return false;
    }

    fn parseMember(self: *JsonStore, member_val: std.json.Value) std.mem.Allocator.Error!?types.Member {
        const name_val = member_val.object.get("name") orelse return null;
        const type_val = member_val.object.get("type") orelse return null;

        if (name_val != .string or type_val != .string) return null;

        var member = types.Member{
            .type = std.meta.stringToEnum(types.MemberType, type_val.string) orelse .fn_decl,
            .name = try self.allocator.dupe(u8, name_val.string),
        };

        if (member_val.object.get("match_hash")) |h| {
            if (h == .string) {
                member.match_hash = try self.allocator.dupe(u8, h.string);
            }
        }

        if (member_val.object.get("signature")) |s| {
            if (s == .string) {
                member.signature = try self.allocator.dupe(u8, s.string);
            }
        }

        // Accept "comment" for the member description.
        // Sanitize: discard stored comments that are leaked LLM prompts rather than
        // real documentation. A comment is treated as absent when it starts with any
        // of the known preamble phrases that a reasoning model writes before its answer.
        const member_comment = member_val.object.get("comment");
        if (member_comment) |d| {
            if (d == .string) {
                if (isLeakedPrompt(d.string)) {
                    self.leaked_prompts_found = true;
                } else {
                    member.comment = try self.allocator.dupe(u8, d.string);
                }
            }
        }

        if (member_val.object.get("returns")) |r| {
            if (r == .string) {
                member.returns = try self.allocator.dupe(u8, r.string);
            }
        }

        if (member_val.object.get("is_pub")) |p| {
            if (p == .bool) {
                member.is_pub = p.bool;
            }
        }

        if (member_val.object.get("line")) |l| {
            if (l == .integer) {
                member.line = @intCast(l.integer);
            }
        }

        if (member_val.object.get("params")) |params_val| {
            if (params_val == .array) {
                var params: std.ArrayList(types.Param) = .{};
                for (params_val.array.items) |param_val| {
                    if (param_val == .object) {
                        var param: types.Param = .{ .name = "" };
                        if (param_val.object.get("name")) |pn| {
                            if (pn == .string) {
                                param.name = try self.allocator.dupe(u8, pn.string);
                            }
                        }
                        if (param_val.object.get("type")) |pt| {
                            if (pt == .string) {
                                param.type = try self.allocator.dupe(u8, pt.string);
                            }
                        }
                        if (param_val.object.get("default")) |pd| {
                            if (pd == .string) {
                                param.default = try self.allocator.dupe(u8, pd.string);
                            }
                        }
                        if (param.name.len > 0) {
                            try params.append(self.allocator, param);
                        }
                    }
                }
                member.params = try params.toOwnedSlice(self.allocator);
            }
        }

        if (member_val.object.get("tags")) |tags_val| {
            if (tags_val == .array) {
                var tags: std.ArrayList([]const u8) = .{};
                for (tags_val.array.items) |tag_val| {
                    if (tag_val == .string) {
                        try tags.append(self.allocator, try self.allocator.dupe(u8, tag_val.string));
                    }
                }
                member.tags = try tags.toOwnedSlice(self.allocator);
            }
        }

        if (member_val.object.get("patterns")) |patterns_val| {
            if (patterns_val == .array) {
                var patterns: std.ArrayList(types.Pattern) = .{};
                for (patterns_val.array.items) |pat_val| {
                    if (pat_val == .object) {
                        const pat_name = pat_val.object.get("name") orelse continue;
                        if (pat_name == .string) {
                            var pat: types.Pattern = .{
                                .name = try self.allocator.dupe(u8, pat_name.string),
                                .type = .Domain,
                            };
                            if (pat_val.object.get("type")) |pt| {
                                if (pt == .string) {
                                    pat.type = std.meta.stringToEnum(types.PatternType, pt.string) orelse .Domain;
                                }
                            }
                            if (pat_val.object.get("ref")) |pr| {
                                if (pr == .string) {
                                    pat.ref = try self.allocator.dupe(u8, pr.string);
                                }
                            }
                            try patterns.append(self.allocator, pat);
                        }
                    }
                }
                member.patterns = try patterns.toOwnedSlice(self.allocator);
            }
        }

        if (member_val.object.get("members")) |nested_val| {
            member.members = try self.parseMembers(nested_val);
        }

        return member;
    }

    pub fn saveGuidance(self: *JsonStore, path: []const u8, doc: types.GuidanceDoc) !void {
        const json_str = try types.jsonifyGuidanceDoc(self.allocator, doc);
        defer self.allocator.free(json_str);

        const dir_path = std.fs.path.dirname(path) orelse return error.InvalidPath;
        try makePathAbsolute(dir_path);

        const file = try std.fs.createFileAbsolute(path, .{ .truncate = true });
        defer file.close();

        try file.writeAll(json_str);
    }

    fn makePathAbsolute(abs_path: []const u8) !void {
        var buf: [4096]u8 = undefined;
        var pos: usize = 0;

        var parts = std.mem.splitScalar(u8, abs_path, std.fs.path.sep);
        var is_first = true;

        while (parts.next()) |part| {
            if (part.len == 0) continue;

            if (is_first) {
                buf[0] = '/';
                @memcpy(buf[1 .. 1 + part.len], part);
                pos = 1 + part.len;
                is_first = false;
            } else {
                buf[pos] = std.fs.path.sep;
                @memcpy(buf[pos + 1 .. pos + 1 + part.len], part);
                pos += 1 + part.len;
            }

            const current = buf[0..pos];
            std.fs.makeDirAbsolute(current) catch |err| switch (err) {
                error.PathAlreadyExists => {},
                else => return err,
            };
        }
    }

    /// Deep-copy a single Member; every string field is independently owned.
    pub fn dupeMember(self: *JsonStore, m: types.Member) !types.Member {
        var copy = m;
        copy.name = try self.allocator.dupe(u8, m.name);
        errdefer self.allocator.free(copy.name);

        if (m.match_hash) |h| {
            copy.match_hash = try self.allocator.dupe(u8, h);
        }
        if (m.signature) |s| {
            copy.signature = try self.allocator.dupe(u8, s);
        }
        if (m.comment) |d| {
            copy.comment = try self.allocator.dupe(u8, d);
        }
        if (m.returns) |r| {
            copy.returns = try self.allocator.dupe(u8, r);
        }
        copy.params = try self.dupeParams(m.params);
        copy.tags = try self.dupeStrings(m.tags);
        copy.patterns = try self.dupePatterns(m.patterns);

        // Recursively deep-copy nested members.
        var nested: std.ArrayList(types.Member) = .{};
        errdefer {
            for (nested.items) |nm| self.freeMember(nm);
            nested.deinit(self.allocator);
        }
        for (m.members) |nm| {
            try nested.append(self.allocator, try self.dupeMember(nm));
        }
        copy.members = try nested.toOwnedSlice(self.allocator);

        return copy;
    }

    fn dupeParams(self: *JsonStore, params: []const types.Param) ![]types.Param {
        const result = try self.allocator.alloc(types.Param, params.len);
        for (params, 0..) |p, i| {
            result[i] = .{
                .name = try self.allocator.dupe(u8, p.name),
                .type = if (p.type) |t| try self.allocator.dupe(u8, t) else null,
                .default = if (p.default) |d| try self.allocator.dupe(u8, d) else null,
            };
        }
        return result;
    }

    pub fn dupeStrings(self: *JsonStore, strs: []const []const u8) ![][]const u8 {
        const result = try self.allocator.alloc([]const u8, strs.len);
        for (strs, 0..) |s, i| {
            result[i] = try self.allocator.dupe(u8, s);
        }
        return result;
    }

    fn dupePatterns(self: *JsonStore, patterns: []const types.Pattern) ![]types.Pattern {
        const result = try self.allocator.alloc(types.Pattern, patterns.len);
        for (patterns, 0..) |p, i| {
            result[i] = .{
                .name = try self.allocator.dupe(u8, p.name),
                .type = p.type,
                .ref = if (p.ref) |r| try self.allocator.dupe(u8, r) else null,
            };
        }
        return result;
    }

    /// Deep-copy a slice of Skills.
    pub fn dupeSkills(self: *JsonStore, skills: []const types.Skill) ![]types.Skill {
        const result = try self.allocator.alloc(types.Skill, skills.len);
        for (skills, 0..) |s, i| {
            result[i] = .{
                .ref = try self.allocator.dupe(u8, s.ref),
                .context = if (s.context) |c| try self.allocator.dupe(u8, c) else null,
            };
        }
        return result;
    }

    pub fn freeMember(self: *JsonStore, member: types.Member) void {
        self.allocator.free(member.name);
        if (member.match_hash) |h| self.allocator.free(h);
        if (member.signature) |s| self.allocator.free(s);
        if (member.comment) |d| self.allocator.free(d);
        if (member.returns) |r| self.allocator.free(r);
        for (member.params) |p| {
            self.allocator.free(p.name);
            if (p.type) |t| self.allocator.free(t);
            if (p.default) |d| self.allocator.free(d);
        }
        self.allocator.free(member.params);
        for (member.tags) |t| self.allocator.free(t);
        self.allocator.free(member.tags);
        for (member.patterns) |p| {
            self.allocator.free(p.name);
            if (p.ref) |r| self.allocator.free(r);
        }
        self.allocator.free(member.patterns);
        for (member.members) |m| self.freeMember(m);
        self.allocator.free(member.members);
    }

    /// Free all heap memory owned by a GuidanceDoc.
    pub fn freeGuidanceDoc(self: *JsonStore, doc: types.GuidanceDoc) void {
        self.allocator.free(doc.meta.module);
        self.allocator.free(doc.meta.source);
        if (doc.comment) |d| self.allocator.free(d);
        for (doc.skills) |s| {
            self.allocator.free(s.ref);
            if (s.context) |c| self.allocator.free(c);
        }
        self.allocator.free(doc.skills);
        for (doc.hashtags) |h| self.allocator.free(h);
        self.allocator.free(doc.hashtags);
        for (doc.used_by) |u| self.allocator.free(u);
        self.allocator.free(doc.used_by);
        for (doc.members) |m| self.freeMember(m);
        self.allocator.free(doc.members);
    }

    pub fn mergeMembers(self: *JsonStore, source: []const types.Member, existing: []const types.Member, preserve_existing_comments: bool) std.mem.Allocator.Error!MergeResult {
        var result: MergeResult = .{};
        var merged: std.ArrayList(types.Member) = .{};
        errdefer {
            for (merged.items) |m| self.freeMember(m);
            merged.deinit(self.allocator);
        }

        // Map name → index into `existing`; no ownership transfer.
        var existing_index: std.StringHashMapUnmanaged(usize) = .{};
        defer existing_index.deinit(self.allocator);

        for (existing, 0..) |member, idx| {
            try existing_index.put(self.allocator, member.name, idx);
        }

        for (source) |src_member| {
            if (existing_index.get(src_member.name)) |idx| {
                const existing_member = existing[idx];
                const merged_member = try self.mergeMember(src_member, existing_member, &result, preserve_existing_comments);
                try merged.append(self.allocator, merged_member);
            } else {
                // New member — deep-copy so this result owns all strings.
                result.members_added += 1;
                result.has_changes = true;
                try merged.append(self.allocator, try self.dupeMember(src_member));
            }
        }

        result.members_removed = existing.len;
        for (source) |s| {
            if (existing_index.contains(s.name)) {
                if (result.members_removed > 0) result.members_removed -= 1;
            }
        }
        if (result.members_removed > 0) {
            result.has_changes = true;
        }

        result.members = try merged.toOwnedSlice(self.allocator);
        return result;
    }

    fn mergeMember(self: *JsonStore, source: types.Member, existing: types.Member, result: *MergeResult, preserve_existing_comments: bool) !types.Member {
        // Start with a fully owned deep copy of source.
        var merged = try self.dupeMember(source);
        errdefer self.freeMember(merged);

        if (source.match_hash) |src_hash| {
            if (existing.match_hash) |ex_hash| {
                if (std.mem.eql(u8, src_hash, ex_hash)) {
                    // Hash unchanged — check whether line number shifted (e.g. comment added above).
                    // merged already holds source.line via dupeMember; just flag the write.
                    if (source.line != existing.line) {
                        result.members_updated += 1;
                        result.has_changes = true;
                    }
                    // Hash unchanged — check whether the source //! doc comment changed.
                    if (merged.comment) |src_doc| {
                        // Source has a new or updated //! comment.
                        const existing_doc = existing.comment orelse "";
                        if (!std.mem.eql(u8, src_doc, existing_doc)) {
                            // Source comment changed — mark as updated so the file gets saved.
                            result.members_updated += 1;
                            result.has_changes = true;
                        }
                        // Keep tags/patterns from existing when hash is unchanged.
                        if (merged.tags.len == 0 and existing.tags.len > 0) {
                            self.allocator.free(merged.tags);
                            merged.tags = try self.dupeStrings(existing.tags);
                        }
                        if (merged.patterns.len == 0 and existing.patterns.len > 0) {
                            self.allocator.free(merged.patterns);
                            merged.patterns = try self.dupePatterns(existing.patterns);
                        }
                    } else {
                        // No //! comment in source — preserve stored comment/tags/patterns only if requested.
                        if (preserve_existing_comments and merged.comment == null) {
                            if (existing.comment) |d| {
                                merged.comment = try self.allocator.dupe(u8, d);
                            }
                        } else if (!preserve_existing_comments and merged.comment == null) {
                            // Not preserving existing comments and source has none:
                            // If there was an existing comment, this is a change (stripping it).
                            if (existing.comment) |d| {
                                if (d.len > 0) {
                                    result.has_changes = true;
                                }
                            }
                        }
                        if (merged.tags.len == 0 and existing.tags.len > 0) {
                            self.allocator.free(merged.tags);
                            merged.tags = try self.dupeStrings(existing.tags);
                        }
                        if (merged.patterns.len == 0 and existing.patterns.len > 0) {
                            self.allocator.free(merged.patterns);
                            merged.patterns = try self.dupePatterns(existing.patterns);
                        }
                    }
                } else {
                    result.members_updated += 1;
                    result.has_changes = true;
                    // Hash changed (implementation changed) — stale-comment detection:
                    // If source has no new //! doc comment (merged.comment is null)
                    // and existing had a non-empty comment, that comment is now stale.
                    // Leave merged.comment as null so AI infill regenerates it fresh.
                    // If source has a new //! doc comment, keep it (author updated it).
                    if (merged.comment == null) {
                        if (existing.comment) |d| if (d.len > 0) {
                            // Stale: existing comment exists but hash changed and no
                            // updated //! comment in source.  Leave null for infill.
                            result.members_stale += 1;
                        };
                    }
                    // (If merged.comment != null, it came from source's //! comment — keep it.)
                }
            } else {
                result.members_updated += 1;
                result.has_changes = true;
            }
        }

        // Recursively merge nested members.
        if (source.members.len > 0 and existing.members.len > 0) {
            // Free the deep-copied nested members from dupeMember before replacing.
            for (merged.members) |nm| self.freeMember(nm);
            self.allocator.free(merged.members);
            const nested_result = try self.mergeMembers(source.members, existing.members, preserve_existing_comments);
            merged.members = nested_result.members;
            if (nested_result.has_changes) result.has_changes = true;
        }

        return merged;
    }
};

pub const MergeResult = struct {
    members: []types.Member = &.{},
    members_added: usize = 0,
    members_updated: usize = 0,
    members_removed: usize = 0,
    /// Members whose hash changed and had a stored comment that is now stale
    /// (blanked so AI infill can regenerate).
    members_stale: usize = 0,
    has_changes: bool = false,
};
