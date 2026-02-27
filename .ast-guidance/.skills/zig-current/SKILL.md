# Zig 0.15.x Agent Skill

## ArrayList/HashMap (Unmanaged by Default)

```zig
var list: std.ArrayList(u8) = .{};        // not .init(allocator)
defer list.deinit(allocator);             // not .deinit()
try list.append(allocator, 'a');           // not .append('a')
const slice = try list.toOwnedSlice(allocator);

var map: std.StringHashMap(u32) = .{};
defer map.deinit(allocator);
try map.put(allocator, "key", 42);
```

## I/O: Mandatory Buffer + Flush

```zig
// Writer (stack buffer)
var buf: [4096]u8 = undefined;
var f = try std.fs.cwd().createFile("out.txt", .{});
defer f.close();
var fw = f.writer(&buf);
const w = &fw.interface;
try w.print("data\n", .{});
try w.flush();

// Reader
var rb = f.reader(&buf);
const r = &rb.interface;
const data = try r.readAllAlloc(allocator, 1024);

// stdout
var out_buf: [4096]u8 = undefined;
var fw = std.fs.File.stdout().writer(&out_buf);
try fw.interface.print("hello\n", .{});
try fw.interface.flush();
```

**Rule**: Always `flush()` before close/return. Never copy the interface—use `&fw.interface`.

## JSON Serialization

```zig
var out = std.Io.Writer.Allocating.init(allocator);
defer out.deinit();
try std.json.Stringify.value(value, .{}, &out.writer);
const json = try out.toOwnedSlice();

// Parse
var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
defer parsed.deinit();
```

## HTTP Client (`std.http.Client`)

Use `client.fetch()` — `client.request()` no longer exists.

### POST with response body capture

```zig
var client = std.http.Client{ .allocator = allocator };
defer client.deinit();

// Capture response into a growable buffer.
var aw: std.Io.Writer.Allocating = .init(allocator);
defer aw.deinit();

const result = try client.fetch(.{
    .method = .POST,
    .location = .{ .url = "http://localhost:11434/api/chat" },
    .extra_headers = &[_]std.http.Header{
        .{ .name = "Content-Type", .value = "application/json" },
    },
    .payload = json_body_slice,
    .response_writer = &aw.writer,
});
// result.status is std.http.Status (.ok == 200)
const body: []const u8 = aw.writer.buffer[0..aw.writer.end];
```

### GET (health-check / discard body)

```zig
const result = try client.fetch(.{
    .method = .GET,
    .location = .{ .url = check_url },
    // Omit response_writer — body is discarded automatically.
});
const ok = result.status == .ok;
```

**Rules**:
- `std.Io.Writer.Allocating` is the standard sink for dynamic response bodies.
- `response_writer` is `?*std.Io.Writer`; pass `&aw.writer` (not `&aw`).
- Omit `response_writer` when you only need the status code.
- Never use shell subprocesses (curl) when Zig native HTTP is available.

## Child Process

```zig
var child = std.process.Child.init(&[_]const u8{"cmd", "arg"}, allocator);
child.stdin_behavior = .Ignore;
child.stdout_behavior = .Pipe;
child.stderr_behavior = .Pipe;
try child.spawn();

var stdout = std.ArrayList(u8).init(allocator);
try child.collectOutput(&stdout, &stderr, 10 * 1024 * 1024);
_ = try child.wait();
```

## Formatting

- Structs with format methods: `{f}` (not `{}`)
- Debug output: `{any}`
- Shell braces literal: `{{` and `}}` (e.g., `sed 's/{x}/y/'` needs `sed 's/{{x}}/y/'`)

## File I/O: Sentinel Slices

```zig
const source = try file.readToEndAllocOptions(
    allocator, max_size, null, .@"1", 0  // 0 = null terminator
);
try std.zig.Ast.parse(allocator, source, .zig);  // source is [:0]const u8
```

## mem.Alignment Enum

```zig
const align: std.mem.Alignment = .@"1";  // not @alignOf(u8) as usize
```

## SHA-256

```zig
var hash_out: [32]u8 = undefined;
std.crypto.hash.sha2.Sha256.hash(data, &hash_out, .{});
const hex = std.fmt.bytesToHex(hash_out, .lower);
```

## String Case

```zig
const lower = try std.ascii.allocLowerString(allocator, str);
```

## Allocators

```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer if (gpa.deinit() == .leak) @panic("leak");
```

## Quick Checklist

| Check | Fix |
|-------|-----|
| `.init(allocator)` | Use `.{}` empty struct literal |
| `.append(item)` | Use `.append(allocator, item)` |
| `file.writer()` missing buffer | Add `&buf` argument |
| No `.flush()` | Add before close/return |
| Passing interface by value | Use `&fw.interface` |
| Struct format `{}` | Use `{f}` |
| Shell sed/awk braces | Escape as `{{` `}}` |
| `readToEndAlloc` for AST | Use `readToEndAllocOptions` with sentinel 0 |
| Alignment as usize | Use `.@"1"` enum |
| `client.request()` | Use `client.fetch()` (request API removed) |
| HTTP response body | Use `std.Io.Writer.Allocating`; pass `&aw.writer` |
| Curl subprocess for HTTP | Use `std.http.Client.fetch()` instead |
