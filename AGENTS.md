# Agent Bootloader — ast-guidance

**Context**: ast-guidance is a Zig-native, deterministic codebase documentation engine.
It analyzes source files (Zig, Python, and future languages) via AST, generates JSON
guidance files in `.ast-guidance/`, and exposes an AI-assisted query layer through the
Makefile.  It is designed to drop into any project with zero footprint beyond the Makefile.

## Prime Directive

1. **Current Zig knowledge**: Read `doc/skills/zig-current/SKILL.md` before writing any Zig
2. **Project structure**: When unsure where things live, run `make explain QUERY="<terms>"`
3. **Never guess**: use `make explain` to locate code — never invent file paths
4. **Config**: model names and provider registry live in `.ast-guidance/ast-guidance-config.json`

---

## Quick Start: RALPH Loop (Discovery → Implementation)

```
1. DISCOVER (make):  make explain QUERY="<short phrase>"
                     Prefer phrases: "sync guidance json" > "json"
                     Scan: module purpose, pattern type, skill list

2. UNDERSTAND (MCP): Read the primary source file(s) from step 1
                     Grep callers: who @import's this file?
                     Ask: do the listed skills actually apply?

3. DECIDE:           If skills match → read them
                     If not → proceed to implementation

4. IMPLEMENT:        Write to src/ast-guidance/ or bin/ (Python providers)
                     Follow source patterns and applicable skills only

5. VERIFY (make):    make pre-commit
                     build → test → guidance sync → lint → STRUCTURE.md
```

---

## Source Layout

```
src/
  ast-guidance/     Zig core engine (AST parser, sync, query, structure, triage)
  common/           Zig shared LLM HTTP client and arg helpers
bin/
  ast-guidance      Compiled binary — zig-out/bin/ast-guidance (via zig build)
  ast-guidance-py   Python AST provider (Python files → .ast-guidance/ JSON)
.ast-guidance/
  ast-guidance-config.json   Model / provider configuration
  .skills/          Structured skill documents (GoF, zig-current, domain-patterns)
  .doc/             Capabilities, diary, inbox
  src/              Generated guidance JSON (mirrors src/ tree)
env/
  mk/               Shared Makefile helpers and per-language target overrides
  mise/             Language-specific mise.toml fragments
doc/
  DESIGN.md         System design reference
```

---

## Skill Reading Strategy

**DO:**
- Run `make explain QUERY="<phrase>"` — phrase searches outperform single terms
- Read the primary source file identified in explain output
- Ask: "What design pattern is used here?" before consulting skills
- Read skills **only** if source analysis confirms they apply

**DON'T:**
- Assume skills apply without validating against source code
- Read `.ast-guidance/**/*.json` directly — use `make query` instead
- Skip source reading before implementation

---

## Capturing Knowledge

### Insights  (`.ast-guidance/.doc/inbox/INSIGHTS.md`)
Append major discoveries here as markdown bullets.  Unprocessed items are
automatically surfaced by `make explain` when relevant to the current query.

### New Capabilities  (`.ast-guidance/.doc/inbox/CAPABILITIES.md`)
Append major new features here as markdown bullets.  Before implementing,
consult `.ast-guidance/.doc/capabilities/*.md` to enforce DRY — an existing
implementation may already cover the need.

### Promoting Knowledge  (`make learn`)
Runs `ast-guidance learn --guidance .ast-guidance` to drain both inbox files
into structured skill / capability documents, then clears promoted items.

---

## Language Provider Convention

Each language provider (`ast-guidance-py`, `ast-guidance-cpp`, …) must:

1. Accept: `sync --file <path> --output <guidance_dir> [--infill]`
2. Accept: `sync --scan <dir>  --output <guidance_dir> [--infill]`
3. Emit canonical JSON at `<guidance_dir>/src/<rel_path>.<ext>.json`
4. Use the schema: `{ meta, comment, skills, hashtags, used_by, members[] }`
5. Read model name from `.ast-guidance/ast-guidance-config.json` → `models.infill`
