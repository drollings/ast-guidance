# guidance: Vision Document

**A Deterministic-First Code Navigation Subagent for AI-Assisted Development**

---

## Executive Summary

guidance is a Rust-native AST-guided code navigation tool that produces `.guidance/src/**/*.json` metadata mirrors and `.guidance.db` SQLite vector search databases. It serves as a deterministic-first subagent for AI coders, optimizing for:
- **Token efficiency**: Minimal context required for frontier model queries
- **RALPH orchestration**: Human-in-the-loop test→lint→fmt→sync→structure→db cycles
- **Sub-100ms queries**: Keyword-first with optional LLM synthesis for long queries
- **Zero marginal cost**: Cached results for repeated queries

The vision is to evolve guidance into a fully-capable subagent that offloads AI coders by maintaining comprehensive codebase knowledge, detecting staleness automatically, and providing deterministic answers where possible while escalating novel queries to local LLM inference.

---

## The Closed Maintenance Cycle

The core purpose of guidance is a **closed incremental loop** that keeps codebase navigation indexes current with every change:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GUIDANCE CLOSED LOOP                                  │
└─────────────────────────────────────────────────────────────────────────────┘

   Source Files (.zig, .py, .rs, .md)
          │
          ▼
   [1] AST Parsing + Member Extraction
          │  tree-sitter (Zig, Python, Rust grammars)
          │  match_hash = SHA-256(signature + comment)
          ▼
   [2] Incremental JSON Sync
          │  should_generate() mtime comparison
          │  match_hash unchanged → preserve existing comments
          │  json_mtime = src_mtime - 1 second (validated marker)
          ▼
   [3] .guidance/src/**/*.json   (metadata mirrors, NOT source of truth)
          │
          ▼
   [4] Database Sync (GuidanceDb.sync_from_dir)
          │  Walk .guidance/src/ → upsert into .guidance.db
          │  Embed cosine similarity vectors
          │  Populate fts_inverted_index
          ▼
   [5] Query Explain (guidance explain "query")
          │  SQLite keyword + vector hybrid search
          │  Staged pipeline: code excerpts + prose + metadata
          ▼
   Codebase Navigation
```

**Each phase has change detection built in** — the loop runs incrementally, not from scratch:

| Phase | Change Detection | DRY Enforcement |
|--------|-----------------|-----------------|
| AST Parsing | Extract fresh match_hash | Source `///` comments not stored in JSON |
| Incremental Sync | `should_generate()` mtime comparison | LLM enhancement only when comment missing |
| JSON Write | Skip if `hash unchanged AND line numbers stable` | Preserve existing tags/patterns |
| Database Sync | Only upsert changed nodes (by path + mtime) | Comments extracted from source on query |
| Query Explain | Return cached results for repeated queries | Never regenerate without source change |

---

## Core Goals

### Goal 1: Deterministic-First Code Navigation

Replace frontier model code browsing with local computation:

```
Traditional AI Coder:     Query → Frontier LLM → Response (expensive, slow)
guidance:                 Query → SQLite Search → Cached Result
                              ↓ (if miss)
                          Local LLM Synthesis → Cache for next time
```

**Key Outcomes:**
- Sub-100ms latency for cached patterns
- Zero API cost for deterministic queries (single keyword, no LLM)
- Local LLM synthesis only for novel queries (>1 word or question patterns)
- Full auditability through `.guidance.db` query log

### Goal 2: RALPH Loop Orchestration

The `guidance check` command enforces the RALPH loop. Each phase runs **incrementally** — only files detected stale by `should_generate()` are reprocessed:

```
test → lint → fmt → sync (stale files only) → structure → db
```

This is deterministic waterfall execution: each stage must pass before the next begins. The `guidance check` command is the pre-commit hook entry point that ensures codebase integrity before any commit.

**Key Invariants — all fully implemented:**
- **JSON mtime is the universal "all stages passed" marker**: A file's JSON mtime advances only when all phases have succeeded for that file
- **Source mtime > JSON mtime = stale file requiring re-sync**: `should_generate()` in `staleness.rs` detects this
- **JSON mtime = source_mtime - 1 second = validated marker**: `save_guidance()` sets this pattern; `should_generate()` skips validated files
- **`match_hash` preservation**: Hash unchanged → preserve existing comments/tags/patterns (`json_store.rs::merge_member`)
- **`guidance sync` processes only stale files**: `cmd_sync` calls `should_generate()` per file
- **`guidance sync --force` reprocesses all**: Bypasses staleness checks
- **`comment_generated` flag tracks LLM-generated comments**: Set by enhancer, used by comment sync phase to write back to source

### Goal 3: Token-Efficient Context for AI Coders

The staged explain pipeline produces optimal context for AI consumption:

```
Query: "how does filterStages work?"
  ↓
Stage Collection:
  ├── Prose: Module comments + member descriptions (semantic context)
  ├── Code: Source excerpts with verified line numbers (implementation)
  ├── Metadata: Keywords, skills, capabilities, used_by (discovery)
  └── Skill: From doc/skills/*/SKILL.md (design patterns)
  ↓
LLM Filter (if >3 words): Relevance pruning
  ↓
LLM Synthesis: Markdown summary with file:line citations
```

**Key Features:**
- `--no-llm`: Fast path for structural output only
- `--filter=auto|force|skip`: Control LLM relevance filtering
- `--output=json`: Structured JSON for local LLM output to control deterministic workflow

### Goal 4: Cross-Language Parity

All supported languages produce identical JSON schemas via tree-sitter AST parsing:

| Field | Supported | Purpose |
|-------|-----------|---------|
| `meta.module` | ✓ | Module name (dot-separated path) |
| `meta.source` | ✓ | Relative path |
| `meta.language` | ✓ | "zig", "python", or "rust" |
| `members[].match_hash` | ✓ | SHA-256 of signature + comment |
| `members[].comment` | ✗ | Not stored in JSON (extracted from source) |
| `members[].line` | ✓ | 1-based line number |
| `members[].signature` | ✓ | Function/struct signature |
| `members[].params` | ✓ | Function parameters |
| `members[].returns` | ✓ | Return type |

### Goal 5: Subagent Structured Output

guidance serves as a deterministic-first subagent providing structured output to frontier orchestrators:

```
guidance explain "query" --output=json
  ↓
{
  "intent": "IDENTIFIER",
  "confidence": 0.95,
  "results": [...],
  "summary": "...",
  "citations": [...],
  "gaps": [...]
}
```

---

## Architecture

### Component 1: Query Processing FSM

The query engine implements a **finite state machine** for deterministic routing:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  QUERY PROCESSING FSM                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  ┌────────┐    ┌──────────┐    ┌───────┐    ┌──────────┐    ┌──────────┐    ┌───────┐  │
│  │ INTAKE │───▶│ CLASSIFY │───▶│ ROUTE │───▶│ VALIDATE │───▶│ ASSEMBLE │───▶│ SYNTH │  │
│  └────────┘    └──────────┘    └───────┘    └──────────┘    └──────────┘    └───────┘  │
│       │            │              │            │               │               │       │
│       ▼            ▼              ▼            ▼               ▼               ▼       │
│  Tokenize        Intent         Search       Result          Token        Prompt LLM   │
│  detection       + Domain      strategy      quality         budget        synthesis   │
│                   extraction                                                           │
│                                                                                        │
│  States (strategy.rs):                                                                 │
│  INTAKE   → Parse query into tokens, detect file paths, detect identifiers             │
│  CLASSIFY → Classify intent: SingleIdentifier, IdentifierLookup, FilePath,             │
│             CapabilityQuery, HowTo, Conceptual, MultiKeyword, GeneralSearch             │
│  ROUTE    → Select search primitive: word_index, anchor_lookup, fts, hybrid, vector    │
│  VALIDATE → Verify results: match_hash, relevance threshold                            │
│  ASSEMBLE → Token-budgeted stage collection                                            │
│  SYNTH    → LLM with grounded excerpts (only if confidence < threshold)                │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

**QueryClass — Three-Dimensional Classification:**

The FSM produces three classification dimensions:

```rust
pub struct QueryClass {
    pub intent: QueryIntent,   // SingleIdentifier, CapabilityQuery, HowTo, etc.
    pub domain: String,        // "coral", "guidance", "dag", "common", "llm"
    pub confidence: f32,       // 0.0-1.0
    pub tokens: Vec<String>,
    pub detected_file_paths: Vec<String>,
    pub detected_identifiers: Vec<String>,
}
```

**Intent Classification Rules (strategy.rs):**

| Intent | Priority | Detection | Search Primitive |
|--------|----------|-----------|-----------------|
| `SingleIdentifier` | 0 | 1 token, camelCase/snake_case | word_index |
| `IdentifierLookup` | 0 | Multi-word with identifier patterns | word_index |
| `FilePath` | 1 | Contains `/`, `\`, or dotted uppercase | anchor_lookup |
| `CapabilityQuery` | 2 | 2-4 word phrases | fts |
| `HowTo` | 4 | Starts with question words | hybrid |
| `Conceptual` | 4 | 5+ word phrases | vector |
| `MultiKeyword` | 5 | 2+ words (fallback) | fts |
| `GeneralSearch` | 6 | Everything else | keyword |

### Component 2: Search Primitives Hierarchy

The query engine implements a **primitives hierarchy** ordered by determinism:

| Rank | Primitive | When | Implementation | Latency | LLM Required |
|------|-----------|------|--------------|---------|---------------|
| 1 | **WordIndex** | Always tried first | Inverted index lookup (exact identifier match) | <1ms | No |
| 2 | **AnchorLookup** | When domain classified | Capability anchors → domain-limited search | <5ms | No |
| 3 | **FTS Keyword** | Domain-limited or fallback | SQLite LIKE + position rank | <10ms | No |
| 4 | **RRF Merge** | Multi-keyword | Reciprocal rank fusion | <30ms | No |
| 5 | **Hybrid Fallback** | After RRF | Vector + keyword weighted | <100ms | Optional |
| 6 | **Vector Only** | Last resort | Cosine similarity only | <200ms | Required |

### Component 3: Grounding Enforcement

**Critical Invariant:** No synthesis without source. The LLM must receive verbatim code excerpts, never file paths.

**Grounding Protocol:**
- Extract source excerpts for matched identifiers
- Include full function/struct body, not summaries
- Always include line numbers for citation
- Prompt enforces: "Use ONLY the provided source excerpts"

### Component 4: Structured Output Schemas

| Mode | Flag | Output Format | Use Case |
|------|------|-------------|---------|
| **Markdown** | (default) | Markdown stages | Human reading |
| **JSON** | `--output=json` | Structured JSON | Local LLM orchestration |
| **Compact** | `--output=compact` | JSON, abbreviated | Token savings |
| **Debug** | `--output=debug` | JSON + metadata | Development |

### Component 5: Sync Pipeline (`SyncEngine`)

The sync pipeline is a **closed incremental loop** — each source file is processed independently, and only stale files are touched:

```
Source File (.zig/.py/.rs/.md)
        │
        ▼
    [AstParser] ─→ tree-sitter parse for each language
        │
        ▼
    Member Extraction ──→ members[] with signatures, line numbers
        │
        ▼
    [match_hash Check] ──→ Hash unchanged? Preserve existing comments/tags/patterns
        │
        ▼
    [LLM Enhancement] ──→ Generate missing comments (comment_generated=true)
        │
        ▼
    Guidance JSON ──→ .guidance/src/<path>.json
        │
        ▼
    [Comment Sync] ──→ Write generated comments back to source
        │
        ▼
    [Database Sync] ──→ .guidance.db (SQLite + vector embeddings)
```

**Incremental Design — fully implemented:**
- `should_generate(src_mtime, json_mtime)`: JSON absent → stale; JSON mtime < source mtime → stale; JSON mtime = source_mtime - 1s → validated, skip
- `match_hash`: SHA-256 of `signature ++ "|||COMMENT|||" ++ comment`; unchanged = preserve existing comments/tags/patterns
- Per-file processing: `guidance sync --file src/foo.rs` processes only the named file
- Comment source of truth: `///` doc comments live in source, NOT in JSON

### Component 6: RALPH Loop (`cmd_check`)

The RALPH loop is the **human-in-the-loop gate** before commits:

```
1. Test   — cargo test --workspace (from config test_commands)
2. Lint   — cargo clippy --workspace (from config lint_commands)
3. Fmt    — cargo fmt --check (from config fmt_commands)
4. Sync   — guidance sync (ONLY stale files, via should_generate check)
5. Structure — regenerate STRUCTURE.md
6. DB     — sync .guidance.db via sync_from_dir
```

**Config-driven commands:** Each stage reads its command from `guidance-config.json`:
```json
{
  "test_commands": { "rs": ["cargo", "test", "--workspace"] },
  "lint_commands": { "rs": ["cargo", "clippy", "--workspace", "--", "-D", "warnings"] },
  "fmt_commands": { "rs": ["cargo", "fmt", "--check"] }
}
```

**Failure Modes:**
- Test failure: Exit 1, print test output
- Lint failure: Exit 1, print lint violations
- Parse error: Continue with warning (unguidable file)

### Component 7: Registry Layer (shared with Coral)

The registry layer provides shared building blocks used by both guidance and coral:

```
src/fluent-wvr/         ──→ Component, WorkUnit, FieldAccess, Describable traits
src/dag/                ──→ TargetRegistry, Target, CapabilityRegistry
src/llm/                ──→ LlmClient, EmbeddingProvider, CachedEmbeddingProvider
src/knowledge/  ──→ WordIndex, TrigramIndex, CsrGraph
src/search-vector/      ──→ GuidanceDb (SQLite hybrid search)
src/common-core/        ──→ Hashing, formatting, string utilities, shell
```

---

## Data Model

### GuidanceDoc (`.guidance/src/**/*.json`)

```rust
pub struct GuidanceDoc {
    pub meta: Meta,                      // module, source, language
    pub comment: Option<String>,         // Module-level doc comment (//!)
    pub keywords: Vec<String>,           // Discovery keywords
    pub skills: Vec<Skill>,              // SKILL.md references
    pub used_by: Vec<String>,            // Reverse import dependencies
    pub members: Vec<Member>,            // Extracted declarations
}
```

### Member (`.guidance/src/**/*.json` members[])

```rust
pub struct Member {
    pub member_type: MemberType,         // FnDecl, Struct, Enum, Test, etc.
    pub name: String,                    // Member name
    pub match_hash: Option<String>,      // SHA-256(signature ++ "|||COMMENT|||" ++ comment)
    pub signature: Option<String>,       // Function/struct signature
    pub params: Vec<Param>,              // Function parameters
    pub returns: Option<String>,         // Return type
    pub line: Option<u32>,               // 1-based line number
    pub comment: Option<String>,         // In-memory only: extracted from source
}
```

**Comment Storage Strategy:**
- **Module-level comments (`//!`)**: Stored in JSON's top-level `comment` field
- **Member comments (`///`)**: Extracted from source on-demand, never stored in JSON
- **Rationale**: Keeps JSON minimal, reduces diff noise, aligns with Rust's design
- **`match_hash`**: Still tracks comment changes for staleness detection

### Stage (Staged Query Pipeline)

```rust
pub struct Stage {
    pub kind: StageKind,        // Prose, Code, Metadata, SkillDoc
    pub content: String,        // Stage text
    pub source: String,         // "src/foo.rs"
    pub line: Option<u32>,      // Source line number
    pub member_name: Option<String>,
    pub member_type: Option<String>,
}
```

### SQLite Schema (`.guidance.db`)

```sql
CREATE TABLE guidance_nodes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    signature TEXT,
    comment TEXT,
    module TEXT,
    language TEXT,
    embedding BLOB
);

CREATE TABLE embedding_cache (
    query_hash TEXT PRIMARY KEY,
    query_text TEXT,
    embedding BLOB
);
```

---

## CLI Reference

### 14 Subcommands

| Command | Purpose | Key Args |
|---------|---------|----------|
| `guidance sync` | Parse source → JSON → DB | `--file`, `--scan`, `--force`, `--no-db`, `--dry-run`, `--watch` |
| `guidance explain` | Semantic search query | `query`, `--db`, `--limit`, `--no-llm`, `--filter` |
| `guidance check` | Multi-stage CI check | `--workspace` |
| `guidance init` | Create `.guidance/` + default config | `dir` |
| `guidance status` | Show staleness counts | `--guidance-dir` |
| `guidance clean` | Remove generated files | `--json-dir`, `--db` |
| `guidance structure` | Generate STRUCTURE.md | `--json-dir` |
| `guidance health` | Comment coverage stats | `--workspace`, `--format` |
| `guidance benchmark` | Query accuracy benchmark | `query`, `--num`, `--no-llm` |
| `guidance test` | Run cargo test | (none) |
| `guidance telemetry` | DB node/embedding counts | `--db`, `--reset` |
| `guidance cache-stats` | DB cache statistics | `--db` |
| `guidance todo` | Print TODO.md items | (none) |
| `guidance diary` | Append diary entry | `text` |
| `guidance commit` | Git commit wrapper | `message`, `--dry-run` |

### Supported Languages

| Extension | Language | Tree-sitter Grammar |
|-----------|----------|-------------------|
| `.zig`, `.zon` | Zig | `tree-sitter-zig` |
| `.py` | Python | `tree-sitter-python` |
| `.rs` | Rust | `tree-sitter-rust` |
| `.md` | Markdown | (plugin-based) |

---

## Workspace Crate Map

```
src/
  bin/
    guidance/          guidance binary (14-subcommand CLI)
    coral/             coral binary (MCP server + ingest CLI)
  guidance/            guidance-core: AST parser, sync engine, query engine, config
  coral/               coral-context: graph DB, cache router, MCP server, WASM runtime
  dag/                 guidance-dag: executor, resolver, work_unit, adapter, middleware
  fluent-wvr/          Fluent WVR: Component, WorkUnit, FieldAccess, Describable traits
  fluent-wvr-macros/   Proc macros for FieldAccess derive
  fluent-concurrency/  WorkerPool, Scope, Zone, Limiter, PriorityQueue, CreditFlow
  llm/                 LLM HTTP client + embeddings (Ollama, OpenAI)
  types/               Shared domain types (GuidanceDoc, Member, FileType, etc.)
  common-core/         General utilities (hashing, formatting, shell, string ops)
  search-vector/       SQLite hybrid search (vector + keyword + RRF)
  knowledge/   WordIndex, TrigramIndex, CsrGraph, QueryCache
  content-node/        LOD slicing, file content annotation
  ontology/            Entity extraction, YAGO taxonomy, capability inference
  rdf/                 Turtle/N-Quads parser, normalization
  wasm_ipc/            WASM IPC binary types
```

---

## Current Implementation Status

### Completed

1. **AST Parsing**: Tree-sitter for Zig, Python, Rust with `AstParser` struct
2. **Incremental Sync**: `match_hash` comparison, mtime-based staleness detection (`staleness.rs`)
3. **Hybrid Search**: SQLite keyword + cosine similarity vector fusion (`GuidanceDb`)
4. **Staged Pipeline**: `Synthesizer::synthesize()`, stage formatting
5. **LLM Filtering**: `LlmFilter` with `NoopLlmFilter` fallback
6. **RALPH Loop**: `guidance check` → test→lint→fmt→sync→structure→db
7. **Fluent WVR Integration**: `WorkerPool` + `TokioRuntime` for parallel AST/DB ops
8. **Comment In-filling**: LLM-generated comment sync to source (`comments.rs`)
9. **Query Classification FSM**: `FsmEngine` with 8 intent types (`strategy.rs`)
10. **WordIndex**: Inverted word index for fast keyword lookup (`knowledge`)
11. **Config-Driven Commands**: `test_commands`, `lint_commands`, `fmt_commands` from JSON
12. **File Watching**: `--watch` mode with debounced filesystem events
13. **Benchmark System**: Query accuracy scoring (relevance, completeness, navigation)

### In Progress

1. **Fluent WVR Pattern Adoption**: WorkUnit impls for AstGenJob/DbSyncJob, Instrumented wrappers
2. **Async I/O Consistency**: Replacing std::fs with tokio::fs in hot paths
3. **DRY Enforcement**: Extracting duplicated WordIndex fallback logic

### Planned

1. **Token-Budgeted Assembly**: Binary search for optimal subset fitting budget
2. **MCP Server**: IDE integration via Model Context Protocol
3. **Git-Aware Snapshot**: Snapshot persistence for <1s index loading
4. **Persistent Query Cache**: TTL-based disk-backed cache for LLM synthesis results
5. **Centroid Classification**: SimHash centroid matching for domain routing

---

## Deterministic-First Strategy

### Query Classification FSM

```
Input Query
      ↓
[INTAKE] → Parse tokens, detect patterns
      ↓
[CLASSIFY] → Intent: SingleIdentifier | CapabilityQuery | FilePath | HowTo | Conceptual
      ↓
[ROUTE] → Select search primitive: word_index → fts → hybrid → vector
      ↓
[VALIDATE] → match_hash valid? relevance >= threshold?
      ↓
[SYNTHESIZE] → LLM with grounded excerpts
```

### Cache Hierarchy

| Cache Level | Content | Hit Rate | Latency | Source |
|-------------|---------|---------|---------|--------|
| L1: WordIndex | Exact identifier | ~40% | <1ms | `knowledge::WordIndex` |
| L2: FTS Keyword | Full-text search | ~15% | <10ms | SQLite LIKE + position rank |
| L3: RRF Merge | Hybrid fusion | ~10% | <30ms | Reciprocal rank fusion |
| L4: LLM Synthesis | Cached summaries | ~10% | <50ms | embedding_cache table |
| Miss | Local LLM | ~15% | 200-800ms | `llm::LlmClient` |

---

## RALPH Loop Invariants

### Test Phase
```
cargo test --workspace
```
**Invariant:** All unit tests must pass.

### Lint Phase
```
cargo clippy --workspace -- -D warnings
```
**Invariant:** Zero clippy warnings.

### Format Phase
```
cargo fmt --check
```
**Invariant:** All source files formatted.

### Sync Phase
```
guidance sync
```
**Invariant:** JSON mtime updated only on success. Only stale files processed.

### Structure Phase
```
guidance structure
```
**Invariant:** `STRUCTURE.md` regenerated idempotently.

### Database Phase
```
guidance sync (with db_sync=true)
```
**Invariant:** `.guidance.db` ready for queries.

---

## Integration with Coral Context

### Shared Modules

| Module | Guidance Use |
|--------|-------------|
| `coral::db` | Library (SQLite graph store) for node storage |
| `coral::cache_router` | Multi-tier query routing |
| `coral::packer` | Context packing with token budget |
| `coral::mcp` | MCP server for IDE integration |
| `coral::wasm_runtime` | WASM plugin execution |
| `llm::LlmClient` | LLM HTTP wrapper |
| `llm::EmbeddingProvider` | Embedding computation |
| `dag::TargetRegistry` | DAG target management |

---

## Success Metrics

| Metric | Current | Target |
|--------|--------|--------|
| Deterministic resolution | ~40% | >60% |
| Query latency (WordIndex) | <1ms | <1ms |
| Query latency (cached) | <100ms | <50ms |
| Query latency (LLM) | 500-1500ms | <800ms |
| Token usage median | Varies | <4000 |
| RALPH loop | ~30s | <20s |

---

## Future Directions

### Priority 1 — Query Speed (immediate runtime impact)

1. **WordIndex O(1)**: Replace SQL LIKE table scans with inverted word index. Target: <1ms per exact identifier lookup.
2. **Persistent Query Cache**: TTL-based disk-backed cache for LLM synthesis results.

### Priority 2 — Startup Speed

3. **Git-Aware Snapshot**: Snapshot persistence for <1s index loading on `guidance sync` startup.

### Priority 3 — Routing Intelligence

4. **Intent FSM Enhancement**: Add domain classification to FSM states.
5. **RRF Merge**: Already implemented; tune k-parameter for optimal fusion.
6. **Grounding Enforcement**: Formal `can_synthesize()` check before LLM synthesis.

### Priority 4 — Future Scale

7. **MCP Server**: IDE integration via Model Context Protocol.
8. **Query Telemetry**: Learning from query history.
9. **Two-Tier Content**: Drawer/closet architecture for topic pointer compression.

---

## Conclusion

guidance evolves into a deterministic-first subagent that transforms local LLM execution from generative model into bounded computation unit:

- **FSM routing** replaces cosmetic classification with actual state transitions
- **WordIndex primary** provides O(1) lookup for common queries
- **Grounding enforcement** eliminates hallucination
- **Structured output** enables frontier orchestration
- **RALPH loop** ensures codebase integrity before every commit

The architecture achieves the vision through pattern convergence:
- Primitives over models (local computation covers 80%+ of queries)
- Cached results for zero marginal cost
- Auditability through query telemetry
- Continuous learning through skill auto-linking

The result is an AI-assisted development tool that grows more capable with every use while remaining fast, deterministic, and auditable.

---

*Vision Document v3.0 — June 2026 (Rust codebase)*
