Comprehensive Analysis: Agentic Documentation & Guidance System

System Completeness Assessment

Architecture Overview

This is a remarkably complete bidirectional knowledge system that bridges AST-level code understanding with LLM-enhanced documentation. The architecture demonstrates sophisticated thinking about the AI-assisted development cycle:

Core Innovation: The system treats documentation as a cached computation with incremental updates, not a separate artifact. The match_hash field (SHA-256 of signature) enables surgical updates—only regenerate descriptions when the API contract changes.

Key Design Wins:

1. Determinism-First Philosophy: AST parsing produces the ground truth (functions, types, signatures, line numbers). LLM enhancement is Phase 2, strictly additive, never authoritative for structural data.
2. Dual-Language Implementation: Python orchestrates (fast prototyping, ecosystem integration), Zig executes (performance, zero-copy parsing). This division is pragmatic, not accidental.
3. Single Source of Truth: The ./.guidance/src/**/*.json files are canonical. STRUCTURE.md, make explain output, and query results are all derived views of this data.
4. Inbox Pattern: doc/guidance/inbox/{INSIGHTS,CAPABILITIES}.md acts as a staging area. Unstructured knowledge flows into bullets, make document promotes them into structured skills. This prevents duplication and ensures insights are captured in the heat of development.
5. Synthetic Comment Scrubbing: The _is_synthetic_comment() function (100+ lines of heuristics) is pure gold—it detects AST fallback templates, LLM preamble leaks, truncated sentences, and conversational reasoning. This prevents garbage-in-garbage-out during LLM infill.

Completeness Breakdown

✅ Fully Implemented:

- AST → JSON Pipeline: Both Python (via ast module) and Zig (custom parser on std.zig.Ast) extract members with signatures, params, return types, visibility
- Incremental Sync: match_hash comparison enables skip-if-unchanged behavior
- LLM Enhancement: Three modes (infill, regen, structure-only) with quality scoring and best-of-N selection
- Skill Attachment: Pattern detection (GoF, Domain) auto-adds skill references to guidance JSON
- Reverse Dependencies: findReverseDeps() scans for @import("foo.zig") patterns, populates used_by field
- Query Engine: make explain and make query use keyword matching + AST analysis + inbox scraping
- STRUCTURE.md Generation: Hierarchical tree view with inline comments from guidance JSON

⚠️ Partially Implemented:

- make document (learn command): Python cmd_learn() exists but Zig cmdDocument() is stubbed (not yet implemented). The Ollama-based promotion logic is Python-only.
- TODO Triage System: cmd_triage() generates TRIAGE.md from TODO.md, intended to feed Aider. Workflow is present but integration requires manual handoff.
- Commit Message Generation: cmd_commit() calls LLM with git diff + recent log, but doesn't auto-commit (requires $EDITOR invocation).

❌ Missing/Incomplete:

- Cross-Language Consistency: Python's PythonProcessor and Zig's SyncProcessor have slightly divergent schemas (e.g., Python emits class_methods, Zig nests them as members[].members). The JSON schema isn't formally specified.
- Skill Validation: No check that skills[].ref paths actually exist. Broken skill references silently persist.
- Deduplication Across Languages: A Python module and Zig module with identical purpose will have separate guidance JSON. No cross-language concept linking.
- Interactive Refinement: make explain is read-only. No make annotate QUERY="foo" to interactively add hashtags or skills.
- Performance Profiling: No telemetry on which guidance files are accessed most (for prioritizing LLM enhancement budget).

---

Python vs. Zig: Functional Parity Analysis

Division of Responsibilities

The design is intentionally asymmetric, not incomplete:

| Feature | Python (bin/guidance.py) | Zig (src/ast-guidance/) | Rationale |

|---------|---------------------------|--------------------------------|-----------|
| AST Parsing | ast.parse() → PythonProcessor | std.zig.Ast.parse() → AstParser | Each language parses its own (no polyglot AST) |
| JSON I/O | Manual json.loads/dumps | JsonStore with arena allocator | Zig optimizes for zero-copy + bulk operations |
| LLM Calls | LLMClient (requests lib) | Enhancer + common/llm.zig | Both implement same protocol (Ollama /v1/completions) |
| Scrubbing | _is_synthetic_comment() (Python) | N/A | Python-only pre-processing before LLM sees data |
| STRUCTURE.md | StructureGenerator walks JSON | N/A | Python-only (Zig stubbed per code comment) |
| Query/Explain | cmd_explain() orchestrator | QueryEngine + cmdExplore() | Redundant implementations—Python is legacy? |
| Inbox Promotion | cmd_learn() with RAG logic | Stubbed | Python-only (requires Ollama embeddings) |

Critical Differences

1. LLM Enhancement Strategy

Python (guidance.py:2488-2563):

- Phase 1: Sync AST → JSON (Zig handles .zig, Python handles .py)
- Phase 1.5: Scrub synthetic comments (scrub_all_json())
- Phase 2: Universal infill over all JSON files (infill_all_json())

Zig (sync.zig:92-153):

- Inline enhancement: During processFile(), immediately after mergeMembers()
- Enhancer is opt-in via --infill or --regen flags
- No scrubbing phase—relies on Python pre-processing

Implication: You cannot run ast-guidance sync --infill on a clean repo without garbage comments. The scrubbing logic must be ported to Zig or called as a pre-step.

2. Query Architecture

Python cmd_explain():

# Keyword match in guidance JSON filenames/modules

# → Load matching JSON, extract functions/classes
# → Search inbox bullets for query term
# → Optional LLM summary pass (--ai flag)
# → Format as markdown sections

Zig cmdQuery():

// QueryEngine.execute():

// → File matches (grep source tree)
// → Guidance matches (load JSON, filter by module/source)
// → Live AST parse (fresh member extraction)
// → Format as compact or JSON

Zig cmdExplore():

// Substring match on meta.module or meta.source

// → Load guidance JSON
// → Re-parse source file for LIVE members (includes enum fields!)
// → Render comprehensive report with skills, used_by, inbox bullets

Key Insight: Zig's explore command does live AST parsing to show members added since last sync. Python's explain relies solely on cached JSON. For rapidly evolving code, Zig is more accurate.

3. Match Hash Algorithm

Python (guidance.py:1847-1878):

def _compute_match_hash(member: Dict) -> str:
    sig = member.get("signature", "") or member.get("name", "")
    # Hash the signature string
    return hashlib.sha256(sig.encode()).hexdigest()

Zig (hash.zig, inferred from match_hash usage):

// Likely: SHA-256 of normalized signature
// (Code not shown in excerpts but referenced in sync.zig:15)

Assumption: Both use signature-based hashing. Risk: If Zig normalizes whitespace differently (e.g., fn foo( a: i32 ) vs fn foo(a: i32)), hashes diverge and comments are re-infilled unnecessarily.

4. Skill Auto-Detection

Python: Pattern detection logic not shown in excerpts (may be in separate module).

Zig (sync.zig:372-442):

fn hasGofPatterns(members: []const types.Member) -> bool
fn hasDomainPatterns(members: []const types.Member) -> bool
fn buildSkills(...) -> []const types.Skill

Recursively walks members[].patterns[] to check for .GoF or .Domain pattern types. If detected, auto-adds doc/skills/{gof,domain}-patterns/SKILL.md references.

Question: Where are patterns initially detected and attached to members? This logic is missing from the excerpts—likely in pattern.zig (not reviewed).

---

make explain Effectiveness for AI Coders

Current Design

The Makefile target likely invokes:

explain:

	@bin/guidance.py explain "$(QUERY)" --ai

This runs Python's cmd_explain() which:

1. Searches guidance JSON for modules/functions matching QUERY
2. Loads member data (signatures, comments, line numbers)
3. Scrapes inbox files for bullet points containing QUERY
4. Optionally calls LLM to synthesize a summary

Output Format (inferred from Python code):

# Query: <term>

## Modules

- src/foo.zig:123 — Comment from guidance JSON

## Functions

- fn bar(x: i32) -> bool — Does X, returns Y

## Recent Knowledge (from inbox)

- Insight about <term> from INSIGHTS.md

## AI Summary (if --ai)

LLM-generated explanation

Strengths

1. Contextual Breadth: Combines AST data (signatures, line numbers), human comments (from source), AI descriptions (cached), and recent insights (inbox bullets). This multi-modal context is ideal for agents.
2. Incremental Discovery: The inbox mechanism captures knowledge as it's generated during task execution, not retrospectively. This is critical for long-running agentic workflows.
3. Skill Pointers: When a query matches a module with skills: [gof-patterns], the agent knows to read doc/skills/gof-patterns/SKILL.md for deeper context. This is chained reasoning without hard-coding.

Weaknesses & Improvements

1. Search Precision

Current: Substring match on module name, function name, or source path.

Problem: Query "hash" matches hash.zig, match_hash field, hashlib import, #hashtags. Too noisy.

Solution: 

- Ranked search: TF-IDF over (module comments + member comments + signatures). Boost exact identifier matches.
- Fuzzy matching: Levenshtein distance for typos ("parsr" → "parser").
- Scope filters: make explain QUERY="parse" SCOPE=functions (ignore modules/types).

2. Semantic Search (Missing)

Current: Keyword-only. Query "validate input" won't match a function named sanitizeUserData() with comment "Cleans and validates input."

Solution:

- Embed member comments via Ollama (e.g., nomic-embed-text) into a vector DB (could reuse CozoDB's kNN).
- make explain QUERY="validate input" --semantic → cosine similarity search.
- Hybrid ranking: BM25 (keyword) + cosine (semantic) with tunable weights.

Lightweight Option: Skip vector DB. At query time, embed the query + top 20 keyword matches, compute cosine in-memory. Fast enough for <1000 modules.

3. Dependency-Aware Context

Current: used_by field shows reverse deps, but make explain doesn't traverse them.

Problem: Agent asks "How is AstParser used?" Current output shows AstParser members but not call sites.

Solution:

- Usage examples: Extract call-site snippets during sync (grep for AstParser.init(, store in JSON).
- Cross-module recommendations: "This function calls foo() in module X — see .guidance/src/X.json."

4. Interactive Refinement

Current: Read-only query.

Proposal: make annotate QUERY="resolver" TAG="graph-traversal" --skill=domain-patterns

- Adds #graph-traversal to tags[] in matching guidance JSON.
- Appends skill reference if not present.
- Agent can teach the system during exploration.

5. Caching & Incremental Updates

Current: Every make explain re-scans all guidance JSON + re-scrapes inbox files.

Solution:

- Inverted index: Build guidance/.index/{keywords,tags,skills}.json mapping terms → file paths.
- Regenerate index only when guidance JSON mtimes change (Makefile dependency).
- Query time: O(1) lookup in index, then load ~3-5 JSON files instead of scanning 100+.

Estimated Speedup: 50-100x for large codebases (1000+ modules).

---

src/target.zig as Makefile Replacement

Current Target System (Inferred)

From excerpts, src/target.zig defines a Target type representing build artifacts. The code wasn't shown, so I'll analyze based on:

1. Zig's build system philosophy (build.zig + std.Build)
2. Makefile targets in project (linting, testing, guidance sync)

Feasibility Analysis

Can Target Replace the Makefile?

Short Answer: Partially—for build tasks, yes. For workflow orchestration, no (without significant design changes).

What Targets Can Be Replaced

✅ Pure Zig Compilation:

// build.zig equivalent of `make build`
const exe = b.addExecutable(.{
    .name = "codebase",
    .root_source_file = .{ .path = "src/main.zig" },
});

✅ Zig-Native Tools:

// `make test`

const tests = b.addTest(.{ .root_source_file = .{ .path = "src/main.zig" } });

const run_tests = b.addRunArtifact(tests);

test_step.dependOn(&run_tests.step);

// `make lint` (if using zig fmt)

const fmt = b.addFmt(.{ .paths = &.{"src"} });

✅ Guidance Sync (with caveats):

// `make guidance`

const guidance_sync = b.addSystemCommand(&.{

    "zig-out/bin/ast-guidance", "sync", "--scan", "src", "--output", "guidance"

});

guidance_sync.step.dependOn(&exe.step); // Run after building ast-guidance

What Cannot Be Replaced (Easily)

❌ Python Orchestration:

- make explain calls bin/guidance.py explain (Python).

- make document calls bin/guidance.py learn with Ollama embeddings.

- Zig's std.Build doesn't run Python scripts ergonomically (need addSystemCommand() wrappers).

❌ Per-File Targets:

Makefiles excel at fine-grained dependencies:

.guidance/src/%.zig.json: src/%.zig

	ast-guidance sync --file $< --output guidance

Zig's build system is artifact-centric, not file-centric. You can't easily express "regenerate JSON only for changed .zig files" without manually tracking mtimes.

Workaround: Use std.Build.Step.run() with a custom step that:

1. Compares mtimes of src/**/*.zig vs .guidance/src/**/*.json.

2. Invokes ast-guidance sync --file <changed> per dirty file.

3. Zig's std.fs.Dir.walk() + std.fs.File.stat() provide primitives.

❌ Conditional Execution:

Makefile:

pre-commit: lint test guidance

	@echo "✓ All checks passed"

Zig equivalent:

const pre_commit = b.step("pre-commit", "Run all checks");
pre_commit.dependOn(&lint.step);
pre_commit.dependOn(&test_step);
pre_commit.dependOn(&guidance_sync.step);

// But: no conditional "skip if no changes" without custom logic

Hybrid Approach (Recommended)

Keep the Makefile as the user interface:

build:
	zig build
test:
	zig build test
guidance:
	zig build guidance-sync
	bin/guidance.py sync --scan src --infill  # Python for scrubbing + LLM
explain:
	@zig-out/bin/ast-guidance query "$(QUERY)" --format compact
pre-commit:
	zig build pre-commit-checks

Implement build logic in build.zig, but let Make handle:

1. Python interop (bin/guidance.py).
2. Environment variables (OLLAMA_HOST, MODEL).
3. Shorthand aliases (make explain Q=foo vs zig build explain -DQUERY=foo).

---

Strategic Recommendations

1. Formalize the JSON Schema

Create doc/schemas/guidance.schema.json (JSON Schema draft-07):

{
  $schema: http://json-schema.org/draft-07/schema#,
  type: object,
  required: [meta, members],
  properties: {
    meta: {
      type: object,
      required: [module, source, language],
      properties: {
        module: { type: string },
        source: { type: string },
        language: { enum: [zig, python] }
      }
    },
    comment: { type: string },
    skills: {
      type: array,
      items: {
        type: object,
        required: [ref],
        properties: {
          ref: { type: string },
          context: { type: string }
        }
      }
    },
    members: { type: array, items: { $ref: #/definitions/Member } }
  },
  definitions: {
    Member: {
      type: object,
      required: [type, name, is_pub],
      properties: {
        type: { enum: [fn_decl, struct, enum, class, method] },
        name: { type: string },
        match_hash: { type: string, pattern: ^[a-f0-9]{64}$ },
        signature: { type: string },
        comment: { type: string },
        line: { type: integer },
        is_pub: { type: boolean },
        members: { type: array, items: { $ref: #/definitions/Member } }
      }
    }
  }
}

Benefit: Both Python and Zig validate against the same schema. Catches divergence early (e.g., Zig emitting fn_private while Python expects function).

2. Port Scrubbing to Zig

The 100-line _is_synthetic_comment() heuristic is too valuable to leave Python-only. Port to src/ast-guidance/scrub.zig:

pub fn isSynthetic(comment: []const u8) bool {

    // Regex patterns via std.mem.indexOf + manual state machine

    // (Zig stdlib has no regex; acceptable for fixed patterns)

}

Then ast-guidance sync can run standalone without Python pre-processing.

3. Unify Query Implementations

Currently:

- Python cmd_explain() → Markdown output

- Zig cmdQuery() → Compact text or JSON

- Zig cmdExplore() → Detailed markdown

Proposal: Deprecate Python's explain. Make Zig the single implementation:

make explain QUERY=foo   →   zig-out/bin/ast-guidance explore "foo"

Why? Zig's live AST parsing is more accurate, and the codebase is already C-speed.

4. Implement Inverted Index

Add ast-guidance index subcommand:

ast-guidance index --scan guidance --output guidance/.index

Outputs:

- guidance/.index/keywords.json: { "parse": ["src/parser.zig", "src/ast_parser.zig"], ... }

- guidance/.index/skills.json: { "gof-patterns": ["src/registry.zig", ...], ... }

Query uses the index:

const matches = index.lookup("parse");  // O(1) hash lookup

for (matches) |path| {

    const doc = store.loadGuidance(path);

    // Render results

}

5. Cross-Language Concept Linking

Problem: src/parser.zig and bin/parser.py both parse JSON, but guidance JSON treats them as unrelated.

Solution: Add equivalents[] field:

{

  meta: { module: bin.parser, language: python },

  equivalents: [

    { module: src.json_parser, language: zig, reason: Same parsing logic, different impl }

  ]

}

Populated manually or via LLM analysis (compare function signatures + comments, detect semantic similarity).

make explain QUERY="parser" shows both implementations side-by-side, teaching the agent about dual implementations.

---

Conclusion: System Maturity

Overall Grade: A- (85/100)

This is a production-grade foundation for agentic development, with thoughtful design choices:

- ✅ Deterministic AST extraction (ground truth)

- ✅ Incremental updates (match_hash)

- ✅ LLM as enhancement, not authority

- ✅ Inbox pattern for knowledge capture

- ✅ Skill-based context linking

Gaps preventing A+:

- Python/Zig parity incomplete (scrubbing, STRUCTURE.md)

- No semantic search (embedding-based)

- Query performance unoptimized (no index)

- Makefile/Target hybrid unclear

- Schema not formalized (drift risk)

Recommended Next Steps:

1. Week 1: Port scrubbing to Zig, formalize schema, validate cross-language output.

2. Week 2: Build inverted index, benchmark query performance (target <100ms).

3. Week 3: Add semantic search (Ollama embeddings + in-memory cosine).

4. Week 4: Implement make annotate for interactive agent teaching.

With these additions, this becomes a reference implementation for agentic codebase intelligence. The inbox pattern and determinism-first philosophy are genuinely novel contributions to the AI-assisted development tooling space.

