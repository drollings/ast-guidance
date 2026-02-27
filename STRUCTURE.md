# AST-Guidance Project Structure

A fast, lightweight code navigation and orchestration framework friendly to
human and human-in-the-loop LLM agentic software engineering.  It is based
on enriched AST, and uses optional AI for documentation which is cached,
idempotent, and upcycled for lightweight searches and local agentic
intelligence.

## Quick Navigation (Coding Assistants)

| Purpose | File | Use When |
|---------|------|----------|
| **Find related code** | `make query QUERY="search terms"` | Searching for code |
| **Check Implementation** | `make explore QUERY="search terms"` | Before implementing anything |
| **Understand patterns** | `doc/capabilities/*.md` | Implementation examples + patterns |
| **Find existing code** | `mcp_grep` or `mcp_lsp_find_references` | Searching for implementations |

## **Attention**: Skills needed to understand files

Skills are referenced per-file in comments below.  The lookup path for the skills is: 
`.ast-guidance/.skills/{skill}/SKILL.md`

So if you find a file you're looking for named file.zig:
`file.zig      # [zig-current, gof-patterns] Summary of files' contents` , 
Then you you must read

```
.ast-guidance/.skills/zig-current/SKILL.md
.ast-guidance/.skills/gof-patterns/SKILL.md
```

---

## Directory Tree (Git-Tracked Files Only)

```
.
├── bin
│   ├── ast-guidance
│   └── ast-guidance-py
├── doc
│   └── DESIGN.md
├── env
│   ├── mise
│   │   ├── mise.go.toml
│   │   ├── mise.php.toml
│   │   ├── mise.pine.toml
│   │   ├── mise.rust.toml
│   │   ├── mise.wasm.toml
│   │   └── mise.zig.toml
│   └── mk
│       ├── targets
│       │   ├── go.mk
│       │   ├── php.mk
│       │   ├── pine.mk
│       │   ├── py.mk
│       │   ├── rust.mk
│       │   └── zig.mk
│       ├── common.mk
│       └── target_language.mk
├── src
│   ├── ast-guidance
│   │   ├── ast_parser.zig                        # Parses Zig AST, extracts member signatures, and manages memory for guidance generation.
│   │   ├── config.zig                            # Defines project config with absolute paths, JSON source, skills, inbox, and model; loads via two-level fallback chain.
│   │   ├── deps.zig                              # Generates dependency graph from Zig AST files, handling paths and file content to build a map of imports.
│   │   ├── enhancer.zig                          # Zig enhancement enhancer for docstring generation, mirroring Python AIDocstringEnhancer with LLM support.
│   │   ├── gitignore.zig                         # Manages Gitignore patterns, loads from files, and cleans up allocators while supporting negations and exclusions.
│   │   ├── hash.zig                              # Implements SHA-256 hashing and signature generation for Zig types, ensuring consistent hashes and deterministic API o...
│   │   ├── json_store.zig                        # [gof-patterns] Manages Zig AST parsing, stores guidance docs, tracks leaked prompts, and supports loadGuidance for LL...
│   │   ├── main.zig                              # This file defines Zig's guidance system, handling subcommands for AST parsing, code generation, debugging, and docume...
│   │   ├── pattern.zig                           # [gof-patterns] Analyzes Zig AST nodes to detect design patterns using text heuristics and node metadata.
│   │   ├── query.zig                             # Manages memory for Zig AST query results, freeing allocators and storing JSON data efficiently.
│   │   ├── structure.zig                         # Generates structured STRUCTURE.md from Zig AST, merging new comments with existing ones.
│   │   ├── sync.zig                              # Handles Zig AST parsing, manages memory, and supports LLM-driven comment enrichment for guidance generation.
│   │   ├── tests.zig                             # [gof-patterns] Tests JSON store merge, query engine behavior, and local time handling in Guidance.
│   │   ├── triage.zig                            # Generates TRIAGE.md from TODO.md using lifecycle detection, risk assessment, and recommended actions.
│   │   └── types.zig                             # Defines data structures for Zig AST analysis, storing members, patterns, signatures, and guidance metadata in a struc...
│   └── common
│       ├── args.zig                                # Handles argument parsing for Zig project, interpreting flags and defaults.
│       └── llm.zig                                 # Handles LLM response formatting, strips thinking blocks, and cleans preamble patterns.
├── AGENTS.md
├── build.zig
├── build.zig.zon
├── LICENSE
├── LICENSE-Commercial-Requirement
├── LICENSE-Contributor-Agreement
├── Makefile
├── mise.toml
├── pyproject.toml
├── README.md
└── requirements.txt
```
