# Benchmark queries for guidance test

## Each query is followed by `---` and a rubric that defines the expected answer.  The rubric is a concise description of what a correct answer must contain.  cmdBenchmark uses the rubric to judge whether the LLM evaluation is accurate.

---

## Short queries (fast path, deterministic AST match)

cmd_explain

- **Rubric**: Must return the `cmd_explain` async function from `src/bin/guidance/src/main.rs` showing the explain subcommand dispatch (line 341 area) and the call site to `query_engine`.

---

cmd_check

- **Rubric**: Must return the `cmd_check` function from `src/bin/guidance/src/main.rs` (line 1019 area) showing how it walks `cfg.test_commands`, `cfg.lint_commands`, and `cfg.fmt_commands` for the RALPH loop.

---

match_hash_from_signature

- **Rubric**: Must return the `match_hash_from_signature` function from `src/guidance/src/sync/staleness.rs` (line 24) which computes a blake3 hex digest of the signature, and explain its role in detecting member changes for incremental sync.

---

should_generate

- **Rubric**: Must return the `should_generate` function from `src/guidance/src/sync/staleness.rs` (line 17) and the helper `is_stale` showing the mtime comparison rules (JSON absent → stale; source mtime > json mtime by more than 1 second → stale).

---

GuidanceDb

- **Rubric**: Must return the `GuidanceDb` struct from `src/search-vector/src/db.rs` listing its key methods (`hybrid_search`, `sync_from_dir`, `keyword_search`, `vector_search`) and the SQLite `guidance_nodes` table it owns.

---

LlmClient

- **Rubric**: Must return the `LlmClient` struct from `src/llm/src/client.rs` (line 77) with `LlmConfig`, `chat_complete`, and the OpenAI/Ollama HTTP routing that powers every LLM call in the CLI.

---

WordIndex

- **Rubric**: Must return the `WordIndex` struct from `src/project-knowledge/src/word_index.rs` (line 52) showing its inverted-index `get`/`insert` API and its role in the L1 (sub-1ms) cache tier.

---

## Natural language queries (LLM synthesis path)

How does the query FSM classify a query?

- **Rubric**: Must explain the `FsmEngine` in `src/guidance/src/query/strategy.rs` walking through `Intake` → `Classify` → `Route` → `Validate` states, and the `QueryIntent` enum (SingleIdentifier, IdentifierLookup, FilePath, CapabilityQuery, HowTo, Conceptual, MultiKeyword, GeneralSearch) with their priority values.

---

How does this code use cosine similarity?

- **Rubric**: Must explain `cosine_similarity` and `knn_brute_force` from `src/search-vector/src/math.rs` are used by `GuidanceDb::hybrid_search` to rank results when a query embedding is provided, fused via RRF with keyword search.

---

How is the RALPH loop implemented?

- **Rubric**: Must explain `cmd_check` in `src/bin/guidance/src/main.rs` (line 1019) iterates over `cfg.test_commands`, `cfg.lint_commands`, and `cfg.fmt_commands` from `ProjectConfig` and runs each as a stage, stopping on the first failure; defaults read from `guidance-config.json`.

---

What capabilities does this codebase have?

- **Rubric**: Must list the workspace crates as capabilities: `common-core` (hashing, io, string, walk, metrics), `fluent-wvr` (Component/WorkUnit traits), `fluent-concurrency` (Scope, Limiter, WorkerPool), `dag` (executor/resolver/target), `llm` (LlmClient, CachedEmbeddingProvider), `search-vector` (GuidanceDb, math), `project-knowledge` (WordIndex, TrigramIndex, CsrGraph), `ontology` (YAGO taxonomy), `coral` (cache_router, L1Cache, MCP server).

---

I want to understand how match_hash_from_signature and should_generate work together to detect staleness.

- **Rubric**: Must explain `should_generate` in `src/guidance/src/sync/staleness.rs` decides whether to re-parse a file by comparing mtimes (JSON absent → stale, source newer by >1s → stale), while `match_hash_from_signature` (blake3 of the member signature) preserves existing comments/tags when only unrelated files changed; together they form the incremental sync gate in `SyncEngine`.

---

## Queries that should escalate as unknown (negative tests)

How does the quantum entanglement protocol work in the coral module?

- **Rubric**: Expected answer is "not found" — quantum entanglement protocol does not exist in this codebase. The coral crate handles caching and MCP, not physics.

---

Show me the implementation of the flux capacitor pattern for warp drive acceleration.

- **Rubric**: Expected answer is "not found" — flux capacitor pattern does not exist in this codebase. There is no warp drive code.
