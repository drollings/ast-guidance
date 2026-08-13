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
`{guidance_dir}/skills/{skill}/SKILL.md`

So if you find a file you're looking for named file.rs:
`file.rs      # [zig-current, gof-patterns] Summary of files' contents` , 
Then you you must read

```
{guidance_dir}/skills/zig-current/SKILL.md
{guidance_dir}/skills/gof-patterns/SKILL.md
```

---

## Directory Tree (Git-Tracked Files Only)

```
.
├── AGENTS.md  # # Coral Router — Development Guide
├── Cargo.toml
├── LICENSE
├── LICENSE-Commercial-Requirement
├── LICENSE-Contributor-Agreement
├── Makefile
├── README.md  # # Fluent Monorepo - a high-speed agentic
├── STRUCTURE.md  # # AST-Guidance Project Structure
├── bin/
│   ├── coral-router-test.py  # #!/usr/bin/env python3
│   ├── gen_simhash_projections.py  # #!/usr/bin/env python3
│   ├── router-mock-tests.sh
│   └── router-wait-health.sh
├── data/
│   └── yamake.json
├── doc/
│   ├── AMBIGUOUS_DAG.md  # # Unified Dependency Resolver — Fro...
│   ├── MEMORY_PLUGIN.md  # # Memory Plugin Architecture — Clea...
│   ├── SUBAGENT.md  # # REVIEW_20260418_LOCAL_SUBAGENT.
│   ├── capabilities/
│   │   ├── ast-indexing/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── config-system/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── coral-cache/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── coral-database/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── coral-ingestion/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── coral-mcp/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── embedding-providers/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── explain-query/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── fluent-concurrency/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── llm-client/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── local-model-decomposition/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── ontology/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── plugin-system/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── rdf-parsing/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── reflection/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── sync-pipeline/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── target-registry/
│   │   │   └── CAPABILITY.md  # ---
│   │   ├── vector-search/
│   │   │   └── CAPABILITY.md  # ---
│   │   └── wasm-tools/
│   │       └── CAPABILITY.md  # ---
│   ├── coral/
│   │   ├── CHANGELOG.md  # # Changelog
│   │   ├── DETAILS.md  # # Coral Context: Detailed Engineering Sp
│   │   ├── OVERVIEW.md  # # Coral Context: Architectural Design Do
│   │   └── VISION.md  # # Coral Context: Architectural Vision
│   ├── guidance/
│   │   ├── ARCHITECTURE.md  # # Architecture Overview
│   │   ├── DESIGN.md  # Comprehensive Analysis: Agentic Document
│   │   ├── MCP.md  # # guidance MCP Server
│   │   ├── VISION.md  # # guidance: Vision Document
│   │   └── schemas/
│   │       └── guidance.schema.json
│   ├── router/
│   │   ├── ARCHITECTURE.md  # # Coral Router — Architecture
│   │   ├── TESTING.md  # # Coral Router — Testing Guide
│   │   └── VISION.md  # # Coral Router — Vision
│   └── skills/
│       ├── common-core/
│       │   └── SKILL.md  # # common-core — Zero-Domain Utility...
│       ├── dag/
│       │   └── SKILL.md  # # fluent-dag — Dependency Graph & D...
│       ├── fluent-concurrency/
│       │   └── SKILL.md  # # `fluent-concurrency` — Lightweigh...
│       ├── fluent-db/
│       │   └── SKILL.md  # # fluent-db — Canonical Database-Ac...
│       ├── fluent-wvr/
│       │   └── SKILL.md  # # Fluent WVR in Rust — The Synthesi...
│       └── subagent/
│           └── SKILL.md  # ---
├── env/
│   ├── categories.json
│   ├── coral-router.json.example
│   ├── mk/
│   │   ├── common.mk
│   │   ├── target_language.mk
│   │   └── targets/
│   │       ├── go.mk
│   │       ├── php.mk
│   │       ├── pine.mk
│   │       ├── py.mk
│   │       ├── rust.mk
│   │       └── zig.mk
│   ├── mock-transcripts.json
│   ├── pii-patterns.json
│   └── workflows/
│       └── charts/
│           ├── bug_triage.md.json
│           └── draft_doc.md.json
└── src/
    ├── Cargo.lock
    ├── bin/
    │   ├── coral/
    │   │   ├── Cargo.toml
    │   │   └── src/
    │   │       └── main.rs  # use clap::{Parser, Subcommand};
    │   ├── coral-router/
    │   │   ├── Cargo.toml
    │   │   └── src/
    │   │       └── main.rs  # //! coral-router — LLM Router & Age...
    │   ├── guidance/
    │   │   ├── Cargo.toml
    │   │   └── src/
    │   │       ├── benchmark.rs  # //! `guidance benchmark` — query ac...
    │   │       ├── commit.rs  # //! Commit message generation — LLM...
    │   │       ├── editor.rs  # //! Editor interaction utilities for hum
    │   │       ├── main.rs  # use std::path::{Path, PathBuf};
    │   │       ├── mcp.rs  # //! MCP (Model Context Protocol) server 
    │   │       └── structure.rs  # use std::collections::BTreeMap;
    │   └── yamake-coral/
    │       ├── Cargo.toml
    │       └── src/
    │           └── main.rs  # use std::process;
    ├── common-core/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── cache.rs  # //! A read-through cache abstraction: ch
    │       ├── config.rs  # //! JSON config loaders: `load_json_or_d
    │       ├── constants.rs  # //! Cross-crate magic numbers (size caps
    │       ├── drift.rs  # //! Bit-set drift analysis: compute "mis
    │       ├── error.rs  # //! Shared leaf error types: `IoError`, 
    │       ├── error_context.rs  # //! Contextual error wrappers: `ErrorCon
    │       ├── format.rs  # //! Human-readable output: `format_json`
    │       ├── git.rs  # //! Git operations — thin wrappers ...
    │       ├── hash.rs  # //! Hashing utilities: `blake3_*`, `sha2
    │       ├── http.rs  # //! Process-wide shared HTTP client (and
    │       ├── interner.rs  # //! Capability registry: thread-safe str
    │       ├── io.rs  # use std::fs;
    │       ├── jsonrpc.rs  # //! Shared JSON-RPC 2.
    │       ├── lib.rs  # #![forbid(unsafe_code)]
    │       ├── metrics.rs  # //! Lock-free latency histogram with 12 
    │       ├── prelude.rs  # //! The common-core prelude — impor...
    │       ├── registry.rs  # //! Generic keyed registry — the ca...
    │       ├── retry.rs  # //! Retry and backoff primitives — ...
    │       ├── runtime.rs  # //! Sync→async runtime bridge: run ...
    │       ├── shell.rs  # //! Subprocess helpers: `run_capture`, `
    │       ├── shell_parser.rs  # //! Safe shell parser: whitespace+quote 
    │       ├── sqlite.rs  # //! Shared SQLite helpers — connect...
    │       ├── string.rs  # //! 20+ string utilities: case-insensiti
    │       ├── sync.rs  # //! Poison-safe locking helpers for `std
    │       ├── time.rs  # //! Time utilities: epoch-second helpers
    │       ├── tokens.rs  # //! Token budget helpers: `estimate_toke
    │       ├── walk.rs  # //! Directory walker: `walk_files` (call
    │       └── watchdog.rs  # use std::collections::VecDeque;
    ├── content-node/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── doc_node.rs  # use crate::file_node::FileContentNode;
    │       ├── file_node.rs  # use std::fmt::Debug;
    │       ├── lib.rs  # //! content-node: Level-of-detail text s
    │       ├── lod.rs  # pub fn generate_lod_slices(full_text: &s
    │       ├── node.rs  # use fluent_types::LOD_COUNT;
    │       └── source_node.rs  # use crate::file_node::FileContentNode;
    ├── coral/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── cache/
    │       │   ├── mod.rs  # pub mod reactor;
    │       │   ├── reactor.rs  # use std::sync::Arc;
    │       │   └── stats.rs  # #[derive(Debug, Clone, serde::Serialize,
    │       ├── cache_l1.rs  # use serde::{Deserialize, Serialize};
    │       ├── cache_router.rs  # use std::sync::Arc;
    │       ├── db/
    │       │   ├── edges.rs  # use fluent_types::{GraphNode, NodeId};
    │       │   ├── embeddings.rs  # use fluent_db::vector::{try_bytes_to_vec
    │       │   ├── hnsw.rs  # use std::collections::HashMap;
    │       │   ├── mod.rs  # pub mod edges;
    │       │   ├── nodes.rs  # use std::collections::HashMap;
    │       │   └── schema.rs  # use fluent_db::error::DbError;
    │       ├── error.rs  # use thiserror::Error;
    │       ├── ingest.rs  # use std::sync::Arc;
    │       ├── knowledge.rs  # //! `KnowledgeCapability` implementation
    │       ├── lib.rs  # //! Coral: Context-graph library for gui
    │       ├── mcp.rs  # use std::path::Path;
    │       ├── packer.rs  # use fluent_types::{ContentNode, NodeId};
    │       ├── test_stubs.rs  # //! Test stubs for coral cache reactor t
    │       ├── tier_units.rs  # use std::sync::Arc;
    │       ├── wasm_runtime.rs  # use std::path::Path;
    │       └── wvr.rs  # //! Fluent WVR integration for Coral cra
    ├── dag/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── adapter.rs  # //! Re-export of `ComponentAdapter` and 
    │       ├── closure.rs  # use std::collections::HashSet;
    │       ├── dep_graph.rs  # //! Pure dependency-graph algorithms sha
    │       ├── error.rs  # use thiserror::Error;
    │       ├── lib.rs  # //! fluent-dag: DAG executor with resolv
    │       ├── middleware.rs  # use std::sync::Arc;
    │       ├── narrowing.rs  # use std::collections::HashSet;
    │       ├── resolver.rs  # //! Capability-aware dependency resolver
    │       ├── target.rs  # use bitvec::vec::BitVec;
    │       ├── target_work_unit.rs  # //! `Target → WorkUnit` bridge.
    │       ├── type_inference.rs  # //! Ontology type-hierarchy inference vi
    │       ├── work_unit.rs  # use bon::Builder;
    │       ├── wvr.rs  # //! Fluent WVR integration for DAG crate
    │       └── yamake_loader.rs  # use bitvec::vec::BitVec;
    ├── db/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── cache.rs  # //! Generic TTL/LRU key-value cache stor
    │       ├── capability.rs  # //! The capability-gated async database 
    │       ├── error.rs  # //! The single database error taxonomy f
    │       ├── hnsw.rs  # //! The canonical HNSW-backed vector ind
    │       ├── lib.rs  # //! # fluent-db — the canonical dat...
    │       ├── migrate.rs  # //! Idempotent schema migrations (M3.2).
    │       ├── pool.rs  # //! The canonical pooled SQLite store (D
    │       ├── query.rs  # //! Typed statement helpers shared by `S
    │       ├── store.rs  # //! The canonical single-connection SQLi
    │       ├── vector.rs  # //! Embedding vector math (D8).
    │       └── wvr.rs  # //! Database `Component`/`WorkUnit` adap
    ├── fluent-concurrency/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── affinity.rs  # //! Affinity-aware priority scheduler.
    │       ├── capability.rs  # //! Concrete capability tokens for files
    │       ├── flow.rs  # //! Credit-based backpressure flow contr
    │       ├── io/
    │       │   ├── db.rs  # //! SQLite-backed database capability (p
    │       │   ├── fs.rs  # //! Capability-gated filesystem I/O (rea
    │       │   ├── mod.rs  # //! Capability-gated I/O primitive engin
    │       │   └── net.rs  # //! Capability-gated network I/O (TCP co
    │       ├── lib.rs  # #![forbid(unsafe_code)]
    │       ├── llm_queue.rs  # //! LLM request queue — async, queu...
    │       ├── pool.rs  # //! Bounded async queue, worker pool, an
    │       ├── queue.rs  # //! A priority queue with a fast path fo
    │       ├── reserve.rs  # //! Available primitive: RAII permit on 
    │       ├── router.rs  # //! A partitioned router that distribute
    │       ├── runtime/
    │       │   ├── mod.rs  # //! Pluggable `Runtime` backends (produc
    │       │   ├── test.rs  # //! Test `Runtime` implementation with p
    │       │   └── tokio.rs  # //! Production `Runtime` implementation 
    │       ├── scope.rs  # //! Structured concurrency via `Scope...
    │       ├── tests/
    │       │   ├── e2e.rs  # use crate::pool::WorkerPool;
    │       │   ├── m1.rs  # use super::*;
    │       │   ├── m2.rs  # use super::*;
    │       │   ├── m3.rs  # use super::*;
    │       │   ├── m4.rs  # use super::*;
    │       │   ├── m5.rs  # // Exercises the capability-gated I/O en
    │       │   └── mod.rs  # use std::sync::atomic::{AtomicUsize, Ord
    │       ├── thread_resource.rs  # //! Per-thread lazy-initialized resource
    │       └── zone.rs  # //! Supervision zone with async retry, d
    ├── fluent-wvr/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── capability.rs  # use std::any::{Any, TypeId};
    │       ├── lib.rs  # #![forbid(unsafe_code)]
    │       ├── macros.rs  # /// Eliminates the 7-line `as_any`/`as_a
    │       ├── metadata.rs  # use serde::{Deserialize, Serialize};
    │       ├── prelude.rs  # //! The fluent-wvr prelude — import...
    │       ├── runtime.rs  # use std::future::Future;
    │       ├── store.rs  # //! Typed in-process handoff accumulator
    │       ├── tests.rs  # use crate::*;
    │       ├── traits.rs  # use std::any::Any;
    │       ├── work.rs  # use std::collections::HashMap;
    │       └── wrapper.rs  # use std::collections::HashMap;
    ├── fluent-wvr-macros/
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs  # #![forbid(unsafe_code)]
    ├── fluent-wvr-testutil/
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs  # //! Test utilities for Fluent WVR crates
    ├── guidance/
    │   ├── Cargo.toml
    │   ├── src/
    │   │   ├── ast_parser.rs  # use std::path::Path;
    │   │   ├── config.rs  # use std::collections::HashMap;
    │   │   ├── enhancer.rs  # use fluent_llm::client::LlmClient;
    │   │   ├── grounding.rs  # //! Grounding enforcement — ensures...
    │   │   ├── lib.rs  # //! Guidance: AST-guided vector search &
    │   │   ├── memory.rs  # //! Memory integration for the guidance 
    │   │   ├── plugin.rs  # use std::path::{Path, PathBuf};
    │   │   ├── query/
    │   │   │   ├── formatter.rs  # use std::fmt::Write;
    │   │   │   ├── identifier.rs  # use common_core::string::{contains_ignor
    │   │   │   ├── llm_filter.rs  # use common_core::string::contains_ignore
    │   │   │   ├── llm_filter_batch.rs  # use super::llm_filter::{LlmFilterBackend
    │   │   │   ├── mod.rs  # pub mod formatter;
    │   │   │   ├── search_backend.rs  # use common_core::string::contains_ignore
    │   │   │   ├── snapshot.rs  # use std::path::Path;
    │   │   │   ├── strategy.rs  # use fluent_types::GuidanceDoc;
    │   │   │   └── synthesize.rs  # use fluent_types::{GuidanceDoc, Member, 
    │   │   ├── query_engine.rs  # use std::path::Path;
    │   │   ├── runtime.rs  # use std::path::{Path, PathBuf};
    │   │   ├── scanner.rs  # use common_core::string::{contains_any, 
    │   │   ├── sync/
    │   │   │   ├── comments.rs  # use std::path::Path;
    │   │   │   ├── json_store.rs  # use std::path::{Path, PathBuf};
    │   │   │   ├── json_writer.rs  # use fluent_types::{GuidanceDoc, Member};
    │   │   │   ├── mod.rs  # pub mod comments;
    │   │   │   └── staleness.rs  # use std::path::Path;
    │   │   └── sync_engine.rs  # use std::path::{Path, PathBuf};
    │   └── tests/
    │       └── e2e_gen_roundtrip.rs  # use fluent_types::MemberType;
    ├── knowledge/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── csr_graph.rs  # pub const CSR_MAGIC: u32 = 0x4752_5343;
    │       ├── freq_table.rs  # use std::fs;
    │       ├── index_header.rs  # pub const INDEX_HEADER_SIZE: usize = 10;
    │       ├── lib.rs  # //! fluent-knowledge: Word/trigram index
    │       ├── query_cache.rs  # //! TTL/LRU query cache delegating to `f
    │       ├── tokenizer.rs  # pub struct WordTokenizer<'a> {
    │       ├── trigram_index.rs  # use crate::index_header::Header;
    │       └── word_index.rs  # use std::collections::HashMap;
    ├── llm/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── anonymize.rs  # pub fn anonymize(text: &str) -> String {
    │       ├── client.rs  # use std::sync::Arc;
    │       ├── constants.rs  # //! Cross-crate limit moved to `common-c
    │       ├── context_packer.rs  # use crate::ChatMessage;
    │       ├── decomposer.rs  # use bon::Builder;
    │       ├── embeddings.rs  # use std::num::NonZeroUsize;
    │       ├── error.rs  # use crate::embeddings::EmbeddingError;
    │       ├── http_class.rs  # use serde::{Deserialize, Serialize};
    │       ├── lib.rs  # //! fluent-llm: LLM HTTP client provider
    │       ├── llm_queue.rs  # //! Default LLM request handler — w...
    │       ├── openai.rs  # //! OpenAI-compatible chat-completion wi
    │       ├── parse.rs  # //! Tolerant JSON parsing for LLM output
    │       ├── pii_patterns.rs  # use std::sync::LazyLock;
    │       └── url.rs  # use thiserror::Error;
    ├── memory-plugin/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── capability.rs  # //! Capability token for explicit memory
    │       ├── lib.rs  # #![forbid(unsafe_code)]
    │       ├── plugins/
    │       │   ├── hindsight/
    │       │   │   └── mod.rs  # //! Hindsight memory plugin — struc...
    │       │   ├── holographic/
    │       │   │   ├── hrr.rs  # //! Holographic Reduced Representations 
    │       │   │   ├── mod.rs  # //! Holographic memory plugin — loc...
    │       │   │   └── store.rs  # //! SQLite-backed fact store with entity
    │       │   ├── honcho/
    │       │   │   └── mod.rs  # //! Honcho memory plugin — cross-se...
    │       │   └── mod.rs  # //! Memory plugin implementations.
    │       ├── registry.rs  # //! Central memory plugin registry.
    │       ├── traits.rs  # //! Core trait definitions for the memor
    │       ├── types.rs  # //! Shared types for the memory plugin s
    │       └── zone.rs  # //! Memory ingestion zone.
    ├── ontology/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── entity.rs  # use std::collections::HashMap;
    │       ├── inference.rs  # use std::collections::{HashMap, HashSet}
    │       ├── lib.rs  # //! guidance-ontology: Entity extraction
    │       ├── mapper.rs  # use std::collections::HashMap;
    │       ├── migration.rs  # #[derive(Debug, Clone)]
    │       └── yago.rs  # pub const NS_YAGO: &str = "http://yago-k
    ├── rdf/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── lexer.rs  # use crate::RdfError;
    │       ├── lib.rs  # //! guidance-rdf: RDF/Turtle/N-Quads par
    │       ├── normalize.rs  # pub struct BlankNodeScope;
    │       ├── nquads.rs  # use crate::lexer::{Lexer, TokenKind};
    │       └── parser.rs  # use std::collections::{HashMap, VecDeque
    ├── requirements.txt
    ├── router/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── audit.rs  # //! Canonical durable-audit surface for 
    │       ├── charts/
    │       │   ├── binding.rs  # //! Entity binding layer — the dete...
    │       │   ├── compile.rs  # //! Chart compiler — turns a valida...
    │       │   ├── execute.rs  # //! Zone-supervised execution of a compi
    │       │   ├── extract.rs  # //! Chart auto-extraction from dispatch 
    │       │   ├── mod.rs  # //! Chart content model — a library...
    │       │   ├── render.rs  # //! Chart template rendering — mini...
    │       │   ├── rubric.rs  # //! Rubric acceptance gate for chart tar
    │       │   ├── select.rs  # //! Chart selection — deterministic...
    │       │   ├── stage.rs  # //! ChartPromptStage — a `Classifie...
    │       │   └── store.rs  # //! ChartStore — loads and holds a ...
    │       ├── cli/
    │       │   ├── commands.rs  # //! Implementation of the `coral-router`
    │       │   ├── gguf.rs  # //! GGUF directory scanning, caching, mo
    │       │   ├── mod.rs  # //! Admin CLI support for Coral Router.
    │       │   └── preset.rs  # //! Rendering of downstream serving conf
    │       ├── config/
    │       │   ├── addr.rs  # //! Address parsing, host equivalence, a
    │       │   ├── builder.rs  # //! Pipeline builder - constructs pipeli
    │       │   ├── classification.rs  # //! Classification-tree configuration
    │       │   ├── escalation.rs  # //! Escalation-ladder configuration
    │       │   ├── filters.rs  # //! Filter types and reject patterns for
    │       │   └── routing.rs  # //! Route resolution and routing configu
    │       ├── config.rs  # //! Router configuration types - deseria
    │       ├── dag_session.rs  # //! Dependency-aware session with DAG st
    │       ├── dispatch/
    │       │   ├── backend.rs  # use std::future::Future;
    │       │   ├── escalation.rs  # //! Escalation-ladder dispatch loop
    │       │   ├── frontier.rs  # use serde_json::Value;
    │       │   └── mod.rs  # pub mod backend;
    │       ├── error.rs  # //! Server-level error type — the s...
    │       ├── filters/
    │       │   ├── injection_detect.rs  # use std::collections::HashSet;
    │       │   ├── luhn.rs  # pub fn luhn_valid(input: &str) -> bool {
    │       │   ├── mod.rs  # pub mod injection_detect;
    │       │   └── regex_filter.rs  # use std::collections::HashMap;
    │       ├── frontier/
    │       │   ├── mod.rs  # pub mod modes;
    │       │   └── modes.rs  # //! Frontier escalation ladder — VI...
    │       ├── hnsw.rs  # /// A single HNSW index handle — th...
    │       ├── instances.rs  # //! Instance-pool grammar generation, ma
    │       ├── knowledge.rs  # //! `KnowledgeCapability` implementation
    │       ├── kv_cache.rs  # //! KV cache snapshot management - two-t
    │       ├── ledger.rs  # //! Full-detail content ledger with LOD 
    │       ├── ledger_guard.rs  # //! Irreversible write-path scrubber for
    │       ├── lib.rs  # //! LLM Router & Agent Orchestration Fra
    │       ├── logging.rs  # //! Structured logging infrastructure fo
    │       ├── metrics.rs  # //! Failure classification for the route
    │       ├── node_store.rs  # //! ContentNodeStore — the shared, referen...
    │       ├── normalize.rs  # //! Request and response normalizatio...
    │       ├── pipeline.rs  # //! Pipeline orchestrator — sequenc...
    │       ├── pipeline_types.rs  # //! Pipeline decision types — struc...
    │       ├── routes/
    │       │   ├── mod.rs  # pub mod plan;
    │       │   ├── plan.rs  # use std::sync::Arc;
    │       │   └── rigor.rs  # //! Rigor route - the fixed-pass blue/re
    │       ├── scheduler.rs  # //! Affinity-aware priority scheduler.
    │       ├── score_matrix.rs  # use std::collections::HashMap;
    │       ├── server/
    │       │   ├── admin.rs  # //! Admin endpoints for the CLI (`coral-
    │       │   ├── dispatch.rs  # use std::collections::HashMap;
    │       │   ├── handler.rs  # use std::collections::HashMap;
    │       │   ├── instances_api.rs  # //! Public `/instances` management API f
    │       │   └── responses.rs  # use std::sync::atomic::AtomicU64;
    │       ├── server.rs  # //! HTTP server exposing the router pipe
    │       ├── server_http_tests.rs  # //! HTTP-level integration tests for the
    │       ├── server_tests.rs  # #[cfg(test)]
    │       ├── session.rs  # //! Session context node schema — b...
    │       ├── stage_tests.rs  # #[cfg(test)]
    │       ├── stages/
    │       │   ├── classifier.rs  # //! Stage 2: ClassifierStage — sing...
    │       │   ├── common.rs  # //! Shared helpers for pipeline stages.
    │       │   ├── deterministic.rs  # use std::collections::HashMap;
    │       │   ├── mod.rs  # pub mod classifier;
    │       │   ├── pipeline_ref.rs  # //! PipelineRefStage — a `WorkUnit`...
    │       │   ├── retry_classifier.rs  # //! RetryClassifier — a `WorkUnit` ...
    │       │   └── tree.rs  # //! Classification-tree engine
    │       ├── streaming.rs  # //! SSE streaming handler — transla...
    │       ├── summarization.rs  # //! Summarization and result acceptance.
    │       ├── supervisor.rs  # //! Managed `llama-server` process super
    │       ├── target_match.rs  # //! Classifier-driven target matching...
    │       ├── telemetry.rs  # //! Structured telemetry events with con
    │       ├── test_stubs.rs  # use std::collections::VecDeque;
    │       ├── test_support.rs  # //! Shared test logging capture.
    │       ├── testing/
    │       │   ├── mock.rs  # use std::collections::HashMap;
    │       │   └── mod.rs  # pub mod mock;
    │       ├── tests/
    │       │   ├── e2e_tests.rs  # //! End-to-end tests for the router pipe
    │       │   ├── golden.rs  # //! Golden test set for the router pipel
    │       │   ├── mod.rs  # //! Router test modules.
    │       │   └── rubric_fixtures.rs  # //! Rubric-based test fixtures for `Resu
    │       ├── transforms/
    │       │   ├── codeword_anonymize.rs  # use std::collections::HashMap;
    │       │   ├── decompose_hypothetical.rs  # use fluent_llm::anonymize;
    │       │   ├── decompose_subtasks.rs  # use fluent_llm::Decomposer;
    │       │   ├── mod.rs  # pub mod codeword_anonymize;
    │       │   ├── none.rs  # use crate::transforms::{TransformError, 
    │       │   ├── pii_anonymize.rs  # use std::collections::HashMap;
    │       │   ├── sanitize.rs  # use common_core::string::{filter_unsafe_
    │       │   ├── secret_mask.rs  # use std::sync::LazyLock;
    │       │   └── tests.rs  # #[cfg(test)]
    │       ├── types.rs  # //! Unified request/response types ...
    │       └── views.rs  # //! Reference-only view layer over the s
    ├── search-vector/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── aliases.rs  # use std::collections::HashMap;
    │       ├── db.rs  # use std::path::Path;
    │       ├── error.rs  # //! Database error taxonomy — re-ex...
    │       ├── lib.rs  # //! search-vector: SQLite hybrid search 
    │       └── math.rs  # //! Embedding vector math — re-expo...
    ├── types/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── knowledge.rs  # //! KnowledgeCapability — the cross...
    │       └── lib.rs  # //! fluent-types: Shared data types (Gui
    └── wasm_ipc/
        ├── Cargo.toml
        └── src/
            └── lib.rs  # //! WASM IPC — Binary schemas for E...
```
