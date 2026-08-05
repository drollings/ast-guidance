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
├── AGENTS.md  # # Agent Bootloader — guidance
├── Cargo.toml
├── LICENSE
├── LICENSE-Commercial-Requirement
├── LICENSE-Contributor-Agreement
├── Makefile
├── README.md  # # The fluent monorepo
├── STRUCTURE.md  # # AST-Guidance Project Structure
├── bin/
│   └── gen_simhash_projections.py  # #!/usr/bin/env python3
├── doc/
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
│   │   ├── DESIGN.md  # Comprehensive Analysis: Agentic Document
│   │   ├── MCP.md  # # guidance MCP Server
│   │   ├── VISION.md  # # guidance: Vision Document
│   │   └── schemas/
│   │       └── guidance.schema.json
│   └── skills/
│       ├── fluent-concurrency/
│       │   └── SKILL.md  # # `fluent-concurrency` — Lightweigh...
│       ├── fluent-wvr/
│       │   └── SKILL.md  # # Fluent WVR in Rust — The Synthesi...
│       ├── gof-patterns/
│       │   └── SKILL.md  # ---
│       ├── subagent/
│       │   └── SKILL.md  # ---
│       └── zig-to-rust/
│           └── SKILL.md  # # Zig to Rust Practices: Master Guidelin
├── env/
│   └── mk/
│       ├── common.mk
│       ├── target_language.mk
│       └── targets/
│           ├── go.mk
│           ├── php.mk
│           ├── pine.mk
│           ├── py.mk
│           ├── rust.mk
│           └── zig.mk
├── extension/
│   ├── README.md  # # Job Copilot — Chromium Extension
│   ├── background.js  # // Job Copilot — background service...
│   ├── content-script.js  # // Job Copilot — content script (DO...
│   ├── manifest.json
│   ├── side-panel.css
│   ├── side-panel.html
│   └── side-panel.js  # // Job Copilot — side panel
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
    │   │       └── main.rs  # use std::sync::Arc;
    │   ├── guidance/
    │   │   ├── Cargo.toml
    │   │   └── src/
    │   │       ├── benchmark.rs  # //! `guidance benchmark` — query ac...
    │   │       ├── commit.rs  # //! Commit message generation — LLM...
    │   │       ├── editor.rs  # //! Editor interaction utilities for hum
    │   │       ├── main.rs  # use std::path::{Path, PathBuf};
    │   │       ├── mcp.rs  # //! MCP (Model Context Protocol) server 
    │   │       └── structure.rs  # use std::collections::BTreeMap;
    │   └── job-copilot-daemon/
    │       ├── Cargo.toml
    │       └── src/
    │           └── main.rs  # use std::io::{BufRead, BufReader};
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
    │       ├── shell.rs  # //! Subprocess helpers: `run_capture`, `
    │       ├── shell_parser.rs  # //! Safe shell parser: whitespace+quote 
    │       ├── sqlite.rs  # //! Shared SQLite helpers — connect...
    │       ├── string.rs  # //! 20+ string utilities: case-insensiti
    │       ├── tokens.rs  # //! Token budget helpers: `estimate_toke
    │       └── walk.rs  # //! Directory walker: `walk_files` (call
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
    │       ├── cache_l1.rs  # use lru::LruCache;
    │       ├── cache_reactor.rs  # use std::sync::Arc;
    │       ├── cache_router.rs  # use std::sync::Arc;
    │       ├── db.rs  # use std::collections::HashMap;
    │       ├── error.rs  # use thiserror::Error;
    │       ├── ingest.rs  # use std::sync::Arc;
    │       ├── lib.rs  # //! Coral: Context-graph library for gui
    │       ├── mcp.rs  # use std::path::Path;
    │       ├── packer.rs  # use common_core::tokens::{estimate_token
    │       ├── test_stubs.rs  # //! Test stubs for coral cache reactor t
    │       ├── tier_units.rs  # use std::sync::{Arc, Weak};
    │       ├── wasm_runtime.rs  # use std::num::NonZeroUsize;
    │       └── wvr.rs  # //! Fluent WVR integration for Coral cra
    ├── dag/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── adapter.rs  # //! Re-export of `ComponentAdapter` and 
    │       ├── dep_graph.rs  # //! Pure dependency-graph algorithms sha
    │       ├── error.rs  # use thiserror::Error;
    │       ├── executor.rs  # use std::collections::HashMap;
    │       ├── lib.rs  # //! fluent-dag: DAG executor with resolv
    │       ├── middleware.rs  # use std::sync::Arc;
    │       ├── resolver.rs  # use std::collections::HashMap;
    │       ├── target.rs  # use bitvec::vec::BitVec;
    │       ├── type_inference.rs  # use bitvec::prelude::*;
    │       ├── work_unit.rs  # use bon::Builder;
    │       └── wvr.rs  # //! Fluent WVR integration for DAG crate
    ├── fluent-concurrency/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── capability.rs  # //! Concrete capability tokens for files
    │       ├── flow.rs  # //! Credit-based backpressure flow contr
    │       ├── io/
    │       │   ├── db.rs  # //! SQLite-backed database capability wi
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
    │       │   ├── m5.rs  # use crate::io::db::DbCapability;
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
    │   │   ├── plugin.rs  # use std::collections::HashMap;
    │   │   ├── query/
    │   │   │   ├── formatter.rs  # use std::fmt::Write;
    │   │   │   ├── identifier.rs  # use common_core::string::contains_ignore
    │   │   │   ├── llm_filter.rs  # use common_core::string::contains_ignore
    │   │   │   ├── llm_filter_batch.rs  # use super::llm_filter::{LlmFilterBackend
    │   │   │   ├── mod.rs  # pub mod formatter;
    │   │   │   ├── search_backend.rs  # use common_core::string::contains_ignore
    │   │   │   ├── snapshot.rs  # use std::path::Path;
    │   │   │   ├── strategy.rs  # use fluent_types::GuidanceDoc;
    │   │   │   └── synthesize.rs  # use fluent_types::{GuidanceDoc, Member, 
    │   │   ├── query_engine.rs  # use std::path::Path;
    │   │   ├── runtime.rs  # use std::path::PathBuf;
    │   │   ├── scanner.rs  # use common_core::string::{contains_any, 
    │   │   ├── sync/
    │   │   │   ├── comments.rs  # use std::path::Path;
    │   │   │   ├── json_store.rs  # use std::path::{Path, PathBuf};
    │   │   │   ├── json_writer.rs  # use fluent_types::{GuidanceDoc, Member};
    │   │   │   ├── mod.rs  # pub mod comments;
    │   │   │   └── staleness.rs  # use std::path::Path;
    │   │   └── sync_engine.rs  # use std::path::{Path, PathBuf};
    │   └── tests/
    │       └── e2e_gen_roundtrip.rs  # use fluent_wvr_testutil::tempdir;
    ├── job-copilot/
    │   ├── Cargo.toml
    │   ├── proptest-regressions/
    │   │   └── server/
    │   │       └── handler.txt
    │   └── src/
    │       ├── components.rs  # use std::sync::Arc;
    │       ├── config.rs  # use std::path::PathBuf;
    │       ├── dispatcher/
    │       │   ├── llm.rs  # use std::sync::{Arc, RwLock};
    │       │   ├── local.rs  # use std::sync::{Arc, OnceLock, RwLock};
    │       │   └── mod.rs  # pub mod llm;
    │       ├── error.rs  # use common_core::error_context::ErrorCon
    │       ├── lib.rs  # //! Job Copilot — local-only human-...
    │       ├── memory.rs  # use std::path::PathBuf;
    │       ├── profile.rs  # use std::path::Path;
    │       ├── prompt/
    │       │   ├── context.rs  # use common_core::tokens::estimate_tokens
    │       │   └── mod.rs  # pub mod context;
    │       ├── sanitize.rs  # use std::sync::OnceLock;
    │       ├── schema.rs  # use serde::{Deserialize, Serialize};
    │       ├── server/
    │       │   ├── audit.rs  # use std::fs::{File, OpenOptions};
    │       │   ├── auth.rs  # use std::collections::HashMap;
    │       │   ├── handler.rs  # use std::sync::Arc;
    │       │   ├── http.rs  # use std::collections::HashMap;
    │       │   ├── mod.rs  # pub mod audit;
    │       │   └── stdio.rs  # use std::io::{self, Read, Write};
    │       └── similarity.rs  # use std::path::{Path, PathBuf};
    ├── llm/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── anonymize.rs  # use std::sync::LazyLock;
    │       ├── client.rs  # use std::sync::{Arc, OnceLock};
    │       ├── constants.rs  # //! Cross-crate limit moved to `common-c
    │       ├── context_packer.rs  # use crate::ChatMessage;
    │       ├── decomposer.rs  # use bon::Builder;
    │       ├── embeddings.rs  # use std::num::NonZeroUsize;
    │       ├── error.rs  # use crate::embeddings::EmbeddingError;
    │       ├── lib.rs  # //! guidance-llm: LLM HTTP client provid
    │       ├── llm_queue.rs  # //! Default LLM request handler — w...
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
    ├── knowledge/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── csr_graph.rs  # pub const CSR_MAGIC: u32 = 0x4752_5343;
    │       ├── freq_table.rs  # use std::fs;
    │       ├── index_header.rs  # pub const INDEX_HEADER_SIZE: usize = 10;
    │       ├── lib.rs  # //! fluent-knowledge: Word/tri
    │       ├── query_cache.rs  # use common_core::hash::fnv1a64;
    │       ├── tokenizer.rs  # pub struct WordTokenizer<'a> {
    │       ├── trigram_index.rs  # use crate::index_header::Header;
    │       └── word_index.rs  # use std::collections::HashMap;
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
    │       ├── agent.rs  # //! Agent registry — keyed on `(mod...
    │       ├── compaction.rs  # //! LOD compaction policy — shrink ...
    │       ├── config.rs  # //! Router configuration types — de...
    │       ├── dag_session.rs  # //! Dependency-aware session with DAG st
    │       ├── dispatch/
    │       │   ├── agent.rs  # use std::sync::Arc;
    │       │   ├── frontier.rs  # use std::collections::HashMap;
    │       │   └── mod.rs  # pub mod agent;
    │       ├── indexer.rs  # //! Adapter/model/frontier indexer ...
    │       ├── kv_cache.rs  # //! KV cache snapshot management — ...
    │       ├── lib.rs  # //! LLM Router & Agent Orchestration Fra
    │       ├── logging.rs  # //! Structured logging infrastructure fo
    │       ├── metrics.rs  # //! Metrics and monitoring for the route
    │       ├── normalize.rs  # //! Request and response normalizatio...
    │       ├── orchestrator.rs  # //! Long-lived orchestrator session ...
    │       ├── pipeline.rs  # //! Pipeline orchestrator — sequenc...
    │       ├── pipeline_types.rs  # //! Pipeline decision types — struc...
    │       ├── scheduler.rs  # //! Affinity-aware priority scheduler.
    │       ├── server.rs  # //! HTTP server exposing the router pipe
    │       ├── server_tests.rs  # #[cfg(test)]
    │       ├── session.rs  # //! Session context node schema — e...
    │       ├── stage_tests.rs  # #[cfg(test)]
    │       ├── stages/
    │       │   ├── common.rs  # //! Shared helpers for pipeline stages.
    │       │   ├── deterministic.rs  # //! Stage 1: DeterministicPreFilter ...
    │       │   ├── guardrail.rs  # //! Stage 4: GuardrailCheck — polic...
    │       │   ├── mod.rs  # pub mod common;
    │       │   ├── planning.rs  # //! Stage 3: PlanningRefinementAgent ...
    │       │   ├── quality_gate.rs  # //! Stage 2: QualityGate — classifi...
    │       │   └── router.rs  # //! Stage 5: RouterStage — selects ...
    │       ├── streaming.rs  # //! SSE streaming handler — transla...
    │       ├── summarization.rs  # //! Summarization and result acceptance.
    │       ├── test_stubs.rs  # use std::collections::VecDeque;
    │       ├── testing/
    │       │   ├── mock.rs  # use std::collections::HashMap;
    │       │   └── mod.rs  # //! Testing utilities for the router pip
    │       ├── tests/
    │       │   ├── e2e_tests.rs  # //! End-to-end tests for the router pipe
    │       │   ├── golden.rs  # //! Golden test set for the router pipel
    │       │   ├── mod.rs  # //! Router test modules.
    │       │   └── rubric_fixtures.rs  # //! Rubric-based test fixtures for `Resu
    │       ├── transforms/
    │       │   ├── decompose_hypothetical.rs  # use fluent_llm::anonymize;
    │       │   ├── decompose_subtasks.rs  # use fluent_llm::Decomposer;
    │       │   ├── mod.rs  # pub mod none;
    │       │   ├── none.rs  # use crate::transforms::{TransformError, 
    │       │   ├── pii_anonymize.rs  # use std::collections::HashMap;
    │       │   └── tests.rs  # #[cfg(test)]
    │       ├── types.rs  # //! Unified request/response types ...
    │       └── watchdog.rs  # use std::collections::VecDeque;
    ├── search-vector/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── aliases.rs  # use std::collections::HashMap;
    │       ├── db.rs  # use std::path::Path;
    │       ├── error.rs  # use thiserror::Error;
    │       ├── lib.rs  # //! search-vector: SQLite hybrid search 
    │       └── math.rs  # pub fn cosine_similarity(a: &[f32], b: &
    ├── types/
    │   ├── Cargo.toml
    │   └── src/
    │       └── lib.rs  # //! fluent-types: Shared data types (Gui
    └── wasm_ipc/
        ├── Cargo.toml
        └── src/
            └── lib.rs  # //! WASM IPC — Binary schemas for E...
```
