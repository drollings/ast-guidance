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
├── AGENTS.md
├── Cargo.toml
├── LICENSE
├── LICENSE-Commercial-Requirement
├── LICENSE-Contributor-Agreement
├── Makefile
├── README.md
├── STRUCTURE.md
├── bin/
│   ├── coral-router-test.py
│   ├── gen_simhash_projections.py
│   ├── router-wait-health.sh
│   └── spacy/
│       └── benchmark/
│           ├── Cargo.lock
│           ├── Cargo.toml
│           ├── bench_py.py
│           ├── run.sh
│           ├── src/
│           │   └── main.rs
│           └── summarize.py
├── data/
│   └── yamake.json
├── doc/
│   ├── AMBIGUOUS_DAG.md
│   ├── MEMORY_PLUGIN.md
│   ├── capabilities/
│   │   ├── ast-indexing/
│   │   │   └── CAPABILITY.md
│   │   ├── config-system/
│   │   │   └── CAPABILITY.md
│   │   ├── coral-cache/
│   │   │   └── CAPABILITY.md
│   │   ├── coral-database/
│   │   │   └── CAPABILITY.md
│   │   ├── coral-ingestion/
│   │   │   └── CAPABILITY.md
│   │   ├── coral-mcp/
│   │   │   └── CAPABILITY.md
│   │   ├── embedding-providers/
│   │   │   └── CAPABILITY.md
│   │   ├── explain-query/
│   │   │   └── CAPABILITY.md
│   │   ├── fluent-concurrency/
│   │   │   └── CAPABILITY.md
│   │   ├── llm-client/
│   │   │   └── CAPABILITY.md
│   │   ├── local-model-decomposition/
│   │   │   └── CAPABILITY.md
│   │   ├── ontology/
│   │   │   └── CAPABILITY.md
│   │   ├── plugin-system/
│   │   │   └── CAPABILITY.md
│   │   ├── rdf-parsing/
│   │   │   └── CAPABILITY.md
│   │   ├── reflection/
│   │   │   └── CAPABILITY.md
│   │   ├── sync-pipeline/
│   │   │   └── CAPABILITY.md
│   │   ├── target-registry/
│   │   │   └── CAPABILITY.md
│   │   ├── vector-search/
│   │   │   └── CAPABILITY.md
│   │   └── wasm-tools/
│   │       └── CAPABILITY.md
│   ├── coral/
│   │   ├── CHANGELOG.md
│   │   ├── DETAILS.md
│   │   ├── OVERVIEW.md
│   │   └── VISION.md
│   ├── guidance/
│   │   ├── ARCHITECTURE.md
│   │   ├── DESIGN.md
│   │   ├── MCP.md
│   │   ├── VISION.md
│   │   └── schemas/
│   │       └── guidance.schema.json
│   ├── router/
│   │   ├── ARCHITECTURE.md
│   │   ├── TESTING.md
│   │   └── VISION.md
│   ├── skills/
│   │   ├── common-core/
│   │   │   └── SKILL.md
│   │   ├── dag/
│   │   │   └── SKILL.md
│   │   ├── fluent-concurrency/
│   │   │   └── SKILL.md
│   │   ├── fluent-db/
│   │   │   └── SKILL.md
│   │   ├── fluent-wvr/
│   │   │   └── SKILL.md
│   │   ├── interlingua/
│   │   │   └── SKILLS.md
│   │   └── subagent/
│   │       └── SKILL.md
│   └── spacy-rs/
│       ├── ARCHITECTURE.md
│       └── VISION.md
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
├── src/
│   ├── Cargo.lock
│   ├── bin/
│   │   ├── coral/
│   │   │   ├── Cargo.toml
│   │   │   └── src/
│   │   │       └── main.rs
│   │   ├── coral-router/
│   │   │   ├── Cargo.toml
│   │   │   └── src/
│   │   │       ├── boot.rs
│   │   │       └── main.rs
│   │   ├── guidance/
│   │   │   ├── Cargo.toml
│   │   │   └── src/
│   │   │       ├── benchmark.rs
│   │   │       ├── commit.rs
│   │   │       ├── editor.rs
│   │   │       ├── main.rs
│   │   │       ├── mcp.rs
│   │   │       └── structure.rs
│   │   └── yamake-coral/
│   │       ├── Cargo.toml
│   │       └── src/
│   │           └── main.rs
│   ├── common-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── cache.rs
│   │       ├── config.rs
│   │       ├── constants.rs
│   │       ├── drift.rs
│   │       ├── error.rs
│   │       ├── error_context.rs
│   │       ├── format.rs
│   │       ├── git.rs
│   │       ├── hash.rs
│   │       ├── http.rs
│   │       ├── interner.rs
│   │       ├── io.rs
│   │       ├── jsonrpc.rs
│   │       ├── lib.rs
│   │       ├── metrics.rs
│   │       ├── prelude.rs
│   │       ├── registry.rs
│   │       ├── retry.rs
│   │       ├── runtime.rs
│   │       ├── shell.rs
│   │       ├── shell_parser.rs
│   │       ├── sqlite.rs
│   │       ├── string.rs
│   │       ├── sync.rs
│   │       ├── time.rs
│   │       ├── tokens.rs
│   │       ├── walk.rs
│   │       └── watchdog.rs
│   ├── content-node/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── doc_node.rs
│   │       ├── file_node.rs
│   │       ├── lib.rs
│   │       ├── lod.rs
│   │       ├── node.rs
│   │       └── source_node.rs
│   ├── coral/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── cache/
│   │       │   ├── mod.rs
│   │       │   ├── reactor.rs
│   │       │   └── stats.rs
│   │       ├── cache_l1.rs
│   │       ├── cache_router.rs
│   │       ├── db/
│   │       │   ├── edges.rs
│   │       │   ├── embeddings.rs
│   │       │   ├── hnsw.rs
│   │       │   ├── mod.rs
│   │       │   ├── nodes.rs
│   │       │   └── schema.rs
│   │       ├── error.rs
│   │       ├── ingest.rs
│   │       ├── knowledge.rs
│   │       ├── lib.rs
│   │       ├── mcp.rs
│   │       ├── packer.rs
│   │       ├── test_stubs.rs
│   │       ├── tests/
│   │       │   ├── common.rs
│   │       │   └── mod.rs
│   │       ├── tier_units.rs
│   │       ├── wasm_runtime.rs
│   │       └── wvr.rs
│   ├── dag/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── adapter.rs
│   │       ├── checkpointed.rs
│   │       ├── closure.rs
│   │       ├── dep_graph.rs
│   │       ├── error.rs
│   │       ├── lib.rs
│   │       ├── middleware.rs
│   │       ├── narrowing.rs
│   │       ├── resolver.rs
│   │       ├── target.rs
│   │       ├── target_work_unit.rs
│   │       ├── tests/
│   │       │   ├── common.rs
│   │       │   └── mod.rs
│   │       ├── type_inference.rs
│   │       ├── work_unit.rs
│   │       ├── wvr.rs
│   │       └── yamake_loader.rs
│   ├── db/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── cache.rs
│   │       ├── capability.rs
│   │       ├── error.rs
│   │       ├── hnsw.rs
│   │       ├── lib.rs
│   │       ├── migrate.rs
│   │       ├── pool.rs
│   │       ├── query.rs
│   │       ├── store.rs
│   │       ├── tests/
│   │       │   ├── common.rs
│   │       │   └── mod.rs
│   │       ├── vector.rs
│   │       └── wvr.rs
│   ├── fluent-concurrency/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── affinity.rs
│   │       ├── batch.rs
│   │       ├── capability.rs
│   │       ├── flow.rs
│   │       ├── io/
│   │       │   ├── db.rs
│   │       │   ├── fs.rs
│   │       │   ├── mod.rs
│   │       │   └── net.rs
│   │       ├── ladder.rs
│   │       ├── lib.rs
│   │       ├── llm_queue.rs
│   │       ├── pool.rs
│   │       ├── queue.rs
│   │       ├── reserve.rs
│   │       ├── router.rs
│   │       ├── runtime/
│   │       │   ├── mod.rs
│   │       │   ├── test.rs
│   │       │   └── tokio.rs
│   │       ├── scope.rs
│   │       ├── stream.rs
│   │       ├── tests/
│   │       │   ├── e2e.rs
│   │       │   ├── m1.rs
│   │       │   ├── m2.rs
│   │       │   ├── m3.rs
│   │       │   ├── m4.rs
│   │       │   ├── m5.rs
│   │       │   └── mod.rs
│   │       └── thread_resource.rs
│   ├── fluent-wvr/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── boundary.rs
│   │       ├── capability.rs
│   │       ├── coerce.rs
│   │       ├── dynamic.rs
│   │       ├── lib.rs
│   │       ├── macros.rs
│   │       ├── metadata.rs
│   │       ├── prelude.rs
│   │       ├── runtime.rs
│   │       ├── store.rs
│   │       ├── test_support.rs
│   │       ├── tests.rs
│   │       ├── traits.rs
│   │       ├── work.rs
│   │       └── wrapper.rs
│   ├── fluent-wvr-macros/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs
│   │   └── tests/
│   │       └── derive_expansion.rs
│   ├── fluent-wvr-testutil/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   ├── guidance/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── ast_parser.rs
│   │   │   ├── config.rs
│   │   │   ├── enhancer.rs
│   │   │   ├── grounding.rs
│   │   │   ├── lib.rs
│   │   │   ├── memory.rs
│   │   │   ├── plugin.rs
│   │   │   ├── query/
│   │   │   │   ├── formatter.rs
│   │   │   │   ├── identifier.rs
│   │   │   │   ├── llm_filter.rs
│   │   │   │   ├── llm_filter_batch.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── search_backend.rs
│   │   │   │   ├── snapshot.rs
│   │   │   │   ├── strategy.rs
│   │   │   │   └── synthesize.rs
│   │   │   ├── query_engine.rs
│   │   │   ├── runtime.rs
│   │   │   ├── scanner.rs
│   │   │   ├── sync/
│   │   │   │   ├── comments.rs
│   │   │   │   ├── json_store.rs
│   │   │   │   ├── json_writer.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── staleness.rs
│   │   │   ├── sync_engine.rs
│   │   │   └── tests/
│   │   │       ├── common.rs
│   │   │       └── mod.rs
│   │   └── tests/
│   │       ├── common/
│   │       │   └── mod.rs
│   │       ├── e2e_gen_roundtrip.rs
│   │       ├── live/
│   │       │   ├── README.md
│   │       │   └── smoke_live.rs
│   │       └── live.rs
│   ├── knowledge/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── csr_graph.rs
│   │       ├── freq_table.rs
│   │       ├── index_header.rs
│   │       ├── lib.rs
│   │       ├── query_cache.rs
│   │       ├── tokenizer.rs
│   │       ├── trigram_index.rs
│   │       └── word_index.rs
│   ├── llm/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── anonymize.rs
│   │   │   ├── client.rs
│   │   │   ├── constants.rs
│   │   │   ├── context_packer.rs
│   │   │   ├── decomposer.rs
│   │   │   ├── embeddings.rs
│   │   │   ├── error.rs
│   │   │   ├── http_class.rs
│   │   │   ├── lib.rs
│   │   │   ├── llm_queue.rs
│   │   │   ├── openai.rs
│   │   │   ├── parse.rs
│   │   │   ├── pii_patterns.rs
│   │   │   └── url.rs
│   │   └── tests/
│   │       ├── live/
│   │       │   ├── README.md
│   │       │   └── smoke_live.rs
│   │       └── live.rs
│   ├── memory-plugin/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── capability.rs
│   │       ├── lib.rs
│   │       ├── plugins/
│   │       │   ├── hindsight/
│   │       │   │   └── mod.rs
│   │       │   ├── holographic/
│   │       │   │   ├── hrr.rs
│   │       │   │   ├── mod.rs
│   │       │   │   └── store.rs
│   │       │   ├── honcho/
│   │       │   │   └── mod.rs
│   │       │   └── mod.rs
│   │       ├── registry.rs
│   │       ├── traits.rs
│   │       └── types.rs
│   ├── ontology/
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   ├── data/
│   │   │   └── yago_classes.json
│   │   ├── src/
│   │   │   ├── entity.rs
│   │   │   ├── inference.rs
│   │   │   ├── lib.rs
│   │   │   ├── mapper.rs
│   │   │   ├── migration.rs
│   │   │   ├── yago.rs
│   │   │   └── yago_loader.rs
│   │   └── tests/
│   │       └── yago_interlingua.rs
│   ├── rdf/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lexer.rs
│   │       ├── lib.rs
│   │       ├── normalize.rs
│   │       ├── nquads.rs
│   │       └── parser.rs
│   ├── requirements.txt
│   ├── router/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── audit.rs
│   │   │   ├── charts/
│   │   │   │   ├── binding.rs
│   │   │   │   ├── compile.rs
│   │   │   │   ├── execute.rs
│   │   │   │   ├── extract.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── render.rs
│   │   │   │   ├── rubric.rs
│   │   │   │   ├── select.rs
│   │   │   │   ├── stage.rs
│   │   │   │   └── store.rs
│   │   │   ├── cli/
│   │   │   │   ├── commands/
│   │   │   │   │   ├── filesystem.rs
│   │   │   │   │   ├── mod.rs
│   │   │   │   │   └── server.rs
│   │   │   │   ├── gguf.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── preset.rs
│   │   │   ├── concept_store_sqlite.rs
│   │   │   ├── config/
│   │   │   │   ├── addr.rs
│   │   │   │   ├── builder.rs
│   │   │   │   ├── classification.rs
│   │   │   │   ├── escalation.rs
│   │   │   │   ├── filters.rs
│   │   │   │   └── routing.rs
│   │   │   ├── config.rs
│   │   │   ├── config_route_tests.rs
│   │   │   ├── dag_session.rs
│   │   │   ├── dispatch/
│   │   │   │   ├── backend.rs
│   │   │   │   ├── escalation/
│   │   │   │   │   ├── assemble.rs
│   │   │   │   │   ├── audit.rs
│   │   │   │   │   ├── mod.rs
│   │   │   │   │   └── modes.rs
│   │   │   │   ├── frontier.rs
│   │   │   │   └── mod.rs
│   │   │   ├── error.rs
│   │   │   ├── filters/
│   │   │   │   ├── injection_detect.rs
│   │   │   │   ├── luhn.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── regex_filter.rs
│   │   │   ├── frontier/
│   │   │   │   ├── mod.rs
│   │   │   │   └── modes.rs
│   │   │   ├── hnsw.rs
│   │   │   ├── instances/
│   │   │   │   ├── api.rs
│   │   │   │   ├── client.rs
│   │   │   │   ├── manager.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── pool.rs
│   │   │   ├── knowledge.rs
│   │   │   ├── kv_cache.rs
│   │   │   ├── ledger/
│   │   │   │   ├── correction_index.rs
│   │   │   │   ├── nlp.rs
│   │   │   │   ├── orchestrator.rs
│   │   │   │   ├── prompt.rs
│   │   │   │   └── tiering.rs
│   │   │   ├── ledger.rs
│   │   │   ├── ledger_guard.rs
│   │   │   ├── lib.rs
│   │   │   ├── logging.rs
│   │   │   ├── metrics.rs
│   │   │   ├── node_store.rs
│   │   │   ├── normalize.rs
│   │   │   ├── pipeline.rs
│   │   │   ├── pipeline_types.rs
│   │   │   ├── routes/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── plan.rs
│   │   │   │   └── rigor/
│   │   │   │       ├── mod.rs
│   │   │   │       └── prompts.rs
│   │   │   ├── scheduler.rs
│   │   │   ├── score_matrix.rs
│   │   │   ├── server/
│   │   │   │   ├── admin.rs
│   │   │   │   ├── dispatch.rs
│   │   │   │   ├── handler.rs
│   │   │   │   ├── instances_api.rs
│   │   │   │   ├── responses.rs
│   │   │   │   └── review.rs
│   │   │   ├── server.rs
│   │   │   ├── server_http_tests.rs
│   │   │   ├── server_tests.rs
│   │   │   ├── session.rs
│   │   │   ├── stage_tests.rs
│   │   │   ├── stages/
│   │   │   │   ├── classifier.rs
│   │   │   │   ├── common.rs
│   │   │   │   ├── deterministic.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── nlp.rs
│   │   │   │   ├── pipeline_ref.rs
│   │   │   │   ├── prompt_parse.rs
│   │   │   │   ├── retry_classifier.rs
│   │   │   │   └── tree/
│   │   │   │       ├── decisions.rs
│   │   │   │       ├── engine.rs
│   │   │   │       ├── mod.rs
│   │   │   │       └── verdict.rs
│   │   │   ├── streaming.rs
│   │   │   ├── summarization.rs
│   │   │   ├── supervisor.rs
│   │   │   ├── supervisor_integration_tests.rs
│   │   │   ├── target_match.rs
│   │   │   ├── telemetry.rs
│   │   │   ├── test_stubs.rs
│   │   │   ├── test_support.rs
│   │   │   ├── testing/
│   │   │   │   ├── mock.rs
│   │   │   │   └── mod.rs
│   │   │   ├── tests/
│   │   │   │   ├── common.rs
│   │   │   │   ├── e2e_tests.rs
│   │   │   │   ├── golden.rs
│   │   │   │   ├── mod.rs
│   │   │   │   └── rubric_fixtures.rs
│   │   │   ├── transforms/
│   │   │   │   ├── codeword_anonymize.rs
│   │   │   │   ├── decompose_hypothetical.rs
│   │   │   │   ├── decompose_subtasks.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── none.rs
│   │   │   │   ├── pii_anonymize.rs
│   │   │   │   ├── sanitize.rs
│   │   │   │   ├── secret_mask.rs
│   │   │   │   └── tests.rs
│   │   │   ├── types.rs
│   │   │   └── views.rs
│   │   └── tests/
│   │       ├── live/
│   │       │   ├── README.md
│   │       │   └── smoke_live.rs
│   │       └── live.rs
│   ├── search-vector/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── aliases.rs
│   │       ├── db.rs
│   │       ├── error.rs
│   │       ├── lib.rs
│   │       └── math.rs
│   ├── spacy-rs/
│   │   ├── Cargo.toml
│   │   ├── build.rs
│   │   ├── data/
│   │   │   └── en_lemmatizer.json
│   │   ├── src/
│   │   │   ├── arc_eager.rs
│   │   │   ├── attrs.rs
│   │   │   ├── concept_store.rs
│   │   │   ├── concept_store_mem.rs
│   │   │   ├── doc.rs
│   │   │   ├── error.rs
│   │   │   ├── hash.rs
│   │   │   ├── interlingua.rs
│   │   │   ├── labels.rs
│   │   │   ├── lang/
│   │   │   │   ├── en/
│   │   │   │   │   ├── exceptions.rs
│   │   │   │   │   ├── num_words.rs
│   │   │   │   │   ├── patterns.rs
│   │   │   │   │   ├── stop_words.rs
│   │   │   │   │   └── tag_map.rs
│   │   │   │   ├── en.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── norm_exceptions.rs
│   │   │   │   └── url.rs
│   │   │   ├── lemma_blob.rs
│   │   │   ├── lemmatizer.rs
│   │   │   ├── lex_attrs.rs
│   │   │   ├── lexeme.rs
│   │   │   ├── lib.rs
│   │   │   ├── llm.rs
│   │   │   ├── morph.rs
│   │   │   ├── pipeline/
│   │   │   │   └── tests.rs
│   │   │   ├── pipeline.rs
│   │   │   ├── review.rs
│   │   │   ├── routing.rs
│   │   │   ├── sentencizer.rs
│   │   │   ├── strings.rs
│   │   │   ├── tag_map.rs
│   │   │   ├── tokenizer.rs
│   │   │   ├── validate.rs
│   │   │   └── vocab.rs
│   │   ├── tests/
│   │   │   ├── arceager_golden.rs
│   │   │   ├── data/
│   │   │   │   └── en_tokenization.json
│   │   │   ├── en_tokenization.rs
│   │   │   ├── live/
│   │   │   │   ├── README.md
│   │   │   │   └── smoke_live.rs
│   │   │   └── live.rs
│   │   └── tools/
│   │       ├── gen_en_exceptions.py
│   │       ├── gen_en_lemma_data.py
│   │       ├── gen_en_regexes.py
│   │       ├── gen_en_tag_map.py
│   │       └── gen_golden_corpus.py
│   ├── types/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── interlingua.rs
│   │       ├── knowledge.rs
│   │       └── lib.rs
│   └── wasm_ipc/
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
└── tools/
    ├── download_yago_taxonomy.sh
    └── gen_yago_classes.py
```
