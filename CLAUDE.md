# Coral Router — Development Guide

**What it is**: a Rust-native LLM request router (crate `fluent-router`, binary
`coral-router`) exposing an OpenAI-compatible HTTP API on `:8079`. Every
request runs a **two-stage pipeline** — `DeterministicPreFilter → Classifier`
(see `src/router/src/pipeline_types.rs`) — that resolves to a direct response,
a routing target, or a rejection. Coral Router is also the **process owner** of
the local inference fleet: it spawns and supervises one `llama-server` process
per model weights file, serves the `/instances` management contract at its own
address, and is the single routing element between those llama-server tasks and
every other OpenAI-compatible endpoint. The llama.cpp router mode is never used.

This is a Rust monorepo with many crates and shared infrastructure; coral-router
is the priority.

## Build-Test Loop

1. **BUILD** — `make router` builds the binary; `make router-start` builds and
   (re)starts the server on `:8079`, waiting for `/health`.

2. **TEST** — `make router-test` runs the fluent-router unit/golden/e2e suites
   and a `--help` dry-run of the built binary.

3. **SMOKE** — `make router-mock` runs the config-synced routing integration
   tests in `fluent-router` (`src/router/src/config_route_tests.rs`): every
   intent/route declared in `env/coral-router.json` is probed against an
   in-process mock server and must dispatch through the `model_group` its
   `routes` entry maps to. Expectations are derived from the config at runtime,
   so the suite cannot drift from it. Spot-check a live server:
   ```
   curl -s http://127.0.0.1:8079/health
   curl -s -X POST http://127.0.0.1:8079/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"local","messages":[{"role":"user","content":"What is 2+2?"}]}'
   ```

4. **VERIFY** — the full gate before landing:
   ```
   cargo build --workspace && cargo test --workspace \
     && cargo clippy --workspace -- -D warnings
   ```

## Make targets

| Target | Purpose |
|---|---|
| `make router` | Build coral-router |
| `make router-start` | Build + (re)start on `:8079`, waiting for `/health` (kills old tree first) |
| `make router-test` | Kill server + fluent-router unit/golden/e2e tests + `--help` dry-run |
| `make router-test-all` | `router-test` + coral-context HNSW benchmarks (slow) |
| `make router-mock` | Config-synced routing integration tests (intent → model_group, derived from `env/coral-router.json`) |
| `make test-<crate>` | One crate's Tier-0/1/2 suite, hermetic (e.g. `test-router`, `test-llm`, `test-db`) |
| `make test-live` | **The only pathway that runs real AI inference** — `#[ignore]` + `live-ai`-gated tests, skips cleanly without env |
| `make <crate>-test-live` | Live-AI tests for one crate (e.g. `router-test-live`, `llm-test-live`) |
| `make yago-load` | Operator action: download the full YaGO 4.5 taxonomy + regenerate the embedded class registry |
| `make review-test` | Async parse-review + interlingua suites (spacy-rs review, router review worker/endpoints, ontology loader, boot reconciliation) |
| `make lint-live-ai` | Hermeticity guard: every `#[ignore]` has a `reason`; no hermetic test dials a non-loopback host |

See `doc/TESTING.md` for the full testing convention and the `test-<crate>` /
`test-live` Make target table.

**AI-inference tests must be `#[ignore]` + `live-ai`-gated and run only via
`make test-live`.** No test in `make test` / `make router-test` /
`make router-mock` / CI may dial a real inference endpoint.

## Import boundaries

Shared library crates may NOT import from `guidance`, `router`, `coral`, or
`wasm_ipc`, as those are reserved for building compiled tools.

## Unit test conventions

When adding unit tests, always follow this same convention — create
`src/<crate>/tests/<module>.rs` first, then the `#[path]` forwarder, never an
inline `mod tests {`.

### Prohibited AI Use Cases or Actions

* Writing comments that reference the agent's current workflow, whether roadmaps, decisions, or milestones.

* Any git write operations (checkout, commit, stash, etc.)

* Any implementation without review of relevant documents including, but not limited to:

  - ./doc/skills/common-core/SKILL.md
  - ./doc/skills/fluent-wvr/SKILL.md
  - ./doc/skills/fluent-concurrency/SKILL.md

* Any implementation without reviewing shared primitives in shared libraries, and any removal of code from without human approval from these libraries:

  - ./src/common-core
  - ./src/fluent-wvr
  - ./src/fluent-concurrency

* Removing suspected unused primitives from shared crates without specific user approval.  Do not ask for such approval without extensive code review.

* Referencing transient roadmaps or milestones in documentation or comments
