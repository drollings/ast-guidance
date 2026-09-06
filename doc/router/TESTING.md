# Workspace Testing Convention

**Single source of truth for how this monorepo's Rust crates are tested.**
It supersedes `doc/router/TESTING.md` (folded in below; that file is now a
pointer). Adopt this convention for all new test code; migrate existing suites
incrementally per `ROADMAP_20260816_TESTS.md`.

---

## 1. Guiding principles

1. **Hermetic by default.** The entire workspace test suite is hermetic: no
   GPU, no weights, no real `llama-server`, no external (non-loopback)
   service. Any test that needs a real model call is walled off behind an
   opt-in live pathway (§4) and can never run in `make test` /
   `make router-test` / `make router-mock` / CI.
2. **DRY.** Every helper that appears ≥2× lives in one sanctioned home (§3).
   Never copy a helper into a new test module.
3. **One assertion per tier.** The same behavior is asserted at most once per
   tier (unit → golden table → HTTP e2e). Extend a table, don't duplicate a
   function.
4. **Fast builds.** Tier-0/1/2 additions must not pull heavy deps
   (`llama.cpp`, `candle`, …). `httpmock` and `tempfile` are workspace deps —
   reuse them.
5. **Import boundaries.** Shared library crates may NOT import `guidance`,
   `coral`, or `wasm_ipc`. `fluent-wvr-testutil` stays leaf-adjacent (only
   `fluent-wvr`, `common-core`, `serde_json`, `tempfile`).

---

## 2. The test pyramid

Rust integration tests in a crate-root `tests/` directory are compiled as
**separate crates linked against the public API** and therefore cannot access
`pub(crate)`/private items — and a file cannot be both a `#[path]`-included
lib module and an integration test. This convention is designed *around* that
constraint.

| Tier | Name | Canonical home | Access | Gate |
|---|---|---|---|---|
| **0** | Module unit tests | inline `#[cfg(test)] mod tests` in each source file | private | `cargo test` / `make test` |
| **1** | Crate unit/integration suites | `src/tests.rs` (or `src/tests/<name>.rs`), wired once via `#[cfg(test)] mod tests;` in `lib.rs` | crate-internal (can use `pub(crate)` test-support) | `cargo test` / `make test-<crate>` |
| **2** | E2E / black-box tests | crate-root `tests/`, `tests/common/mod.rs` for shared fixtures | public API only | `cargo test` / `make test-<crate>` |
| **3** | Live AI-inference tests | `tests/live/<name>_live.rs` | public API only | **opt-in only** — `#[ignore]` + `#[cfg(feature = "live-ai")]`, run via `make test-live`; NEVER in `make test` / `make router-test` / `make router-mock` / CI |

- **Tier 0 inline** — tests sit next to the code they verify; private access;
  incremental compilation per module.
- **Tier 1 `src/tests.rs`** — the single crate-wide suite module for
  multi-module behavior and suites needing `pub(crate)` test-support
  (`stage_tests`, `server_http_tests`, `config_route_tests`, `m1..m5`, `e2e`).
  Collapse scattered top-level test files into it.
- **Tier 2 crate-root `tests/`** — the single home for all e2e tests.
  `tests/common/mod.rs` holds that crate's public-API test fixtures.
- **Tier 3 live** — real-inference tests are explicitly walled off (feature +
  `#[ignore]`) so they can never run in the default paths.

### Directory layout (canonical)

```text
src/<crate>/
  src/lib.rs                 // #[cfg(test)] mod tests;  (Tier 1 suites)
  src/tests.rs               // OR src/tests/mod.rs + src/tests/<suite>.rs (Tier 1)
  src/tests/test_support.rs  // pub(crate) helpers shared by Tier-1 suites (opt.)
  src/testing/               // ONLY if non-test code needs the helpers (router's mock-mode infra — keep as-is)
  tests/                     // Tier 2 e2e
    common/mod.rs            // shared e2e fixtures (public API only)
    <area>_e2e.rs            // one file per e2e area
    live/                    // Tier 3 — only present when live-ai tests exist
      <area>_live.rs
```

### Naming rules

- Tier-1 suites: `<area>_tests.rs` (e.g. `stage_tests.rs`, `pipeline_tests.rs`).
- Tier-2 e2e files: `<area>_e2e.rs` (e.g. `server_http_e2e.rs`,
  `config_route_e2e.rs`, `gen_roundtrip_e2e.rs`).
- Tier-3 live files: `<area>_live.rs`.
- Test fns: `test_<behavior>[_<variant>]` or `golden_<category>` for
  table-driven corpora; no `test_` prefix on table rows.
- Crate-local test-support modules are always named `test_support` (Tier 1)
  or `common` (Tier 2) so implementers and reviewers know where to look.

---

## 3. Shared test infrastructure (DRY, SOLID)

Two homes only — no ad-hoc third:

1. **`fluent-wvr-testutil`** — the *cross-crate* test-support crate for
   trait-level stubs. Owns `StubComponent`, `PassthroughUnit`,
   `impl_component_for_test!`, `tempdir`, `make_tree`. This is the ONLY
   allowed cross-crate home. Each crate uses it via `[dev-dependencies]`.
2. **Per-crate `test_support`** (Tier 1, `#[cfg(test)] pub(crate)`) or
   `tests/common` (Tier 2, public) — for *crate-typed* fixtures:
   - `fluent-dag`: `make_bitset`, `make_registry`, linear/diamond graph
     fixtures → `src/tests/common.rs`.
   - `fluent-db`: `db_caps()`, `in_memory_pool()`, `conn()` →
     `src/tests/common.rs`.
   - `coral-context`: `make_node()` + shared reactor/router builders.
   - `guidance-core`: `make_test_doc()`.
   - `fluent-wvr`: crate-local `test_support.rs` (cannot depend on
     `fluent-wvr-testutil` — dependency cycle — so its own support module is
     the sanctioned exception).
   - `fluent-router`: `tests/common/mod.rs` for `TestServer`, `post_chat`,
     `make_config`, `ServerDeps` builder, SSE upstream fixtures. The
     always-compiled `testing/` module is the `--mock` binary's runtime
     infra, not test code — leave it untouched.

The dependency-inversion rule: **tests depend on the shared trait, never on
another crate's private test harness.** A stub for `ChatBackend`,
`EmbeddingProvider`, or `WorkUnit` that two crates need goes to
`fluent-wvr-testutil` (or its trait's own crate, e.g. `fluent-llm::testing`,
matching the router `testing` precedent). A stub that one crate needs is local.

### Router (folded from `doc/router/TESTING.md`)

| Layer | Location | What it tests |
|-------|----------|---------------|
| Unit tests | `src/router/src/stage_tests.rs` | Individual pipeline stages in isolation |
| Golden tests | `src/router/src/tests/golden.rs` | Labeled corpus — intent, PII, adversarial cases |
| E2E tests | `src/router/src/tests/e2e_tests.rs` | Full pipeline with `TranscriptProvider` (no LLM) |
| Server tests | `src/router/src/server_tests.rs` | `RouterServer` construction and config |
| Config-synced | `src/router/src/config_route_tests.rs` | Config-derived intent→model_group probes (`make router-mock`) |
| Mock mode | `coral-router start --mock` | Real HTTP server, real pipeline, transcript-driven |

Quick commands:

```sh
# All fluent-router tests (unit + golden + e2e + server + config-synced)
make router-test

# Config-synced routing integration tests (intent → model_group)
make router-mock

# Workspace-wide
cargo test --workspace
```

---

## 4. AI-inference isolation rules (binding)

1. A test that needs a **real model call** (any endpoint that performs
   inference on model weights — local `llama-server`, Ollama, frontier) must
   be marked `#[ignore]` **and** `#[cfg(feature = "live-ai")]` and placed in
   `tests/live/<area>_live.rs` (or `src/tests/live/` for Tier-1 suites needing
   private access).
2. Every crate that ships live tests declares a `live-ai = []` feature and a
   documented env contract (e.g. `LLM_BASE_URL`, `LLM_MODEL`, `LLAMA_SERVER`)
   with a `skip`-not-fail policy when the env contract is absent (use
   `std::env::var` guard + early `return`, never panic).
3. Live tests run **only** via `make test-live` (or `make <crate>-test-live`).
   They are excluded from `make test`, `make router-test`, `make router-mock`,
   `make pre-commit`, and `.github/workflows/ci.yml`.
4. **Hermeticity is the default and must be preserved.** Any new mock-backed
   test that could accidentally dial the network must use unreachable/refused
   loopback (`127.0.0.1:1`) or an in-process stub — never `localhost:11434`
   or a real public host.
5. `#[ignore]` is reserved for: live-AI tests (rule 1) and benchmarks. Every
   `#[ignore]` test must carry a `reason = "..."` string explaining how to
   run it.

---

## 5. Makefile / CI contract

| Target | Runs | Notes |
|---|---|---|
| `make test` | `cargo test --workspace` | Hermetic; no AI. |
| `make router-test` | `cargo test -p fluent-router` | Hermetic; no AI. |
| `make router-mock` | `cargo test -p fluent-router config_route_tests` | Hermetic; config-synced. |
| `make router-test-all` | `router-test` + `cargo test -p coral-context --features hnsw-bench -- --ignored` | Benchmark only, no AI. |
| `make test-<crate>` | Tier-0/1/2 suite of one crate | Per-crate gate; hermetic, no AI. |
| `make test-live` | live-ai crates with `--features live-ai --test live -- --ignored` | **The only pathway that runs real inference.** Skips cleanly when the env contract is absent. |
| `make <crate>-test-live` | One crate's live-ai tests (e.g. `router-test-live`) | Same skip-not-fail policy. |
| `make lint-live-ai` | `bin/live-ai-guard.sh` | Hermeticity guard: every `#[ignore]` has a `reason`; no hermetic test dials a non-loopback host. |
| `make lint` | `cargo clippy --workspace -- -D warnings` | Static gate. |

CI (`.github/workflows/ci.yml`) runs `cargo clippy --workspace -- -D warnings`,
`cargo test --workspace`, a fast hermetic `router-mock` job, and the
`live-ai-guard` lint. It must never run live-AI tests.

---

## Appendix A — Baseline metrics (2026-08-16, start of roadmap)

Counts use the roadmap method `rg -c "#\[test\]|#\[tokio::test\]|#\[test_util::test\]"`
(one per matching line, summed per crate). Values below reflect **post-M3**
state; the M0 baseline is the start-of-roadmap figure in parentheses.

### Per-crate test counts

| Crate (package) | Test lines (M0 → post-M3) |
|---|---|
| `fluent-router` | 749 → 797 |
| `fluent-concurrency` | 58 → 67 |
| `fluent-db` | 90 → 90 |
| `fluent-dag` | 107 → 119 |
| `coral-context` | 86 → 114 |
| `fluent-wvr` | 140 → 157 |
| `fluent-wvr-macros` | 18 → 25 (18 src + 7 `tests/derive_expansion.rs`) |
| `guidance-core` | 170 → 181 |
| `bin/guidance` | 31 → 39 |
| `fluent-llm` | 194 → 191 |
| `common-core` | 345 → 345 |
| `search-vector` | 10 → 10 |
| `bin/coral-router` | 3 → 3 |
| `bin/coral` | 0 → 0 |
| `content-node` | 3 → 3 |
| `fluent-wvr-testutil` | 0 → 0 |
| **Workspace total** | **~2,185 → ~2,141** |

### Ignored tests

Every `#[ignore]` carries a `reason`. Currently:

- **1 benchmark**: the coral-context HNSW 100K-node KNN benchmark
  (`src/coral/src/db/mod.rs`), feature-gated `hnsw-bench`, run via
  `make router-test-all`.
- **3 live-AI smoke tests** (feature-gated `live-ai` + `#[ignore]`, compiled
  only under the feature, run via `make test-live` / `make <crate>-test-live`):
  `fluent-llm` `tests/live/smoke_live.rs`, `guidance-core`
  `tests/live/smoke_live.rs`, `fluent-router` `tests/live/smoke_live.rs`.
  Each skips cleanly when `LLM_BASE_URL`/`LLM_MODEL` are absent.

### Duplicated-helper sites flagged (target: zero identical copies)

Status after M1: the helpers below were consolidated to their sanctioned homes
(`src/tests/common.rs` per crate, or `fluent-wvr-testutil`), so the remaining
sites are the canonical implementations — the "copies" column is now 1 each
unless noted.

| Helper | Canonical home (after M1) |
|---|---|
| `make_bitset` | `fluent-dag` `src/tests/common.rs` (was 4 copies) |
| `make_registry` | `fluent-dag` `src/tests/common.rs` (was 2 copies) |
| `db_caps()` | `fluent-db` `src/tests/common.rs` (was 3 copies) |
| `in_memory_pool()` | `fluent-db` `src/tests/common.rs` (was 2 copies) |
| `conn()` | `fluent-db` `src/tests/common.rs` (was 2 copies) |
| `make_tree` | `common_core::walk::make_tree`; testutil delegates (was 2 impls) |
| `make_test_config`/`make_pipeline`/`make_request`/`route` | router `tests/golden.rs` vs `tests/e2e_tests.rs` (see M2.1) |
| `post_chat`/`post_raw`/`get`/`TestServer`/`make_config` | router `src/tests/common.rs` (was 2 copies each) |
| `ServerDeps` builder (`test_deps` + variants) | router `src/tests/common.rs` (was 6 hand-built sites) |
| `make_test_doc()` | `guidance-core` `src/tests/common.rs` (was 2 impls + ≥8 literal sites) |
| `make_test_member()` | `guidance-core` `src/tests/common.rs` (was 1 impl + 5 sites) |
| `ContentNode` literals | `coral-context` `src/tests/common.rs` `make_node()` (was ~30 sites) |
| `make_batch` | `fluent-concurrency` `src/tests/mod.rs` (was ~25 inline setups) |
| `TestComponent` + wrapper stubs | `fluent-wvr` `src/test_support.rs` (was `tests.rs` + 6 inline stubs) |
| `repair_json` | `fluent-llm` `src/parse.rs` delegates structural repair to `fluent_wvr::boundary::repair_boundary` (M2.9 — pruned 7 mirror-image repair tests from `parse.rs`; kept only llm-specific truncation cases: `close_open_containers`/`drop_incomplete_tail`/first-value extraction) |

