# fluent-onnx — Architecture

*The working overview of the current implementation, written for code
assistants and maintainers. For the aspirational brief and the router-side
wiring see [`doc/router/VISION.md`](../router/VISION.md) and
[`doc/router/ARCHITECTURE.md`](../router/ARCHITECTURE.md).*

`fluent-onnx` (`src/fluent-onnx/`) is the ONNX / `ort` serving crate: the
in-process ONNX fleet that Coral Router composes. It supplies the ONNX
configuration schema, a session registry with llama-fleet-parity residency, a
generative `CausalLm` decoder with grammar-constrained sampling, and the
encoder / two-tower / PII / ColBERT workers. It is `spacy-rs`-free and
`guidance`/`coral`/`wasm_ipc`-free (import-boundary rule); the router is the
adapter that turns its sessions into `fluent_llm::client::ChatBackend`s.

## Module map

| Module | Role |
|---|---|
| `config.rs` | Pure, ort-free schema: `OnnxTask`, `Quant`, `ResidencyPolicy`, `OnnxConfig`, `OnnxRole`/`OnnxFleetConfig`/`OnnxRoleConfig`, `LlmIo`, `AnnotationHeads`, and the `instances` block. Validation runs ort-free (hermetic). |
| `session.rs` | `OrtSessionRegistry` (one session per model key, type-erased `SessionHandle`s, `SessionLoader` DIP seam for hermeticity). The real `OrtSessionLoader` (feature `onnx`) builds `ort` sessions **including execution-provider selection**. Residency parity: `last_used`, `resident_bytes`, `release`, `unloadable_keys`, `residency_report`. |
| `residency.rs` | `OrtResidencyLoop` — the CPU-RAM residency sibling of the llama sidecar: idle release + working-set LRU eviction; `Always`/pinned never released. |
| `llm.rs` | `OrtLlmSession` (feature `onnx`): chat-template, tokenize, prefill, per-token decode loop over the KV + conv-state contract (`LlmIo`), grammar-constrained sampling. `OnnxChatCompletion` is the "run a chat call" entry. |
| `grammar.rs` | `Grammar`, `JsonObjectGrammar`/`JsonArrayGrammar`/`BatchPromptGrammar`, `HuggingFaceVocab`, `grammar_from_json_schema` — structural JSON automata so invalid output shapes are impossible at generation time. |
| `context.rs`, `context_pool.rs` | `OnnxContext`/`OnnxKVCache` (the shared `fluent-llm::runtime` `LlmContext`/`LlmKVCache` contracts) and `OnnxContextPool` (one weights load, N named context windows with per-context KV). |
| `tokenizer.rs` | The HuggingFace `tokenizers` LFM tokenizer bridge. |
| `encoder.rs`, `annotate.rs`, `two_tower.rs`, `pii.rs`, `colbert.rs`, `overlay.rs` | The non-generative workers: mean-pool encoder, trained-annotation heads, two-tower router/policy-linter, PII token-classifier, ColBERT retriever, prompt-router overlays. |
| `error.rs` | `OrtError`. |
| `align.rs` | `SpacyTokenAligner` — LFM subword → spaCy span alignment. |

Feature gating: the config/lifecycle/registry types are **ort-free** (the crate
compiles and its hermetic config tests run with `--no-default-features`); the
real `OrtSessionLoader` and every worker are behind feature `onnx` (default-on).
Real-inference tests live in `tests/live/` under feature `live-ai`, `#[ignore]`d,
run only via `make test-live` / `make ort-test-live`.

## The registry and residency

`OrtSessionRegistry` stores one type-erased `SessionHandle` per model key and
delegates loading to an injected `SessionLoader`. The real
`OrtSessionLoader` builds an `ort::session::Session` from an `OnnxConfig`
(optimization level, intra threads, execution provider — see below) and stores
it behind a `Mutex` so every worker serializes runs on the shared session.

Residency mirrors the llama fleet: `resident: true` → `ResidencyPolicy::Always`
(loads at boot, refuses unload); `resident: false` → `Unloadable` (lazy load on
first use, releasable). `OrtResidencyLoop` polls the registry's
`residency_report`, releases idle `Unloadable` entries, and enforces the
working-set budget by evicting the LRU-largest first. Pinned/`Always` entries
are never released.

## Execution-provider selection

`OnnxConfig.execution_provider` (default `"cpu"`) selects the ONNX Runtime
execution provider. The loader (`src/fluent-onnx/src/session.rs`,
`OrtSessionLoader::load`) wires:

| Config value | EP | Notes |
|---|---|---|
| `"cpu"` | `CPUExecutionProvider` | The hermetic, deterministic default (`intra_threads: 1`). |
| `"gpu"` \| `"migraphx"` | `MIGraphXExecutionProvider` | AMD ROCm GPU. The linked runtime is probed first via `ExecutionProvider::is_available()`; a build without MIGraphX **fails open to CPU with a loud, actionable warning** — a `"gpu"` request is never silently served. |
| anything else | `CPUExecutionProvider` | Loud warning that the provider is not wired. |

### Why MIGraphX and not ROCm

ONNX Runtime **removed the `ROCMExecutionProvider` upstream in ORT 1.23**. The
AMD-supported GPU path is now the **MIGraphX execution provider**, developed in
the ROCm fork of onnxruntime (`github.com/ROCm/onnxruntime`) and bound by this
`ort` crate version as `ort::ep::MIGraphX` (behind the `migraphx` feature). On
an AMD Radeon / ROCm system, `"gpu"` therefore maps to MIGraphX.

### Build-time resolution (`lax-feature-matching`)

The workspace pins `ort = { version = "2.0.0-rc.13", features =
["migraphx", "lax-feature-matching"] }`.

- `migraphx` compiles `ort::ep::MIGraphX` in.
- The `download-binaries` prebuilt-binary resolver ships **no AMD-GPU dist**
  (its table covers CPU/CUDA/TensorRT/directml/webgpu). Without
  `lax-feature-matching`, requesting `migraphx` would fail the build; with it,
  the resolver falls back to the closest dist — the CPU build — with a debug
  log. So the default build links a CPU-only onnxruntime, and the loader's
  `is_available()` probe reports MIGraphX absent → `"gpu"` fails open to CPU.

### Making `"gpu"` actually engage

To run ONNX on the GPU, link a **MIGraphX-enabled onnxruntime** (an AMD ROCm
build) and rebuild so the loader's probe reports MIGraphX available:

- **Static**: set `ORT_LIB_PATH` to the directory holding the ROCm
  `libonnxruntime.a` (ort-sys links it before consulting `download-binaries`).
- **Dynamic**: build with `load-dynamic` and set `ORT_DYLIB_PATH` to the ROCm
  `libonnxruntime.so`.

MIGraphX is not part of a stock ROCm install; it must be built from the ROCm
fork or installed from AMD's prebuilt wheels. `make onnx-gpu-check` reports
whether the currently-linked onnxruntime exposes MIGraphX — the single fact
that decides whether `execution_provider: "gpu"` engages the GPU.

## Contracts that must not break

- **Fail-open, but loud.** A mistyped or unsupported `execution_provider`
  degrades to CPU with a loud warning; a `"gpu"` request on a runtime without
  MIGraphX logs an actionable warning (never a silent CPU run). A *declared*
  model that fails to load is a loud boot error.
- **Ort-free by default.** The config/lifecycle/registry code and its hermetic
  tests compile without the `onnx` feature; only the loader and workers are
  feature-gated.
- **Residency parity.** `Always`/pinned entries are never released; working-set
  eviction is LRU-largest against the onnx RAM budget.
- **Single loader, one provider decision.** Every ONNX model (encoder,
  two-tower, PII, ColBERT, and the generative LLM) goes through
  `OrtSessionLoader::load`, so provider selection is decided in exactly one
  place.
- **Hermetic by default.** Only `make test-live` / `make ort-test-live` /
  `make onnx-gpu-check` touch a real onnxruntime; no hermetic test probes one.

## Verification

- `cargo test -p fluent-onnx` — hermetic unit/golden suites (ort-free config
  tests + stub-loader registry tests).
- `cargo clippy -p fluent-onnx` / `cargo clippy --workspace -- -D warnings`.
- `make onnx-gpu-check` — provider-availability probe against the linked
  onnxruntime (CPU-only report expected without a MIGraphX build).
- `make ort-test-live` — real model inference (`#[ignore]` + `live-ai`).