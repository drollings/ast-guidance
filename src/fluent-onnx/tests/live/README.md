# fluent-onnx live-AI tests

Opt-in tests that perform **real ONNX inference**. They are compiled only when
the `live-ai` feature is enabled and run exclusively via
`make test-live` / `make ort-test-live`. They never run under
`make test` / `make router-test` / `make router-mock` / CI.

## Env contract

- `ORT_LIVE_ENCODER_MODEL` — path to the Encoder model directory (contains
  `onnx/` with the `.onnx` artifacts and `tokenizer.json`). Required by every
  encoder test.
- `ORT_LIVE_ENCODER_REFERENCE` — optional; a second directory used for the
  q8-vs-reference drift band. Point it at an fp32 export when one exists; the
  on-disk q8/q4 pair is the fallback (the base Encoder export ships only
  q8/q4 today). Absent → the drift test skips.
- `ORT_LIVE_PROMPT_ROUTER_MODEL` — path to the Prompt-Router model directory
  (`onnx/` artifacts + `config.json` `head` block + `tokenizer.json`).
  Required by every two-tower test.
- `ORT_LIVE_PROMPT_ROUTER_REFERENCE` — optional; a second Prompt-Router
  directory for the q8-vs-reference probability delta. Absent → the delta
  test falls back to the on-disk q8/q4 pair.
- `ORT_LIVE_PII_MODEL` — path to the PII-Detector directory (contains the
  `model*.onnx` artifacts, `config.json` with `id2label`, `tokenizer.json`).
  Required by every PII test.
- `ORT_LIVE_PII_REFERENCE` — optional; a second PII directory for the
  quant-vs-reference span-flip measurement (the on-disk fp32 `model.onnx` is
  the reference when absent).
- `ORT_LIVE_POLICY_MODEL` — path to the Policy-Linter directory (`onnx/`
  artifacts + `config.json` `head` block + `tokenizer.json`). Required by
  every Policy-Linter test.
- `ORT_LIVE_POLICY_LABELS` — path to a policy-labels JSON file (a JSON array
  of rule strings).
- `ORT_LIVE_POLICY_REFERENCE` — optional; a second Policy-Linter directory for
  the quant-vs-reference threshold-flip measurement (falls back to the on-disk
  q4 artifact — the export ships no fp32, README-documented).
- `ORT_LIVE_LLM_MODEL` — path to the generative `CausalLm` model directory
  (contains `onnx/` with the q4 artifacts + `config.json` + `tokenizer.json`).
  Defaults to the ROADMAP wiring point
  `/ai/models/lfm2/2.6b/LiquidAI/LFM2.5-2.6B-ONNX`. Required by every LLM
  test (`llm_live.rs`): the IO-contract probe, the KV-cached decoder
  determinism test, and the grammar-constrained JSON-prefix test.

## Skip-not-fail

When the model env vars are unset the tests skip cleanly (early `return`,
never a panic). This keeps `make test-live` green on machines with no ONNX
models.

## What it exercises

- dims / non-empty embeddings from a real q8 Encoder session
- bit-identical determinism across repeat runs (`intra_op_threads=1`)
- the q8-vs-reference cosine drift band on a fixed sample (recorded; each
  consumer gates its own quantization acceptance in M5/M6)
- p50/p95-style single-call latency, recorded against a generous budget
- two-tower softmax normalization + top-1 agreement on 4 clearly-typed
  reference cases (**smoke baseline only** — the redirect evidence requires
  the ≥100-case zero-shot eval corpus, ROADMAP §2.6a)
- the Prompt-Router q8-vs-reference max-Δ probability band (README: 0.0910)
- PII-Detector recall on a known-PII golden corpus + the recorded
  quant-vs-reference span-flip rate (ROADMAP §3.2; each pre-filter gates its
  own threshold with the recorded rate)
- Policy-Linter threshold hits on a violating/benign pair + the recorded
  quant-vs-reference threshold-flip rate (ROADMAP §3.1; the README documents
  3-in-6 flips for q8-vs-fp32)