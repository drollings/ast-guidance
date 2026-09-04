# spacy-rs live-AI tests

Opt-in tests that perform **real AI inference**. They are compiled only when
the `live-ai` feature is enabled and run exclusively via
`make test-live` / `make spacy-test-live`. They never run under
`make test` / `make router-test` / `make router-mock` / CI.

## Env contract

- `SPACY_LIVE_LLM_URL` — an OpenAI-compatible chat-completions endpoint that
  is **finetuned to emit the §10.1 annotation JSON array** (the
  [`AnnotationRecord::contract`](../../src/llm.rs) schema) for a given token
  list. Plain chat models will not produce a valid array and the test will
  fail the 7-check gate.

## Skip-not-fail

When `SPACY_LIVE_LLM_URL` is unset the test skips cleanly (early `return`,
never a panic). This keeps `make test-live` green on machines with no
inference endpoint.

## What it exercises

`NlpPipeline` → `LlmRequestQueue` (ResultPool-backed fan-out) → the §10.1
prompt built from the deterministic token list → the 7-check gate → the
stage DAG under `SupervisedBatch`. Assertions are structural only.
