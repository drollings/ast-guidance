#!/usr/bin/env bash
# llm-boundary-check — ROADMAP_20260903_LLM M0.1 lint (advisory until M1–M8 green).
#
# Fails on:
#   1. Code importing LLM-leak symbols from `common_core` (the L1–L8
#      Table-0 surface that M1–M8 moves into `fluent-llm` / router):
#        common_core::string::{strip_think*, strip_thinking*, StreamingThinkFilter, drain_sse*}
#        common_core::tokens::*, common_core::cache::{ResponseCache, CachedResponse},
#        common_core::sqlite::{init_embedding_cache, EMBEDDING_CACHE},
#        common_core::constants::{MAX_EMBEDDING, DEFAULT_TOTAL_TIMEOUT, DEFAULT_IDLE_TIMEOUT, DEFAULT_RETRY_INTERVAL},
#        common_core::http::{error_value, add_cors, CORS_},
#        common_core::telemetry::{ToolName, ProviderCategory, FeatureName}
#      Excluded: the definition crate itself (`src/common-core/src/`,
#      `src/common-core/tests/` golden locks that M1–M8 move, and
#      `src/common-core/tests/llm_boundary.rs` which asserts the shim
#      contract through M10) and the migration target (`src/llm/`,
#      `src/router/` — their remaining old-path imports are the M1–M8
#      work list, tracked by `src/llm/tests/no_domain_imports.rs` instead).
#      This lint fires only on NEW consumers outside those trees.
#   2. `src/llm/src/**` or `src/common-core/src/**` importing domain crates
#      (router|coral|guidance|types|dag|db|ontology|rdf|spacy|wasm_ipc).
#
# Table-0 parity corpus (M0.2 baseline — goldens that must stay byte-identical
# through M0–M10; moves relocate them, never fork them):
#   src/common-core/tests/string.rs (think + SSE/CJK goldens)
#   src/common-core/tests/tokens.rs
#   src/common-core/tests/cache.rs
#   src/common-core/tests/sqlite.rs
#   src/common-core/tests/http.rs
#   src/common-core/tests/constants.rs
#   src/common-core/tests/telemetry.rs
#   src/router/tests/streaming.rs (think-block assertions)
#   src/llm/tests/context_packer.rs + src/llm/tests/openai.rs + src/llm/tests/http_class.rs
#
# Usage: bin/llm-boundary-check.sh [--deny]
#   default (no flag): advisory — prints violations, exits 0.
#   --deny:            exits 1 on any violation (enabled after M1–M8 green).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DENY=0
if [ "${1:-}" = "--deny" ]; then DENY=1; fi

FAIL=0

# ── Check 1: LLM-leak imports from common_core in new code ──────────────
LEAK_PAT='use common_core::(string::(strip_think|strip_thinking|StreamingThinkFilter|drain_sse)|tokens::|cache::(ResponseCache|CachedResponse)|sqlite::(init_embedding_cache|EMBEDDING_CACHE)|constants::(MAX_EMBEDDING|DEFAULT_TOTAL_TIMEOUT|DEFAULT_IDLE_TIMEOUT|DEFAULT_RETRY_INTERVAL)|http::(error_value|add_cors|CORS_)|telemetry::(ToolName|ProviderCategory|FeatureName))'
LEAK_HITS="$(grep -rnE "$LEAK_PAT" src/ --include='*.rs' 2>/dev/null \
  | grep -v 'src/common-core/' \
  | grep -v 'src/llm/' \
  | grep -v 'src/router/' \
  | grep -v 'src/coral/' \
  | grep -v 'src/search-vector/' \
  | grep -v 'src/bin/' \
  || true)"
if [ -n "$LEAK_HITS" ]; then
  printf '%s\n' "$LEAK_HITS"
  echo "--"
  echo "llm-boundary: NEW LLM-leak imports from common_core found above (use fluent-llm::… instead)."
  FAIL=1
fi

# ── Check 2: domain imports in llm / common-core ─────────────────────────
DOMAIN_PAT='use (fluent_)?(router|coral|guidance|guidance_core|guidance_ontology|coral_context|fluent_types|fluent_dag|fluent_db|fluent_llm|ontology|rdfs?|spacy|spacy_rs|wasm_ipc|search_vector)(::|;)'
DOMAIN_HITS="$(grep -rnE "$DOMAIN_PAT" src/llm/src/ src/common-core/src/ --include='*.rs' 2>/dev/null | grep -v '//.*use ' || true)"
if [ -n "$DOMAIN_HITS" ]; then
  printf '%s\n' "$DOMAIN_HITS"
  echo "--"
  echo "llm-boundary: domain-crate imports in src/llm/src or src/common-core/src (forbidden both directions, M0.1)."
  FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
  if [ "$DENY" -eq 1 ]; then
    echo "llm-boundary: FAIL (--deny)"
    exit 1
  fi
  echo "llm-boundary: advisory warnings above (exit 0 until M1–M8 green)."
  exit 0
fi
echo "llm-boundary: clean."
