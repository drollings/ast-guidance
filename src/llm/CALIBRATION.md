# fluent-llm calibration report — ROADMAP_20260903_LLM M10

Measured 2026-09-04 on HEAD. Locked by `src/llm/tests/calibration.rs`
(`--test calibration`, 8/8 green); any silent retune of the moved heuristics
fails those tests first. This milestone wrote no production code: it measures,
locks, and records. It blocks M11 (shim removal) and any new
caching/persisting of heuristic outputs.

Confidence vs task-value (§1) applies to every number below: producer
signals (think blocks, queue saturation) never stand in for task outcome
(budget fit, answer correctness, cache freshness), and transport
completeness (SSE framing) is neither.

## M10.1 — Token weights (task-value budget fit, not confidence)

Corpus: 200 deterministic samples (fixed-seed LCG `0x20260903`, no RNG
dependency) — 50 prose / 50 code / 50 CJK / 50 emoji-mixed, 19,588 bytes
total. Fingerprint (class sizes, byte total, first sample per class) is
locked in `m10_1_corpus_fingerprint`; the generator is the checked-in
corpus in effect.

Reference: in-test density model with cl100k-ish densities —
ASCII-alnum 0.25 (shared with production BY CONSTRUCTION, both ≈ tiktoken
English density; absolute ASCII accuracy is not independently verified,
but any ASCII retune moves the locked divergence), ASCII symbols 0.8,
CJK 1.2, emoji 1.5, whitespace 0.15, control 0. The reference is itself an
estimate, not ground-truth tiktoken: the report measures divergence
between two estimators.

| metric | measured | locked as |
|---|---|---|
| mean abs error (est vs ref) | 6.2267 tokens | `±1e-3` band |
| samples within ±20% | 93/200 (46.50%) | exact count |
| prose class mean err | 0.7950 | `±1e-3` band |
| code class mean err | 8.9100 | `±1e-3` band |
| CJK class mean err | 11.7590 | `±1e-3` band |
| emoji class mean err | 3.4430 | `±1e-3` band |

Attribution: prose agrees (shared density); code diverges (production
counts ASCII symbols at 0.25 vs tiktoken ~1 each — a ~4x undercount on
symbol-heavy code); CJK diverges systematically (0.67 vs 1.2, ~44%
under by design: "1.5 chars/token"); emoji diverges (1.0 vs 1.5).
If the bar fails, the weights stay as-is (documented estimate) — no
retune in a move PR, per milestone rule.

Truncation (`truncate_to_budget`, 196/200 samples exercised):
corpus max overshoot over budget = **10 tokens** (locked exact).
Attribution (`m10_1_truncate_multibyte_attribution`, all locked exact):

| input | estimate | budget (est/2) | overshoot |
|---|---|---|---|
| ASCII `"word "×200` | 220 | 110 | 1 (the `...` suffix) |
| CJK `"漢"×300` | 201 | 100 | **102** |
| emoji `"😀"×200` | 200 | 100 | **101** |

Cause: `truncate_to_budget` scales `text.len()` (BYTES) but keeps that
many CHARS (`chars().take(target_len)`), so multibyte truncation is a
near-no-op that appends `"..."`. ASCII holds the budget; CJK/emoji
roughly double it. The one-line fix (`chars().count()` instead of
`len()`) is follow-up work, NOT an M10 production change.
**Verdict: truncation has NOT earned budget-critical use on multibyte
text** (context-window overflow direction). ASCII-only budgeting holds
within the `...`-suffix slack.

Controls: whitespace-heavy (`"   \t\n  \t   \n "`) estimates exactly 1,
control chars (`\0\1\2`) exactly 0 — locked, no inflation.

## M10.2 — Think stripping (self-doubt artifact, not correctness)

Full-spec set (8 recall + 11 controls), via `common_core::calibration`:

| metric | measured |
|---|---|
| true positives / false negatives | 8 / 0 (recall 1.0000 — all tagged/DeepSeek forms strip) |
| false positives / true negatives | **3 / 8** |
| precision | 0.7273 (8/11) |
| FPR | 0.2727 (3/11) |
| `passes_gate()` (≥0.90 / ≤0.05) | **false** |

The 3 FPs are the roadmap-specified tick controls — inline code
carrying think tags is task content, and the stripper deletes it:

- `"run \`<think>foo</think>\` now"` → `"run \`\` now"`
- `"use \`<think>\` tag"` → `"use \`"` (unclosed marker strips to end)
- `"call \`<thinking>reason</thinking>\` twice"` → `"call \`\` twice"`

`strip_tag_pairs` matches byte subsequences with no code-span
awareness. A confident answer with no think block and a wrong answer
with a stripped block are both possible — presence/absence of a think
block says nothing about answer quality.

The M1-subset (same 8 recall + the 6 pre-existing controls, no tick
cases) is fully clean: FP 0/6, `passes_gate()` true (locked) — the
previously-earned behavior is intact.
**Verdict: think-stripping has NOT earned truncate/cache/persist trust.**
The code-span guard is follow-up work (no M10 production change by
milestone rule). Until then, stripping applies to model-output
reasoning traces only — never to user-supplied or code-carrying text
without human review.

## M10.3 — SSE framing (completeness, neither axis)

253 split-points (every byte offset as a 2-chunk split, mid-codepoint
splits included, plus full byte-by-byte feeds) over 3 multi-line
CJK/emoji payloads: all reassemble losslessly, zero U+FFFD, empty tail
buffers (locked exact count). No-`\n` chunks drain zero lines and
preserve bytes verbatim.
**Verdict: earned** — framing completeness holds; it remains never a
correctness verdict (a complete frame can carry a wrong answer).

## M10.4 — Cache identity (freshness, not endorsement)

11 probes, all green (locked individually): fresh set hits with the
stored value; key format `{model}:{64 lowercase hex}`; cross-model
same-text MISSES; unknown key MISSES; expired entry MISSES; TTL
boundary locked (`age == ttl` misses on `>=`, `age == ttl - 1` hits);
neighboring keys never cross-talk (per-model + per-request isolation);
absent/malformed backend MISSES.
**Verdict: earned** — a hit is key-equality + TTL freshness, never an
endorsement of the cached output.

## M11 implications

- SSE framing and cache identity are earned for their documented uses.
- Think-stripping and multibyte truncation are NOT earned for
  truncate/cache/persist decisions; both limitations are locked by tests
  above and must be resolved (code-span guard; `chars().count()` fix) in
  follow-up work with their own calibration proof — never silently.
- M11 (shim deletion) proceeds with these limitations recorded; the
  `#[deprecated]` shims it removes are behavior-identical copies either
  way (parity-locked in M1–M9), so shim removal changes no heuristic
  behavior.
