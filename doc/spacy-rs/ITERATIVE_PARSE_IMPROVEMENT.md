# Iterative Deterministic Parse Improvement (ArcEager)

**Goal:** raise the hermetic parse-bench scores (`make spacy-parse-benchmark`)
iteration by iteration — either by parsing more accurately **or** by accurately
flagging ambiguity for LLM enrichment. Both outcomes count; silent confident
misparses count against you.

**Non-goals:** touching the LLM/refiner path, changing reference annotations to
match the parser, lowering floors, adding dependencies to `spacy-rs`.

---

## 1. Read this first (mandatory)

Every iteration begins by re-reading, not by coding:

- `doc/skills/common-core/SKILL.md`, `doc/skills/fluent-wvr/SKILL.md`,
  `doc/skills/fluent-concurrency/SKILL.md` — zero-cost-idiom, FSM, and
  pass-shape rules. Oracle edits are hot-path code; the idioms are binding.
- `doc/skills/interlingua/SKILL.md` — the ID scheme and stage-DAG contracts
  your rules must not disturb.
- `src/spacy-rs/src/arc_eager.rs` header (`Honest framing`, `Head convention`,
  `POS heuristics`) and the oracle (`DeterministicOracle::score`,
  ~line 440–560) — the mechanism you are changing.
- `src/spacy-rs/src/labels.rs` — the closed dep inventory. `mark`, `cop`,
  `ccomp`, `advcl`, `parataxis`, `relcl`, `neg`, `conj`, `acomp`, `oprd`
  **already exist**; most have no oracle rule producing them (they fall to
  `_ => 1.0`). Check the inventory before inventing a label.
- Shared primitives before new code: `src/common-core`, `src/fluent-wvr`,
  `src/fluent-concurrency`. Do not remove anything from them; do not add
  without human approval.

## 2. The loop (follow it literally)

```
1. make spacy-parse-benchmark
   → read the per-category scoreboard. Pick the worst category that is NOT
     already at its explainable ceiling (see §5).
2. Diagnose ONE error class inside that category.
   → run the inspector: cargo run -p spacy-rs --example parse -- "<sentence>"
   → map each misparse to a mechanism: missing oracle rule? POS fallback?
     missing label? genuine ambiguity? Name the exact `score()` arm or
     `infer_pos` branch responsible. If you cannot name it, you are not ready.
3. Write the failing golden case FIRST in tests/arceager_golden.rs
   (attachment-level assertions: exact dep + head for the tokens at issue)
   plus a must-NOT-fire control (a nearby sentence the rule must leave alone).
   → cargo test -p spacy-rs --test arceager_golden → confirm red.
4. Implement the smallest deterministic change that greens it:
   oracle weight arm, POS contextual rule, or validator signal (§4).
5. make spacy-parse-benchmark
   → target category must rise; EVERY other category must hold its floor.
     Any regression anywhere = revert the rule, not the floor.
6. Re-pin floors: recompute exact k/N counts from the scoreboard output and
   update tests/data/parse_bench.floors.json (full precision, never rounded).
7. cargo test -p spacy-rs && cargo clippy --workspace -- -D warnings
   && make lint-live-ai
```

One error class per iteration. One rule per change. If step 5 shows a
regression, the rule is wrong — shrink its scope (lexical guard, POS guard,
lookahead guard), don't widen anything else to compensate.

## 3. Metric semantics (read before chasing hundredths)

- **UPOS**: token POS string equality. Ceiling-setter: the oracle keys every
  arc off POS pairs, so +1 UPOS ≈ +1 UAS downstream.
- **UAS**: relative-head-offset equality (`token.i + head == head_index`).
  References use the parser's relative convention (root `head == 0`).
- **LAS**: head AND dep-label equality, in the parser's closed vocabulary.
- **lemma**: counted only where the ref pins one; omitted lemmas auto-credit.
- Scoreboard prints `k/N` counts — floors are exact fractions, comparison
  epsilon is `1e-9`. Per-category N is 29–48 tokens: read ordering and
  halves, never hundredths.
- `scored_items` must equal the dataset size (66). If it drops, you have a
  tokenization drift, not an improvement — investigate, never adjust the
  floor down.

## 4. The two tracks: repair vs. flag

**Track A — repair (preferred when the error is systematic).**
Closed-class syntax is finite and fixable: subordinators (`mark`),
contractions (`neg`), semicolons (`parataxis`), copular contexts (`cop` +
predicate POS), contextual verb detection (bare infinitive after
causative/perception verbs, `to` + X, modal/aux + X). Each fix is an oracle
arm with lexical/POS/lookahead guards. Priority order, by measured product
impact: verb detection → contractions → copulas → `mark`/subordination →
`parataxis`/`relcl` (see the benchmark's category ranking).

**Track B — flag (mandatory when the error is genuine ambiguity).**
PP attachment, coordination scope, garden paths, and lexically underdetermined
tags are information-theoretically unfixable by rules — do NOT write a rule
that guesses. Instead, emit a deterministic uncertainty signal the ladder can
act on: near-tie margins (already plumbed to `ParseConfidence`), a structural
validator finding (e.g. finite verb without `nsubj`), or a `RefineReason` that
fires the refiner. A flagged ambiguity that routes to the LLM is a success;
measure it by the must-fire control in step 3 (ambiguous input → signal
present), paired with a must-NOT-fire control (clean input → signal absent).

## 5. Known ceilings (do not chase these)

- Open-class verb detection is mitigable, never closable — the closed verb
  list has a documented false-negative class. Contextual patterns raise the
  ceiling; they don't remove it.
- `role_coverage` is non-monotonic in quality (wrong labels satisfy it).
  Never use it as evidence that a parse is good; prefer the bench LAS.
- The bench refs are reviewed drafts, not a gold treebank: structure-validated
  100%, content spot-checked. If a ref is linguistically wrong, fix the REF
  (with a written justification in your summary), then re-pin — never tune a
  rule to match a bad ref, and never "fix" a ref to match the parser.

## 6. Worked example (the shape of a good iteration)

Target: `contraction` LAS 0.152. Diagnosis: `n't` → POS `X` (no rule),
everything downstream attaches `dep`. Golden case first: `Don't help them.`
asserting `n't → part/neg → help`, plus control `Do help them.` (no `neg`
without `n't`). Change: one closed rule — `n't` (and `not` after aux) → PART
with a `neg` arc to the following verb, scored above the `dep` fallback.
Verify: contraction LAS rises, all other floors hold (especially `command`,
which shares imperatives), re-pin floors, full gate green.

## 7. Anti-patterns (each has happened or nearly has)

- Lowering a floor to make red green. Floors move up or stay.
- Editing a ref to agree with the parser. Refs answer to UD, not to output.
- A rule without a must-NOT-fire control. Unscoped rules always leak.
- Rounding floors to 3 decimals (`0.594 < 0.594` false-reds the ratchet).
- Touching `tests/data/parse_bench.json` item texts to dodge drift — drift
  means the tokenizer changed; treat it as a signal, not an inconvenience.
- New `spacy-rs` dependencies, or imports from `guidance`/`router`/`coral`/
  `wasm_ipc`/`ontology`/`rdf`. The crate's hermeticity is load-bearing;
  `cargo check -p spacy-rs` must show no such edge.

## 8. Command reference

```sh
make spacy-parse-benchmark                                   # the loop's meter
cargo run -p spacy-rs --example parse -- "<sentence>"       # inspector
cargo test -p spacy-rs --test arceager_golden               # golden cases
cargo test -p spacy-rs                                      # hermetic suite
cargo clippy --workspace -- -D warnings && make lint-live-ai
# New references (gated; drafts only — review, correct, pin, re-bench):
make spacy-test-live   # requires SPACY_LIVE_LLM_URL (+ SPACY_LIVE_LLM_MODEL)
```

**Definition of done for one iteration:** target category up, all floors
held, floors re-pinned at exact counts, golden case + both controls green,
full gate green, and a summary stating which track (repair/flag) the change
belongs to and why the next-worst category is or isn't actionable.
