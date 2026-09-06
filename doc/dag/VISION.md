# Unified Dependency Resolver — From Build Graphs to LLM Workflow Orchestration

> **API reference**: Concrete type signatures, construction examples, and
> benchmarking tests live in `doc/skills/dag/SKILL.md`. This document describes
> the abstract algorithm and design rationale.

## What Makes This Resolver Different

Traditional dependency resolvers — the kind that powers `make`, `cargo`, or
`npm` — assume that every dependency name maps to exactly one thing. When
you say `depends: ["parser", "codegen"]`, the resolver finds the single
target named `parser` and the single target named `codegen`, ensures they
run first, and produces an execution order.

That model breaks when dependencies are described by **capability** rather
than by identity. An LLM workflow might declare: `depends: ["code_review",
"translation", "data_extraction"]` — where none of those are the name of a
specific tool, but rather descriptions of **what the target needs**. Multiple
tools can provide `code_review`. Multiple agents can provide `translation`.
The resolver needs to choose **which one** to use, not just include them all.

This resolver extends Kahn's topological sort (the classic algorithm for
dependency ordering) with a **provider-selection policy** that can:
- **Include all providers** — the classic build-graph semantics (every
  provider produces a distinct artifact, all are needed)
- **Narrow to one provider** — the capability-graph semantics (only one
  concrete implementation should be selected per abstract capability)

The NarrowOne mode is the novel contribution. It applies a narrowing
pipeline — essential-first, strict-satisfaction, locality, no-dep-preference —
to decide which provider to keep when multiple targets claim the same
capability. If narrowing cannot reduce the field to exactly one, it reports
a structured `AmbiguousDependency` error with the full list of candidates.

## Abstract vs Concrete Targets

A **concrete target** is a named, executable thing — a script, a binary, a
WASM plugin, an API endpoint, a database migration. It has an identity and
can be invoked directly. In the yamake model these are `File` or `Phony`
targets. Concrete targets **self-provide**: if a target named `code_review`
is concrete, any other target that depends on `"code_review"` will find it.

An **abstract target** is a pure capability descriptor. It has no executable
body — it names a capability and, implicitly, the set of concrete targets
that provide it. Abstract targets do **not** self-provide (doing so would
create cycles in the dependency graph). They act as the semantic glue
between "what I need" and "who provides it."

```
Concrete:  "rust_analyzer"  provides: ["lsp", "code_analysis"]
           "pyright"       provides: ["lsp", "type_checking"]

Abstract:  "lsp"           depends: []  (just a capability label)
```

When a workflow step depends on `"lsp"`, the resolver must pick one of
`rust_analyzer` or `pyright` — it cannot run both (they'd fight over the
same editor socket). The narrow-one policy makes this selection.

## What Becomes Possible

### LLM Workflow Orchestration

The primary use case this resolver unlocks is **LLM-driven workflow
construction**. An LLM can describe a multi-step pipeline in terms of
capabilities it needs at each step:

```
Step 1:  "extract_entities"     depends: ["raw_text"]
         provides: ["entities", "relationships"]

Step 2:  "summarize"            depends: ["entities"]
         provides: ["summary", "key_points"]

Step 3:  "translate"            depends: ["summary"]
         provides: ["localized_output"]
```

The LLM describes **what** each step needs and produces, not **which**
implementation to use. A separate registry maps concrete tool instances
(which may be WASM plugins, remote APIs, or native functions) to the
capabilities they provide. The resolver's job is to:

1. Find all transitive dependencies (Kahn's topological ordering)
2. For each capability with multiple providers, narrow to exactly one
3. Report structured ambiguity when the LLM's description is under-
   specified ("I need `code_review` but there are 7 tools for that —
   which one?")

### Dynamic Tool Composition

Tools can be registered and deregistered at runtime. A WASM plugin that
provides `"data_visualization"` can be loaded dynamically, and the resolver
will pick it when a workflow step needs that capability — without any
recompilation or static wiring.

### Multi-Model Agent Pipelines

Different LLM backends (OpenAI, Anthropic, local Ollama models) provide
the same capabilities (e.g., `"text_generation"`, `"function_calling"`,
`"embedding"`) with different cost/latency/quality tradeoffs. The narrowing
pipeline can express preferences: prefer essential targets, prefer targets
whose own dependencies are already satisfied (locality), prefer zero-
dependency targets. This lets the resolver be the dispatch layer in an
agent orchestration system.

### Self-Healing Workflow Graphs

When a provider is removed (a plugin crashes, an API goes down, a tool is
deprecated), the resolver can re-resolve the same workflow and find an
alternative provider for each contested capability — without changing the
LLM-generated workflow description.

---

## Kahn's Algorithm — The Backbone

Kahn's algorithm is the classic method for topological sorting of directed
acyclic graphs. It solves one problem and one problem only: **given a set
of nodes where some must run before others, produce a linear order that
respects every dependency.**

```
Algorithm (Kahn, 1962):
  1. Count in-degree (number of incoming edges) for every node.
  2. Emit all nodes with in-degree 0 (they have no prerequisites).
  3. When a node is emitted, decrement the in-degree of every node
     that depends on it.
  4. If a node's in-degree reaches 0, it's ready to emit.
  5. If the graph contains a cycle, some nodes will never reach
     in-degree 0 → CircularDependency error.
```

This resolver uses Kahn's algorithm as its final step. After the
transitive closure is computed and any narrowing decisions are made, Kahn's
produces the execution order. The algorithm itself is **unchanged** from
classic build systems — what's new is what feeds into it.

**What Kahn's does NOT do:**

- It does not select between alternative providers — it runs every node it
  receives.
- It does not understand capabilities — it only understands edges.
- It does not handle ambiguity — cycles are the only error it detects.

The narrowing layer sits **on top** of Kahn's and answers the questions
Kahn's cannot: "Which provider should I use?" and "What do I do when I
can't decide?"

---

## Narrowing — Selecting One Provider Among Many

When a capability has multiple providers, the resolver applies a four-stage
filter pipeline. Each stage can remove candidates. After a stage removes at
least one candidate, the pipeline continues with the reduced set. The
pipeline stops as soon as one candidate remains.

### Stage 1 — Essential

If any candidate is marked `essential: true`, every non-essential candidate
is dropped. Essential marks a target as "must-use when available." If all
candidates share the same essential status, no one is removed.

### Stage 2 — Strict Satisfaction

Keep only candidates whose **every** dependency is already present in the
accumulated `full_provides` set. A candidate that depends on nothing
(no-dependency) always passes this stage — it's already fully satisfied.
This stage prefers targets that can run immediately because everything they
need is already available.

### Stage 3 — Locality

Keep only candidates that have **at least one** dependency already in
`full_provides`. Unlike Stage 2, this stage explicitly **excludes**
no-dependency candidates. This is the locality heuristic from
yamake-old.py: prefer providers that are "close to" work already done,
because in real systems locality predicts relevance.

### Stage 4 — No-Dependency (Strict Reduction)

If more than one candidate remains and some have no dependencies at all,
replace the set with only the no-dependency candidates. This only fires
when it **strictly reduces** the candidate count — if all remaining
candidates already have no dependencies, nothing changes.

### When Narrowing Fails

If all four stages run and more than one candidate remains, the resolver
returns an `AmbiguousDependency` error containing:

```
ambiguous dependency: 'code_review' could be provided by
  review_board, pair_programmer, linter_bot
```

The error is **structured** — callers receive the capability name and the
full list of remaining candidates — rather than a generic "too many
providers" message. This lets orchestrators fall back to LLM re-prompting
("I found 3 tools for code_review — which one fits this PR's context?"),
human-in-the-loop selection, or default-preference heuristics.

### Why Single-Provider Isn't Always the Answer

Sometimes narrowing to one provider is **impossible by design**: the
capability is genuinely ambiguous in the current context. For example, an
LLM might describe a step needing `"embedding"` — and the registry has
`openai_embedding`, `local_sentence_transformer`, and `cohere_embedding`,
all with no dependencies and none marked essential. The narrowing pipeline
exhausts its stages and reports ambiguity.

In this case the orchestrator has options:
- Re-prompt the LLM with the candidate list for a more specific description
- Use a default-preference ranking (fastest, cheapest, most accurate)
- Ask a human operator to choose
- Expand the workflow with an additional step that disambiguates

This is not a failure mode — it's the resolver telling you exactly where
the specification is under-determined, with enough information to fix it.

---

## Self-Provision Model

A concrete target (File or Phony) automatically provides a capability named
after itself. This bridges two models of dependency:

- **Name-based**: A step says `depends: ["spell_check"]` meaning "run the
  target literally named spell_check."
- **Capability-based**: A step says `depends: ["spell_check"]` meaning
  "I need spell-checking capability, provide it however you like."

Self-provision makes name-based dependencies work without explicit
registration: if `spell_check` is a concrete target, the resolver finds it
by the name-match fallback when no explicit capability provider is found.

Abstract targets do **not** self-provision. A target named `"spell_check"`
that is abstract represents a pure capability descriptor, not an
executable. Adding self-provision to abstracts would create cycles (the
abstract provides itself, depends on itself, and the resolver loops).

---

## Data Model

Every registered target has these fields:

| Field | Type | Description |
|---|---|---|
| `id` | i64 | Unique numeric ID (bit index in capability vectors) |
| `name` | string | Target name (also used for implicit-name fallback) |
| `target_type` | `"file"` \| `"abstract"` \| `"phony"` | Concrete or abstract |
| `depends` | string[] | Capabilities this target requires |
| `provides` | string[] | Capabilities this target provides |
| `essential` | bool | Preference marker for narrowing (Stage 1) |

Names are interned through a `CapabilityRegistry` which assigns each unique
string a stable bit index. Dependencies and provides are stored as
compact `BitVec` sets — membership tests are O(1) bit operations, not
string comparisons.

---

## Architecture

```
DependencyResolver
  ├── TargetRegistry       (target metadata, depends/provides BitVecs)
  ├── CapabilityRegistry   (string → bit index interning)
  ├── closure module       transitive-closure DFS (shared primitive)
  ├── narrowing module     narrowing pipeline + error construction
  └── Kahn's sort          topological ordering (classic algorithm)

resolve(targets):
  1. Validate seeds exist → error on unknown names. Empty → empty plan.
  2. Fast-path: compute full closure; if no ambiguity exists in the
     closure (every capability has ≤1 provider and no abstract seeds),
     delegate directly to Kahn's — zero narrowing overhead.
  3. All policy: transitive-closure → Kahn's sort. Classic semantics.
  4. NarrowOne fixpoint loop:
     - Expand the resolved set by satisfying each target's dependencies.
     - 0 providers → strict error, or implicit name fallback, or skip
       (non-strict mode).
     - 1 provider → add it unconditionally.
     - N providers → run narrowing pipeline. Winner added, losers
       recorded in a persistent rejected set.
     - Abstract targets get their own self-provision narrowing branch.
  5. Final closure expansion: re-run the transitive closure with the
     satisfied and rejected guards active. This ensures a narrowing
     loser cannot sneak back in through a different dependency path
     that leads to the same capability.
  6. Kahn's topological sort → ExecutionPlan (order + target names).
```

### The Durability Invariant

Once a target is rejected by narrowing for a given capability, it is
recorded in a `rejected` bitset. The final transitive closure checks this
set and skips any rejected target. This prevents the "include-all leak"
where a target that lost narrowing for capability X is re-discovered
through a different dependency path that also leads to X.

Rejection is **capability-scoped**, not target-scoped. A target rejected
for capability X can still enter the plan through a different capability Y
that it uniquely provides and that no narrowing decision has excluded. This
matters in LLM workflows where a single tool may provide multiple
capabilities.

---

## The Resolver API

```rust
use fluent_dag::resolver::{DependencyResolver, ProviderSelection};
use fluent_dag::yamake_loader::load_yamake_config;

let json = std::fs::read_to_string("workflow_targets.json")?;
let (registry, caps) = load_yamake_config(&json);

// NarrowOne — for LLM workflows and capability graphs
let resolver = DependencyResolver::with_narrowing(&registry, &caps)
    .with_strict(true);

match resolver.resolve(&["extract_entities", "translate"]) {
    Ok(plan) => {
        // plan.order: topologically-sorted target ids
        // plan.target_names: ["preprocess", "extract_entities",
        //                     "summarize", "translate"]
        for name in &plan.target_names {
            dispatch_workflow_step(name)?;
        }
    }
    Err(ResolverError::AmbiguousDependency { name, candidates }) => {
        // The LLM's description was under-determined.
        // Re-prompt with the candidate list for clarification.
        eprintln!("cannot decide which '{name}' provider to use: {candidates:?}");
        ask_llm_to_disambiguate(&name, &candidates)?;
    }
    Err(ResolverError::MissingDependency(msg)) => {
        // A needed capability has no provider at all.
        eprintln!("{msg}");
        // Could fall back to LLM generating a new tool for this.
    }
    Err(e) => eprintln!("resolution error: {e}"),
}

// ProviderSelection::All — classic build-graph semantics
let classic = DependencyResolver::new(&registry);
let plan = classic.resolve(&["build"]).unwrap();
```

### Policy Selection

| `ProviderSelection` | Semantics | Use case |
|---|---|---|
| `All` (default) | Include every provider of every capability | Build graphs, compilation pipelines, data processing DAGs where all outputs are needed |
| `NarrowOne` | Pick exactly one provider per contested capability | LLM workflows, agent orchestration, tool selection, capability graphs |

`NarrowOne` without a `CapabilityRegistry` is a programming error — the
resolver returns `Err(MissingDependency(...))` rather than panicking, so
the caller can provide a clear diagnostic.

---

## Error Types

```rust
/// A capability has no registered provider.
/// e.g., step needs "code_review" but no tool provides it.
MissingDependency(String)

/// Multiple providers survive all narrowing stages.
/// e.g., 3 tools claim "code_review" and none can be preferred.
AmbiguousDependency {
    name: String,           // the contested capability
    candidates: Vec<String>, // providers that remain after narrowing
}

/// A requested target name does not exist in the registry.
TargetNotFound(String)

/// The dependency graph contains a cycle (detected by Kahn's).
CircularDependency
```

---

## How to Test

```bash
# Unit tests (96 tests)
cargo test -p fluent-dag

# Full E2E battery (21 scenarios)
cargo run --bin yamake-coral -- test

# Compare policies for specific inputs
cargo run --bin yamake-coral -- compare confuse bee
cargo run --bin yamake-coral -- compare confuse stoat
cargo run --bin yamake-coral -- compare confuse_a_cat

# Run a single policy
cargo run --bin yamake-coral -- classic confuse bee
cargo run --bin yamake-coral -- ambiguous confuse bee
```

### Key scenarios

| Input | `All` policy | `NarrowOne` policy | What it demonstrates |
|---|---|---|---|
| `confuse bee` | 2 targets (seeds) | 4 targets (selects `distract_a_bee`) | Locality picks the provider whose own deps are satisfied by `bee` |
| `confuse stoat` | 2 targets (seeds) | 3 targets (selects `stun_a_stoat`) | Same locality heuristic with different context |
| `confuse_a_cat` | 15 targets (full tree) | Ambiguity error for `cognitive` | 4 providers for `cognitive` cannot be disambiguated |
| `distract_a_bee` | 3 targets | 3 targets (identical) | Zero-overhead fast path — no contested capabilities |
| `confuse puma` | 2 targets | Ambiguity error for `staff` | `puzzle_a_puma` requires `magic_tricks` which requires `staff` — 2 providers |

### Performance

The resolver is benchmarked at three levels:
- **Linear chain (200 nodes)**: resolves in <10ms, ~0.05µs/node
- **Breadth-first fan-out (100 leaves)**: ~1400µs total, ~14µs/node
- **Deep diamond (height=80)**: ~3400µs total, 240 nodes resolved

The fast-path check (no contested capabilities + no abstract seeds)
delegates directly to Kahn's with zero narrowing overhead — measured at
~4µs for `distract_a_bee` on both `All` and `NarrowOne` paths.

---

## Design Properties

- **One algorithm, two policies**: The same resolver handles both build
  graphs and capability graphs. Switching between them is a single enum
  value.
- **Deterministic output**: Narrowing sorts candidates by name before
  applying filters. The same input always produces the same output.
- **Fast-path delegation**: When no ambiguity exists, NarrowOne produces
  identical output to All with identical performance.
- **Structured errors**: Ambiguity errors carry the candidate list, not
  just a message. Callers can use this information for re-prompting,
  human-in-the-loop decisions, or automated fallback strategies.
- **Durability guard**: Once a provider loses narrowing for a capability,
  it stays excluded — even if the transitive closure would rediscover it
  through a different dependency path.
- **Implicit name fallback**: A concrete target whose name matches a
  capability name is automatically found as a provider, bridging
  name-based and capability-based dependency models.
