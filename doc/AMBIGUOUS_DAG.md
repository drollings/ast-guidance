# AmbiguousDependencyResolver

Technical reference for the yamake-old.py-style provider disambiguation resolver,
located at `src/dag/src/ambiguous_resolver.rs`.

## Purpose

`AmbiguousDependencyResolver` resolves dependency graphs where a single
capability (e.g. `animal`, `confuse`, `feline`) can be provided by multiple
targets. The classic `DependencyResolver` includes **all** providers in the
execution plan — which is correct for build graphs where each provider
produces a distinct artifact, but wrong for abstract capability graphs where
only one concrete provider should be selected.

The resolver applies set-logic narrowing rules ported from
`yamake-old.py:Enqueue()` to select exactly one provider when possible, and
reports structured ambiguity when narrowing fails.

## Architecture

```
AmbiguousDependencyResolver
 ├── TargetRegistry       (target metadata, depends/provides BitVecs)
 ├── CapabilityRegistry   (string → bit index interning)
 └── DependencyResolver   (Kahn's topological sort — composed, not duplicated)

resolve(targets):
  1. Fast-path: if no abstract targets and single-provider-per-capability,
     delegate to inner DependencyResolver directly (zero overhead).
  2. Disambiguation loop: iterate over the combined target set, expanding
     providers for unsatisfied dependencies. When multiple providers exist
     for a capability, call narrow_providers().
  3. Expand transitive closure: walk depends chains from the resolved set,
     adding any remaining implicit providers.
  4. Feed the final target set to DependencyResolver::resolve() for Kahn's
     topological sort + cycle detection.
```

## Narrowing rules (`narrow_providers`)

Given a list of candidate providers for a single capability, the function
applies these filters in order, stopping when exactly one candidate remains:

| Priority | Filter | Description |
|---|---|---|
| 1 | Essential | If any candidates have `essential: true`, drop non-essentials |
| 2 | Strict sat | Keep candidates whose **all** deps are in the current `full_provides` set |
| 3 | Locality | Keep candidates whose **any** dep is in `full_provides` (the locality heuristic from yamake-old.py) |
| 4 | No-dep | Prefer candidates with **no** dependencies |

If multiple candidates survive all filters, the resolver returns
`AmbiguousDependency { name, candidates }` with human-readable names.

## Self-provision of concrete targets

The `yamake_loader` adds an implicit self-provision bit for every concrete
(File/Phony) target. This bridges yamake-old.py's model (where `depends`
references Target objects directly) with the Rust model (where `depends`
references capability bits).

Concrete targets self-provide because other targets can name them in
`depends` lists. Example: `confuse_a_cat` depends on `cat`, `stage`, `cannon`
— these are all concrete targets that must appear in the build plan.

Abstract targets do **not** self-provide. Self-provision of abstracts would
create cycles: `confuse_a_cat` provides `agency`, `agency` provides `staff`,
`staff` depends on `agency` (through `stage_hands → stage → confuse_a_cat`).

## Yamake data model

The `yamake.json` file at the repo root defines 52 targets covering the test
scenarios from `yamake.yaml`. Each target has:

| Field | Type | Description |
|---|---|---|
| `id` | i64 | Unique numeric ID (also used as bit index) |
| `name` | string | Target name |
| `target_type` | `"file"` \| `"abstract"` \| `"phony"` | Concrete (exists) vs abstract (capability only) |
| `depends` | string[] | Target/capability names this target requires |
| `provides` | string[] | Capability names this target provides |
| `essential` | bool | Whether this target participates in narrowing preference |

The loader (`src/dag/src/yamake_loader.rs`) interns all names through
`CapabilityRegistry`, converts string lists to `BitVec`, and auto-adds
self-provision for concrete targets.

## How to test

### Unit tests

```bash
cargo test -p fluent-dag
```

74 tests covering:
- `test_linear_chain_no_ambiguity` — pure Kahn's path, no abstraction
- `test_disambiguate_with_locality_preference` — `confuse+bee → distract_a_bee`, `confuse+stoat → stun_a_stoat`
- `test_ambiguity_reported_when_multiple_providers` — two providers for `animal`, no context to pick
- `test_circular_dependency_detected` — cycle detection preserved
- `test_perf_deep_chain_no_abstraction` — 200-node linear chain in <100ms

### E2E test battery (against yamake.json)

```bash
cargo run --bin yamake-coral -- test
```

21 scenarios validating both classic and ambiguous resolvers against the full
yamake target graph.

### Compare classic vs ambiguous for a specific input

```bash
cargo run --bin yamake-coral -- compare confuse bee
cargo run --bin yamake-coral -- compare confuse stoat
cargo run --bin yamake-coral -- compare confuse_a_cat
```

### Run a single resolver directly

```bash
cargo run --bin yamake-coral -- classic confuse bee
cargo run --bin yamake-coral -- ambiguous confuse bee
```

### Release-mode performance

```bash
cargo build --bin yamake-coral --release
target/release/yamake-coral compare confuse bee
```

## Key test scenarios and expected results

| Input | Classic | Ambiguous | Notes |
|---|---|---|---|
| `confuse bee` | 2 targets (seeds) | 4 targets (includes `distract_a_bee`) | Locality: bee provides `insect`+`color_vision` |
| `confuse stoat` | 2 targets (seeds) | 3 targets (includes `stun_a_stoat`) | Locality: stoat provides `mammal` |
| `confuse_a_cat` | 15 targets (full tree) | 15 targets (100% overlap) | No ambiguity — single concrete path |
| `distract_a_bee` | 3 targets | 3 targets (identical) | Zero-overhead fast path |
| `confuse puma` | 2 targets | Ambiguity error for `staff` | `puzzle_a_puma` requires `magic_tricks` which requires `staff` |

## Error types

`AmbiguousDependencyResolver` uses the same `ResolverError` enum as
`DependencyResolver` (`src/common-core/src/error.rs`), adding one variant:

```rust
#[error("ambiguous dependency: '{name}' could be provided by {}", candidates.join(", "))]
AmbiguousDependency {
    name: String,
    candidates: Vec<String>,
}
```

## API

```rust
use fluent_dag::ambiguous_resolver::AmbiguousDependencyResolver;
use fluent_dag::yamake_loader::load_yamake_config;

let json = std::fs::read_to_string("yamake.json")?;
let (registry, caps) = load_yamake_config(&json);

let resolver = AmbiguousDependencyResolver::new(&registry, &caps)
    .with_strict(true);

match resolver.resolve(&["confuse", "bee"]) {
    Ok(plan) => println!("{:?}", plan.target_names),
    Err(ResolverError::AmbiguousDependency { name, candidates }) => {
        eprintln!("ambiguous: {name} → {}", candidates.join(", "));
    }
    Err(e) => eprintln!("error: {e}"),
}
```

## Design notes

- `AmbiguousDependencyResolver` composes `DependencyResolver` for Kahn's sort;
  no topological logic is duplicated.
- The fast-path check (`needs_disambig`) scans the transitive closure for
  any capability with multiple providers. If none exist, the resolver
  delegates to `DependencyResolver::resolve` directly with zero overhead
  (confirmed by benchmarks: `distract_a_bee` resolves in ~4µs on both paths).
- `narrow_providers` sorts by name before applying filters to ensure
  deterministic output across platforms.
- The `while changed` loop in `resolve()` iterates until no new providers
  are added to the combined set. For the yamake test graph, this converges
  in 1-3 iterations.
