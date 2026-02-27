# Insights Inbox

Reusable knowledge acquired during workflows. Run `make document` to promote into `guidance/skills/`.
Unprocessed items are surfaced automatically by `make explain QUERY=<term>` when relevant.

<!-- Add new insights as bullets below:
- CozoDB CozoScript prefix distinction: Data manipulation / DDL uses single `:` prefix (e.g. `:create`, `:put`, `?[...] :=`). System/admin commands use double `::` prefix (e.g. `::index create`, `::relations`, `::columns`). Using `:index` instead of `::index` causes "unexpected input at 0..0" parse errors.
- CozoDB graph-algo build: graph_builder 0.4.0 (pulled in by `graph-algo` feature) has a rayon API incompatibility — `Box<[T]>::into_par_iter()` now yields `IntoIter` not `slice::Iter`, breaking `.copied()`. Fix: patch `graph_builder/src/input/edgelist.rs` to use `.par_iter().copied()` and override via `[patch.crates-io]` in the workspace Cargo.toml. Build with `--features minimal,graph-algo`.
- CozoDB build: cozo-core uses rayon only when graph-algo feature is enabled. Without it, rayon::spawn and par_iter calls in lib.rs and query/eval.rs must be guarded with `#[cfg(all(not(target_arch = "wasm32"), feature = "graph-algo"))]` to compile cleanly with `--features minimal`.
- CozoDB u128 storage: CozoDB has no INT128 native type. Store Zig u128 (UUIDv7, bitmasks) as two Int columns (hi: i64, lo: i64). NodeId.fromU128/toU128 in db.zig handles the split via @bitCast on u64.
- CozoDB KNN at edge scale: No native ANN index in CozoDB. Fetch all rows with List<Float> embedding column, compute cosine distance in Zig (dot product / norm product), sort and take top-K. Correct and fast for ≤100K nodes.
- Hybrid Resolution: The @popCount pattern for capability distance (§2.3) replaces expensive Python scoring loops with a single hardware instruction. The key invariant: `missing = depends_mask & ~available_traits`, then `@popCount(missing)` gives the Hamming distance to satisfiability.
- ECS over OOP: Using flat Zig structs with u128 IntegerBitSet masks instead of Python class hierarchies eliminates pointer chasing and enables SIMD-friendly bulk operations on the mask array.
- cmdExplore M2 architecture: `ExploreDoc` (owned dupe), `runGather` (struct-fn trick for Zig "closure"), `docMatchesTerms` (satisfied-terms tracking), `extractSourceExcerpt` (stops at top-level decl), `grepFile` (case-insensitive, skips comment lines). Dedup uses `StringHashMapUnmanaged` keyed on `ed.source` (stable pointer in ExploreDoc), not the transient abs path from `std.fs.path.join` (which has a defer free). `[THINK]` / `<think>` / preamble stripping is in `src/common/llm.zig`.
- ArrayList toOwnedSlice + defer deinit double-free: After calling `list.toOwnedSlice(allocator)`, the ArrayList no longer owns the buffer. A `defer list.deinit(allocator)` placed before the `toOwnedSlice` call will fire and free the transferred memory, silently zeroing the result. Fix: remove the `defer deinit` when `toOwnedSlice` is the intended exit path.

