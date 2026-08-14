---
name: subagent
description: Deterministic-First FSM Subagent for Edge LLMs — pattern compliance reference
---

# Subagent FSM — Pattern Compliance Reference

**For AI coders and human engineers working on the `src/subagent/` module.**

---

## Architecture

The subagent implements a deterministic-first FSM that orchestrates tool calls for `guidance todo run`. It resolves 70%+ of iterations without any LLM call by pattern-matching checklist items to tool actions, filling parameters via in-process `guidance explain`, and routing unknown items through a single grammar-constrained batch LLM call (O(1) per batch, not O(N) per item).

```
INTAKE → CLASSIFY → ROUTE → VALIDATE → EXECUTE → REFLECT → SYNTH → INTAKE
   │         │         │                                 │
   │    unknown items   │                            scratchpad
   │    batch LLM       │                            accumulates
   │         │          unfilled                       observations
   │    batch_classify  params                         │
   │                    → route_llm                    │
   │                    → LLM infill                  │
```

---

## Module Layout

| File | Purpose | Pattern |
|------|---------|---------|
| `types.zig` | FsmState, ActionType, ToolParams, SubagentConfig, IterationProfile | Typed Opaque Handles, Value Types |
| `builder.zig` | SubagentConfig fluent builder | Fluent Builder (Pattern 1) |
| `classify.zig` | Deterministic + batch LLM classification | Function-pointer array (QueryMatch) |
| `route.zig` | Template → explain → LLM infill pipeline | ExplainCache (QueryCache dedup) |
| `validate.zig` | Schema + path + command allowlist validation | ErrorContext |
| `execute.zig` | VTable dispatch + WorkUnit per-tool execution | VTable (Pattern 4), WorkUnit |
| `reflect.zig` | Scratchpad with O(1) ring buffer eviction | Ring Buffer |
| `synthesize.zig` | Context-isolated summarization via ContextPacker | Context Isolation |
| `guardrails.zig` | Loop detection, failure limits, no-progress | FNV-1a output hashing |
| `grammar.zig` | GBNF grammar constraints for LLM output | Grammar-Constrained |
| `fsm.zig` | Main FSM loop + crash safety + profiling | Real FSM (switch dispatch) |
| `todo.zig` | Todo lifecycle (cmdTodoNew, cmdTodoTriage, cmdTodoRun) | CLI dispatch |

---

## Pattern Compliance

| Pattern | Usage | Location |
|---------|-------|----------|
| **Fluent Builder** | SubagentConfig builder with `.build()` terminal | `builder.zig` |
| **Comptime Reflection** | Not used directly (no DynamicEditable needed) | — |
| **Comptime Wrappers** | `retryCall` for bash execution (planned) | `execute.zig` |
| **VTable** | `SubagentTool {ptr, vtable}` handle for tool dispatch | `execute.zig` |
| **WorkUnit** | `WorkUnit(Handler)` for per-unit arena cleanup + concurrent execution | `execute.zig`, `classify.zig` |
| **ExecutionBackend** | `SyncBackend` for tests, `ZioBackend` for production | `fsm.zig` |
| **Arena-Backed** | Per-iteration arena with `reset(.retain_capacity)` between iterations | `fsm.zig` |
| **Typed Opaque Handles** | `IterationId`, `StepId` as `enum(i64) { _ }` | `types.zig` |
| **Function-pointer array** | `TodoAction` for classification | `classify.zig` |
| **Context Isolation** | Parent sees only `SummarizedContext`, never raw output | `synthesize.zig` |
| **ReAct Scratchpad** | Per-iteration observation accumulation with O(1) ring eviction | `reflect.zig` |
| **Batch Classification** | All unknowns classified in O(1) LLM call with constitutional validation | `classify.zig` |
| **Grammar-Constrained** | Ollama GBNF grammar for batch LLM output + constitutional validation | `grammar.zig` |
| **In-Process Explain** | `ExplainFn` callback returns structured results, no subprocess | `route.zig` |
| **ExplainCache** | `QueryCache`-backed explain result deduplication | `route.zig` |
| **Crash Safety** | DIARY.md persistence + CHECKLIST.md state | `fsm.zig` |
| **Allowlist Security** | `shell_parser.parseCommand` for proper argv tokenization | `validate.zig` |
| **PII Anonymization** | Planned: `anonymizeContext` before LLM batch calls | `classify.zig` |
| **QueryCache Dedup** | `fnv1a64`-keyed cache for explain results across iterations | `route.zig` |
| **ContextPacker** | Budget-aware stage assembly for LLM prompts | `synthesize.zig` |
| **ContextCompressor** | 3-phase scratchpad compression preserving tail | `synthesize.zig` |
| **Real FSM** | State-driven switch dispatch, not sequential pipeline | `fsm.zig` |
| **Iteration Profiling** | Per-iteration timing (`IterationProfile`) for deterministic vs LLM path time | `types.zig`, `fsm.zig` |

---

## Anti-Patterns Avoided

- **Cosmopolitan Polymorphism**: Classification uses function-pointer array, not vtable
- **Stack-allocated vtable**: `ToolVTable` is `const` global, never stack-local
- **Arena for long-lived data**: Only per-iteration intermediates; `SubagentResult` escapes to caller allocator
- **LLM for deterministic decisions**: Pattern matching first, LLM only for batch unknowns
- **Unbounded context**: Token budget enforced via ContextPacker; scratchpad bounded at max_entries
- **Per-item LLM calls**: Unknowns batched into single O(1) call with constitutional validation
- **Bash blocklist**: Commands validated against allowlist with proper argv parsing
- **Overcounting no-progress**: Only increments when FNV-1a output hash matches previous
- **O(n) scratchpad eviction**: Ring buffer, not orderedRemove(0)
- **Sequential pipeline**: `FsmState` enum drives real state transitions via switch
- **Subprocess explain**: `ExplainFn` callback returns structured results in-process

---

## Configuration

```zig
const config = try subagent.SubagentBuilder(allocator)
    .workspace("/path/to/project")
    .dbPath("/path/to/.guidance.db")
    .guidanceDir("/path/to/.guidance")
    .apiUrl("http://localhost:11434")
    .model("qwen2.5-coder:7b")
    .maxIterations(20)
    .scratchpadMaxEntries(10)
    .build();
```

## Backend Selection

```zig
// Tests: SyncBackend for deterministic execution
var sync = concurrency.SyncBackend{};
const backend = sync.backend();
var result = try subagent.runSubagentWithBackend(allocator, config, callbacks, backend);

// Production: ZioBackend for concurrent tool execution
const zb = try concurrency.ZioBackend.builder()
    .withPermits(4)
    .build(allocator);
defer zb.deinit();
const backend = zb.backend();
var result = try subagent.runSubagentWithBackend(allocator, config, callbacks, backend);
```

## FSM State Transitions

```
intake ──(has item)──▶ classify ──(known)──▶ route ──(filled)──▶ validate ──▶ execute ──▶ reflect ──▶ synth ──▶ intake
   │                      │                      │
   │                      │(unknown)             │(unfillable)
   │                      ▼                      ▼
   │               batch_classify             route_llm
   │                      │                      │
   │                      ▼                      │
   │                   classify ────────────────┘
   │
   └──(no items)──▶ done

validate ──(invalid params)──▶ escalate
execute  ──(guardrail halt)──▶ escalate

Any state ──(guardrail trigger)──▶ escalate
Any state ──(user cancel via Group)──▶ done
```

## Scratchpad Ring Buffer

The scratchpad uses O(1) append and eviction. When `max_entries` is reached, the oldest entry is overwritten instead of using `orderedRemove(0)` which is O(n).

```zig
var sp = Scratchpad.init(allocator, 10);  // max 10 entries
defer sp.deinit();

try sp.append(.{ .iteration = 1, .item_text = "run tests", .action = .bash, .observation = "all passed", .reasoning = "", .success = true });
const ctx = try sp.formatContext(allocator);
defer allocator.free(ctx);
```

## ExplainCache

Session-scoped cache backed by `common/hash.QueryCache` (FNV-1a64 key) for deduplication of repeated explain queries across iterations.

```zig
var cache = ExplainCache.init(allocator);
defer cache.deinit();

// Route with cache
const result = try routeParamsCached(allocator, item, action, null, explain_fn, llm_fn, db_path, workspace, allowlist, &cache);

// Check cache stats
const stats = cache.stats();
std.log.info("explain cache: {d} hits, {d} misses", .{stats.hits, stats.misses});
```

## Iteration Profiling

Adds per-iteration profiling with `IterationProfile`:

```zig
pub const IterationProfile = struct {
    iteration: u16,
    state_entered: FsmState,
    deterministic_time_us: u64,
    llm_time_us: u64,
    total_time_us: u64,
    action: ActionType,
    used_cache: bool,
};
```

Each iteration of `runSubagentWithBackend` records:
- Time spent in deterministic paths (pattern matching, template expansion)
- Time spent in LLM calls
- Total iteration time
- Whether the explain cache was hit

Results are available in `SubagentResult.profiles`.