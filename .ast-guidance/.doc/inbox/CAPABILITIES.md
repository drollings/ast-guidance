# Capabilities Inbox

Major new features implemented. Run `make document` to promote into `guidance/capabilities/`.

<!-- Add new capabilities as bullets below:
- CozoDB backend: `src/cozo.zig` wraps the CozoDB C API (`cozo_c.h`) providing open/close/query/exec with RAII result handling. `src/db.zig` implements the unified Library (context_nodes, targets, depends_on, provides_capability, neighbor_of, wasm_tools), HydrationPipeline (in-Zig cosine KNN + edge persistence), and ContextPacker (LOD selection by graph distance). Schema DDL lives in `src/schema.zig` as CozoScript `:create` statements. CozoDB replaces both pgvector and LadybugDB/Cypher.
- Hybrid resolution: TraitSet bitmasks (u128) stored in CozoDB as Int hi/lo pairs; Datalog recursive rules traverse depends_on edges; final @popCount validation runs in Zig.

