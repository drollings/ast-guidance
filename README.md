# ast-guidance

A Zig-native, deterministic codebase documentation engine.  It analyzes source
files via AST, generates structured JSON guidance in `.ast-guidance/`, and
exposes an AI-assisted query layer through the Makefile — with minimal footprint
so it can drop into any project.

## Authorship and copyright

This code is authored by Daniel Rollings, February 2026, with a mixture of
elements from previous hand-written projects in Python and C++, rendered
into Zig with ease of extensibility into other languages.

It is released under a dual GPL/Commercial license.  See below.

## What it does

- **Bottom-up documentation**: Parses source files (Zig, Python, …) and emits
  per-file `*.json` guidance files that capture module purpose, function
  signatures, design patterns, reverse dependencies, and AI-generated comments.
- **STRUCTURE.md synthesis**: Aggregates all guidance JSON into a human-legible
  codebase map that AI agents can traverse without MCP or tool calls.
- **Incremental RALPH loop**: The Makefile chains `build → test → guidance sync
  → lint → STRUCTURE.md` with per-file stamp files so only changed files are
  re-processed.
- **Multi-language via providers**: `bin/ast-guidance-py` handles Python; future
  providers (`ast-guidance-cpp`, `ast-guidance-php`, …) follow the same
  `sync --file / --scan` contract.
- **Knowledge management**: `make explain`, `make query`, `make learn`, and
  `make diary` give AI agents a structured way to read, annotate, and promote
  codebase knowledge without hallucinating file paths.

## Quick start

```bash
# Install toolchain (requires mise)
mise install          # installs Zig + Python + uv from mise.toml

# Set up Python provider venv
make env-init

# Build the Zig binary
make build            # → zig-out/bin/ast-guidance

# Run the full RALPH loop gate
make pre-commit

# Query the guidance index
make explain QUERY="sync guidance json"
make query   QUERY="ring buffer"
```

## Source layout

```
src/
  ast-guidance/   Zig core engine
  common/         Shared LLM HTTP client
bin/
  ast-guidance-py Python AST provider
.ast-guidance/
  ast-guidance-config.json  Model / provider configuration
  .skills/                  Design-pattern skill documents
  .doc/                     Capabilities, diary, inbox
  src/                      Generated guidance JSON
env/
  mk/             Makefile helpers + per-language overrides
  mise/           Language-specific mise.toml fragments
doc/
  DESIGN.md       System design reference
```

## Adding a new language provider

Create `bin/ast-guidance-<lang>` and ensure it accepts:

```
ast-guidance-<lang> sync --file <path> --output <guidance_dir> [--infill]
ast-guidance-<lang> sync --scan <dir>  --output <guidance_dir> [--infill]
```

Output JSON must follow the canonical schema:

```json
{
  "meta":     { "module": "…", "source": "…", "language": "…" },
  "comment":  "one-line module description",
  "skills":   [],
  "hashtags": [],
  "used_by":  [],
  "members":  [ { "type", "name", "is_pub", "line", "signature", "comment", … } ]
}
```

Register the provider in `.ast-guidance/ast-guidance-config.json` under
`providers`.

## Configuration

`.ast-guidance/ast-guidance-config.json` controls model selection and
provider registration.  All Makefile targets read from this file.

## License

Licensing & Usage

This software is dual-licensed, meaning you must choose the appropriate
license for your use case.  This model ensures the software remains free and
open for the community, while ensuring sustainable development through
commercial support from large organizations.

Option A: Community License (GNU GPLv3)

If you are building an open-source application, a hobby project, or are an
individual developer, you may use this software for free under the terms of
the GNU General Public License v3.0 (GPLv3).

Obligations: If you distribute your software, you must open-source your
entire application under the GPLv3.

Disclaimer: Provided "AS IS" with absolutely no warranty, no legal liability,
and no technical support.

Option B: Commercial License

If you are developing proprietary, closed-source software, you cannot legally
use the GPLv3 license without open-sourcing your own codebase.  You must
purchase a Commercial License if you meet any of the following criteria:

You wish to embed this software in a proprietary, closed-source product.

Your Legal Entity (including parent companies and affiliates) generates gross
annual revenue exceeding $1,000,000 USD.

You require usage for more than one (1) developer seat.

You require technical support, indemnification, or liability waivers.
