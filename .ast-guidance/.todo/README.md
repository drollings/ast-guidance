# .ast-guidance/.todo/ — Work Item Lifecycle

Each subdirectory is a **work item** that moves through a dependency chain:

```
TODO.md → TRIAGE.md → WORK.md → COMPLETE.md → COMMITTED.md
```

## Lifecycle Stages

| Stage | File | Created by | Meaning |
|-------|------|------------|---------|
| **TODO** | `TODO.md` | Human / agent | Work identified, not yet analyzed |
| **TRIAGE** | `TRIAGE.md` | `make triage ITEM=<name>` | Risk assessed, steps planned |
| **WORK** | `WORK.md` | Human / agent | Implementation in progress |
| **COMPLETE** | `COMPLETE.md` | Human / agent | Implementation done, tests pass |
| **COMMITTED** | `COMMITTED.md` | `make commit` | Changes committed to git |

## Creating a Work Item

```bash
mkdir .ast-guidance/.todo/my-feature
# Write the work description:
cat > .ast-guidance/.todo/my-feature/TODO.md << 'EOF'
# Add feature X

## Goal
Describe what needs to be done.

## Files to change
- src/foo.py
- bin/guidance.py

## Acceptance criteria
- Tests pass
- STRUCTURE.md updated
EOF

# Triage it (generates TRIAGE.md with risk assessment + steps):
make triage ITEM=my-feature
```

## Committing with AI-summarized message

```bash
git add .
make commit   # Opens editor with AI-generated commit message
```

## Makefile targets

| Target | Description |
|--------|-------------|
| `make triage ITEM=<name>` | Generate TRIAGE.md for a TODO work item |
| `make commit` | AI-summarize staged diff → open editor → commit |
| `make explain QUERY=<term>` | Explain a module, function, or concept |
| `make diary NOTE="..."` | Append timestamped diary entry |
| `make learn` | Drain inbox files into structured knowledge |
| `make structure` | Regenerate STRUCTURE.md from all guidance JSON |
