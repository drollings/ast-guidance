# guidance Makefile
# Context: Rust-native AST-guided SQLite FTS5 database generator for NullClaw
# Maintainer: AI & Human Co-Pilot
#
# RALPH LOOP:
#   build → guidance check (validate) → guidance sync (generate)
#     └─► test → lint → fmt (from config) → staleness check → structure → db check
#           └─► .guidance/src/**/*.json  (JSON mtime = universal marker)
#                 └─► .explain.db
#                       └─► STRUCTURE.md
#
# Key invariant: $(TARGET_BIN) depends ONLY on source files — never on
# STRUCTURE.md or markers that themselves depend on the binary.
# Change detection: source mtime vs guidance JSON mtime (no separate marker files).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := pre-commit

TARGET_BIN  := guidance
CORAL_ROUTER_BIN := target/debug/coral-router
CONFIG      := .guidance/guidance-config.json
INSTALLDIR  := $(HOME)/.local/bin

RUST_SRC_DIR := src
GUIDANCE_DIR  := .guidance
GUIDANCE_DB   := .guidance.db
ENV_DIR      := .env
HASH_DIR     := $(ENV_DIR)/.make_hashes

# Verbosity control: V=1 enables shell echo
V ?= 0
Q := $(if $(filter 1,$V),,@)

# ==============================================================================
# MISE INTEGRATION
# ==============================================================================

include env/mk/common.mk
-include env/mk/targets/$(TARGET_LANG).mk

# ==============================================================================
# HELP
# ==============================================================================

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  %-22s%s\n", $$1, $$2 } /^##@/ { printf "\n%s\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Intelligence Layer

# Database target: rebuilt by guidance sync automatically.
# This rule is kept for direct `make .guidance.db` invocations.
$(GUIDANCE_DB): | $(CARGO_BIN)
	$(Q)echo "Syncing database: $@"
	$(Q)$(TARGET_BIN) sync --workspace . --json-dir $(GUIDANCE_DIR) --db $@
	$(Q)touch $@

.PHONY: commit
commit: $(CARGO_BIN) | STRUCTURE.md ## Generate AI commit message from staged diff + guidance JSON context
	$(Q)$(TARGET_BIN) commit $(if $(DRY_RUN),--dry-run) $(if $(DEBUG),--debug)

##@ Environment

.PHONY: venv
venv: $(VENV) ## Install / verify Python dependencies

$(HASH_DIR):
	$(Q)mkdir -p $(HASH_DIR)

$(VENV): requirements.txt | $(HASH_DIR)
	$(Q)echo "Syncing Python environment..."
	$(Q)if [ ! -d $(VENV) ]; then $(UV) venv $(VENV); fi
	$(Q)$(UV) pip install --no-cache -q -r requirements.txt
	$(Q)$(UV) pip install --no-cache -q ruff pytest pytest-cov
	$(Q)touch $(VENV)
	$(Q)echo "Python environment ready."

.PHONY: check-prereqs
check-prereqs: ## Verify prerequisites (cargo, uv)
	@which cargo > /dev/null || (echo "cargo not found. Install via rustup: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"; exit 1)
	@which uv  > /dev/null || (echo "uv not found.  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1)
	@echo "Prerequisites satisfied"

.PHONY: env-init
env-init: check-prereqs venv ## Initialize development environment
	@echo "Environment ready."

.PHONY: clean
clean: clean-db ## Remove build artifacts and markers (keeps venv and .guidance config)
	$(Q)rm -rf target $(HASH_DIR)

##@ Guidance Management

.PHONY: clean-db
clean-db: ## Remove build artifacts and markers (keeps venv and .guidance config)
	$(Q)rm -f $(GUIDANCE_DB)

.PHONY: clean-all
clean-all: clean ## Remove stale JSON files from .guidance/src
	$(Q)find $(GUIDANCE_DIR)/src -name '*.json' -type f -exec rm -rf {} \; || true

.PHONY: explain
explain: $(GUIDANCE_DB) ## Explain a module, function, or concept  make explain QUERY="sma"
	@if [ -z "$(QUERY)" ]; then \
		echo "❌ Usage: make explain QUERY=\"<module or function or concept>\""; \
		echo "   Examples:"; \
		echo "     make explain QUERY=\"sma\""; \
		echo "     make explain QUERY=\"ring buffer\""; \
		echo "     make explain QUERY=\"ast_parser\""; \
		exit 1; \
	fi
	$(Q)$(TARGET_BIN) explain --guidance $(GUIDANCE_DIR) "$(QUERY)"

##@ Rust Build & RALPH Loop

RUST_SRC_FILES := $(shell find $(RUST_SRC_DIR) -name '*.rs' 2>/dev/null)

# ── Binary ───────────────────────────────────────────────────────────────────

CARGO_BIN := target/debug/$(TARGET_BIN)

$(CARGO_BIN) $(CORAL_ROUTER_BIN): $(RUST_SRC_FILES)
	$(Q)echo "Building guidance + coral-router"
	$(Q)cargo build
	$(Q)echo "Build complete"

.PHONY: install
install: $(CARGO_BIN) $(CORAL_ROUTER_BIN)
	$(Q)mkdir -p $(INSTALLDIR)
	$(Q)cp $(CARGO_BIN) $(INSTALLDIR)/guidance
	$(Q)cp $(CORAL_ROUTER_BIN) $(INSTALLDIR)/coral-router
	$(Q)echo "Installed guidance and coral-router in $(INSTALLDIR)"

# ── Router Targets ────────────────────────────────────────────────────────────

.PHONY: router
router: $(CORAL_ROUTER_BIN) ## Build coral-router

.PHONY: router-test
router-test: $(CORAL_ROUTER_BIN) ## Run all router unit, golden, and e2e mock tests; validate binary
	$(Q)echo "Running fluent-router unit + e2e mock tests"
	$(Q)cargo test -p fluent-router
	$(Q)echo "Running coral-router dry-run (--help)"
	$(Q)cargo run --bin coral-router -- --help
	$(Q)echo "All router tests passed."

# ── Standard Targets ──────────────────────────────────────────────────────────

.PHONY: test
test: ## Run unit tests across the Rust source in src
	$(Q)cargo test --workspace

.PHONY: lint
lint: ## Run clippy across the Rust source in src on all .rs files
	$(Q)cargo clippy --workspace -- -D warnings

.PHONY: health
health: ## Run cargo tarpaulin and verify 85% coverage
	$(Q)cargo tarpaulin --workspace --fail-under 85

.PHONY: format
format: ## Run rustfmt across the Rust source in src on all .rs files
	$(Q)cargo fmt --all

# ── STRUCTURE.md ─────────────────────────────────────────────────────────────
# Generated by guidance structure after sync.
# The target below is kept for direct `make STRUCTURE.md` invocations.

STRUCTURE.md: $(GUIDANCE_DB) | $(CARGO_BIN)
	$(Q)$(TARGET_BIN) structure --json-dir $(GUIDANCE_DIR) 2>&1 | grep -E "STRUCTURE|Generated|✓" || true
	$(Q)touch STRUCTURE.md

##@ Gate Targets

# Full RALPH loop: build → check (validate) → sync (generate)
# guidance check runs config commands (test/lint/fmt) and validates staleness.
# guidance sync regenerates JSON + DB for stale files.
.PHONY: pre-commit build
build: $(CARGO_BIN) $(CORAL_ROUTER_BIN) ## Build guidance and coral-router binaries

pre-commit: STRUCTURE.md $(CARGO_BIN) $(CORAL_ROUTER_BIN) ## Run full RALPH loop
	$(Q)$(TARGET_BIN) sync --workspace .
	$(Q)echo "✓ All checks passed. Ready to commit."

