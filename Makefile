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
CORAL_ROUTER_CONFIG := env/coral-router.json
# Extract bind address from the config file (source of truth), with a sane
# fallback when the config is unreadable at parse time.
CORAL_ROUTER_BIND_ADDR := $(shell python3 -c "import json; print(json.load(open('$(CORAL_ROUTER_CONFIG)'))['server']['bind_addr'])" 2>/dev/null || echo "127.0.0.1:8079")
CORAL_ROUTER_HEALTH_URL := http://$(CORAL_ROUTER_BIND_ADDR)/health
CORAL_ROUTER_MOCK_HEALTH_URL := http://127.0.0.1:8078/health
# How long to poll /health after (re)starting the router. Real mode spawns
# managed llama-servers at boot, so the default is generous; override with
# ROUTER_START_TIMEOUT_S=<n>. Mock mode skips supervision and boots fast.
ROUTER_START_TIMEOUT_S ?= 300
ROUTER_MOCK_TIMEOUT_S ?= 30
ROUTER_WAIT_SCRIPT := bin/router-wait-health.sh
ROUTER_LOG := /tmp/coral-router.out
ROUTER_MOCK_LOG := /tmp/coral-router-mock.out
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

$(CARGO_BIN): $(RUST_SRC_FILES)
	$(Q)echo "Building guidance"
	$(Q)cargo build --bin guidance

$(CORAL_ROUTER_BIN): $(RUST_SRC_FILES)
	$(Q)echo "Building coral-router"
	$(Q)cargo build --bin coral-router

.PHONY: install
install: $(CARGO_BIN) $(CORAL_ROUTER_BIN)
	$(Q)mkdir -p $(INSTALLDIR)
	$(Q)cp $(CARGO_BIN) $(INSTALLDIR)/guidance
	$(Q)cp $(CORAL_ROUTER_BIN) $(INSTALLDIR)/coral-router
	$(Q)echo "Installed guidance and coral-router in $(INSTALLDIR)"

# ── Router Targets ────────────────────────────────────────────────────────────

.PHONY: router
router: $(CORAL_ROUTER_BIN) ## Build coral-router (fast, no run; the base for every other router target)

# Kill any running coral-router and wait for it to actually exit before the
# caller starts a fresh one. The router is the process owner of its spawned
# llama-servers and handles SIGTERM gracefully (stops the supervisor first),
# so a plain kill never orphans serving processes.
#
# Match by exact process name (`pkill -x`) AND by the router config file path
# (`pkill -f 'coral-router[.]json'`): the binary is often invoked through a
# symlink (e.g. the `gguf_tool` alias) whose comm(2) name is the symlink, so
# `killall coral-router` would silently miss it and leave a stale router on
# the port. The `[.]` character class keeps the second pattern from matching
# its own literal text (or a calling shell that merely mentions the word
# `coral-router`), so it only ever kills the router process.
define stop-router
	$(Q)echo "Stopping any running router"
	$(Q)pkill -x coral-router 2>/dev/null || true
	$(Q)pkill -f 'coral-router[.]json' 2>/dev/null || true
	$(Q)for i in $$(seq 1 75); do pgrep -f 'coral-router[.]json' >/dev/null 2>&1 || break; sleep 0.2; done
endef

.PHONY: router-stop
router-stop: ## Stop coral-router (SIGTERM, so it stops its managed llama-servers first)
	$(stop-router)

# ── Router Benchmark ──────────────────────────────────────────────────────────

CORAL_ROUTER_TEST_SCRIPT := bin/coral-router-test.py

.PHONY: router-benchmark
router-benchmark: router-start ## Score the live router (routing accuracy, TTFT, VRAM) via bin/coral-router-test.py; reads env/coral-router.json for routes + expectations
	$(Q)python3 $(CORAL_ROUTER_TEST_SCRIPT) --config $(CORAL_ROUTER_CONFIG)

.PHONY: doc-check
doc-check: ## Doc consistency lint — types named in skill/router docs must exist in source
	$(Q)bin/doc-check.sh --types

.PHONY: router-test
router-test: $(CORAL_ROUTER_BIN) ## Run fluent-router unit/golden/e2e tests + a --help dry-run of the built binary (stops a running router first)
	$(stop-router)
	$(Q)echo "Running fluent-router unit + golden + e2e mock tests"
	$(Q)cargo test -p fluent-router
	$(Q)echo "Validating coral-router --help"
	$(Q)$(CORAL_ROUTER_BIN) --help > /dev/null && echo "All router tests passed." || echo "ERRROR: coral-router did NOT successfully run."

.PHONY: router-test-all
router-test-all: $(CORAL_ROUTER_BIN) ## router-test + coral-context HNSW benchmarks (large, slow; stops a running router first)
	$(stop-router)
	$(Q)echo "Running fluent-router unit + golden + e2e mock tests"
	$(Q)cargo test -p fluent-router
	$(Q)echo "Running coral-context tests with HNSW benchmarks"
	$(Q)cargo test -p coral-context --features hnsw-bench -- --ignored --nocapture
	$(Q)echo "Validating coral-router --help"
	$(Q)$(CORAL_ROUTER_BIN) --help > /dev/null && echo "All router tests passed." || echo "ERRROR: coral-router did NOT successfully run."

ROUTER_MOCK_TEST_SCRIPT := bin/router-mock-tests.sh

.PHONY: router-start
router-start: $(CORAL_ROUTER_BIN) ## Build (if needed), (re)start coral-router in real mode on :8079, and wait for /health (stops the old tree first)
	$(stop-router)
	$(Q)echo "Starting coral-router"
	$(Q)nohup $(CORAL_ROUTER_BIN) start -c $(CORAL_ROUTER_CONFIG) > $(ROUTER_LOG) 2>&1 &
	$(Q)bash $(ROUTER_WAIT_SCRIPT) $(CORAL_ROUTER_HEALTH_URL) $(ROUTER_START_TIMEOUT_S) $(ROUTER_LOG)

.PHONY: router-mock
router-mock: $(CORAL_ROUTER_BIN) $(ROUTER_MOCK_TEST_SCRIPT) ## Build, start a mock router on :8078, run the 29 curl smoke-tests (leaves that server running)
	$(stop-router)
	$(Q)nohup $(CORAL_ROUTER_BIN) start -c $(CORAL_ROUTER_CONFIG) --host 127.0.0.1 --port 8078 --mock env/mock-transcripts.json > $(ROUTER_MOCK_LOG) 2>&1 &
	$(Q)bash $(ROUTER_WAIT_SCRIPT) $(CORAL_ROUTER_MOCK_HEALTH_URL) $(ROUTER_MOCK_TIMEOUT_S) $(ROUTER_MOCK_LOG)
	$(Q)ROUTER_BASE_URL=http://127.0.0.1:8078 bash $(ROUTER_MOCK_TEST_SCRIPT)

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

