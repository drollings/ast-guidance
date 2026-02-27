# ast-guidance Makefile
# Context: Zig-native codebase documentation system with Python language providers
# Maintainer: AI & Human Co-Pilot
#
# RALPH LOOP DISTRIBUTED: This Makefile implements the review/iteration phases
# of the RALPH Loop. The agent writes code → deterministic tools (reviewer) run
# build/test/guidance/lint → stamp gate progress → iterate until pass.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := pre-commit

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PYTHON      := python3
VENV        := .venv
BIN         := $(VENV)/bin
PYTHON_VENV := $(BIN)/python
UV          := uv

DOCUMENTOR  := $(shell which ast-guidance)
TARGET_BIN  := zig-out/bin/ast-guidance
AST_PY      := bin/ast-guidance-py
CONFIG      := .ast-guidance/ast-guidance-config.json

SRC_DIR      := ./src
GUIDANCE_DIR := .ast-guidance
ENV_DIR      := .env
HASH_DIR     := $(ENV_DIR)/.make_hashes

# Marker directories — store per-file success stamps for incremental builds
BUILD_MARKER_DIR    := zig-out/mark/build
TEST_MARKER_DIR     := zig-out/mark/test
GUIDANCE_MARKER_DIR := zig-out/mark/guidance
LINT_MARKER_DIR     := zig-out/mark/lint

# Verbosity control: V=1 enables shell echo
V ?= 0
Q := $(if $(filter 1,$V),,@)

# ==============================================================================
# MISE INTEGRATION
# ==============================================================================

include env/mk/common.mk

# Conditionally include target language overrides
-include env/mk/targets/$(TARGET_LANG).mk

# ==============================================================================
# HELP
# ==============================================================================

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  %-22s%s\n", $$1, $$2 } /^##@/ { printf "\n%s\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Intelligence Layer

.PHONY: query
query: $(DOCUMENTOR) ## Query guidance index  make query QUERY="ring buffer"
	@if [ -z "$(QUERY)" ]; then echo "Usage: make query QUERY=\"<term>\""; exit 1; fi
	$(Q)$(DOCUMENTOR) query "$(QUERY)"

.PHONY: explain
explain: $(DOCUMENTOR) ## Explain a module, function, or concept  make explain QUERY="sma"
	@if [ -z "$(QUERY)" ]; then echo "Usage: make explain QUERY=\"<term>\""; exit 1; fi
	$(Q)$(DOCUMENTOR) explain "$(QUERY)" --guidance $(GUIDANCE_DIR)

.PHONY: diary
diary: $(DOCUMENTOR) ## Append a diary entry  make diary NOTE="fixed bug"
	@if [ -z "$(NOTE)" ]; then echo "Usage: make diary NOTE=\"<what you did>\""; exit 1; fi
	$(Q)$(DOCUMENTOR) diary "$(NOTE)" --guidance $(GUIDANCE_DIR)

.PHONY: learn
learn: $(DOCUMENTOR) ## Drain INSIGHTS.md / CAPABILITIES.md inbox into structured knowledge
	$(Q)$(DOCUMENTOR) learn --guidance $(GUIDANCE_DIR)

.PHONY: commit
commit: $(DOCUMENTOR) ## AI-summarize staged diff, open editor, then commit
	$(Q)$(DOCUMENTOR) commit

.PHONY: triage
triage: $(DOCUMENTOR) ## Triage a TODO work item  make triage ITEM=my-feature
	@if [ -z "$(ITEM)" ]; then echo "Usage: make triage ITEM=<work-item-name>"; exit 1; fi
	$(Q)$(DOCUMENTOR) triage $(GUIDANCE_DIR)/.todo/$(ITEM)/TODO.md --guidance $(GUIDANCE_DIR)

##@ Python Language Provider (ast-guidance-py)

.PHONY: py-sync
py-sync: $(VENV) ## Sync guidance JSON for Python source files
	$(Q)$(PYTHON_VENV) $(AST_PY) sync --scan src/ --output $(GUIDANCE_DIR)

.PHONY: py-sync-infill
py-sync-infill: $(VENV) ## Sync Python guidance with AI comment infill
	$(Q)$(PYTHON_VENV) $(AST_PY) sync --scan src/ --output $(GUIDANCE_DIR) --infill

.PHONY: py-lint
py-lint: $(VENV) ## Lint Python with ruff
	$(Q)$(BIN)/ruff check src/ bin/

.PHONY: py-fmt
py-fmt: $(VENV) ## Format Python with ruff
	$(Q)$(BIN)/ruff format src/ bin/ && $(BIN)/ruff check --fix src/ bin/

.PHONY: py-test
py-test: $(VENV) ## Run Python tests
	$(Q)$(PYTHON_VENV) -m pytest tests/ -v

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
check-prereqs: ## Verify prerequisites (zig, uv)
	@which zig > /dev/null || (echo "zig not found. Install via mise: mise install zig"; exit 1)
	@which uv  > /dev/null || (echo "uv not found.  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1)
	@echo "Prerequisites satisfied"

.PHONY: env-init
env-init: check-prereqs venv ## Initialize development environment (uv venv + zig check)
	@echo "Environment ready."

.PHONY: clean
clean: ## Remove build artifacts and markers (keeps venv)
	$(Q)rm -rf zig-out/.zig-cache $(HASH_DIR) $(TARGET_BIN)
	$(Q)rm -rf $(BUILD_MARKER_DIR) $(TEST_MARKER_DIR) $(GUIDANCE_MARKER_DIR) $(LINT_MARKER_DIR)
	$(Q)find . -type d -name ".zig-cache" -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-all
clean-all: clean ## Nuclear cleanup (includes venv)
	$(Q)rm -rf $(VENV) $(HOME)/.cache/uv/

##@ Zig Build & RALPH Loop (incremental per-file)

$(TARGET_BIN): STRUCTURE.md
	$(Q)mkdir -p $(BUILD_MARKER_DIR) $(TEST_MARKER_DIR) $(GUIDANCE_MARKER_DIR) $(LINT_MARKER_DIR)
	$(Q)zig build

.PHONY: install
install: $(DOCUMENTOR)
	cp $(TARGET_BIN) $(DOCUMENTOR)

# Find all Zig source files
ZIG_SRC_FILES := $(shell find $(SRC_DIR) -name '*.zig' 2>/dev/null)

# Derive marker sets from source files
BUILD_MARKERS    := $(patsubst $(SRC_DIR)/%.zig,$(BUILD_MARKER_DIR)/%.zig,$(ZIG_SRC_FILES))
TEST_MARKERS     := $(patsubst $(SRC_DIR)/%.zig,$(TEST_MARKER_DIR)/%.zig,$(ZIG_SRC_FILES))
GUIDANCE_MARKERS := $(patsubst $(SRC_DIR)/%.zig,$(GUIDANCE_MARKER_DIR)/%.zig,$(ZIG_SRC_FILES))
LINT_MARKERS     := $(patsubst $(SRC_DIR)/%.zig,$(LINT_MARKER_DIR)/%.zig,$(ZIG_SRC_FILES))

# Module flags for ast-guidance tests: supplies the 'common' module.
GUIDANCE_MODULE_FLAGS = --dep common -Mroot=$< -Mcommon=$(SRC_DIR)/common/llm.zig -lc

# Generate Zig dependency file from AST import analysis
# Maps @import edges onto build-marker prerequisites so upstream changes propagate.
ZIG_DEPEND_FILE := zig.depend

$(ZIG_DEPEND_FILE): $(SRC_DIR) $(DOCUMENTOR)
	$(Q)$(DOCUMENTOR) deps --src $(SRC_DIR) \
	  | sed \
	      -e 's|__finish____src__|$(BUILD_MARKER_DIR)/|g' \
	      -e 's|__success____src__|$(BUILD_MARKER_DIR)/|g' \
	      -e 's|__src__|$(SRC_DIR)/|g' \
	  > $(ZIG_DEPEND_FILE)

-include $(ZIG_DEPEND_FILE)

.PRECIOUS: $(BUILD_MARKERS) $(TEST_MARKERS) $(GUIDANCE_MARKERS) $(LINT_MARKERS)

# Linear dependency chain per file:
#   src/file.zig
#     → zig-out/mark/build/file.zig    (zig build)
#     → zig-out/mark/test/file.zig     (zig test)
#     → zig-out/mark/guidance/file.zig (ast-guidance sync --infill)
#     → zig-out/mark/lint/file.zig     (zig fmt check)
#     → STRUCTURE.md

# Per-file build marker
$(BUILD_MARKER_DIR)/%.zig: $(SRC_DIR)/%.zig | $(BUILD_MARKER_DIR)
	$(Q)mkdir -p $(dir $@)
	$(Q)echo "Building: $<"
	$(Q)zig build 2>&1 || exit 1
	$(Q)touch $@

# Per-file test marker — ast-guidance/* needs the common module flag
$(TEST_MARKER_DIR)/ast-guidance/%.zig: $(SRC_DIR)/ast-guidance/%.zig | $(BUILD_MARKER_DIR)
	$(Q)mkdir -p $(dir $@)
	$(Q)echo "Testing:  $<"
	$(Q)zig test $(GUIDANCE_MODULE_FLAGS) 2>&1 || exit 1
	$(Q)touch $@

# Per-file test marker — common/* and other top-level modules
$(TEST_MARKER_DIR)/%.zig: $(SRC_DIR)/%.zig | $(BUILD_MARKER_DIR) $(TEST_MARKER_DIR)
	$(Q)mkdir -p $(dir $@)
	$(Q)echo "Testing:  $<"
	$(Q)zig test $< -lc 2>&1 || exit 1
	$(Q)touch $@

# Per-file guidance marker (AI sync --infill fills blank comments)
$(GUIDANCE_MARKER_DIR)/%.zig: $(SRC_DIR)/%.zig $(TEST_MARKER_DIR)/%.zig | $(DOCUMENTOR)
	$(Q)mkdir -p $(dir $@)
	$(Q)echo "Guidance: $<"
	$(Q)$(DOCUMENTOR) sync --file $< --output $(GUIDANCE_DIR) --infill
	$(Q)[ -f $(GUIDANCE_DIR)/src/$*.zig.json ] || touch $(GUIDANCE_DIR)/src/$*.zig.json
	$(Q)touch $@

# Per-file lint marker (non-AI sync, then zig fmt check)
$(LINT_MARKER_DIR)/%.zig: $(SRC_DIR)/%.zig $(GUIDANCE_MARKER_DIR)/%.zig | $(LINT_MARKER_DIR)
	$(Q)mkdir -p $(dir $@)
	$(Q)echo "Linting:  $<"
	$(Q)$(DOCUMENTOR) sync --file $< --output $(GUIDANCE_DIR)
	$(Q)zig fmt $< 2>&1
	$(Q)zig fmt --check $< 2>&1 || (echo "Format errors in $<. Run 'zig fmt $<' to fix."; exit 1)
	$(Q)touch $@

# Marker directory creation
$(BUILD_MARKER_DIR) $(GUIDANCE_MARKER_DIR) $(LINT_MARKER_DIR) $(TEST_MARKER_DIR):
	$(Q)mkdir -p $@

# Regenerate STRUCTURE.md once all lint markers are current
STRUCTURE.md: $(LINT_MARKERS)
	$(Q)$(DOCUMENTOR) structure --guidance $(GUIDANCE_DIR) 2>&1 | grep -E "STRUCTURE|Generated|✓" || true
	$(Q)touch STRUCTURE.md

##@ Gate Targets

# Full RALPH loop gate: build → test → guidance → lint → STRUCTURE.md
.PHONY: pre-commit
pre-commit: $(TARGET_BIN) ## Run all checks (RALPH loop gate)
	$(Q)echo "All checks passed."

.PHONY: fmt
fmt: ## Format all Zig source files
	$(Q)zig fmt $(SRC_DIR)/

.PHONY: build
build: $(DOCUMENTOR) ## Build ast-guidance binary

.PHONY: test
test: ## Run all Zig unit tests
	$(Q)zig build test

.PHONY: test-integration
test-integration: $(DOCUMENTOR) ## Run ast-guidance integration smoke tests
	$(Q)bash tests/guidance_integration.sh
