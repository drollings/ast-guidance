# mk/targets/py.mk
# Python-specific build targets

.PHONY: fmt-target
fmt-target:
	$(call mise_or_cmd,fmt,uv run ruff format src/ tests/ && uv run ruff check --fix src/ tests/)

.PHONY: lint-target
lint-target:
	@$(MAKE) lint

.PHONY: test-target
test-target:
	$(call mise_or_cmd,test,$(PYTHON_VENV) -m pytest tests/ -v)

.PHONY: build-target
build-target:
	$(call mise_or_cmd,build,uv build)

# ==============================================================================
# WASM Build Targets (Python → Not Supported)
# ==============================================================================
# Python is interpreted and does not directly compile to WASM.
# These targets provide informational output about alternatives.

.PHONY: wasm-fmt
wasm-fmt:
	$(call mise_or_cmd,fmt,uv run ruff format src/ tests/ && uv run ruff check --fix src/ tests/)

.PHONY: wasm-lint
wasm-lint:
	@$(MAKE) lint

.PHONY: wasm-test
wasm-test:
	@echo "⚠️  Python does not directly compile to WASM"
	@echo "   Alternative: Use Pyodide (Python in WASM via Emscripten)"
	@echo "   Recommended: Transpile Python → Rust/Go, then build WASM"
	$(call mise_or_cmd,test,uv run pytest tests/ -v) || true

.PHONY: wasm-build
wasm-build:
	@echo "❌ Python does not support WASM compilation"
	@echo "   Workaround 1: Use Pyodide (Python interpreter compiled to WASM)"
	@echo "   Workaround 2: Transpile Python → Rust (use mk/targets/rust.mk wasm-build)"
	@echo "   Workaround 3: Transpile Python → Go (use mk/targets/go.mk wasm-build)"
	@exit 1
