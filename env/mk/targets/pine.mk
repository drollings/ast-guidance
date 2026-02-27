# mk/targets/pine.mk
# Pine Script v6-specific targets

.PHONY: fmt-target
fmt-target:
	@echo "⚠ Pine Script has no standard formatter"

.PHONY: lint-target
lint-target:
	@echo "⚠ Pine Script linting requires TradingView IDE"

.PHONY: test-target
test-target:
	@echo "⚠ Pine Script testing requires TradingView backtesting"

.PHONY: build-target
build-target:
	@echo "✓ Pine Script is interpreted (no build step)"

# ==============================================================================
# WASM Build Targets (Pine Script → Not Supported)
# ==============================================================================
# Pine Script is TradingView-specific and does not compile to WASM.
# These targets provide informational output about alternatives.

.PHONY: wasm-fmt
wasm-fmt:
	@echo "⚠ Pine Script has no standard formatter"

.PHONY: wasm-lint
wasm-lint:
	@echo "⚠ Pine Script linting requires TradingView IDE"

.PHONY: wasm-test
wasm-test:
	@echo "⚠️  Pine Script does not support WASM compilation"
	@echo "   Pine Script is specific to TradingView platform"
	@echo "   Recommended: Transpile Pine Script → Rust/Go, then build WASM"

.PHONY: wasm-build
wasm-build:
	@echo "❌ Pine Script does not support WASM compilation"
	@echo "   Pine Script is an indicator/strategy DSL for TradingView"
	@echo "   Workaround: Transpile Pine Script → Rust (use mk/targets/rust.mk wasm-build)"
	@echo "   Workaround: Transpile Pine Script → Go (use mk/targets/go.mk wasm-build)"
	@exit 1
