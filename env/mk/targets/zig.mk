# mk/targets/zig.mk
# Zig-specific build targets

.PHONY: fmt-target
fmt-target:
	cd target && zig fmt .

.PHONY: lint-target
lint-target:
	@echo "⚠ Zig has no separate linter; compile checks types"

.PHONY: test-target
test-target:
	cd target && zig test src/main.zig

.PHONY: build-target
build-target:
	cd target && zig build -Doptimize=ReleaseFast

# ==============================================================================
# WASM Build Targets (Zig → wasm32-wasi)
# ==============================================================================

.PHONY: wasm-fmt
wasm-fmt:
	cd target && zig fmt .

.PHONY: wasm-lint
wasm-lint:
	@echo "⚠ Zig has no separate linter; running compile check for WASM target"
	cd target && zig build-exe src/main.zig -target wasm32-wasi --check

.PHONY: wasm-test
wasm-test:
	@echo "⚠️  WASM tests require wasmtime runtime"
	@echo "   Running: cd target && zig test src/main.zig -target wasm32-wasi"
	cd target && zig test src/main.zig -target wasm32-wasi || true

.PHONY: wasm-build
wasm-build:
	cd target && zig build-exe src/main.zig -target wasm32-wasi -O ReleaseFast -femit-bin=bin/app.wasm
	@echo "✓ WASM binary built: target/bin/app.wasm"
