# mk/targets/rust.mk
# Rust-specific build targets

.PHONY: fmt-target
fmt-target:
	cd target && cargo fmt

.PHONY: lint-target
lint-target:
	cd target && cargo clippy --all-targets -- -D warnings

.PHONY: test-target
test-target:
	cd target && cargo test --all-features

.PHONY: build-target
build-target:
	cd target && cargo build --release

# ==============================================================================
# WASM Build Targets (Rust → wasm32-wasi)
# ==============================================================================

.PHONY: wasm-fmt
wasm-fmt:
	cd target && cargo fmt --all

.PHONY: wasm-lint
wasm-lint:
	cd target && cargo clippy --target wasm32-wasi --all-targets -- -D warnings

.PHONY: wasm-test
wasm-test:
	@echo "⚠️  WASM tests require wasmtime or browser runtime"
	@echo "   Running: cd target && cargo test --target wasm32-wasi"
	cd target && cargo test --target wasm32-wasi

.PHONY: wasm-build
wasm-build:
	cd target && cargo build --target wasm32-wasi --release
	@echo "✓ WASM binary built: target/.cargo/wasm32-wasi/release/*.wasm"
