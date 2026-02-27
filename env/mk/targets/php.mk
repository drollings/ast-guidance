# mk/targets/php.mk
# PHP-specific build targets

.PHONY: fmt-target
fmt-target:
	cd target && php-cs-fixer fix . --rules=@PSR12

.PHONY: lint-target
lint-target:
	cd target && phpstan analyse src/

.PHONY: test-target
test-target:
	cd target && vendor/bin/phpunit tests/

.PHONY: build-target
build-target:
	@echo "✓ PHP is interpreted (no build step)"

# ==============================================================================
# WASM Build Targets (PHP → Not Directly Supported)
# ==============================================================================
# PHP does not natively compile to WASM. These targets provide informational
# output about available alternatives.

.PHONY: wasm-fmt
wasm-fmt:
	cd target && php-cs-fixer fix . --rules=@PSR12

.PHONY: wasm-lint
wasm-lint:
	cd target && phpstan analyse src/

.PHONY: wasm-test
wasm-test:
	@echo "⚠️  PHP does not directly compile to WASM"
	@echo "   Alternative: Use PHP via Emscripten (complex build)"
	@echo "   Recommended: Transpile PHP → Rust/Go, then build WASM"
	cd target && vendor/bin/phpunit tests/ || true

.PHONY: wasm-build
wasm-build:
	@echo "❌ PHP does not support WASM compilation"
	@echo "   Workaround 1: Transpile PHP → Rust (use mk/targets/rust.mk wasm-build)"
	@echo "   Workaround 2: Transpile PHP → Go (use mk/targets/go.mk wasm-build)"
	@echo "   Workaround 3: Use Emscripten with PHPWASM"
	@exit 1
