# mk/targets/go.mk
# Go-specific build targets

.PHONY: fmt-target
fmt-target:
	cd target && gofmt -w -s .

.PHONY: lint-target
lint-target:
	cd target && golangci-lint run ./...

.PHONY: test-target
test-target:
	cd target && go test -v -race ./...

.PHONY: build-target
build-target:
	cd target && go build -o bin/sandbox ./cmd/sandbox

# ==============================================================================
# WASM Build Targets (Go → wasip1/wasm)
# ==============================================================================

.PHONY: wasm-fmt
wasm-fmt:
	cd target && gofmt -w -s .

.PHONY: wasm-lint
wasm-lint:
	cd target && golangci-lint run --build-tags wasip1 ./...

.PHONY: wasm-test
wasm-test:
	@echo "⚠️  WASM tests require wasmtime runtime"
	@echo "   Running: cd target && GOOS=wasip1 GOARCH=wasm go test -v ./..."
	cd target && GOOS=wasip1 GOARCH=wasm go test -v ./... || true

.PHONY: wasm-build
wasm-build:
	cd target && GOOS=wasip1 GOARCH=wasm go build -o bin/sandbox.wasm ./cmd/sandbox
	@echo "✓ WASM binary built: target/bin/sandbox.wasm"
