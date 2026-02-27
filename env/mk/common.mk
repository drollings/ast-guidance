# mk/common.mk
# Shared helpers for target detection and mise invocation

# Detect mise (optional)
MISE := $(shell command -v mise 2>/dev/null)

# Read target from .env/mk/target_language.mk file (default: py)
TARGET_LANG ?= $(shell test -f .env/mk/target_language.mk && cat .env/mk/target_language.mk || echo py)

# mise installation command (profile-based)
MISE_INSTALL = $(if $(MISE), \
	if [ -f ".env/mise/mise.$(TARGET_LANG).toml" ]; then \
		mise -E $(TARGET_LANG) install; \
	else \
		mise install; \
	fi, \
	echo "⚠ mise not found. Install: https://mise.jdx.dev")

# Helper: Run mise task if available, fallback to direct command
# Usage: $(call mise_or_cmd,task_name,fallback_command)
define mise_or_cmd
	@if [ -n "$(MISE)" ]; then \
		mise run $(1); \
	else \
		$(2); \
	fi
endef
