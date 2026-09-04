#!/usr/bin/env bash
# live-ai-guard — hermeticity / live-AI lint (roadmap M4.5).
#
# Enforces the two AI-inference isolation rules from doc/TESTING.md §4:
#   1. Every `#[ignore]` test attribute must carry a `reason` string (rule 5),
#      so ignored tests are always self-describing about how to run them.
#   2. No *hermetic* (non-live) test source may reference a dialable
#      non-loopback host URL literal (rule 4), so a mock-backed test can never
#      accidentally fire real inference. Loopback and reserved non-dialable
#      hosts are exempt so config fixtures don't false-positive.
#
# Scope for rule 2: test-harness `.rs` files only (paths containing
# `/tests/`, `/test_support`, `/test_stub`, or a `tests.rs`/`*_tests.rs`
# stem), excluding the `tests/live/` live-AI directories (rule 1 exempts them).
#
# Usage:  bin/live-ai-guard.sh            # exit 0 clean, non-zero on violation
#
# Wired into `make lint-live-ai` and CI (it is cheap and fast).

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

ROOT="$(pwd)"
FAIL=0

# ── Check 1: every `#[ignore]` must carry a reason ─────────────────────────
# A reason is present if the attribute is `#[ignore = "..."]` or
# `#[ignore(reason = "...")]`. A bare `#[ignore]` is a violation. The
# negative lookbehind for a backtick excludes prose mentions (e.g.
# `` is `#[ignore]`d ``) in doc comments and READMEs.
while IFS= read -r line; do
  file="${line%%:*}"; rest="${line#*:}"; lineno="${rest%%:*}"
  echo "FAIL: ${file}:${lineno}: bare #[ignore] has no reason (add '= \"...\"' or '(reason = \"...\")')"
  FAIL=1
done < <(grep -rnP '(?<!`)#\[ignore\]' "$ROOT/src" 2>/dev/null || true)

# ── Check 2: no dialable non-loopback host in hermetic test code ───────────
# Reserved / non-dialable host suffixes that never reach the network:
#   .test, .example, .invalid, .localhost, .local  (RFC 6761 / RFC 6762)
# Loopback is 127.0.0.1, localhost, [::1].
if command -v python3 >/dev/null 2>&1; then
  # Emit all candidate URL literals from hermetic test-harness files, then let
  # a tiny python filter decide which hosts are genuinely dialable. grep's
  # "no match" exit (1) is neutralized with `|| true`; the violation set is
  # driven purely by python's printed output.
  FILES="$(find "$ROOT/src" -name '*.rs' \
    \( -path '*/tests/*' -o -path '*/tests.rs' -o -name '*_tests.rs' \
       -o -path '*/test_support*' -o -path '*/test_stub*' \) \
    ! -path '*/tests/live/*')"
  if [ -n "$FILES" ]; then
    URLS="$(grep -rhoE 'http[s]?://[A-Za-z0-9.:_-]+' $FILES 2>/dev/null || true)"
    VIOLATIONS="$(printf '%s\n' "$URLS" | python3 -c '
import re, sys
LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}
RESERVED_TLDS = ("test", "example", "invalid", "localhost", "local")
RESERVED_EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net")
# RDF/ontology namespace hosts used purely as content-addressing identifiers
# (e.g. `hash_iri("http://schema.org/Person")`) — never dialed in hermetic
# tests; the IRIs are hashed as strings, not fetched as endpoints.
RESERVED_NAMESPACE_HOSTS = ("schema.org", "yago-knowledge.org", "www.w3.org", "w3.org")
bad = []
for line in sys.stdin:
    host = re.sub(r"^[^:]+://", "", line.strip())
    host = host.split("/", 1)[0].split(":", 1)[0].strip("[]").lower()
    if host in LOOPBACK:
        continue
    if "." not in host:
        continue                      # no dot => not a real dialable FQDN
    if host.rsplit(".", 1)[-1] in RESERVED_TLDS:
        continue                      # RFC 6761 reserved / non-dialable
    if host.endswith(RESERVED_EXAMPLE_DOMAINS):
        continue                      # RFC 2606 reserved example domains
    if host in RESERVED_NAMESPACE_HOSTS:
        continue                      # content-addressing namespace IRIs
    bad.append(line.strip())
for u in sorted(set(bad)):
    print(f"FAIL: dialable non-loopback host in hermetic test: {u}")
sys.exit(1 if bad else 0)
' || true)"
    if [ -n "$VIOLATIONS" ]; then
      printf '%s\n' "$VIOLATIONS"
      FAIL=1
    fi
  fi
else
  echo "WARN: python3 not found; skipping dialable-host check"
fi

if [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "live-ai guard failed. Fix the violations above, or run them via"
  echo "'make test-live' only (live tests are #[ignore] + live-ai-gated)."
  exit 1
fi
echo "live-ai guard: clean."