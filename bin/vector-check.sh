#!/usr/bin/env bash
# vector-check — HNSW/error-taxonomy consolidation lint (roadmap M8).
#
# Fails if any deleted re-export shim or second threshold/error source
# regrows anywhere under `src/`, or the raw HNSW graph types escape their two
# canonical homes:
#   0. `hnsw_rs::` / `DistCosine` outside `common-core/sqlite.rs` + `db/hnsw.rs`.
#   1. `search_vector::math` / `search_vector::error` code paths
#      (prose uses the hyphenated "search-vector" form, which never matches).
#   2. `crate::hnsw::` paths (the deleted `fluent_router::hnsw` shim).
#   3. Fresh `1.0 - d` distance→similarity closures outside the canonical
#      kernel (`fluent_db::vector::{distance_to_similarity[_clamped],
#      scored_hits}` in `src/db/src/vector.rs`, the one allowed home).
#   4. The deleted shim files themselves.
#   5. A second `DEFAULT_HNSW_THRESHOLD` / `HnswParams` / `AdaptiveHnsw`
#      definition (single threshold source).
#   6. A second `From<rusqlite::Error>` impl (single error taxonomy in
#      `src/db/src/error.rs`).
#
# Usage:  bin/vector-check.sh [root]   # exit 0 clean, non-zero on violation
#   `root` overrides the tree to scan (default: repo root) so the checks can
#   be red-teamed against fixture trees without touching the repo.
#
# Wired into `make lint-vector` and CI (it is cheap, fast, and needs no
# compilation — same shape as `bin/live-ai-guard.sh`).

set -uo pipefail

ROOT="${1:-$(dirname "$0")/..}"
cd "$ROOT" || exit 1

FAIL=0
fail() { echo "FAIL: $1"; FAIL=1; }

# ── Check 0: raw HNSW graph types stay in their canonical homes ─────────
while IFS= read -r line; do
  fail "${line}: raw HNSW type outside common-core/sqlite.rs + db/hnsw.rs (compose HnswIndex / make_hnsw instead)"
done < <(grep -rn "hnsw_rs::\|DistCosine" src --include="*.rs" 2>/dev/null | grep -v "^src/common-core/src/sqlite\.rs:" | grep -v "^src/db/src/hnsw\.rs:" || true)

# ── Check 1: no search-vector re-export shim paths ──────────────────────
while IFS= read -r line; do
  fail "${line}: search-vector math/error shim path regrew (use fluent_db::vector / fluent_db::error::DbError)"
done < <(grep -rn "search_vector::math\|search_vector::error" src --include="*.rs" 2>/dev/null || true)

# ── Check 2: no router hnsw shim paths ──────────────────────────────────
while IFS= read -r line; do
  fail "${line}: crate::hnsw:: path regrew (use fluent_db::hnsw::…)"
done < <(grep -rn "crate::hnsw::" src --include="*.rs" 2>/dev/null || true)

# ── Check 3: no fresh distance→similarity closures ──────────────────────
# Forbidden shapes: `1.0 - distance`, `1.0 - dist`, `1.0 - d)`/`,`,
# `1.0 - f64::from(dist)`. The canonical kernel home is exempt — the mapping
# may exist in exactly one place.
while IFS= read -r line; do
  fail "${line}: distance→similarity closure outside the canonical kernel (use distance_to_similarity[_clamped] / scored_hits)"
done < <(grep -rn -E "1\.0 - (distance|dist\b|d\)|d,|f64::from\(dist\)|f32)" src --include="*.rs" 2>/dev/null | grep -v "^src/db/src/vector\.rs:" || true)

# ── Check 4: deleted shim files stay deleted ────────────────────────────
for f in src/router/src/hnsw.rs src/search-vector/src/math.rs src/search-vector/src/error.rs; do
  if [ -e "$f" ]; then
    fail "$f: deleted re-export shim regrew"
  fi
done

# ── Check 5: single threshold source ────────────────────────────────────
for def in "pub const DEFAULT_HNSW_THRESHOLD" "pub struct HnswParams" "pub struct AdaptiveHnsw"; do
  count=$(grep -rn -- "$def" src --include="*.rs" 2>/dev/null | wc -l)
  if [ "$count" -ne 1 ]; then
    fail "expected exactly 1 definition of \`$def\`, found $count"
  fi
done

# ── Check 6: single rusqlite error taxonomy ─────────────────────────────
impls=$(grep -rn "impl From<rusqlite" src --include="*.rs" 2>/dev/null || true)
count=$(echo "$impls" | grep -c . || true)
if [ "$count" -ne 1 ]; then
  fail "expected exactly 1 From<rusqlite::Error> impl, found $count"
elif ! echo "$impls" | grep -q "^src/db/src/error\.rs:"; then
  fail "From<rusqlite::Error> impl lives outside src/db/src/error.rs: $impls"
fi

if [ "$FAIL" -eq 0 ]; then
  echo "ok: raw HNSW types contained; no re-export shims, no second threshold/error source, no fresh distance closures"
fi
exit "$FAIL"
