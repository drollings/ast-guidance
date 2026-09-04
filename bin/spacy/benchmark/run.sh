#!/usr/bin/env bash
# spaCy vs spacy-rs parity benchmark runner.
#
# Builds the Rust release harness and runs both tokenizers over the shared
# golden corpus fixture (`src/spacy-rs/tests/data/en_tokenization.json`, pinned
# to spaCy 3.8.15), then prints a speed + memory comparison table.
#
# Env:
#   SPACY_PYTHON        python interpreter with spaCy installed (default: python3)
#   SPACY_BENCH_PASSES  in-process passes over the corpus per run (default: 300)
#   SPACY_BENCH_REPS    runs per side for the median (default: 3)
#   SPACY_BENCH_OUT     directory for per-run outputs (default: mktemp -d)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
FIXTURE="${REPO}/src/spacy-rs/tests/data/en_tokenization.json"

PASSES="${SPACY_BENCH_PASSES:-300}"
REPS="${SPACY_BENCH_REPS:-3}"
SPACY_PYTHON="${SPACY_PYTHON:-python3}"

if [[ ! -f "${FIXTURE}" ]]; then
    echo "ERROR: golden corpus fixture not found: ${FIXTURE}" >&2
    exit 1
fi

echo "==> building spacy-rs benchmark harness (release)"
# Cargo discovers `.cargo/config.toml` from the CWD, but run.sh must behave the
# same from anywhere; the monorepo's config sets `build.target-dir = "target"`,
# so pin the shared repo target (env override allowed) for a deterministic,
# incrementally-cached artifact location.
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${REPO}/target}"
export CARGO_TARGET_DIR
cargo build --release --quiet --manifest-path "${HERE}/Cargo.toml"
BIN="${CARGO_TARGET_DIR}/release/spacy-bench"
if [[ ! -x "${BIN}" ]]; then
    echo "ERROR: built spacy-bench binary not found at ${BIN}" >&2
    exit 1
fi

echo "==> checking spaCy availability via ${SPACY_PYTHON}"
"${SPACY_PYTHON}" -c "import spacy" >/dev/null 2>&1 || {
    echo "ERROR: ${SPACY_PYTHON} cannot import spaCy (install it, or set SPACY_PYTHON=<python-with-spacy>)" >&2
    exit 1
}

OUT="${SPACY_BENCH_OUT:-$(mktemp -d /tmp/spacy-bench.XXXXXX)}"
mkdir -p "${OUT}"

echo "==> running spacy-rs (${REPS}x, ${PASSES} passes each)"
for i in $(seq 1 "${REPS}"); do
    "${BIN}" "${PASSES}" > "${OUT}/rust.${i}.txt"
done

echo "==> running spaCy via ${SPACY_PYTHON} (${REPS}x, ${PASSES} passes each)"
for i in $(seq 1 "${REPS}"); do
    "${SPACY_PYTHON}" "${HERE}/bench_py.py" "${PASSES}" > "${OUT}/py.${i}.txt"
done

echo
"${SPACY_PYTHON}" "${HERE}/summarize.py" \
    "spacy-rs" "${OUT}"/rust.*.txt -- "spaCy" "${OUT}"/py.*.txt
echo
echo "per-run outputs: ${OUT}"