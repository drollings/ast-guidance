#!/usr/bin/env bash
# ROADMAP_20260826_INTERLINGUA_V2 M11.1 — download the YaGO taxonomy.
#
# Fetches the YaGO 4.x taxonomy export from yago-knowledge.org into
# data/yago-taxonomy.tsv, then pipes it through `gen_yago_classes.py` to
# produce ontology/data/yago_classes.json (the 130k-class registry embedded at
# build time).
#
# The repository ships a curated sample (ontology/data/yago_classes.json) so
# the loader/build.rs pipeline is exercised hermetically; running this script
# is an operator action to swap in the full taxonomy.
#
# Usage:
#   tools/download_yago_taxonomy.sh
#
# Output:
#   data/yago-taxonomy.tsv          raw export (intermediate)
#   src/ontology/data/yago_classes.json   the class registry

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_TSV="$ROOT/data/yago-taxonomy.tsv"
OUT_JSON="$ROOT/src/ontology/data/yago_classes.json"

# YaGO 4.5 ships as a set of N-Triples files; the taxonomy (classes +
# rdfs:subClassOf) is the tractable subset for the interlingua registry.
# The exact URL is a placeholder for the published export; adjust to the
# current yago-knowledge.org layout.
YAGO_TAXONOMY_URL="${YAGO_TAXONOMY_URL:-https://yago-knowledge.org/data/yago4.5/taxonomy.nt.gz}"

mkdir -p "$ROOT/data"

# sha256 pin per roadmap E5
SHA_FILE="$ROOT/env/yago.sha256"
if [ -f "$SHA_FILE" ]; then
  EXPECTED=$(awk '{print $1}' "$SHA_FILE")
  echo "pinned sha256: $EXPECTED"
fi

echo "downloading YaGO taxonomy from $YAGO_TAXONOMY_URL ..."
curl -fL "$YAGO_TAXONOMY_URL" -o "$OUT_TSV.gz"
if [ -f "$SHA_FILE" ]; then
  echo "$EXPECTED  $OUT_TSV.gz" | sha256sum -c - || { echo "sha256 mismatch — abort"; exit 1; }
fi
gunzip -f "$OUT_TSV.gz"

echo "generating $OUT_JSON ..."
python3 "$ROOT/tools/gen_yago_classes.py" "$OUT_TSV" "$OUT_JSON"

echo "done: $(wc -l < "$OUT_JSON") classes"