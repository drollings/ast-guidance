#!/usr/bin/env python3
"""Generator for the English rule-lemmatizer tables.

Reads the ``en_lemma_rules`` / ``en_lemma_exc`` / ``en_lemma_index`` JSON from
the installed ``spacy-lookups-data`` package (v1.0.5) and emits
``../../env/en_lemmatizer.json`` — a single auditable JSON that mirrors
the upstream tables keyed by the lowercased UPOS names (adj/noun/verb/adv/
punct). ``build.rs`` compiles this into the versioned binary blob embedded via
``include_bytes!``, so the crate never carries the data as Rust source.

Run:
    /tmp/opencode/spacy-venv/bin/python3 src/spacy-rs/tools/gen_en_lemma_data.py
"""
from __future__ import annotations

import gzip
import json
import os

import spacy_lookups_data  # noqa: E402

OUT = "../../env/en_lemmatizer.json"
DATA = os.path.join(os.path.dirname(spacy_lookups_data.__file__), "data")


def load(name: str):
    with gzip.open(os.path.join(DATA, name + ".json.gz"), "rt") as f:
        return json.load(f)


rules = load("en_lemma_rules")
exc = load("en_lemma_exc")
index = load("en_lemma_index")

n_exc = sum(len(m) for m in exc.values())
n_index = sum(len(words) for words in index.values())
print(f"rules: {len(rules)} pos, {sum(len(r) for r in rules.values())} rules")
print(f"exc: {len(exc)} pos, {n_exc} surfaces")
print(f"index: {len(index)} pos, {n_index} words")

payload = {
    "lemma_rules": rules,
    "lemma_exc": exc,
    "lemma_index": index,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"wrote {OUT}")