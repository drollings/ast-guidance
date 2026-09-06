#!/usr/bin/env python3
"""ROADMAP_20260826_INTERLINGUA_V2 M11.2 — YaGO taxonomy TSV → yago_classes.json.

Parses a YaGO taxonomy export and emits the class registry consumed by
`YaGoLoader` (ontology/src/yago_loader.rs). Input format (TSV, one line per
class):

    <class-iri>\t<label>\t[<superclass-iri>]

Tabs and labels that appear twice are resolved first-wins. The output JSON is
the canonical `[{iri, label, superclass}, ...]` array.

Usage:
    tools/gen_yago_classes.py <taxonomy.tsv> <out.json>
"""
import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    tsv_path, out_path = sys.argv[1], sys.argv[2]

    classes: dict[str, dict] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            iri = parts[0].strip()
            if not iri or iri in classes:
                continue  # first-wins
            label = parts[1].strip() if len(parts) > 1 else iri.rsplit("/", 1)[-1]
            superclass = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            classes[iri] = {"iri": iri, "label": label, "superclass": superclass}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(classes.values()), f, ensure_ascii=False, indent=2)

    print(f"wrote {len(classes)} classes to {out_path}")


if __name__ == "__main__":
    main()