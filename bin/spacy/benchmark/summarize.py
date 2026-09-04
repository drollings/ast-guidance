#!/usr/bin/env python3
"""Summarize raw spacy-rs / spaCy benchmark outputs into the comparison table.

Usage:
    summarize.py RUST_LABEL RUST_OUT.txt... -- PY_LABEL PY_OUT.txt...

Each side's numeric fields are median-averaged across the given output files
and printed side-by-side with spacy-rs:sPaCy ratios, plus a parity gate that
cross-checks the corpus stats and the Workload-B round-trip checksums.
"""

import sys
from statistics import median


def parse(path):
    vals = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("==="):
                continue
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            vals[k] = v
    return vals


def med_int(vals, key):
    try:
        return int(median([int(r[key]) for r in vals]))
    except (KeyError, ValueError):
        return None


def med_float(vals, key):
    try:
        return median([float(r[key]) for r in vals])
    except (KeyError, ValueError):
        return None


def hex_of(vals, key):
    xs = {r.get(key) for r in vals}
    return xs.pop() if len(xs) == 1 else None


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        print("usage: summarize.py RUST_LABEL RUST_OUT... -- PY_LABEL PY_OUT...", file=sys.stderr)
        sys.exit(2)
    sep = argv.index("--")
    rust_label, rust_files = argv[0], argv[1:sep]
    py_label, py_files = argv[sep + 1], argv[sep + 2:]

    rust = [parse(p) for p in rust_files]
    py = [parse(p) for p in py_files]

    r = {
        "cases": med_int(rust, "corpus_cases"),
        "tokens": med_int(rust, "corpus_tokens"),
        "orths": med_int(rust, "distinct_orths"),
        "passes": med_int(rust, "passes"),
        "a_tps": med_float(rust, "a_tokens_per_s"),
        "a_ns": med_float(rust, "a_ns_per_token"),
        "a_el": med_float(rust, "a_elapsed_s"),
        "b_rps": med_float(rust, "b_roundtrips_per_s"),
        "b_el": med_float(rust, "b_elapsed_s"),
        "startup": med_float(rust, "startup_s"),
        "total": med_float(rust, "total_s"),
        "peak_kb": med_float(rust, "rss_peak_kb"),
        "base_kb": med_float(rust, "rss_baseline_kb"),
        "inc_kb": med_float(rust, "rss_incremental_kb"),
        "b_csum": hex_of(rust, "b_checksum"),
    }
    p = {
        "cases": med_int(py, "corpus_cases"),
        "tokens": med_int(py, "corpus_tokens"),
        "orths": med_int(py, "distinct_orths"),
        "passes": med_int(py, "passes"),
        "a_tps": med_float(py, "a_tokens_per_s"),
        "a_ns": med_float(py, "a_ns_per_token"),
        "a_el": med_float(py, "a_elapsed_s"),
        "b_rps": med_float(py, "b_roundtrips_per_s"),
        "b_el": med_float(py, "b_elapsed_s"),
        "startup": med_float(py, "startup_s"),
        "total": med_float(py, "total_s"),
        "peak_kb": med_float(py, "rss_peak_kb"),
        "base_kb": med_float(py, "rss_baseline_kb"),
        "inc_kb": med_float(py, "rss_incremental_kb"),
        "b_csum": hex_of(py, "b_checksum"),
    }

    def speed(metric, fmt, higher_better=True):
        a, b = r[metric], p[metric]
        if a is None or b is None or b == 0:
            return None
        ratio_val = a / b if higher_better else b / a
        return fmt(a), fmt(b), ratio_val

    def memory(metric, fmt):
        a, b = r[metric], p[metric]
        if a is None or b is None or b == 0:
            return None
        return fmt(a), fmt(b), b / a  # spaCy / spacy-rs

    def mb(kb):
        return f"{kb / 1024:.1f} MB"

    print("=== spaCy vs spacy-rs parity benchmark ===")
    print(f"  dataset : src/spacy-rs/tests/data/en_tokenization.json")
    print(f"  stats   : {r['cases']} cases / {r['tokens']} tokens / {r['orths']} distinct orths")
    print(f"  runs    : {len(rust_files)} runs x {r['passes']} passes each (medians)")
    print()

    print(f"  {'metric':<34}{rust_label:>14}{py_label:>14}{'ratio':>10}")
    print("  " + "-" * 70)
    row = speed("a_tps", lambda v: f"{v:,.0f}")
    if row:
        print(f"  {'Workload A tokens/s':<34}{row[0]:>14}{row[1]:>14}{row[2]:>9.2f}x")
    row = speed("a_ns", lambda v: f"{v:,.0f}", higher_better=False)
    if row:
        tag = " slower" if row[2] > 1.0 else ""
        print(f"  {'Workload A ns/token':<34}{row[0]:>14}{row[1]:>14}{row[2]:>9.2f}x{tag}")
    row = speed("b_rps", lambda v: f"{v:,.0f}")
    if row:
        print(f"  {'Workload B round-trips/s':<34}{row[0]:>14}{row[1]:>14}{row[2]:>9.2f}x")
    row = speed("startup", lambda v: f"{v:.3f}s", higher_better=False)
    if row:
        tag = " slower" if row[2] > 1.0 else ""
        print(f"  {'Startup':<34}{row[0]:>14}{row[1]:>14}{row[2]:>9.2f}x{tag}")
    row = speed("total", lambda v: f"{v:.3f}s", higher_better=False)
    if row:
        tag = " slower" if row[2] > 1.0 else ""
        print(f"  {'End-to-end wall':<34}{row[0]:>14}{row[1]:>14}{row[2]:>9.2f}x{tag}")
    for metric, name in (("base_kb", "Peak RSS baseline"), ("peak_kb", "Peak RSS during batch"),
                         ("inc_kb", "Peak RSS incremental")):
        row = memory(metric, mb)
        if row:
            tag = " smaller" if row[2] >= 1.0 else " larger"
            print(f"  {name:<34}{row[0]:>14}{row[1]:>14}{row[2]:>9.2f}x{tag}")
    print()

    ok = (
        r["cases"] == p["cases"] and r["tokens"] == p["tokens"] and r["orths"] == p["orths"]
        and r["b_csum"] is not None and r["b_csum"] == p["b_csum"]
    )
    gate = "PASS" if ok else "FAIL"
    note = (
        f"corpus stats agree; Workload-B round-trip checksum = 0x{r['b_csum']} "
        "(token-count integrity per case is asserted inside each harness)"
    )
    print(f"  parity gate : {gate} — {note}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()