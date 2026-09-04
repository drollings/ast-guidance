#!/usr/bin/env python3
"""Parity benchmark: pinned spaCy English tokenizer + StringStore against the
shared golden corpus (`src/spacy-rs/tests/data/en_tokenization.json`).

Workload A (tokenize): tokenize every corpus case and materialize exactly the
attribute surface the golden fixture records (text/idx/whitespace/lower/shape/
prefix/suffix/norm + the 17 lexeme flags), folding everything into a checksum.

Workload B (strings): intern every distinct orth, then serialize + deserialize
the store (the first-wins round-trip both suites assert).

The fixture is located relative to this file (env `SPACY_RS_FIXTURE` overrides).
Usage: bench_py.py [passes]
"""

import json
import os
import resource
import sys
import time

import spacy

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FIXTURE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "src", "spacy-rs", "tests", "data", "en_tokenization.json")
)
FIXTURE = os.environ.get("SPACY_RS_FIXTURE", _DEFAULT_FIXTURE)

# The fixture is generated from the pinned reference; warn on a different
# version because the token-count integrity assertion may then legitimately
# fail (that is the parity gate, not a harness bug).
if spacy.__version__ != "3.8.15":
    print(
        f"# warning: spaCy {spacy.__version__} != pinned 3.8.15; "
        "the token-count integrity check may fail (parity gate)",
        file=sys.stderr,
    )


def peak_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kB on Linux


def fold_token_attrs(checksum, t):
    """Materialize the golden surface: orth/idx/spacy/norm/lower/shape/prefix/
    suffix + the 17 flags, and fold them into the checksum."""
    checksum += len(t.text) + t.idx + len(t.whitespace_) + len(t.norm_)
    checksum += len(t.lower_) + len(t.shape_) + len(t.prefix_) + len(t.suffix_)
    for flag in (
        t.is_alpha, t.is_ascii, t.is_digit, t.is_lower, t.is_punct, t.is_space,
        t.is_title, t.is_upper, t.like_url, t.like_num, t.like_email, t.is_stop,
        t.is_bracket, t.is_quote, t.is_left_punct, t.is_right_punct, t.is_currency,
    ):
        checksum += int(flag)
    return checksum


def tokenize_pass(nlp, corpus):
    checksum = 0
    for c in corpus:
        doc = nlp(c["text"])
        for t in doc:
            checksum = fold_token_attrs(checksum, t)
    return checksum


def main():
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    baseline = peak_rss_kb()
    t_start = time.monotonic()

    corpus = json.load(open(FIXTURE, encoding="utf-8"))
    total_tokens = sum(len(c["tokens"]) for c in corpus)
    nlp = spacy.blank("en")  # tokenizer only — exactly what generated the fixture

    for c in corpus:  # integrity + warmup
        doc = nlp(c["text"])
        assert len(doc) == len(c["tokens"]), f"parity: {c['text']!r}"
    warm_checksum = tokenize_pass(nlp, corpus)
    startup = time.monotonic() - t_start

    t0 = time.monotonic()
    a_checksum = 0
    for _ in range(passes):
        a_checksum += tokenize_pass(nlp, corpus)
    a_elapsed = time.monotonic() - t0
    a_cases = passes * len(corpus)
    a_tokens = passes * total_tokens

    distinct = sorted({t["orth"] for c in corpus for t in c["tokens"]})
    from spacy.strings import StringStore

    store = StringStore()
    for o in distinct:
        store.add(o)
    t1 = time.monotonic()
    b_checksum = 0
    for _ in range(passes):
        s = StringStore()
        for o in distinct:
            s.add(o)
        b = s.to_bytes()
        reloaded = StringStore().from_bytes(b)
        b_checksum += len(reloaded)
    b_elapsed = time.monotonic() - t1

    peak = peak_rss_kb()
    total = time.monotonic() - t_start

    print("=== spaCy parity benchmark ===")
    print(f"passes: {passes}")
    print(f"corpus_cases: {len(corpus)}")
    print(f"corpus_tokens: {total_tokens}")
    print(f"distinct_orths: {len(distinct)}")
    print(f"startup_s: {startup:.3f}")
    print(f"a_cases: {a_cases}")
    print(f"a_tokens: {a_tokens}")
    print(f"a_elapsed_s: {a_elapsed:.4f}")
    print(f"a_cases_per_s: {a_cases / a_elapsed:.1f}")
    print(f"a_tokens_per_s: {a_tokens / a_elapsed:.1f}")
    print(f"a_ns_per_token: {a_elapsed * 1e9 / a_tokens:.0f}")
    print(f"a_checksum: {a_checksum:x}")
    print(f"b_elapsed_s: {b_elapsed:.4f}")
    print(f"b_roundtrips_per_s: {passes / b_elapsed:.1f}")
    print(f"b_checksum: {b_checksum:x}")
    print(f"rss_baseline_kb: {baseline}")
    print(f"rss_peak_kb: {peak}")
    print(f"rss_incremental_kb: {peak - baseline}")
    print(f"warm_checksum: {warm_checksum:x}")
    print(f"total_s: {total:.3f}")


if __name__ == "__main__":
    main()