//! ROADMAP_20260903_LLM M10 — calibration gate (tests + measurements).
//!
//! This milestone writes NO production code: it measures the moved
//! heuristics, locks the numbers, and records them in
//! `src/llm/CALIBRATION.md`. It blocks M11 (shim removal) and any new
//! caching/persisting of heuristic outputs.
//!
//! Confidence vs task-value (§1) — read before trusting any number here:
//! - `estimate_tokens` weights are a task-value budget fit, not producer
//!   confidence: a low estimate never means "the model is sure", and an
//!   over-budget truncation is data loss even when the producer was confident.
//! - Think-block presence is a producer self-doubt artifact, not answer
//!   quality: a confident answer may carry no think block and a wrong answer
//!   may carry a stripped one. Stripping must never delete task content.
//! - SSE framing is transport completeness (neither axis): a complete frame
//!   can carry a wrong answer; an incomplete frame is never a verdict.
//! - Cache identity is freshness, not endorsement: a hit is key-equality
//!   (`{model}:{sha256(request)}` + TTL), never a correctness vote.
//!
//! Method: every section below asserts LOCKED measured values (a silent
//! retune breaks the test and forces a `CALIBRATION.md` update — never a
//! quiet behavior change). The shared `common_core::calibration` harness is
//! composed, never re-implemented.

use common_core::calibration::{calibrate_threshold, CalibrationReport};
use fluent_llm::cache::{CachedResponse, ResponseCache};
use fluent_llm::openai::drain_sse_lines;
use fluent_llm::thinking::strip_thinking_blocks;
use fluent_llm::tokens::{estimate_tokens, TokenBudget};

// ─── M10.1: deterministic 200-sample corpus ────────────────────────────────
// Fixed-seed LCG (no RNG dependency): the corpus bytes are identical on every
// run, so the fingerprint below is a checked-in corpus in effect — any
// generator change fails loudly. Seed and class sizes are part of the lock.

const CORPUS_SEED: u64 = 0x2026_0903;
const CLASS_SIZES: [usize; 4] = [50, 50, 50, 50]; // prose, code, CJK, emoji

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const PROSE_WORDS: &[&str] = &[
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "router",
    "request", "response", "model", "context", "window", "budget", "cache",
    "queue", "worker", "config", "server", "client", "retry", "timeout",
    "stream", "chunk", "token", "prompt", "answer", "review", "policy",
    "session", "ledger", "index", "search", "report", "sales", "plan",
    "route", "chart", "stage", "filter", "score", "embed", "classify",
    "summarize", "transform", "Pack", "my", "box", "with", "five",
];

const CODE_TOKENS: &[&str] = &[
    "fn", "let", "mut", "if", "else", "for", "in", "return", "struct",
    "impl", "pub", "use", "match", "while", "loop", "self", "true",
    "false", "None", "Some", "Ok", "Err", "vec", "String", "u64", "i32",
    "bool", "push", "len", "new", "map", "filter", "collect", "async",
    "await", "spawn", "config", "queue", "cache", "token", "{", "}", "(",
    ")", "[", "]", ";", "=", "=>", "::", "->", ".", ",",
];

const CJK_CHARS: &[char] = &[
    '安', '全', '世', '界', '日', '本', '語', '韓', '国', '中', '文', '報',
    '告', '売', '上', '計', '画', '路', '由', '検', '索', '応', '答', '設',
    '定', '文', '字', '混', '合', '서', '울', '한', '글', '테', '스', '트',
    'あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し',
];

const EMOJI: &[&str] = &[
    "🌍", "🌏", "🌎", "😀", "🎉", "🎊", "✅", "❌", "⚠️", "🔍", "📊", "📝",
    "🚀", "💡", "🧪", "📦", "🔧", "📌", "🎯", "💬", "📎", "🗂️", "⏱️", "🧠",
];

/// Class 0 prose, 1 code, 2 CJK, 3 emoji/mixed. Returns 200 samples.
fn build_corpus() -> Vec<(usize, String)> {
    let mut rng = Lcg(CORPUS_SEED);
    let mut out = Vec::with_capacity(200);
    for _ in 0..CLASS_SIZES[0] {
        let n = 8 + rng.below(18);
        let words: Vec<&str> = (0..n)
            .map(|_| PROSE_WORDS[rng.below(PROSE_WORDS.len())])
            .collect();
        out.push((0, words.join(" ")));
    }
    for _ in 0..CLASS_SIZES[1] {
        let n = 20 + rng.below(41);
        let mut toks: Vec<String> = (0..n)
            .map(|_| CODE_TOKENS[rng.below(CODE_TOKENS.len())].to_string())
            .collect();
        // Deterministic newlines + indentation every ~8 tokens.
        let mut s = String::new();
        for (i, t) in toks.drain(..).enumerate() {
            if i > 0 {
                if i % 8 == 0 {
                    s.push('\n');
                    s.push_str("    ");
                } else {
                    s.push(' ');
                }
            }
            s.push_str(&t);
        }
        out.push((1, s));
    }
    for _ in 0..CLASS_SIZES[2] {
        let n = 5 + rng.below(26);
        let mut s: String = (0..n)
            .map(|_| CJK_CHARS[rng.below(CJK_CHARS.len())])
            .filter(|&c| c != ' ')
            .collect();
        // 30% of CJK samples carry 1–3 trailing English words (mixed).
        if rng.below(10) < 3 {
            s.push(' ');
            let m = 1 + rng.below(3);
            let words: Vec<&str> = (0..m)
                .map(|_| PROSE_WORDS[rng.below(PROSE_WORDS.len())])
                .collect();
            s.push_str(&words.join(" "));
        }
        out.push((2, s));
    }
    for _ in 0..CLASS_SIZES[3] {
        let n = 2 + rng.below(9);
        let mut s: String = (0..n)
            .map(|_| EMOJI[rng.below(EMOJI.len())])
            .collect::<Vec<_>>()
            .join("");
        // Half the emoji samples carry prose (mixed).
        if rng.below(2) == 0 {
            let m = 3 + rng.below(8);
            let words: Vec<&str> = (0..m)
                .map(|_| PROSE_WORDS[rng.below(PROSE_WORDS.len())])
                .collect();
            s.push(' ');
            s.push_str(&words.join(" "));
        }
        out.push((3, s));
    }
    out
}

// ─── M10.1: tiktoken-style reference (in-test, documented estimate) ─────────
// Density-based, per Unicode class, with cl100k-ish densities where they
// differ from the production weights. Class ranges mirror the production
// table (Unicode facts, not model choices); the WEIGHTS are the model:
// ASCII-alnum shares the production 0.25 BY CONSTRUCTION (both approximate
// tiktoken's ~4 chars/token English density — documented limitation: ASCII
// absolute accuracy is not independently verified here, but any ASCII
// retune still moves the locked divergence), while CJK (1.2), emoji (1.5),
// ASCII symbols (0.8) and the rest use tiktoken densities the production
// table deliberately simplifies away.
// This reference is itself an estimate, NOT ground-truth tiktoken output —
// the report measures divergence between two estimators, and the gate locks
// that divergence so a silent retune of the production weights fails loudly.

fn ref_class(c: char) -> f64 {
    match c {
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{20000}'..='\u{2A6DF}'
        | '\u{2A700}'..='\u{2B739}'
        | '\u{2B740}'..='\u{2B81D}'
        | '\u{2B820}'..='\u{2CEA1}'
        | '\u{2CEB0}'..='\u{2EBE0}'
        | '\u{30000}'..='\u{3134A}'
        | '\u{31350}'..='\u{323AF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{2F800}'..='\u{2FA1F}'
        | '\u{2E80}'..='\u{2EFF}'
        | '\u{3000}'..='\u{303F}'
        | '\u{31C0}'..='\u{31EF}'
        | '\u{3200}'..='\u{33FF}'
        | '\u{FE30}'..='\u{FE4F}'
        | '\u{FF00}'..='\u{FFEF}'
        | '\u{AC00}'..='\u{D7AF}'
        | '\u{3040}'..='\u{30FF}' => 1.2, // CJK: tiktoken ~1-2/char
        '\u{1F300}'..='\u{1F9FF}'
        | '\u{1FA00}'..='\u{1FA6F}'
        | '\u{1FA70}'..='\u{1FAFF}'
        | '\u{2600}'..='\u{27BF}'
        | '\u{FE00}'..='\u{FE0F}'
        | '\u{200D}'
        | '\u{E0000}'..='\u{E007F}'
        | '\u{1F000}'..='\u{1F02F}'
        | '\u{1F0A0}'..='\u{1F0FF}'
        | '\u{1F100}'..='\u{1F1FF}'
        | '\u{1F200}'..='\u{1F2FF}' => 1.5, // emoji: tiktoken ~1-3 each
        ' ' | '\t' | '\n' | '\r' | '\u{200B}' | '\u{00A0}' => 0.15,
        '\u{0000}'..='\u{001F}' | '\u{007F}'..='\u{009F}' => 0.0,
        c if c.is_ascii_alphanumeric() => 0.25,
        c if c.is_ascii_punctuation() => 0.8, // tiktoken: symbols ~1 each
        _ => 0.35, // accented Latin etc.: tiktoken ~1 per 1-2 chars
    }
}

fn reference_tokens(text: &str) -> f64 {
    text.chars().map(ref_class).sum()
}

#[test]
fn m10_1_corpus_fingerprint() {
    let corpus = build_corpus();
    assert_eq!(corpus.len(), 200);
    for (class, want) in [0, 1, 2, 3].iter().zip(CLASS_SIZES) {
        assert_eq!(
            corpus.iter().filter(|(c, _)| c == class).count(),
            want,
            "class {class}"
        );
    }
    let total_bytes: usize = corpus.iter().map(|(_, s)| s.len()).sum();
    // Locked corpus fingerprint (measured; see CALIBRATION.md). Any generator
    // change — seed, word lists, class sizes — fails here first.
    assert_eq!(total_bytes, 19_588, "corpus byte fingerprint");
    assert_eq!(
        corpus[0].1,
        "index quick brown chart stream ledger fox score session lazy classify token session index five stream dog"
    );
    assert_eq!(
        corpus[50].1,
        "return async collect String let while String ::\n    Ok async push :: len , vec false\n    vec fn . match queue mut return config\n    ) len collect = -> true token queue"
    );
    assert_eq!(corpus[100].1, "い上応文告上混え한報路検おさ界語売");
    assert_eq!(
        corpus[150].1,
        "🚀🎊🧠🎯💬 the Pack client brown filter dog budget timeout policy with"
    );
    // Loops back: every sample is non-empty (reference is > 0 on content).
    assert!(corpus.iter().all(|(_, s)| !s.is_empty()));
}

#[test]
fn m10_1_token_weight_divergence() {
    let corpus = build_corpus();
    let mut abs_err = 0.0;
    let mut within20 = 0usize;
    let mut class_err = [[0.0f64; 2]; 4]; // [sum_err, n] per class
    for (class, text) in &corpus {
        let est = estimate_tokens(text) as f64;
        let reference = reference_tokens(text);
        let err = (est - reference).abs();
        abs_err += err;
        if err <= 0.20 * reference.max(1.0) {
            within20 += 1;
        }
        class_err[*class][0] += err;
        class_err[*class][1] += 1.0;
    }
    let n = corpus.len() as f64;
    let mean_abs_err = abs_err / n;
    let pct_within20 = within20 as f64 / n * 100.0;
    // Locked divergence (measured HEAD values; see CALIBRATION.md for the
    // per-class attribution). IEEE-754 f64 with fixed op order is
    // bit-deterministic, so the epsilon band only tolerates representation
    // noise — any weight retune moves these and fails loudly.
    assert!(
        (mean_abs_err - 6.2267).abs() < 1e-3,
        "mean abs err moved: {mean_abs_err:.6}"
    );
    assert_eq!(within20, 93, "samples within ±20% moved: {pct_within20:.2}%");
    for (class, name, want) in
        [(0, "prose", 0.7950), (1, "code", 8.9100), (2, "cjk", 11.7590), (3, "emoji", 3.4430)]
    {
        let class_mean = class_err[class][0] / class_err[class][1];
        assert!(
            (class_mean - want).abs() < 1e-3,
            "class {name} mean err moved: {class_mean:.6}"
        );
    }
}

#[test]
fn m10_1_truncate_within_budget() {
    let corpus = build_corpus();
    let mut max_overshoot = 0u64;
    let mut truncated = 0usize;
    for (_, text) in &corpus {
        let est = estimate_tokens(text);
        if est <= 2 {
            continue;
        }
        let budget = (est / 2).max(1);
        let out = TokenBudget(budget as usize).truncate_to_budget(text);
        let out_est = estimate_tokens(&out);
        max_overshoot = max_overshoot.max(out_est.saturating_sub(budget));
        truncated += 1;
    }
    // Locked (measured HEAD values; see CALIBRATION.md). The ceiling is
    // multibyte-driven (see `m10_1_truncate_multibyte_attribution`): ASCII
    // truncations overshoot by ~1 (the `...` suffix), CJK/emoji by ~100.
    assert_eq!(truncated, 196, "truncation coverage moved");
    assert_eq!(max_overshoot, 10, "overshoot ceiling moved: {max_overshoot}");
}

#[test]
fn m10_1_truncate_multibyte_attribution() {
    // Attribution for the corpus overshoot: `truncate_to_budget` scales
    // `text.len()` (BYTES) but keeps that many CHARS, so multibyte text
    // keeps ~3-4x too much. ASCII-only truncations hold the budget; CJK-only
    // ones do not. Locked as the cause record (see CALIBRATION.md); the
    // one-line fix (`chars().count()` instead of `len()`) is follow-up work,
    // not an M10 production change.
    for (name, text, want) in [
        ("ascii", "word ".repeat(200), 1u64),
        ("cjk", "漢".repeat(300), 102u64),
        ("emoji", "😀".repeat(200), 101u64),
    ] {
        let est = estimate_tokens(&text);
        let budget = (est / 2).max(1);
        let out = TokenBudget(budget as usize).truncate_to_budget(&text);
        let overshoot = estimate_tokens(&out).saturating_sub(budget);
        assert_eq!(overshoot, want, "TRUNC_{name} overshoot moved (est={est} budget={budget})");
    }
}

#[test]
fn m10_1_whitespace_control_no_inflate() {
    // Control group: whitespace/control-heavy strings stay fractional/zero.
    let ws = "   \t\n  \t   \n ";
    assert_eq!(estimate_tokens(ws), 1, "whitespace control moved");
    assert_eq!(estimate_tokens("\u{0}\u{1}\u{2}"), 0, "control-char control moved");
}

// ─── M10.2: think-stripping precision/recall ────────────────────────────────

struct ThinkCase {
    input: &'static str,
    should_strip: bool,
}

const THINK_RECALL: &[ThinkCase] = &[
    ThinkCase { input: "<think>reason</think>result", should_strip: true },
    ThinkCase { input: "before <think>reason</think> after", should_strip: true },
    ThinkCase { input: "<thinking>a</thinking>B<thinking>c</thinking>D", should_strip: true },
    ThinkCase { input: "[THINK]hidden[/THINK]visible", should_strip: true },
    ThinkCase {
        input: " thinking let me check response\nThe answer is 4",
        should_strip: true,
    },
    ThinkCase { input: "start<thinking>unclosed", should_strip: true },
    ThinkCase { input: "<think>unclosed", should_strip: true },
    ThinkCase { input: "A<think>B</think>C<think>D</think>E", should_strip: true },
];

const THINK_CONTROLS_M1: &[&str] = &[
    "rethinking a plan",
    "normal text response here",
    "Hello  world",
    "unclosed-without-marker text",
    "use <div> tags in html",
    "",
];

// M10.2 roadmap controls: inline code in ticks. The first three FIRE on the
// current implementation (measured, see CALIBRATION.md) — they are kept in
// the control set with their true labels so the finding stays loud; the
// last two never fire.
const THINK_CONTROLS_TICKS: &[&str] = &[
    "run `<think>foo</think>` now",
    "use `<think>` tag",
    "call `<thinking>reason</thinking>` twice",
    "`thinking` about it",
    "a `code` span",
];

fn think_report(cases: &[ThinkCase], controls: &[&str]) -> CalibrationReport {
    let mut all: Vec<(&str, bool)> =
        cases.iter().map(|c| (c.input, c.should_strip)).collect();
    all.extend(controls.iter().map(|s| (*s, false)));
    calibrate_threshold(&all, |(input, _)| fired(input), |(_, label)| *label, 0.5)
}

fn fired(input: &&str) -> f64 {
    if strip_thinking_blocks(input) != *input {
        1.0
    } else {
        0.0
    }
}

#[test]
fn m10_2_think_precision_recall() {
    // Full-spec set: M1 controls + roadmap tick controls. Measured on HEAD:
    // the three tagged tick cases fire (task-content deletion — see
    // CALIBRATION.md); everything else holds.
    let full = think_report(THINK_RECALL, &[THINK_CONTROLS_M1, THINK_CONTROLS_TICKS].concat());
    // Locked (measured HEAD values; see CALIBRATION.md): recall is total,
    // but the three tagged tick controls fire — precision 8/11, FPR 3/11,
    // `passes_gate()` false. Think-stripping has NOT earned truncate/cache/
    // persist trust; the code-span guard is follow-up work (no M10
    // production change by milestone rule).
    assert_eq!((full.tp, full.r#fn), (8, 0), "recall must be total");
    assert_eq!((full.fp, full.tn), (3, 8), "measured FP locked");
    assert!((full.precision - 8.0 / 11.0).abs() < 1e-12, "{}", full.precision);
    assert_eq!(full.recall, 1.0);
    assert!((full.fpr - 3.0 / 11.0).abs() < 1e-12, "{}", full.fpr);
    assert!(!full.passes_gate(), "gate must stay red until the FP is fixed");
    // M1-subset: the previously-earned behavior stays clean.
    let m1 = think_report(THINK_RECALL, THINK_CONTROLS_M1);
    assert_eq!((m1.fp, m1.tn), (0, 6));
    assert!(m1.passes_gate(), "M1-subset must pass: {m1:?}");
}

// ─── M10.3: SSE framing properties ──────────────────────────────────────────

const SSE_PAYLOADS: &[&str] = &[
    "data: 안녕하세요 🌍🌏🌎\n",
    "data: {\"delta\": \"日本語テスト😀\"}\ndata: more\n",
    "event: ping\r\ndata: 混合mix😀done\n",
];

fn want_lines(payload: &str) -> Vec<String> {
    let mut lines: Vec<String> = payload
        .split('\n')
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect();
    // A trailing newline terminates the last line; it is not an extra line.
    if payload.ends_with('\n') {
        lines.pop();
    }
    lines
}

#[test]
fn m10_3_sse_arbitrary_splits_reassemble() {
    let mut split_points = 0usize;
    for payload in SSE_PAYLOADS {
        let bytes = payload.as_bytes();
        let want = want_lines(payload);
        // Every byte offset as a 2-chunk split (mid-codepoint splits included).
        for split in 0..=bytes.len() {
            let mut buf = Vec::new();
            let mut got = drain_sse_lines(&mut buf, &bytes[..split]);
            got.extend(drain_sse_lines(&mut buf, &bytes[split..]));
            assert_eq!(got, want, "2-way split at {split} of {payload:?}");
            assert!(buf.is_empty(), "tail at split {split}");
            split_points += 1;
        }
        // Byte-by-byte feed (worst-case chunking).
        let mut buf = Vec::new();
        let mut got = Vec::new();
        for b in bytes {
            got.extend(drain_sse_lines(&mut buf, &[*b]));
        }
        assert_eq!(got, want, "byte-wise feed of {payload:?}");
        assert!(buf.is_empty(), "tail after byte-wise feed");
        split_points += bytes.len();
    }
    for line in SSE_PAYLOADS.iter().flat_map(|p| want_lines(p)) {
        assert!(!line.contains('\u{FFFD}'), "no replacement char: {line:?}");
    }
    // Locked: every 2-way split plus every byte-wise step reassembles.
    assert_eq!(split_points, 253, "SSE split-point coverage moved");
    // No-newline chunk drains zero and preserves bytes (transport
    // completeness is never a verdict).
    let mut buf = Vec::new();
    assert!(drain_sse_lines(&mut buf, "partial — 安".as_bytes()).is_empty());
    assert_eq!(buf, "partial — 安".as_bytes());
}

// ─── M10.4: cache identity ──────────────────────────────────────────────────

fn mem_store() -> (
    std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, CachedResponse>>>,
    impl Fn(&str) -> Option<CachedResponse>,
    impl Fn(&str, &CachedResponse),
) {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    let store = Arc::new(Mutex::new(HashMap::<String, CachedResponse>::new()));
    let s1 = store.clone();
    let s2 = store.clone();
    let check = move |k: &str| s1.lock().unwrap().get(k).cloned();
    let persist = move |k: &str, v: &CachedResponse| {
        s2.lock().unwrap().insert(k.to_string(), v.clone());
    };
    (store, check, persist)
}

#[test]
fn m10_4_cache_identity_probes() {
    use std::time::Duration;
    let (store, check, persist) = mem_store();
    let cache = ResponseCache::new(Some(Duration::from_secs(60)), check, persist);
    let req = r#"{"messages":[{"role":"user","content":"hi"}]}"#;

    // Fresh set hits with the stored value.
    cache.set("modelA", req, serde_json::json!({"ok": true}));
    assert_eq!(
        cache.get("modelA", req).map(|e| e.response_json),
        Some(serde_json::json!({"ok": true}))
    );
    // Key format probe: `{model}:{64 lowercase hex}`.
    let observed: Vec<String> = store.lock().unwrap().keys().cloned().collect();
    assert_eq!(observed.len(), 1);
    assert!(observed[0].starts_with("modelA:"), "key: {}", observed[0]);
    assert_eq!(observed[0].len(), "modelA:".len() + 64);

    // Cross-model same-text must MISS (identity, not similarity).
    assert!(cache.get("modelB", req).is_none());
    // Unknown key must MISS.
    assert!(cache.get("modelA", r#"{"other":1}"#).is_none());
    // Expired entry must MISS even on identical text.
    {
        let mut map = store.lock().unwrap();
        for entry in map.values_mut() {
            entry.stored_at_secs -= 61;
        }
    }
    assert!(cache.get("modelA", req).is_none());

    // TTL boundary: age == ttl misses (`>=`), age == ttl - 1 hits.
    let now = common_core::time::now_secs();
    {
        let mut map = store.lock().unwrap();
        for entry in map.values_mut() {
            entry.stored_at_secs = now - 60;
        }
    }
    assert!(cache.get("modelA", req).is_none(), "age == ttl must miss");
    {
        let mut map = store.lock().unwrap();
        for entry in map.values_mut() {
            entry.stored_at_secs = now - 59;
        }
    }
    assert!(cache.get("modelA", req).is_some(), "age == ttl-1 must hit");

    // Key independence: neighboring keys never cross-talk.
    cache.set("modelA", r#"{"q":1}"#, serde_json::json!(1));
    cache.set("modelA", r#"{"q":2}"#, serde_json::json!(2));
    cache.set("modelB", r#"{"q":1}"#, serde_json::json!(3));
    assert_eq!(
        cache.get("modelA", r#"{"q":1}"#).map(|e| e.response_json),
        Some(serde_json::json!(1))
    );
    assert_eq!(
        cache.get("modelA", r#"{"q":2}"#).map(|e| e.response_json),
        Some(serde_json::json!(2))
    );
    assert_eq!(
        cache.get("modelB", r#"{"q":1}"#).map(|e| e.response_json),
        Some(serde_json::json!(3))
    );

    // Backend-absent (malformed/uncached) must MISS.
    let gone = ResponseCache::new(None, |_: &str| None, |_: &str, _: &CachedResponse| {});
    assert!(gone.get("modelA", req).is_none());
}
