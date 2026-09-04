//! Opt-in live-AI reference generator for the parse bench.
//!
//! This test performs REAL model calls. It is compiled only when the
//! `live-ai` feature is enabled and is `#[ignore]`d so it can never run under
//! `make test` / `make router-test` / `make router-mock` / CI. Run it via
//! `make test-live` (or `make spacy-test-live`).
//!
//! Env contract (same shape as `refine_live.rs`):
//! - `SPACY_LIVE_LLM_URL` — an OpenAI-compatible chat-completions endpoint.
//! - `SPACY_LIVE_LLM_MODEL` — model name (defaults to `local`).
//!
//! When the URL is absent the test skips cleanly (early `return`, never
//! panic). For every dataset item in `tests/data/parse_bench.json` that has
//! no pinned reference yet, it asks the model for UD annotations in the
//! ladder's own `AnnotationRecord` JSON shape, validates the reply (one
//! record per deterministic token, verbatim texts), and merges the new refs
//! into `tests/data/parse_bench.refs.json`.
//!
//! Generated refs are DRAFTS: review the diff, correct by hand, then re-run
//! `make spacy-parse-benchmark`. The bench (hermetic) is the ratchet; this
//! generator only proposes.

use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// The live-endpoint URL; unset (or empty) skips the test.
fn live_url() -> Option<String> {
    std::env::var("SPACY_LIVE_LLM_URL").ok().filter(|u| !u.is_empty())
}

fn live_model() -> String {
    std::env::var("SPACY_LIVE_LLM_MODEL").unwrap_or_else(|_| "local".to_string())
}

const SYSTEM_PROMPT: &str = r#"You are a Universal Dependencies annotator. Given TOKENS (exactly tokenized, one per line as `i: text`), emit a JSON array with exactly one object per token, in order, no extra text:
{"text": <verbatim token>, "pos": <upos>, "dep": <label>, "head": <relative offset>, "lemma": <base form lowercase>}
- pos: one of adj adp adv aux cconj det intj noun num part pron propn punct sconj sym verb x.
- dep: one of nsubj dobj iobj compound prep pobj det aux root punct dep amod advmod cc conj ccomp mark cop advcl parataxis relcl neg poss case discourse appos acomp attr oprd obj obl xcomp csubj expl nummod acl amod.
- head: token_index + head == head_index; the single root has head 0. Prefer precise labels over `dep`; use `dep` only when truly uncertain.
- Imperatives head the verb as root; copulas (`is/was/become/taste`) take dep `cop` with the predicate adjective/noun as root; subordinating `as/if/because/when` take dep `mark`; negatives like `n't` take dep `neg`; second clauses after `;` attach as `parataxis`.
Example:
TOKENS:
0: She
1: runs
JSON: [{"text":"She","pos":"pron","dep":"nsubj","head":1,"lemma":"she"},{"text":"runs","pos":"verb","dep":"root","head":0,"lemma":"run"}]"#;

/// Extract the first JSON array from a reply (tolerates markdown fences).
fn extract_array(text: &str) -> Option<String> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end > start {
        Some(text[start..=end].to_string())
    } else {
        None
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "live-AI: requires SPACY_LIVE_LLM_URL; run via `make test-live`"]
async fn generate_missing_parse_bench_refs() {
    let Some(url) = live_url() else {
        eprintln!("skipping: SPACY_LIVE_LLM_URL unset");
        return;
    };
    let model = live_model();

    let raw = std::fs::read_to_string(data_dir().join("parse_bench.json")).expect("dataset readable");
    let dataset: serde_json::Value = serde_json::from_str(&raw).expect("dataset parses");
    let raw = std::fs::read_to_string(data_dir().join("parse_bench.refs.json")).expect("refs readable");
    let mut refs: serde_json::Value = serde_json::from_str(&raw).expect("refs parse");
    let have = refs
        .get("refs")
        .and_then(|r| r.as_object())
        .map(|o| o.len())
        .unwrap_or(0);

    // Deterministic tokenization (the texts refs must match verbatim).
    let pipe = spacy_rs::NlpPipeline::en_default().expect("en pipeline");
    let client = reqwest::Client::new();
    let mut added = 0usize;
    let mut skipped = 0usize;

    let items = dataset.get("items").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    for item in items {
        let id = item.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
        let text = item.get("text").and_then(|v| v.as_str()).unwrap_or_default().to_string();
        if refs.get("refs").and_then(|r| r.get(&id)).is_some() {
            continue;
        }
        let doc = match pipe.process_sync(&text, None) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{id}: deterministic tokenize failed ({e}) — skipping");
                skipped += 1;
                continue;
            }
        };
        let tokens: Vec<String> = (0..doc.len()).map(|i| doc.token_text(i)).collect();
        let numbered = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{i}: {t}"))
            .collect::<Vec<_>>()
            .join("\n");
        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format!("TOKENS:\n{numbered}")},
            ],
        });
        let reply = match client.post(&url).json(&body).send().await {
            Ok(r) => r.text().await.unwrap_or_default(),
            Err(e) => {
                eprintln!("{id}: http error ({e}) — skipping");
                skipped += 1;
                continue;
            }
        };
        // OpenAI envelope or raw text.
        let content = serde_json::from_str::<serde_json::Value>(&reply)
            .ok()
            .and_then(|v| {
                v.get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(str::to_string)
            })
            .unwrap_or(reply);
        let Some(array) = extract_array(&content) else {
            eprintln!("{id}: no JSON array in reply — skipping");
            skipped += 1;
            continue;
        };
        let records: Vec<serde_json::Value> = match serde_json::from_str(&array) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{id}: reply is not a JSON array ({e}) — skipping");
                skipped += 1;
                continue;
            }
        };
        // Validate: one record per token, verbatim texts.
        let texts_ok = records.len() == tokens.len()
            && records.iter().zip(tokens.iter()).all(|(r, t)| {
                r.get("text").and_then(|v| v.as_str()) == Some(t.as_str())
            });
        if !texts_ok {
            eprintln!("{id}: record texts/len mismatch vs tokenizer — skipping (do not pin)");
            skipped += 1;
            continue;
        }
        refs["refs"][&id] = serde_json::Value::Array(records);
        added += 1;
        eprintln!("{id}: draft ref accepted (REVIEW BEFORE PINNING)");
    }

    if added > 0 {
        let out = serde_json::to_string_pretty(&refs).expect("refs serialize");
        std::fs::write(data_dir().join("parse_bench.refs.json"), out).expect("refs writable");
    }
    eprintln!("ref-gen done: had {have} refs, added {added} drafts, skipped {skipped}.");
    eprintln!("Next: `git diff` the refs file, correct by hand, re-run `make spacy-parse-benchmark`.");
}
