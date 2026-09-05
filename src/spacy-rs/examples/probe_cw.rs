//! Temporary probe: per-item content errors joined with oracle tie signal.
use std::collections::BTreeMap;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Item { id: String, category: String, text: String }
#[derive(Debug, Deserialize)]
struct Dataset { items: Vec<Item> }
#[derive(Debug, Deserialize)]
struct RefRec { #[allow(dead_code)] text: String, pos: String, dep: String, head: i32 }
#[derive(Debug, Deserialize)]
struct Refs { refs: BTreeMap<String, Vec<RefRec>> }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let dataset: Dataset = serde_json::from_str(&std::fs::read_to_string(dir.join("parse_bench.json"))?)?;
    let refs: Refs = serde_json::from_str(&std::fs::read_to_string(dir.join("parse_bench.refs.json"))?)?;
    let pipe = spacy_rs::NlpPipeline::en_default()?;
    let owned: Vec<String> = std::env::args().skip(1).collect();
    // Content errors: exclude punct + prep/case convention.
    let mut content: Vec<(String, String, usize, usize)> = vec![];
    for item in &dataset.items {
        let Some(gold) = refs.refs.get(&item.id) else { continue };
        let (_doc, result) = pipe.process_sync_with_confidence(
            &item.text, None, None, spacy_rs::RefinePolicy::default())?;
        let set = result.records;
        if set.0.len() != gold.len() { continue; }
        let ties = result.parse_confidence.as_ref().map(|c| c.oracle_tie_count).unwrap_or(999);
        let mut cerr = 0;
        for (p, g) in set.0.iter().zip(gold.iter()) {
            if g.pos == "punct" { continue; }
            if p.pos == "adp" && g.pos == "adp" && (g.dep == "case" || p.dep == "prep") { continue; }
            if p.pos != g.pos || p.head != g.head || p.dep != g.dep { cerr += 1; }
        }
        if cerr > 0 {
            content.push((item.category.clone(), item.id.clone() + " :: " + &item.text, cerr, ties));
        }
    }
    content.sort_by(|a, b| b.2.cmp(&a.2));
    for r in &content {
        println!("{:<20} cerr={} ties={} | {}", r.0, r.2, r.3, r.1);
    }
    if !owned.is_empty() {
        println!("\n=== TOKEN DETAIL ===");
        for item in &dataset.items {
            if !owned.iter().any(|f| f == &item.id) { continue; }
            let Some(gold) = refs.refs.get(&item.id) else { continue };
            let (_doc, result) = pipe.process_sync_with_confidence(
                &item.text, None, None, spacy_rs::RefinePolicy::default())?;
            let set = result.records;
            println!("--- {} :: {}", item.id, item.text);
            for (p, g) in set.0.iter().zip(gold.iter()) {
                let mark = if p.pos != g.pos || p.head != g.head || p.dep != g.dep { "  << MISMATCH" } else { "" };
                println!("  {:<10} pred {}/{}/{} | ref {}/{}/{}{}", p.text, p.pos, p.dep, p.head, g.pos, g.dep, g.head, mark);
            }
        }
    }
    Ok(())
}
