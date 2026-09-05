//! Interactive ArcEager parse inspector (fully offline — no LLM).
//!
//! ```sh
//! cargo run -p spacy-rs --example parse -- "The cat sat on the mat."
//! ```
//!
//! Runs the deterministic ladder (tokenizer → rule → ArcEager → sentencizer)
//! with the default `RefinePolicy::default()` (`mode: Off`, refiners never
//! consulted) and prints the token table (text / UPOS / head / dep / lemma),
//! the winning rung provenance, the parse confidence, and the full
//! `AnnotationSet` JSON — i.e. what the router's `NlpStage` sees before any
//! refiner could touch it.

use spacy_rs::{NlpPipeline, RefinePolicy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let text = if text.is_empty() {
        "The cat sat on the mat.".to_string()
    } else {
        text
    };

    let pipe = NlpPipeline::en_default()?;
    let (_doc, result) =
        pipe.process_sync_with_confidence(&text, None, None, RefinePolicy::default())?;

    // The table reads the same `AnnotationSet` the JSON below serializes,
    // so the two views cannot disagree (`head` is the relative offset from
    // the record contract: `token.i + head == head_index`).
    println!("{:<4} {:<16} {:<8} {:<6} {:<10} lemma", "i", "text", "upos", "head", "dep");
    for (i, rec) in result.records.0.iter().enumerate() {
        println!(
            "{:<4} {:<16} {:<8} {:<6} {:<10} {}",
            i, rec.text, rec.pos, rec.head, rec.dep, rec.lemma
        );
    }
    println!("\nwinning rung : {:?}", result.source);
    if let Some(pc) = &result.parse_confidence {
        println!(
            "confidence  : overall={:.3} role_coverage={:.3} ties={} margins={:?}",
            pc.overall, pc.role_coverage, pc.oracle_tie_count, pc.oracle_margins
        );
    }
    println!("\nannotations :\n{}", serde_json::to_string_pretty(&result.records)?);
    Ok(())
}
