use common_core::tokens::estimate_tokens;

/// LOD level labels mapping `generate_lod_slices` indices.
const LOD_LABELS: &[&str] = &["Source", "Detailed", "Summary", "Brief", "Tiny", "Name"];

/// Select the best level of detail that fits within `token_budget`.
///
/// Returns `(level_index, lod_label)`. Level 0 is the full source text;
/// higher levels are progressively shorter sentence-boundary truncations.
/// If nothing fits, returns the shortest level (index 4, "Tiny").
pub fn select_lod(page_context: &str, token_budget: usize) -> (usize, &'static str) {
    let slices = content_node::generate_lod_slices(page_context);
    // Levels 0–4 map to indices 0–4; skip level 5 (alias of full text).
    let limit = 5.min(slices.len());
    for (level, slice) in slices.iter().take(limit).enumerate() {
        if estimate_tokens(slice) <= token_budget {
            return (level, LOD_LABELS[level]);
        }
    }
    // Budget is too tight — return the shortest level.
    let fallback = 4.min(slices.len() - 1);
    (fallback, LOD_LABELS[fallback])
}

/// Chunk `page_context` to fit within `token_budget` using LOD slicing.
///
/// Returns the selected chunk. Empty input returns empty string.
pub fn chunk_page_context(page_context: &str, token_budget: usize) -> String {
    if page_context.is_empty() {
        return String::new();
    }
    let slices = content_node::generate_lod_slices(page_context);
    // Find best level (0–4).
    let limit = 5.min(slices.len());
    for slice in slices.iter().take(limit) {
        if estimate_tokens(slice) <= token_budget {
            return slice.clone();
        }
    }
    // Fallback to shortest.
    slices[4.min(slices.len() - 1)].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_context_returns_full_text() {
        let short = "This is a short context.";
        let result = chunk_page_context(short, 300);
        assert_eq!(result, short);
    }

    #[test]
    fn short_context_select_lod_level_zero() {
        let short = "Short text.";
        let (level, label) = select_lod(short, 300);
        assert_eq!(level, 0);
        assert_eq!(label, "Source");
    }

    #[test]
    fn long_context_returns_truncated() {
        // ~2400 chars → exceeds 300-token budget at level 0.
        let long = "Sentence. ".repeat(300);
        let long = long.trim_end();
        let result = chunk_page_context(long, 300);
        // Should be truncated from the full text.
        assert!(result.len() < long.len());
    }

    #[test]
    fn tight_budget_returns_shortest() {
        // Very tight budget — should fall back to Tiny LOD (40ch).
        let long = "A sentence with enough words to be longer than forty characters. ".repeat(50);
        let long = long.trim_end();
        let result = chunk_page_context(long, 5);
        assert!(result.len() <= 40);
    }

    #[test]
    fn empty_context_returns_empty() {
        assert_eq!(chunk_page_context("", 300), "");
    }

    #[test]
    fn select_lod_returns_known_label() {
        let (_, label) = select_lod("Some text", 100);
        assert!(LOD_LABELS.contains(&label));
    }

    #[test]
    fn tight_budget_select_lod_level_four() {
        let long = "Word ".repeat(500);
        let long = long.trim_end();
        let (level, label) = select_lod(long, 3);
        assert_eq!(level, 4);
        assert_eq!(label, "Tiny");
    }
}
