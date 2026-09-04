use content_node::lod::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_lod_slices_short_text() {
        let slices = generate_lod_slices("Hello");
        assert_eq!(slices.len(), 6);
        for s in &slices {
            assert_eq!(s, "Hello");
        }
    }

    #[test]
    fn generate_lod_slices_truncates_summary() {
        let long = "A. ".repeat(500);
        let text = &long[..long.len() - 2];
        let slices = generate_lod_slices(text);
        assert!(slices[0].len() > 800);
        assert!(slices[1].len() <= 800);
        assert!(slices[2].len() <= 240);
        assert!(slices[3].len() <= 80);
        assert!(slices[4].len() <= 40);
    }

    #[test]
    fn generate_lod_slices_sentence_boundary() {
        let text = "First sentence. Second sentence that is longer. Third sentence.";
        let slices = generate_lod_slices(text);
        assert!(slices[1].ends_with('.'));
    }
}
