pub fn generate_lod_slices(full_text: &str) -> Vec<String> {
    let targets = [usize::MAX, 800, 240, 80, 40, usize::MAX];
    let mut slices = Vec::with_capacity(targets.len());
    for &max_chars in &targets {
        if max_chars == usize::MAX || full_text.len() <= max_chars {
            slices.push(full_text.to_string());
        } else {
            slices.push(common_core::string::truncate_at_sentence(
                full_text, max_chars,
            ));
        }
    }
    slices
}
