use std::collections::HashMap;

use guidance_llm::anonymize;

use crate::transforms::{rewrite_text_messages, TransformError, TransformStrategy};
use crate::types::RouterRequest;

pub struct PiiAnonymize;

impl TransformStrategy for PiiAnonymize {
    fn name(&self) -> &str {
        "pii_anonymize"
    }

    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError> {
        let mut anonymize_map: HashMap<String, String> = HashMap::new();

        let mut transformed = rewrite_text_messages(request, |content| {
            let anonymized = anonymize(content);
            build_anonymize_map(content, &anonymized, &mut anonymize_map);
            Ok(anonymized)
        })?;

        if !anonymize_map.is_empty() {
            let map_obj: serde_json::Map<String, serde_json::Value> = anonymize_map
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect();
            transformed
                .metadata
                .insert("anonymize_map".into(), serde_json::Value::Object(map_obj));
        }

        Ok(transformed)
    }
}

fn build_anonymize_map(original: &str, anonymized: &str, map: &mut HashMap<String, String>) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let orig_bytes = original.as_bytes();
    let anon_bytes = anonymized.as_bytes();
    let mut orig_idx = 0;
    let mut anon_idx = 0;

    while orig_idx < orig_bytes.len() && anon_idx < anon_bytes.len() {
        if orig_bytes[orig_idx] == anon_bytes[anon_idx] {
            orig_idx += 1;
            anon_idx += 1;
            continue;
        }

        let remaining_orig = &original[orig_idx..];
        let placeholder = extract_placeholder(&anonymized[anon_idx..]);
        if let Some(ph) = placeholder {
            let ph_len = ph.len();
            let matched_len = find_matching_len(remaining_orig, ph);
            if matched_len > 0 {
                let original_value = &remaining_orig[..matched_len];
                let count = counts.entry(ph).or_insert(0);
                let key = if *count == 0 {
                    ph.to_string()
                } else {
                    format!("{ph}_{count}")
                };
                *count += 1;
                map.insert(key, original_value.to_string());

                orig_idx += matched_len;
                anon_idx += ph_len;
                continue;
            }
        }

        orig_idx += 1;
        anon_idx += 1;
    }
}

fn extract_placeholder(s: &str) -> Option<&str> {
    if s.starts_with('[') {
        if let Some(end) = s.find(']') {
            let candidate = &s[..=end];
            if candidate.len() <= 64 {
                return Some(candidate);
            }
        }
    }
    None
}

fn find_matching_len(text: &str, _placeholder: &str) -> usize {
    let non_anon_end = text
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric() || c.is_ascii_punctuation())
        .map_or(text.len(), |(i, c)| i + c.len_utf8());
    non_anon_end.max(1)
}
