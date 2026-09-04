use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    Person,
    Project,
    Location,
    Uncertain,
}

#[derive(Debug, Clone)]
pub struct EntityFreq {
    pub name: String,
    pub frequency: usize,
    pub entity_type: EntityType,
}

use std::sync::LazyLock;

static ENTITY_STOPLIST: LazyLock<std::collections::HashSet<&'static str>> = LazyLock::new(|| {
    let mut s = std::collections::HashSet::new();
    for w in &[
        "The",
        "This",
        "That",
        "When",
        "Where",
        "What",
        "Why",
        "Who",
        "Which",
        "How",
        "Then",
        "There",
        "Here",
        "Now",
        "Just",
        "Also",
        "Some",
        "Such",
        "Each",
        "Every",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ] {
        s.insert(*w);
    }
    s
});

fn is_capitalized(s: &str) -> bool {
    s.chars().next().is_some_and(char::is_uppercase)
}

fn is_entity_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-' || c == '\'' || c == '.'
}

pub fn candidate_entity_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if is_entity_word_char(c) {
            current.push(c);
        } else {
            if current.len() >= 2
                && is_capitalized(&current)
                && !ENTITY_STOPLIST.contains(current.as_str())
            {
                words.push(current.clone());
            }
            current.clear();
        }
    }
    if current.len() >= 2 && is_capitalized(&current) && !ENTITY_STOPLIST.contains(current.as_str())
    {
        words.push(current);
    }
    words
}

pub fn extract_entities(content: &str, min_frequency: usize) -> Vec<EntityFreq> {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for word in candidate_entity_words(content) {
        *freq.entry(word).or_insert(0) += 1;
    }
    let mut result: Vec<EntityFreq> = freq
        .into_iter()
        .filter(|(_, count)| *count >= min_frequency)
        .map(|(name, frequency)| EntityFreq {
            name,
            frequency,
            entity_type: EntityType::Uncertain,
        })
        .collect();
    result.sort_by_key(|b| std::cmp::Reverse(b.frequency));
    result
}

#[cfg(test)]
#[path = "../tests/entity.rs"]
mod tests;
