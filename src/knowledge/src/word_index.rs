use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct WordHit {
    pub doc_id: u32,
    pub line_num: u32,
}

pub const WORD_INDEX_MAGIC: u32 = 0x574F_5244;
pub const WORD_INDEX_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct DocRegistry {
    path_to_id: HashMap<String, u32>,
    id_to_path: Vec<String>,
}

impl DocRegistry {
    pub fn new() -> Self {
        Self {
            path_to_id: HashMap::new(),
            id_to_path: Vec::new(),
        }
    }

    pub fn get_or_create(&mut self, path: &str) -> u32 {
        if let Some(&id) = self.path_to_id.get(path) {
            return id;
        }
        let id = self.id_to_path.len() as u32;
        self.path_to_id.insert(path.to_string(), id);
        self.id_to_path.push(path.to_string());
        id
    }

    pub fn path_for_id(&self, id: u32) -> &str {
        self.id_to_path.get(id as usize).map_or("", String::as_str)
    }

    pub fn count(&self) -> u32 {
        self.id_to_path.len() as u32
    }
}

impl Default for DocRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct WordIndex {
    index: HashMap<String, Vec<WordHit>>,
    registry: DocRegistry,
    file_words: HashMap<u32, HashMap<String, ()>>,
}

impl WordIndex {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            registry: DocRegistry::new(),
            file_words: HashMap::new(),
        }
    }
}

impl Default for WordIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl WordIndex {
    pub fn hit_path(&self, hit: &WordHit) -> &str {
        self.registry.path_for_id(hit.doc_id)
    }

    pub fn index_file(&mut self, path: &str, content: &str) {
        let doc_id = self.registry.get_or_create(path);
        let mut file_word_set: HashMap<String, ()> = HashMap::new();
        for (line_num, line) in content.lines().enumerate() {
            let tokens: Vec<&str> = crate::tokenizer::WordTokenizer::new(line).collect();
            for token in tokens {
                let normalized = token.to_lowercase();
                if normalized.len() < 2 {
                    continue;
                }
                let sub_tokens = crate::tokenizer::split_identifier(token);
                for sub in &sub_tokens {
                    self.index_one_token(sub, doc_id, line_num as u32, &mut file_word_set);
                }
                self.index_one_token(&normalized, doc_id, line_num as u32, &mut file_word_set);
            }
        }
        self.file_words.insert(doc_id, file_word_set);
    }

    fn index_one_token(
        &mut self,
        token: &str,
        doc_id: u32,
        line_num: u32,
        file_word_set: &mut HashMap<String, ()>,
    ) {
        if token.len() < 2 {
            return;
        }
        file_word_set.insert(token.to_string(), ());
        self.index
            .entry(token.to_string())
            .or_default()
            .push(WordHit { doc_id, line_num });
    }

    pub fn search(&self, word: &str) -> Vec<&WordHit> {
        let lower = word.to_lowercase();
        self.index
            .get(&lower)
            .map(|hits| hits.iter().collect())
            .unwrap_or_default()
    }

    pub fn search_prefix(&self, prefix: &str) -> Vec<&WordHit> {
        let lower = prefix.to_lowercase();
        let mut results = Vec::new();
        for (word, hits) in &self.index {
            if word.starts_with(&lower) {
                results.extend(hits.iter());
            }
        }
        results
    }

    pub fn remove_file(&mut self, path: &str) {
        if let Some(&doc_id) = self.registry.path_to_id.get(path) {
            if let Some(words) = self.file_words.remove(&doc_id) {
                for word in words.keys() {
                    if let Some(hits) = self.index.get_mut(word) {
                        hits.retain(|h| h.doc_id != doc_id);
                        if hits.is_empty() {
                            self.index.remove(word);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_index_and_search() {
        let mut wi = WordIndex::new();
        wi.index_file("test.txt", "hello world");
        let hits = wi.search("hello");
        assert!(!hits.is_empty());
        assert_eq!(wi.hit_path(hits[0]), "test.txt");
    }

    #[test]
    fn doc_registry_roundtrip() {
        let mut reg = DocRegistry::new();
        let id = reg.get_or_create("/path/to/file.rs");
        assert_eq!(reg.path_for_id(id), "/path/to/file.rs");
    }

    #[test]
    fn search_prefix() {
        let mut wi = WordIndex::new();
        wi.index_file("a.txt", "hello world");
        let hits = wi.search_prefix("hel");
        assert!(!hits.is_empty());
    }

    #[test]
    fn remove_file() {
        let mut wi = WordIndex::new();
        wi.index_file("a.txt", "hello world");
        wi.remove_file("a.txt");
        assert!(wi.search("hello").is_empty());
    }
}
