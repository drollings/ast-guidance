pub struct WordTokenizer<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> WordTokenizer<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }
}

impl<'a> Iterator for WordTokenizer<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.bytes.len() && !self.bytes[self.pos].is_ascii_alphanumeric() {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_alphanumeric() {
            self.pos += 1;
        }
        let token = std::str::from_utf8(&self.bytes[start..self.pos]).ok()?;
        Some(token)
    }
}

pub fn split_identifier(ident: &str) -> Vec<String> {
    let bytes = ident.as_bytes();
    if bytes.is_empty() || bytes.len() < 2 {
        return Vec::new();
    }
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i] as char;
        if !c.is_ascii_alphanumeric() {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            i += 1;
            continue;
        }
        if c.is_ascii_uppercase() && !current.is_empty() {
            let next_is_lower = i + 1 < bytes.len() && (bytes[i + 1] as char).is_ascii_lowercase();
            if next_is_lower {
                parts.push(std::mem::take(&mut current));
            }
        }
        current.push(c);
        i += 1;
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

pub fn normalize_char(c: char) -> char {
    match c {
        'A'..='Z' => (c as u8 + 32) as char,
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_basic() {
        let tokens: Vec<&str> = WordTokenizer::new("hello world").collect();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn tokenizer_with_symbols() {
        let tokens: Vec<&str> = WordTokenizer::new("hello, world! test").collect();
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }

    #[test]
    fn split_identifier_snake_case() {
        let parts = split_identifier("hello_world");
        assert!(parts.contains(&"hello".to_string()));
        assert!(parts.contains(&"world".to_string()));
    }

    #[test]
    fn split_identifier_camel_case() {
        let parts = split_identifier("helloWorld");
        assert!(parts.contains(&"hello".to_string()));
        assert!(parts.contains(&"World".to_string()));
    }

    #[test]
    fn split_identifier_pascal_case() {
        let parts = split_identifier("HelloWorld");
        assert!(parts.contains(&"Hello".to_string()));
        assert!(parts.contains(&"World".to_string()));
    }

    #[test]
    fn split_identifier_short_returns_empty() {
        let parts = split_identifier("a");
        assert!(parts.is_empty());
    }

    #[test]
    fn normalize_char_cases() {
        assert_eq!(normalize_char('A'), 'a');
        assert_eq!(normalize_char('z'), 'z');
    }
}
