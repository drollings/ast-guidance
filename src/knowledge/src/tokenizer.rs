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
#[path = "../tests/tokenizer.rs"]
mod tests;
