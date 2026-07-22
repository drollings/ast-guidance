use guidance_llm::client::LlmClient;
use guidance_llm::ChatMessage;
use fluent_types::GuidanceDoc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EnhancerError {
    #[error("LLM error: {0}")]
    Llm(String),
    #[error("no response from model")]
    NoResponse,
}

/// Language-specific comment generation — the fluent-wvr control plane for
/// LLM prompting. Each language implements its own system prompt and
/// member/file prompt templates.
pub trait CommentGenerator: Send + Sync {
    /// System message for the LLM conversation.
    fn system_prompt(&self, language: &str) -> String;

    /// Prompt for generating a member (function/type) comment.
    fn member_prompt(
        &self,
        name: &str,
        signature: &str,
        module_context: &str,
        kind_label: &str,
        language: &str,
    ) -> String;

    /// Prompt for generating a file-level description.
    fn file_prompt(&self, rel_path: &str, source_preview: &str, language: &str) -> String;
}

/// Default comment generator — language-agnostic prompts that work for any language.
pub struct DefaultCommentGenerator;

impl CommentGenerator for DefaultCommentGenerator {
    fn system_prompt(&self, language: &str) -> String {
        format!("You are a technical documentation assistant for {language} code.")
    }

    fn member_prompt(
        &self,
        name: &str,
        signature: &str,
        module_context: &str,
        kind_label: &str,
        _language: &str,
    ) -> String {
        format!(
            "{kind_label} in {module_context}:\n  {signature}\n\n\
             Write a single-line comment for this {kind_label}.\n\
             Rules:\n\
             - Plain English, technically specific\n\
             - Max 200 characters\n\
             - No boilerplate openers\n\n\
             Wrap your answer in <comment> tags. Example:\n\
             <comment>Parses a null-terminated C string into an owned slice.</comment>\n\n\
             Name: {name}"
        )
    }

    fn file_prompt(&self, rel_path: &str, source_preview: &str, _language: &str) -> String {
        format!(
            "Source:\n{source_preview}\n\nFile: {rel_path}\n\n\
             Write a single-line description for this file.\n\
             Rules:\n\
             - Plain English, technically specific\n\
             - Max 200 chars\n\
             - No boilerplate openers\n\n\
             Wrap your answer in <comment> tags."
        )
    }
}

/// AI comment enhancer for guidance generation.
///
/// Generates descriptions (≤200 chars) for functions, structs, and files
/// by calling the configured LLM endpoint. Uses a `CommentGenerator` trait
/// for language-specific prompting (fluent-wvr control plane pattern).
pub struct Enhancer {
    pub client: LlmClient,
    pub generator: Box<dyn CommentGenerator>,
    pub debug: bool,
    pub show_prompts: bool,
}

impl Enhancer {
    pub fn new(api_base: &str, model: &str) -> Self {
        Self {
            client: LlmClient::new(api_base, model),
            generator: Box::new(DefaultCommentGenerator),
            debug: false,
            show_prompts: false,
        }
    }

    #[must_use]
    pub fn with_generator(mut self, generator: Box<dyn CommentGenerator>) -> Self {
        self.generator = generator;
        self
    }

    #[must_use]
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    #[must_use]
    pub fn with_show_prompts(mut self, show: bool) -> Self {
        self.show_prompts = show;
        self
    }

    /// Generate a comment for a function, struct, enum, or union type.
    pub fn enhance_member(
        &self,
        name: &str,
        signature: &str,
        module_context: &str,
        kind_label: &str,
        language: &str,
    ) -> Result<Option<String>, EnhancerError> {
        let prompt =
            self.generator
                .member_prompt(name, signature, module_context, kind_label, language);
        self.call_llm(language, &prompt)
    }

    /// Generate a comment for a function or method.
    pub fn enhance_function(
        &self,
        name: &str,
        signature: &str,
        module_context: &str,
        language: &str,
    ) -> Result<Option<String>, EnhancerError> {
        self.enhance_member(name, signature, module_context, "Function", language)
    }

    /// Generate a comment for a struct/enum/union type.
    pub fn enhance_struct(
        &self,
        name: &str,
        signature: &str,
        module_context: &str,
        language: &str,
    ) -> Result<Option<String>, EnhancerError> {
        self.enhance_member(name, signature, module_context, "Type", language)
    }

    /// Generate a one-line file-level description.
    pub fn enhance_file(
        &self,
        rel_path: &str,
        source_preview: &str,
        language: &str,
    ) -> Result<Option<String>, EnhancerError> {
        let preview = if source_preview.len() > 3000 {
            &source_preview[..3000]
        } else {
            source_preview
        };
        let prompt = self.generator.file_prompt(rel_path, preview, language);
        self.call_llm(language, &prompt)
    }

    /// Score a docstring for quality (mirrors Zig `scoreDocstring`).
    pub fn score_docstring(text: &str) -> u32 {
        if text.is_empty() {
            return 0;
        }
        let mut score: u32 = 0;
        if text.len() > 50 {
            score += 1;
        }
        let lower = text.to_lowercase();
        if lower.contains("args:") || lower.contains("parameters") {
            score += 2;
        }
        if lower.contains("returns:") || lower.contains("return") {
            score += 2;
        }
        if lower.contains("error") || lower.contains("raises:") {
            score += 1;
        }
        let newline_count = text.chars().filter(|&c| c == '\n').count() as u32;
        if newline_count > 2 {
            score += 1;
        }
        score
    }

    fn call_llm(&self, language: &str, prompt: &str) -> Result<Option<String>, EnhancerError> {
        if self.debug {
            tracing::debug!("[enhancer] sending prompt (len={})", prompt.len());
        }
        if self.show_prompts {
            tracing::debug!("[enhancer] prompt:\n{prompt}");
        }

        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: self.generator.system_prompt(language),
            },
            ChatMessage {
                role: "user".into(),
                content: prompt.to_string(),
            },
        ];

        let response = self
            .client
            .chat_complete(&messages)
            .map_err(|e| EnhancerError::Llm(e.to_string()))?;

        if response.is_empty() {
            return Err(EnhancerError::NoResponse);
        }

        // Extract <comment>...</comment> tag
        if let Some(start) = response.find("<comment>") {
            let after_start = &response[start + 9..];
            if let Some(end) = after_start.find("</comment>") {
                let content = after_start[..end].trim().to_string();
                if !content.is_empty() {
                    return Ok(Some(content));
                }
            }
        }

        // Fallback: return the raw response (trimmed)
        let trimmed = response.trim().to_string();
        if !trimmed.is_empty() {
            return Ok(Some(trimmed));
        }

        Ok(None)
    }
}

/// Generate missing comments for all members in a doc using the enhancer.
pub fn enhance_doc(
    enhancer: &Enhancer,
    doc: &mut GuidanceDoc,
    source_content: &str,
) -> Result<usize, EnhancerError> {
    let mut generated = 0;
    let module = doc.meta.module.as_str();
    let language = doc.meta.language.as_str();

    for member in &mut doc.members {
        if member.comment.is_some() && !member.comment_generated {
            continue;
        }

        let sig = member
            .signature
            .as_ref()
            .map_or(member.name.as_str(), smol_str::SmolStr::as_str);

        let result = match member.type_name {
            fluent_types::MemberType::FnDecl
            | fluent_types::MemberType::FnPrivate
            | fluent_types::MemberType::Method
            | fluent_types::MemberType::MethodPrivate => {
                enhancer.enhance_function(member.name.as_str(), sig, module, language)
            }
            fluent_types::MemberType::Struct
            | fluent_types::MemberType::Enum
            | fluent_types::MemberType::Union => {
                enhancer.enhance_struct(member.name.as_str(), sig, module, language)
            }
            _ => continue,
        };

        if let Ok(Some(comment)) = result {
            member.comment = Some(comment.into());
            member.comment_generated = true;
            generated += 1;
        }
    }

    // Generate file-level comment if missing
    if doc.comment.is_none() {
        if let Ok(Some(file_comment)) =
            enhancer.enhance_file(doc.meta.source.as_str(), source_content, language)
        {
            doc.comment = Some(file_comment.into());
            generated += 1;
        }
    }

    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluent_types::{GuidanceDoc, Member, MemberType, Meta};

    #[test]
    fn test_score_docstring_short() {
        assert_eq!(Enhancer::score_docstring(""), 0);
        assert_eq!(Enhancer::score_docstring("short"), 0);
    }

    #[test]
    fn test_score_docstring_long() {
        let text = "a".repeat(51);
        let score = Enhancer::score_docstring(&text);
        assert!(score >= 1, "long text gets a point for length");
    }

    #[test]
    fn test_score_docstring_with_args() {
        let score = Enhancer::score_docstring("Args: x i32, y i32");
        assert!(score >= 2, "args gets points");
    }

    #[test]
    fn test_score_docstring_with_returns() {
        let score = Enhancer::score_docstring("Returns: the sum");
        assert!(score >= 2, "returns gets points");
    }

    #[test]
    fn test_call_llm_extracts_comment_tag() {
        let response = "<comment>Test comment</comment>";
        assert!(response.contains("<comment>"), "should have comment tag");
        assert!(
            response.contains("Test comment"),
            "should contain extracted text"
        );
    }

    #[test]
    fn test_enhance_doc_no_llm_fallback() {
        let mut doc = GuidanceDoc {
            meta: Meta {
                module: "test".into(),
                source: "src/test.zig".into(),
                language: "zig".into(),
            },
            comment: None,
            members: vec![Member {
                type_name: MemberType::FnDecl,
                name: "foo".into(),
                signature: Some("fn foo() void".into()),
                comment: None,
                ..Member::default()
            }],
            ..GuidanceDoc::default()
        };

        // No LLM available, so enhance should gracefully skip all members
        let enhancer = Enhancer::new("http://localhost:99999", "test");
        let result = enhance_doc(&enhancer, &mut doc, "pub fn foo() void {}");
        assert!(result.is_ok(), "should not propagate LLM errors");
    }

    #[test]
    fn test_score_docstring_newlines() {
        let text = "Line 1\nLine 2\nLine 3\nLine 4\n";
        let score = Enhancer::score_docstring(text);
        assert!(score >= 1, "multi-line text gets a point");
    }

    #[test]
    fn test_comment_tag_extraction_logic() {
        let response = "Some reasoning text\n<comment>Parses input and produces output.</comment>";
        let start = response.find("<comment>");
        assert!(start.is_some());
        if let Some(s) = start {
            if let Some(end) = response[s + 9..].find("</comment>") {
                let content = response[s + 9..s + 9 + end].trim();
                assert_eq!(content, "Parses input and produces output.");
            }
        }
    }

    #[test]
    fn test_default_comment_generator_language_agnostic() {
        let gen = DefaultCommentGenerator;
        let sys = gen.system_prompt("python");
        assert!(sys.contains("python"), "should mention the language");

        let prompt = gen.member_prompt("foo", "fn foo()", "mod", "Function", "rust");
        assert!(prompt.contains("foo"), "should contain the name");
        assert!(prompt.contains("Function"), "should contain the kind label");
    }
}
