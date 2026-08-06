//! fluent-types: Shared data types (GuidanceDoc, Member, FileType, etc.).
//! No business logic — only type definitions, serde, and validation.

use internment::ArcIntern;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

pub mod knowledge;

pub use knowledge::{KnowledgeCapability, KnowledgeError};

pub const LOD_COUNT: usize = 6;

/// A deterministic context hit — a short-circuit fact/cached answer that
/// lets the escalation ladder avoid a frontier call.  `content` is the
/// verified answer/fact; `source` names the cache that produced it (e.g. 
/// "coral-context" or "ledger") for audit provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextHit {
    pub source: String,
    pub content: String,
    #[serde(default)]
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Deterministic-fact lookup used by the escalation ladder **before** engaging
/// the frontier (VISION: "a `ContextHit` short-circuits").
///
/// This is the cross-crate boundary for context retrieval: `fluent-router`
/// consumes context through this trait only and never imports coral /
/// knowledge / ontology / rdf. Concrete implementations (coral-context's
/// cache, the router's ledger-backed cache) are composed at the binary.
pub trait ContextCache: Send + Sync {
    /// Look up a deterministic answer/fact for `query`. `None` means the
    /// cache has no short-circuit hit and the ladder must escalate.
    fn lookup(&self, query: &str) -> Option<ContextHit>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub i64);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub ArcIntern<str>);

impl NodeId {
    pub fn from_int(i: i64) -> Self {
        Self(i)
    }
    pub fn as_int(self) -> i64 {
        self.0
    }
}

impl SessionId {
    pub fn new(s: impl Into<ArcIntern<str>>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileType {
    Source,
    Markdown,
    Config,
    Data,
    Pdf,
    Audio,
    Unknown,
}

impl FileType {
    pub fn from_extension(ext: &str) -> Self {
        let known_source = [
            ".zig", ".zon", ".py", ".rs", ".go", ".ts", ".tsx", ".js", ".jsx", ".c", ".cpp", ".h",
            ".hpp", ".lua", ".rb", ".java", ".swift", ".kt",
        ];
        let known_markdown = [".md", ".markdown", ".mdx"];
        let known_config = [
            ".json", ".toml", ".yaml", ".yml", ".ini", ".env", ".cfg", ".conf",
        ];
        let known_data = [".csv", ".tsv", ".sqlite", ".db", ".sql", ".parquet"];
        let known_pdf = [".pdf"];
        let known_audio = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma"];
        let ext = ext.to_lowercase();
        if known_source.iter().any(|e| ext == *e) {
            return Self::Source;
        }
        if known_markdown.iter().any(|e| ext == *e) {
            return Self::Markdown;
        }
        if known_config.iter().any(|e| ext == *e) {
            return Self::Config;
        }
        if known_data.iter().any(|e| ext == *e) {
            return Self::Data;
        }
        if known_pdf.iter().any(|e| ext == *e) {
            return Self::Pdf;
        }
        if known_audio.iter().any(|e| ext == *e) {
            return Self::Audio;
        }
        Self::Unknown
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Markdown => "markdown",
            Self::Config => "config",
            Self::Data => "data",
            Self::Pdf => "pdf",
            Self::Audio => "audio",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemberType {
    #[default]
    FnDecl,
    FnPrivate,
    Struct,
    Enum,
    Union,
    EnumField,
    TestDecl,
    ComptimeBlock,
    Method,
    MethodPrivate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    Depends,
    Provides,
    Imports,
    Exports,
    Calls,
    CalledBy,
    Extends,
    Implements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    Prose,
    Code,
    Metadata,
    Insight,
    SkillDoc,
    CapabilityDoc,
    NotFound,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Param {
    #[serde(default)]
    pub name: SmolStr,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_name: Option<SmolStr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<SmolStr>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Member {
    #[serde(default)]
    pub type_name: MemberType,
    #[serde(default)]
    pub name: SmolStr,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub match_hash: Option<SmolStr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<SmolStr>,
    #[serde(default)]
    pub params: Vec<Param>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returns: Option<SmolStr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<SmolStr>,
    #[serde(default)]
    pub tags: Vec<SmolStr>,
    #[serde(default)]
    pub is_pub: bool,
    #[serde(default)]
    pub members: Vec<Member>,
    #[serde(default)]
    pub equivalents: Vec<SmolStr>,
    #[serde(skip)]
    pub line: Option<u32>,
    #[serde(default)]
    pub comment_generated: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Span {
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Skill {
    #[serde(default)]
    pub ref_path: SmolStr,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<SmolStr>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Meta {
    #[serde(default)]
    pub module: SmolStr,
    #[serde(default)]
    pub source: SmolStr,
    #[serde(default = "default_language")]
    pub language: SmolStr,
}

fn default_language() -> SmolStr {
    SmolStr::new("zig")
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilityEval {
    #[serde(default)]
    pub capability_name: SmolStr,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub evaluated_at_hash: SmolStr,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GuidanceDoc {
    #[serde(default)]
    pub meta: Meta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<SmolStr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<SmolStr>,
    #[serde(default)]
    pub keywords: Vec<SmolStr>,
    #[serde(default)]
    pub skills: Vec<Skill>,
    #[serde(default)]
    pub capabilities: Vec<SmolStr>,
    #[serde(default)]
    pub hashtags: Vec<SmolStr>,
    #[serde(default)]
    pub used_by: Vec<SmolStr>,
    #[serde(default)]
    pub members: Vec<Member>,
    #[serde(default)]
    pub equivalents: Vec<SmolStr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capability_eval: Option<CapabilityEval>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentNode {
    // ── Core fields ──
    pub id: Option<NodeId>,
    pub name: SmolStr,
    pub source: String,
    #[serde(default)]
    pub lod: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<u8>>,
    // ── Session fields ──
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_lod: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<NodeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_status: Option<StepStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
}

impl ContentNode {
    pub fn content(&self) -> Option<&str> {
        self.lod.first().map(String::as_str)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnHit {
    pub node_id: NodeId,
    pub distance: f32,
    pub name: SmolStr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub node_id: NodeId,
    pub name: SmolStr,
    pub depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    pub kind: StageKind,
    pub content: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub query: String,
    pub stages: Vec<Stage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    pub filepath: String,
    pub members_added: usize,
    pub members_updated: usize,
    pub members_removed: usize,
    pub has_changes: bool,
    pub comments_generated: bool,
    pub source_modified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMatch {
    pub filename: SmolStr,
    pub filepath: SmolStr,
    #[serde(default)]
    pub description: SmolStr,
    #[serde(default)]
    pub line_context: SmolStr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASTAnalysis {
    pub filepath: SmolStr,
    #[serde(default)]
    pub functions: Vec<Member>,
    #[serde(default)]
    pub classes: Vec<Member>,
    #[serde(default)]
    pub imports: Vec<SmolStr>,
    #[serde(default)]
    pub patterns_detected: Vec<SmolStr>,
    #[serde(default)]
    pub token_count: usize,
    #[serde(default)]
    pub signature_preview: SmolStr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidanceInfo {
    pub path: SmolStr,
    #[serde(default)]
    pub comment: SmolStr,
    #[serde(default)]
    pub functions: Vec<Member>,
    #[serde(default)]
    pub classes: Vec<Member>,
    #[serde(default)]
    pub skills: Vec<SmolStr>,
    #[serde(default)]
    pub tags: Vec<SmolStr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetType {
    File,
    Phony,
    Abstract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorKind {
    Native,
    Docker,
    Wasm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmTool {
    pub name: SmolStr,
    pub path: String,
    pub capabilities: Vec<SmolStr>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_roundtrip() {
        let id = NodeId::from_int(42);
        assert_eq!(id.as_int(), 42);
    }
    #[test]
    fn session_id_roundtrip() {
        let id = SessionId::new("test_session");
        assert_eq!(id.as_str(), "test_session");
    }
    #[test]
    fn types_are_distinct() {
        fn takes_node(_: NodeId) {}
        fn takes_session(_: &SessionId) {}
        takes_node(NodeId::from_int(1));
        let sess = SessionId::new("test");
        takes_session(&sess);
    }
    #[test]
    fn file_type_from_extension() {
        assert_eq!(FileType::from_extension(".zig"), FileType::Source);
        assert_eq!(FileType::from_extension(".md"), FileType::Markdown);
        assert_eq!(FileType::from_extension(".json"), FileType::Config);
        assert_eq!(FileType::from_extension(".xyz"), FileType::Unknown);
    }
    #[test]
    fn file_type_to_str() {
        assert_eq!(FileType::Source.to_str(), "source");
        assert_eq!(FileType::Markdown.to_str(), "markdown");
        assert_eq!(FileType::Config.to_str(), "config");
        assert_eq!(FileType::Data.to_str(), "data");
        assert_eq!(FileType::Pdf.to_str(), "pdf");
        assert_eq!(FileType::Audio.to_str(), "audio");
        assert_eq!(FileType::Unknown.to_str(), "unknown");
    }
    #[test]
    fn file_type_from_extension_data_pdf_audio() {
        assert_eq!(FileType::from_extension(".csv"), FileType::Data);
        assert_eq!(FileType::from_extension(".db"), FileType::Data);
        assert_eq!(FileType::from_extension(".pdf"), FileType::Pdf);
        assert_eq!(FileType::from_extension(".mp3"), FileType::Audio);
        assert_eq!(FileType::from_extension(".wav"), FileType::Audio);
    }
    #[test]
    fn member_serde_roundtrip() {
        let m = Member {
            type_name: MemberType::FnDecl,
            name: SmolStr::new("foo"),
            match_hash: None,
            signature: Some(SmolStr::new("fn foo()")),
            params: vec![],
            returns: None,
            comment: Some(SmolStr::new("test function")),
            tags: vec![SmolStr::new("public")],
            is_pub: true,
            members: vec![],
            equivalents: vec![],
            line: Some(10),
            comment_generated: false,
        };
        let json = serde_json::to_string(&m).unwrap();
        let m2: Member = serde_json::from_str(&json).unwrap();
        assert_eq!(m.name, m2.name);
        assert_eq!(m.is_pub, m2.is_pub);
    }
    #[test]
    fn guidance_doc_default_language_on_deserialize() {
        let json = r#"{"meta":{"module":"test","source":"test.zig"}}"#;
        let doc: GuidanceDoc = serde_json::from_str(json).unwrap();
        assert_eq!(doc.meta.language.as_str(), "zig");
    }
    #[test]
    fn guidance_doc_serde_roundtrip() {
        let doc = GuidanceDoc {
            meta: Meta {
                module: SmolStr::new("test"),
                source: SmolStr::new("test.zig"),
                language: SmolStr::new("zig"),
            },
            comment: Some(SmolStr::new("module comment")),
            detail: None,
            keywords: vec![SmolStr::new("api")],
            skills: vec![],
            capabilities: vec![],
            hashtags: vec![],
            used_by: vec![],
            members: vec![],
            equivalents: vec![],
            capability_eval: None,
        };
        let json = serde_json::to_string(&doc).unwrap();
        let doc2: GuidanceDoc = serde_json::from_str(&json).unwrap();
        assert_eq!(doc.meta.module, doc2.meta.module);
        assert_eq!(doc.comment, doc2.comment);
    }
    #[test]
    fn knn_hit_creation() {
        let hit = KnnHit {
            node_id: NodeId::from_int(1),
            distance: 0.95,
            name: SmolStr::new("test"),
        };
        assert_eq!(hit.node_id.as_int(), 1);
    }
    #[test]
    fn stage_kind_serde() {
        let kinds = [StageKind::Prose, StageKind::Code, StageKind::NotFound];
        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let k2: StageKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, k2);
        }
    }
    #[test]
    fn target_type_serde() {
        let t = TargetType::File;
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "\"file\"");
        let t2: TargetType = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }
    #[test]
    fn executor_kind_serde() {
        let e = ExecutorKind::Wasm;
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(json, "\"wasm\"");
    }
    #[test]
    fn content_node_defaults() {
        let node = ContentNode {
            id: None,
            name: SmolStr::new("root"),
            source: "full text".into(),
            lod: vec!["hello".into()],
            embedding: None,
            capabilities: None,
            session_id: None,
            request_id: None,
            role: None,
            turn_index: None,
            accepted: None,
            acceptance_score: None,
            active_lod: None,
            parent_id: None,
            step_id: None,
            step_status: None,
            metadata: None,
            created_at: None,
        };
        assert_eq!(node.lod.len(), 1);
        assert_eq!(node.content(), Some("hello"));
        assert!(node.embedding.is_none());
    }
    #[test]
    fn wasm_tool_serde() {
        let tool = WasmTool {
            name: SmolStr::new("tokenizer"),
            path: "/usr/local/bin/guidance-tokenizer.wasm".into(),
            capabilities: vec![SmolStr::new("tokenize")],
        };
        let json = serde_json::to_string(&tool).unwrap();
        let t2: WasmTool = serde_json::from_str(&json).unwrap();
        assert_eq!(tool.name, t2.name);
    }
    #[test]
    fn context_hit_roundtrip() {
        let hit = ContextHit {
            source: "ledger".into(),
            content: "2+2 is 4".into(),
            score: 0.97,
            metadata: Some(serde_json::json!({ "node_id": 42 })),
        };
        let json = serde_json::to_string(&hit).unwrap();
        let back: ContextHit = serde_json::from_str(&json).unwrap();
        assert_eq!(back, hit);
        assert_eq!(back.source, "ledger");
    }
    #[test]
    fn context_hit_defaults_metadata_to_none() {
        let hit: ContextHit = serde_json::from_str(r#"{"source":"c","content":"x","score":0.5}"#)
            .unwrap();
        assert!(hit.metadata.is_none());
        assert!((hit.score - 0.5).abs() < f32::EPSILON);
    }
}
