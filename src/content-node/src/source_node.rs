use crate::file_node::FileContentNode;
use crate::node::{ContentNode, LodLevel, NodeType, NodeTypeInfo};
use fluent_types::GuidanceDoc;
use fluent_wvr::prelude::*;

const SOURCE_LOD_LABELS: &[&str] = &[
    "path",
    "AST member summaries",
    "full GuidanceDoc JSON",
    "",
    "",
    "",
];

#[derive(Debug)]
pub struct SourceCodeContentNode {
    inner: FileContentNode,
    ast_doc: Option<GuidanceDoc>,
}

impl SourceCodeContentNode {
    pub fn new(inner: FileContentNode) -> Self {
        Self {
            inner,
            ast_doc: None,
        }
    }

    #[must_use]
    pub fn with_ast(mut self, doc: GuidanceDoc) -> Self {
        self.ast_doc = Some(doc);
        self
    }

    pub fn inner(&self) -> &FileContentNode {
        &self.inner
    }
    pub fn ast_doc(&self) -> Option<&GuidanceDoc> {
        self.ast_doc.as_ref()
    }
}

impl ContentNode for SourceCodeContentNode {
    fn node_type(&self) -> NodeType {
        NodeType::SourceCode
    }
    fn lod(&self, level: LodLevel) -> Option<&str> {
        match level {
            LodLevel::Source | LodLevel::Name => self.inner.path().to_str(),
            LodLevel::Detailed => self.ast_doc.as_ref().map(|_| "<AST>"),
            LodLevel::Summary => self
                .ast_doc
                .as_ref()
                .map(|d| d.comment.as_deref().unwrap_or("")),
            _ => None,
        }
    }
    fn set_lod(&mut self, _level: LodLevel, _value: &str) {}
    fn lod_label(&self, level: LodLevel) -> Option<&str> {
        SOURCE_LOD_LABELS.get(level as usize).copied()
    }
    fn type_info(&self) -> NodeTypeInfo {
        NodeTypeInfo {
            kind: NodeType::SourceCode,
            name: "SourceCodeContentNode",
            lod_labels: SOURCE_LOD_LABELS,
        }
    }
}

impl WorkUnit for SourceCodeContentNode {
    fn name(&self) -> &str {
        "SourceCodeContentNode"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        WorkOutput::typed("content node loaded", &self.inner.path().to_string_lossy())
    }
}

impl FieldAccess for SourceCodeContentNode {
    fn set_field(&mut self, name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(name.to_string()))
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "path" => Ok(self.inner.path().to_string_lossy().to_string()),
            "has_ast" => Ok((self.ast_doc.is_some()).to_string()),
            _ => Err(FieldError::NotFound(name.to_string())),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["path", "has_ast"]
    }
}

impl Describable for SourceCodeContentNode {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path":    { "type": "string", "description": "Source file path" },
                "has_ast": { "type": "boolean", "description": "Whether AST doc is loaded" }
            }
        })
    }
}

impl_component!(SourceCodeContentNode);
