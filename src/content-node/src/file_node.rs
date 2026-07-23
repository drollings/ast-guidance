use std::fmt::Debug;
use std::path::{Path, PathBuf};

use crate::node::{ContentNode, LodLevel, NodeType, NodeTypeInfo};
use fluent_wvr::prelude::*;

const FILE_LOD_LABELS: &[&str] = &["path", "inode+hash", "", "", "", ""];

#[derive(Debug)]
pub struct FileContentNode {
    path: PathBuf,
    inode: u64,
    hash: [u8; 32],
}

impl FileContentNode {
    pub fn new(path: PathBuf, inode: u64, hash: [u8; 32]) -> Self {
        Self { path, inode, hash }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn inode(&self) -> u64 {
        self.inode
    }
    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }
}

impl ContentNode for FileContentNode {
    fn node_type(&self) -> NodeType {
        NodeType::File
    }
    fn lod(&self, level: LodLevel) -> Option<&str> {
        match level {
            LodLevel::Name | LodLevel::Source => Some(self.path.to_str()?),
            _ => None,
        }
    }
    fn set_lod(&mut self, _level: LodLevel, _value: &str) {}
    fn lod_label(&self, level: LodLevel) -> Option<&str> {
        FILE_LOD_LABELS.get(level as usize).copied()
    }
    fn type_info(&self) -> NodeTypeInfo {
        NodeTypeInfo {
            kind: NodeType::File,
            name: "FileContentNode",
            lod_labels: FILE_LOD_LABELS,
        }
    }
}

impl WorkUnit for FileContentNode {
    fn name(&self) -> &str {
        "FileContentNode"
    }
    fn depends(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn provides(&self) -> &[ArcIntern<str>] {
        &[]
    }
    fn execute(&self, _ctx: &WorkContext) -> Result<WorkOutput, WorkError> {
        WorkOutput::typed("content node loaded", &self.path.to_string_lossy())
    }
}

impl FieldAccess for FileContentNode {
    fn set_field(&mut self, name: &str, _value: &str) -> Result<(), FieldError> {
        Err(FieldError::NotFound(name.to_string()))
    }
    fn get_field(&self, name: &str) -> Result<String, FieldError> {
        match name {
            "path" => Ok(self.path.to_string_lossy().to_string()),
            "inode" => Ok(self.inode.to_string()),
            "hash" => Ok(common_core::hash::hex_encode(&self.hash)),
            _ => Err(FieldError::NotFound(name.to_string())),
        }
    }
    fn field_names(&self) -> &'static [&'static str] {
        &["path", "inode", "hash"]
    }
}

impl Describable for FileContentNode {
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path":  { "type": "string", "description": "File path" },
                "inode": { "type": "integer", "description": "Inode number" },
                "hash":  { "type": "string", "description": "Blake3 hash (hex)" }
            }
        })
    }
}

impl_component!(FileContentNode);