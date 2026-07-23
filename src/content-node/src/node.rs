use fluent_types::LOD_COUNT;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    Source = 0,
    Detailed = 1,
    Summary = 2,
    Brief = 3,
    Tiny = 4,
    Name = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    File,
    SourceCode,
    Document,
}

pub struct NodeTypeInfo {
    pub kind: NodeType,
    pub name: &'static str,
    pub lod_labels: &'static [&'static str],
}

pub trait ContentNode: Debug + Send + Sync {
    fn node_type(&self) -> NodeType;
    fn lod_count(&self) -> usize {
        LOD_COUNT
    }
    fn lod(&self, level: LodLevel) -> Option<&str>;
    fn set_lod(&mut self, level: LodLevel, value: &str);
    fn lod_label(&self, level: LodLevel) -> Option<&str>;
    fn type_info(&self) -> NodeTypeInfo;
}