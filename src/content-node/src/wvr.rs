//! Fluent WVR integration for `content-node`.
//!
//! Provides the `ContentNodeRef` wrapper type with downcasting support
//! for implementing `WorkUnit` and related traits from `fluent-wvr`.
//! New content-node types that need Fluent WVR integration should implement
//! the traits in their own module.

use crate::node::{ContentNode, LodLevel, NodeType, NodeTypeInfo};
use std::fmt::Debug;

#[derive(Debug)]
pub struct ContentNodeRef {
    inner: Box<dyn ContentNode>,
}

impl ContentNodeRef {
    pub fn new(node: impl ContentNode + 'static) -> Self {
        Self {
            inner: Box::new(node),
        }
    }

    pub fn node_type(&self) -> NodeType {
        self.inner.node_type()
    }
    pub fn lod(&self, level: LodLevel) -> Option<&str> {
        self.inner.lod(level)
    }
    pub fn type_info(&self) -> NodeTypeInfo {
        self.inner.type_info()
    }

    pub fn downcast_ref<T: ContentNode + 'static>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref::<T>()
    }

    pub fn downcast_mut<T: ContentNode + 'static>(&mut self) -> Option<&mut T> {
        self.inner.as_any_mut().downcast_mut::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_node::FileContentNode;
    use crate::node::NodeType;
    use std::path::PathBuf;

    #[test]
    fn content_node_ref_downcasting() {
        let node = FileContentNode::new(
            PathBuf::from("test.txt"),
            42,
            common_core::hash::blake3_hash(b"test"),
        );
        let wrapper = ContentNodeRef::new(node);
        assert_eq!(wrapper.node_type(), NodeType::File);
        let downcast = wrapper.downcast_ref::<FileContentNode>();
        assert!(downcast.is_some());
        assert_eq!(downcast.unwrap().inode(), 42);
    }
}
