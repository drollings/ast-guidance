//! ServingConfig sub-config (M8) — models, sidecar, onnx, gguf.
use serde::{Deserialize, Serialize};
use crate::config::{ModelEntry, SidecarConfig};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServingConfig {
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    #[serde(default)]
    pub sidecar: Option<SidecarConfig>,
    #[serde(default)]
    pub onnx: Option<String>,
}
