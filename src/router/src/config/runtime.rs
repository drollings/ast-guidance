//! RuntimeConfig (M8).
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub rigor: Option<crate::config::RigorConfig>,
}
