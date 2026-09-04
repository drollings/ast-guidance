//! LedgerGroupConfig (M8).
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LedgerGroupConfig {
    #[serde(default)]
    pub ledger: Option<crate::config::LedgerConfig>,
}
