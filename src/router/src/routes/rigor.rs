use std::sync::Arc;

use crate::kv_cache::KvCacheManager;
use crate::orchestrator::OrchestratorSession;

pub struct RigorRoute {
    kv_cache: Option<Arc<KvCacheManager>>,
}

#[derive(Debug, Clone)]
pub struct RigoResult {
    pub blue_answer: String,
    pub red_objections: Vec<RedObjection>,
    pub judge_verdict: JudgeVerdict,
    pub frontier_escalation: bool,
}

#[derive(Debug, Clone)]
pub struct RedObjection {
    pub category: String,
    pub description: String,
    pub severity: f64,
}

#[derive(Debug, Clone)]
pub enum JudgeVerdict {
    Accept,
    AcceptWithCaveats { caveats: Vec<String> },
    Reject { reasons: Vec<String> },
}

impl Default for RigorRoute {
    fn default() -> Self {
        Self::new()
    }
}

impl RigorRoute {
    #[must_use]
    pub fn new() -> Self {
        Self { kv_cache: None }
    }

    #[must_use]
    pub fn with_kv_cache(mut self, cache: Arc<KvCacheManager>) -> Self {
        self.kv_cache = Some(cache);
        self
    }

    /// Execute the 3-pass rigor protocol (§7):
    /// 1. Blue team: produce candidate answer
    /// 2. Checkpoint KV cache (so we can rewind if red team wins)
    /// 3. Red team: adversarial scrutiny
    /// 4. Judge: score blue vs red, produce verdict
    /// 5. If judge confidence low → frontier escalation (§8.4)
    /// 6. If red team wins materially → interview user
    pub fn execute(
        &self,
        _session: &OrchestratorSession,
        _user_message: &str,
    ) -> Result<RigoResult, RigorError> {
        Err(RigorError::NotImplemented)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RigorError {
    #[error("not yet implemented")]
    NotImplemented,
    #[error("blue team error: {0}")]
    BlueTeam(String),
    #[error("red team error: {0}")]
    RedTeam(String),
    #[error("judge error: {0}")]
    Judge(String),
}
