use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use common_core::retry::{PollResult, PollWithBackoff};

/// Unified health probe for a llama-server instance.
///
/// Collapses the previous five spellings (`probe_health`, `probe_health_with_client`,
/// `wait_healthy`, `wait_healthy_with_client`, plus the manual `liveness_failures`
/// counter in `supervise`) into one code path. The single `PollWithBackoff` generic
/// is the only poll shape.
pub struct HealthProbe {
    pub client: reqwest::Client,
    pub base_url: String,
    pub poll: PollWithBackoff,
    pub threshold: u32,
    pub interval: Duration,
}

impl HealthProbe {
    pub fn new(
        client: reqwest::Client,
        base_url: String,
        poll: PollWithBackoff,
        threshold: u32,
        interval: Duration,
    ) -> Self {
        Self {
            client,
            base_url,
            poll,
            threshold,
            interval,
        }
    }

    pub fn for_wait(client: reqwest::Client, base_url: String) -> Self {
        let poll = PollWithBackoff::new(super::HEALTH_POLL, 1)
            .with_max_failures((super::HEALTH_TIMEOUT.as_secs()) as u32);
        Self::new(client, base_url, poll, 0, super::HEALTH_POLL)
    }

    pub fn for_liveness(client: reqwest::Client, base_url: String, interval: Duration, threshold: u32) -> Self {
        let poll = PollWithBackoff::new(interval, 1);
        Self::new(client, base_url, poll, threshold, interval)
    }

    /// Single health probe primitive: GET /health → 2xx.
    pub async fn probe_once(&self) -> bool {
        self.client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
    }

    /// Wait until healthy, driven by the shared `PollWithBackoff` config.
    pub async fn wait_until_healthy(&self) -> Result<(), String> {
        let base_url = self.base_url.clone();
        match self
            .poll
            .run(|| async { self.probe_once().await })
            .await
        {
            PollResult::Ready => Ok(()),
            PollResult::Exhausted { .. } => Err(format!(
                "llama-server did not become healthy on {base_url} within deadline"
            )),
        }
    }

    /// Post-boot liveness loop: probes every `poll.base` interval, counting consecutive
    /// failures against `threshold`. Returns `true` when the threshold is tripped (hung).
    pub async fn wait_for_hung(&self, stopping: &AtomicBool) -> bool {
        let mut failures: u32 = 0;
        loop {
            // Use the poll's base interval as the liveness poll interval (constant when cap=1)
            tokio::time::sleep(self.poll_interval()).await;
            if stopping.load(Ordering::Relaxed) {
                return false;
            }
            if self.probe_once().await {
                failures = 0;
            } else {
                failures += 1;
                if failures >= self.threshold {
                    return true;
                }
            }
        }
    }

    fn poll_interval(&self) -> Duration {
        self.interval
    }
}
