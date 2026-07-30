//! Route resolution and routing configuration types.
//! `RoutingConfig` is the resolved routing table used by the classifier stage.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::ModelEntry;
use crate::score_matrix::ScoreMatrix;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRef {
    pub group: String,
    #[serde(default = "default_pipelines")]
    pub pipelines: Vec<String>,
}

fn default_pipelines() -> Vec<String> {
    vec!["default".into()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub routes: HashMap<String, RouteRef>,
    pub models: HashMap<String, ModelEntry>,
    pub model_groups: HashMap<String, Vec<String>>,
    pub system_prompt: String,
    pub safety_threshold: f64,
    pub default_route: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_matrix: Option<ScoreMatrix>,
}

impl RoutingConfig {
    pub fn resolve_route(
        &self,
        route_name: &str,
        min_complexity: Option<u8>,
    ) -> Option<(&ModelEntry, String)> {
        let route_ref = self
            .routes
            .get(route_name)
            .or_else(|| self.routes.get(&self.default_route));

        let route_ref = match route_ref {
            Some(r) => Some(r),
            None => {
                return self.models.get(route_name).map(|entry| {
                    let name = entry.name.clone().unwrap_or_else(|| route_name.to_string());
                    tracing::info!(target: "router.config", route = %route_name, model = %name,
                        "route resolved as direct model"
                    );
                    (entry, name)
                }).or_else(|| {
                    tracing::warn!(target: "router.config", route = %route_name,
                        default = %self.default_route,
                        "no route or model found for target"
                    );
                    None
                });
            }
        };

        let route_ref = route_ref?;

        let model_names = self.model_groups.get(route_ref.group.as_str());
        let Some(model_names) = model_names else {
            tracing::warn!(target: "router.config", route = %route_name, group = %route_ref.group, "model group not found for route");
            return None;
        };

        tracing::debug!(target: "router.config",
            route = %route_name,
            group = %route_ref.group,
            model_count = model_names.len(),
            min_complexity = ?min_complexity,
            "resolving route"
        );

        let candidates: Vec<(&String, &ModelEntry)> = model_names
            .iter()
            .filter_map(|n| self.models.get(n).map(|m| (n, m)))
            .filter(|(_, m)| {
                m.intelligence >= min_complexity.unwrap_or(0)
            })
            .collect();

        if candidates.is_empty() {
            tracing::debug!(target: "router.config", route = %route_name, "no candidates passed complexity filter, falling back to cheapest in group");
            model_names
                .iter()
                .filter_map(|n| self.models.get(n).map(|m| (n, m)))
                .min_by(|(_, a), (_, b)| {
                    (a.cost_input + a.cost_output)
                        .partial_cmp(&(b.cost_input + b.cost_output))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(entry_key, entry)| {
                    let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
                    tracing::info!(target: "router.config", route = %route_name, model = %name, "route resolved (cheapest fallback)");
                    (entry, name)
                })
        } else {
            let (entry_key, entry) = candidates
                .into_iter()
                .min_by(|(_, a), (_, b)| {
                    (a.cost_input + a.cost_output)
                        .partial_cmp(&(b.cost_input + b.cost_output))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })?;
            let name = entry.name.clone().unwrap_or_else(|| entry_key.clone());
            tracing::info!(target: "router.config", route = %route_name, model = %name, "route resolved");
            Some((entry, name))
        }
    }
}
