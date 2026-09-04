//! The public `/instances` aggregation surface Coral Router exposes at its OWN
//! address as the single sidecar entry point (the managed servers bind to
//! `127.0.0.1` and are never exposed directly).
//!
//! Public instance ids are `<model_id>:<instance_name>`; `total` is summed
//! with 64-bit arithmetic with each model's shared weights counted once. This
//! module holds the `InstancePool` aggregation/proxy methods (`aggregate`,
//! `list_models`, `create`, and the per-instance operation proxies).

use std::sync::Arc;

use serde_json::Value;

use super::client::{InstanceError, InstanceInfo, InstanceTotals, SnapshotInfo};
use super::manager::{
    instance_name_from_server_id, resume_snapshot_name, InstanceManager,
};
use super::instance_aliases;
use super::pool::InstancePool;
use fluent_types::instance_id::InstanceId;

impl InstancePool {

    /// Resolve the public instance id grammar `<model_id>:<name>` (or a bare
    /// `<model_id>` + name) to `(model key, instance name)`. `None` when the
    /// model is unmanaged, the id has more than one `:`, or the instance name
    /// breaks the `[A-Za-z0-9._-]` grammar.
    pub fn resolve_instance_id(&self, id: &str) -> Option<(String, String)> {
        let parsed = InstanceId::parse(id).ok()?;
        self.managers.get(&(**parsed.model()).to_string())?;
        Some(((**parsed.model()).to_string(), (**parsed.name()).to_string()))
    }

    /// `GET /instances` - the aggregate envelope across every managed model.
    /// `model: Some(...)` scopes the response to one model. Instance ids are
    /// `<model_id>:<name>`; snapshot entries are tagged with their owning
    /// `model`; `total` sums each server's envelope with 64-bit arithmetic.
    /// Plain (no-instance-grammar) models contribute a synthesized footprint
    /// (their shared weights; 0 when the fork reports the model sleeping).
    pub async fn aggregate(&self, model: Option<&str>) -> Result<Value, InstanceError> {
        let mut instances = Vec::new();
        let mut snapshots: Vec<Value> = Vec::new();
        let mut total = InstanceTotals::default();
        for model_key in self.managers.keys() {
            let Some(manager) = self.managers.get(&model_key).map(|m| m.as_ref().clone()) else {
                continue;
            };
            if let Some(filter) = model {
                if filter != model_key.as_str() {
                    continue;
                }
            }
            let Some((envelope, _plain)) = manager.list_with_fallback().await else {
                tracing::debug!(
                    target: "router.instances",
                    model = %model_key,
                    "aggregate /instances poll skipped - server down",
                );
                continue;
            };
            for info in envelope.instances {
                let instance_name = instance_name_from_server_id(&info.id);
                let instance_id = format!("{model_key}:{instance_name}");
                let aliases =
                    instance_aliases(&model_key, &instance_id, &info.group, info.is_default);
                let mut entry = serde_json::to_value(&info).unwrap_or_default();
                if let Value::Object(ref mut obj) = entry {
                    obj.insert("id".into(), Value::String(instance_id));
                    obj.insert(
                        "aliases".into(),
                        Value::Array(aliases.into_iter().map(Value::String).collect()),
                    );
                    // The fork knows nothing of `resume`; Coral Router tracks it
                    // and overlays the router-side flag on the envelope.
                    obj.insert(
                        "resume".into(),
                        Value::Bool(manager.resume_for(instance_name)),
                    );
                    // Surface the weights-file identity so `coral-router ps`
                    // shows `id`/`arch`/`quant` without the weights file on the
                    // CLI's host. Overlaid per-instance (like `model_bytes`).
                    let idn = manager.weights_identity();
                    obj.insert("short_id".into(), Value::String(idn.short_id.clone()));
                    obj.insert("arch".into(), Value::String(idn.arch.clone()));
                    obj.insert("quant".into(), Value::String(idn.quant.clone()));
                }
                instances.push(entry);
            }
            for snap in envelope.snapshots {
                let mut entry = serde_json::to_value(&snap).unwrap_or_default();
                if let Value::Object(ref mut obj) = entry {
                    obj.insert("model".into(), Value::String(model_key.clone()));
                }
                snapshots.push(entry);
            }
            total.model = total.model.saturating_add(envelope.total.model);
            total.context = total.context.saturating_add(envelope.total.context);
            total.compute = total.compute.saturating_add(envelope.total.compute);
            total.total = total.total.saturating_add(envelope.total.total);
        }
        Ok(serde_json::json!({
            "instances": instances,
            "snapshots": snapshots,
            "total": total,
        }))
    }

    /// `GET /v1/models` - one entry per instance across every managed model,
    /// plus aliases for the bare model, group, and `latest` forms. Plain
    /// (no-instance-grammar) models contribute one synthesized entry.
    pub async fn list_models(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for model_key in self.managers.keys() {
            let Some(manager) = self.managers.get(&model_key).map(|m| m.as_ref().clone()) else {
                continue;
            };
            let Some((envelope, _plain)) = manager.list_with_fallback().await else {
                continue;
            };
            let created = common_core::now_secs();
            for info in envelope.instances {
                let instance_name = instance_name_from_server_id(&info.id);
                let instance_id = format!("{model_key}:{instance_name}");
                let mut entry = serde_json::json!({
                    "id": instance_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "coral-router",
                    "n_ctx": info.n_ctx,
                    "parallel": info.parallel,
                    "pinned": info.pinned,
                    "resume": manager.resume_for(instance_name),
                    "is_default": info.is_default,
                    "state": info.state,
                    "last_used": info.last_used,
                });
                entry["aliases"] = Value::Array(
                    instance_aliases(&model_key, &instance_id, &info.group, info.is_default)
                        .into_iter()
                        .map(Value::String)
                        .collect(),
                );
                out.push(entry);
            }
        }
        out
    }

    /// `POST /instances` - allocate a NEW context on `model_key`'s server.
    /// `resume` is router-side (the fork knows nothing of it): recorded here so
    /// the aggregate reports it and eviction snapshots the context first.
    pub async fn create(
        &self,
        model_key: &str,
        name: &str,
        group: &str,
        ctx_size: u64,
        parallel: Option<u32>,
        pinned: bool,
        is_default: bool,
        resume: bool,
    ) -> Result<InstanceInfo, InstanceError> {
        let manager = self
            .managers
            .get(&model_key.to_string())
            .ok_or_else(|| InstanceError::Rejected {
                status: 404,
                body: format!("unknown model: {model_key}"),
            })?;
        let info = manager
            .client()
            .create(name, group, ctx_size, parallel, pinned, is_default)
            .await?;
        manager.set_resume(name, resume);
        let mut info = info;
        info.resume = resume;
        Ok(info)
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn destroy(&self, model_key: &str, name: &str) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?
            .client()
            .destroy(name, false)
            .await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn pin(&self, model_key: &str, name: &str) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?.client().pin(name).await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn unpin(&self, model_key: &str, name: &str) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?.client().unpin(name).await
    }

    /// Set the preserve-on-evict flag for a context (router-side). Disabling
    /// also deletes any `-resume` snapshot the context left behind - the router
    /// concluding the work is done.
    pub async fn set_resume(
        &self,
        model_key: &str,
        name: &str,
        enabled: bool,
    ) -> Result<(), InstanceError> {
        let manager = self
            .managers
            .get(&model_key.to_string())
            .ok_or_else(|| InstanceError::Rejected {
                status: 404,
                body: format!("unknown model: {model_key}"),
            })?;
        if !enabled {
            let _ = manager
                .client()
                .delete_snapshot(name, &resume_snapshot_name(name))
                .await;
        }
        manager.set_resume(name, enabled);
        crate::audit::emit(
            "instances",
            serde_json::json!({
                "action": "set_resume",
                "instance": format!("{model_key}:{name}"),
                "enabled": enabled,
            }),
        );
        Ok(())
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn resize(
        &self,
        model_key: &str,
        name: &str,
        ctx_size: u64,
    ) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?.client().resize(name, ctx_size).await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn save_snapshot(
        &self,
        model_key: &str,
        instance: &str,
        name: &str,
    ) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?
            .client()
            .save_snapshot(instance, name)
            .await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn list_snapshots(
        &self,
        model_key: &str,
        instance: &str,
    ) -> Result<Vec<SnapshotInfo>, InstanceError> {
        self.manager_checked(model_key)?
            .client()
            .list_snapshots(instance)
            .await
    }

    /// Proxy a per-instance operation to the owning model's server.
    pub async fn delete_snapshot(
        &self,
        model_key: &str,
        instance: &str,
        name: &str,
    ) -> Result<(), InstanceError> {
        self.manager_checked(model_key)?
            .client()
            .delete_snapshot(instance, name)
            .await
    }

    fn manager_checked(
        &self,
        model_key: &str,
    ) -> Result<Arc<InstanceManager>, InstanceError> {
        self.managers
            .get(&model_key.to_string())
            .map(|m| m.as_ref().clone())
            .ok_or_else(|| InstanceError::Rejected {
                status: 404,
                body: format!("unknown model: {model_key}"),
            })
    }
}