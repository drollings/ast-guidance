use crate::transforms::{TransformError, TransformStrategy};
use crate::types::RouterRequest;

pub struct NoTransform;

impl TransformStrategy for NoTransform {
    fn name(&self) -> &str {
        "none"
    }

    fn transform(
        &self,
        request: &RouterRequest,
        _pii_classes: &[String],
    ) -> Result<RouterRequest, TransformError> {
        Ok(request.clone())
    }
}
