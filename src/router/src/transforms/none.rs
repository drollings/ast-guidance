use crate::transforms::{TransformError, TransformStrategy};
use crate::types::RouterRequest;

pub struct NoTransform;

impl TransformStrategy for NoTransform {
    fn name(&self) -> &str {
        "none"
    }

    fn transform(&self, request: &RouterRequest) -> Result<RouterRequest, TransformError> {
        Ok(request.clone())
    }
}
