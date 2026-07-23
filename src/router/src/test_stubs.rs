use std::collections::VecDeque;
use std::sync::Mutex;

use guidance_llm::client::ChatBackend;
use guidance_llm::{ChatMessage, LlmError};

pub struct StubChatBackend {
    responses: Mutex<VecDeque<String>>,
}

impl StubChatBackend {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }

    pub fn always(response: impl Into<String>) -> Self {
        Self::new(vec![response.into()])
    }
}

impl ChatBackend for StubChatBackend {
    fn chat_complete(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        let mut queue = self.responses.lock().unwrap();
        queue.pop_front().ok_or(LlmError::NoResponse)
    }
}
