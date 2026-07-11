// Job Copilot — background service worker (MV3)
// Handles native messaging, hotkey dispatch, and side panel management.

const NATIVE_HOST = "io.github.anomalyco.job_copilot";

// Cache of last analysis per tab for side panel restoration.
const lastAnalysisByTabId = new Map();

// ── Install ─────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  console.log(`Job Copilot v${chrome.runtime.getManifest().version} installed`);
});

// ── Hotkey handling ─────────────────────────────────────────────

chrome.commands.onCommand.addListener(async (command) => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;
  chrome.tabs.sendMessage(tab.id, { kind: "hotkey", action: command });
});

// ── Message routing ─────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.kind === "analyze") {
    // Forward to native host.
    chrome.runtime.sendNativeMessage(NATIVE_HOST, msg.payload, (response) => {
      if (chrome.runtime.lastError) {
        console.error("Native host error:", chrome.runtime.lastError.message);
        sendResponse({ error: chrome.runtime.lastError.message });
        return;
      }
      // Cache the analysis for this tab.
      if (sender.tab?.id) {
        lastAnalysisByTabId.set(sender.tab.id, response);
      }
      // Relay response to content script.
      chrome.tabs.sendMessage(sender.tab.id, {
        kind: "analysisResult",
        requestId: msg.payload.request_id,
        result: response,
      });
      sendResponse({ ok: true });
    });
    return true; // async sendResponse
  }

  if (msg.kind === "feedback") {
    // Forward feedback to native host.
    chrome.runtime.sendNativeMessage(
      NATIVE_HOST,
      {
        jsonrpc: "2.0",
        id: crypto.randomUUID(),
        method: "session.feedback",
        params: msg.params,
      },
      () => {
        if (chrome.runtime.lastError) {
          console.error("Feedback error:", chrome.runtime.lastError.message);
        }
      }
    );
    return false;
  }

  if (msg.kind === "getAnalysis") {
    // Side panel requesting cached analysis for a tab.
    const analysis = lastAnalysisByTabId.get(msg.tabId) || null;
    sendResponse({ analysis });
    return false;
  }
});

// ── Side panel open on action click ─────────────────────────────

chrome.action.onClicked.addListener((tab) => {
  if (tab.id) {
    chrome.sidePanel.open({ tabId: tab.id });
  }
});
