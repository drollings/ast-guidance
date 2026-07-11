// Job Copilot — side panel
// Renders the field list with editable suggestions, confidence bars, and action buttons.

(() => {
  "use strict";

  const fieldList = document.getElementById("field-list");
  const statusBadge = document.getElementById("status");
  let currentTabId = null;
  let currentResult = null;

  // ── Initialize ────────────────────────────────────────────────

  async function init() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) {
      showEmpty("No active tab.");
      return;
    }
    currentTabId = tab.id;

    // Request cached analysis from background.
    chrome.runtime.sendMessage(
      { kind: "getAnalysis", tabId: currentTabId },
      (response) => {
        if (chrome.runtime.lastError) {
          showEmpty("Could not connect to background script.");
          return;
        }
        if (response?.analysis) {
          renderAnalysis(response.analysis);
        } else {
          showEmpty("Waiting for form analysis…");
        }
      }
    );
  }

  // ── Rendering ─────────────────────────────────────────────────

  function showEmpty(message) {
    fieldList.innerHTML = `<p class="empty-state">${escapeHtml(message)}</p>`;
    statusBadge.textContent = "Idle";
    statusBadge.className = "status-badge status-idle";
  }

  function renderAnalysis(result) {
    currentResult = result;
    const prefilled = result.prefilled || [];
    const skipped = result.skipped || [];
    const totalFields = prefilled.length + skipped.length;

    if (totalFields === 0) {
      showEmpty("No form fields detected.");
      return;
    }

    statusBadge.textContent = `${prefilled.length} suggested`;
    statusBadge.className = "status-badge status-ok";

    let html = "";

    // Render prefilled fields.
    for (const item of prefilled) {
      html += renderField(item.field_id, item.value, item.confidence, item.source, item.reasoning);
    }

    // Render skipped fields.
    for (const item of skipped) {
      html += renderSkippedField(item.field_id, item.reason, item.suggested_action);
    }

    fieldList.innerHTML = html;
    attachEventListeners();
  }

  function renderField(fieldId, value, confidence, source, reasoning) {
    const pct = Math.round(confidence * 100);
    const sourceClass = source === "resume" ? "source-resume" : "source-llm";
    const sourceLabel = source === "resume" ? "Resume" : "LLM";
    return `
      <div class="field-card" data-field-id="${escapeAttr(fieldId)}">
        <div class="field-header">
          <span class="field-label">${escapeHtml(fieldId)}</span>
          <span class="source-badge ${sourceClass}">${sourceLabel}</span>
        </div>
        <input type="text" class="field-input" value="${escapeAttr(value)}" data-field-id="${escapeAttr(fieldId)}">
        <div class="confidence-bar-container">
          <div class="confidence-bar" style="width: ${pct}%"></div>
          <span class="confidence-label">${pct}%</span>
        </div>
        <div class="field-reasoning">${escapeHtml(reasoning)}</div>
        <div class="field-actions">
          <button class="btn btn-apply" data-field-id="${escapeAttr(fieldId)}">Apply</button>
          <button class="btn btn-skip" data-field-id="${escapeAttr(fieldId)}">Skip</button>
        </div>
      </div>
    `;
  }

  function renderSkippedField(fieldId, reason, suggestedAction) {
    const reasonLabel = {
      sensitive_type: "Sensitive",
      no_match: "No match",
      llm_error: "LLM error",
      llm_refused: "LLM refused",
      context_too_untrusted: "Untrusted context",
    }[reason] || reason;

    return `
      <div class="field-card field-skipped" data-field-id="${escapeAttr(fieldId)}">
        <div class="field-header">
          <span class="field-label">${escapeHtml(fieldId)}</span>
          <span class="source-badge source-skip">${reasonLabel}</span>
        </div>
        <div class="field-skipped-note">${escapeHtml(suggestedAction)}</div>
      </div>
    `;
  }

  // ── Event handling ────────────────────────────────────────────

  function attachEventListeners() {
    // Apply buttons.
    fieldList.querySelectorAll(".btn-apply").forEach((btn) => {
      btn.addEventListener("click", () => {
        const fieldId = btn.dataset.fieldId;
        const input = fieldList.querySelector(
          `.field-input[data-field-id="${fieldId}"]`
        );
        if (input) {
          applyValue(fieldId, input.value);
        }
      });
    });

    // Skip buttons.
    fieldList.querySelectorAll(".btn-skip").forEach((btn) => {
      btn.addEventListener("click", () => {
        const fieldId = btn.dataset.fieldId;
        sendFeedback(fieldId, "skipped", "");
      });
    });

    // Track edits for feedback.
    fieldList.querySelectorAll(".field-input").forEach((input) => {
      input.addEventListener("change", () => {
        const fieldId = input.dataset.fieldId;
        sendFeedback(fieldId, "overridden", input.value);
      });
    });
  }

  function applyValue(fieldId, value) {
    chrome.tabs.sendMessage(currentTabId, {
      kind: "apply",
      field_id: fieldId,
      value,
    });
    sendFeedback(fieldId, "accepted", value);
  }

  async function sendFeedback(fieldId, action, value) {
    const finalValueHash = value ? await sha256hex(value) : "";
    chrome.runtime.sendMessage({
      kind: "feedback",
      params: {
        request_id: currentResult?.request_id || "",
        field_id: fieldId,
        action,
        final_value_hash: finalValueHash,
      },
    });
  }

  // ── Utilities ─────────────────────────────────────────────────

  async function sha256hex(text) {
    const encoder = new TextEncoder();
    const data = encoder.encode(text);
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function escapeAttr(str) {
    return str.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // ── Boot ──────────────────────────────────────────────────────

  init();
})();
